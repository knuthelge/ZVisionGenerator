import { flushSync } from 'svelte';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { galleryStore } from '$lib/state/gallery.svelte';
import { draft } from '$lib/state/draft.svelte';
import { jobStore } from '$lib/state/job.svelte';
import { galleryMemory, workspaceMemory } from '$lib/state/pageMemory';
import { projectPick } from '$lib/state/projectPick.svelte';
import { projectsStore } from '$lib/state/projects.svelte';
import { hasPendingReference, takeReference } from '$lib/state/pendingReference.svelte';
import type { GalleryAsset, GalleryPage as GalleryPageResponse, MoveResult, SavedVoice, WorkspaceContext } from '$lib/types';

const galleryApiMocks = vi.hoisted(() => ({
  getGallery: vi.fn<() => Promise<GalleryPageResponse>>(),
  deleteAsset: vi.fn<(assetId: string) => Promise<void>>(),
  moveAssets: vi.fn<(ids: string[], project: string | null) => Promise<MoveResult>>(),
}));

const projectApiMocks = vi.hoisted(() => ({ listProjects: vi.fn(), createProject: vi.fn() }));
vi.mock('$lib/api/projects', () => projectApiMocks);
const workspaceApiMocks = vi.hoisted(() => ({
  getWorkspaceCoreContext: vi.fn<() => Promise<WorkspaceContext>>(),
}));

vi.mock('$lib/api/workspace', async (importOriginal) => ({ ...(await importOriginal<typeof import('$lib/api/workspace')>()), getWorkspaceCoreContext: workspaceApiMocks.getWorkspaceCoreContext }));

const routerMocks = vi.hoisted(() => ({
  params: {} as Record<string, string>,
  replace: vi.fn<(page: string, params?: Record<string, string>) => void>(),
  navigate: vi.fn<(page: string, params?: Record<string, string>) => void>(),
}));

const toastMocks = vi.hoisted(() => ({
  addToast: vi.fn<(message: string, tone: 'success' | 'warning' | 'error', options?: { action?: { label: string; run: () => void } }) => string | undefined>(),
  dismissToast: vi.fn<(id: string) => void>(),
}));

const confirmMocks = vi.hoisted(() => ({
  requestConfirm: vi.fn(async () => true),
}));

vi.mock('$lib/components/molecules/confirm.svelte', () => confirmMocks);

// A test sets the layout before it mounts the page.
const layoutState = vi.hoisted(() => ({ mode: 'wide' as 'wide' | 'tall' | 'compact' }));
vi.mock('$lib/state/layoutMode.svelte', async (importOriginal) => {
  const actual = await importOriginal<typeof import('$lib/state/layoutMode.svelte')>();
  return {
    ...actual,
    layout: new Proxy(actual.layout, {
      get: (target, key) => (key === 'mode' ? layoutState.mode : key === 'compact' ? layoutState.mode === 'compact' : Reflect.get(target, key)),
    }),
  };
});
const voiceApiMocks = vi.hoisted(() => ({
  saveRecordingVoice: vi.fn<(form: FormData) => Promise<SavedVoice>>(),
}));

vi.mock('$lib/api/voices', async (importOriginal) => ({ ...(await importOriginal<typeof import('$lib/api/voices')>()), saveRecordingVoice: voiceApiMocks.saveRecordingVoice }));

beforeEach(() => {
  confirmMocks.requestConfirm.mockReset();
  confirmMocks.requestConfirm.mockResolvedValue(true);
});

vi.mock('$lib/api/gallery', async (importOriginal) => {
  const actual = await importOriginal<typeof import('$lib/api/gallery')>();
  return {
    ...actual,
    getGallery: galleryApiMocks.getGallery,
    deleteAsset: galleryApiMocks.deleteAsset,
    moveAssets: galleryApiMocks.moveAssets,
  };
});

vi.mock('$lib/state/router.svelte', () => ({
  router: {
    get params(): Record<string, string> {
      return routerMocks.params;
    },
    replace: routerMocks.replace,
    navigate: routerMocks.navigate,
  },
}));

vi.mock('$lib/state/toasts.svelte', () => ({
  addToast: toastMocks.addToast,
  dismissToast: toastMocks.dismissToast,
}));

import GalleryPage from './GalleryPage.svelte';
import { makeAudioAsset, makeImportedAudio, makeSpeechAsset, withAudio } from '../../test-utils/audio';
import { mountComponent, settle } from '../../test-utils/mount';
import { optionLabels, pickOption } from '../../test-utils/select';
import { makeContext, makeImageDefaults, makeVideoDefaults } from '../../test-utils/workspace';

class MockIntersectionObserver {
  static instances: MockIntersectionObserver[] = [];

  observe = vi.fn();
  disconnect = vi.fn();

  constructor(private readonly callback: IntersectionObserverCallback) {
    MockIntersectionObserver.instances.push(this);
  }

  trigger(isIntersecting: boolean): void {
    this.callback([{ isIntersecting } as IntersectionObserverEntry], this as unknown as IntersectionObserver);
  }
}

function makeAsset(overrides: Partial<GalleryAsset> = {}): GalleryAsset {
  return {
    id: 'nested/asset.png',
    url: '/media/asset.png',
    thumbnail_url: '/media/thumb.png',
    filename: 'asset.png',
    created_at: '2026-04-24T12:00:00Z',
    project: null,
    workflow: 'txt2img',
    prompt: 'A calm shoreline at dusk',
    model: 'zit',
    reuse_workspace_url: '#/workspace?workflow=txt2img&prompt=A%20calm%20shoreline%20at%20dusk',
    has_reusable_config: true,
    media_type: 'image',
    ...overrides,
  };
}

/** The open asset viewer, which is where the active asset's details now live. */
function getViewer(): HTMLElement | null {
  return document.querySelector('[data-testid="asset-viewer"]');
}

function queryButtonByName(container: ParentNode, name: string): HTMLButtonElement | null {
  return Array.from(container.querySelectorAll('button')).find((button) => button.textContent?.trim() === name) ?? null;
}

function deferred<T>(): { promise: Promise<T>; resolve: (value: T) => void; reject: (reason?: unknown) => void } {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, resolve, reject };
}

function selectAssetForBatch(container: ParentNode, asset: GalleryAsset): HTMLInputElement {
  const checkbox = container.querySelector(`input[aria-label="Select ${asset.filename}"]`) as HTMLInputElement | null;
  expect(checkbox).not.toBeNull();
  checkbox!.checked = true;
  checkbox!.dispatchEvent(new Event('change', { bubbles: true }));
  return checkbox!;
}

function deleteAssetFromCard(container: ParentNode, asset: GalleryAsset): void {
  const more = container.querySelector(`button[aria-label="More actions for ${asset.filename}"]`) as HTMLButtonElement | null;
  expect(more).not.toBeNull();
  more!.dispatchEvent(new MouseEvent('click', { bubbles: true }));
  flushSync();
  const button = document.querySelector('[role="menu"] [data-action="delete"]') as HTMLButtonElement | null;
  expect(button).not.toBeNull();
  button!.dispatchEvent(new MouseEvent('click', { bubbles: true }));
}

let page: ReturnType<typeof mountComponent>;
let target: HTMLElement;

function mountPage(): void {
  page = mountComponent(GalleryPage);
  target = page.target;
}

/** Click the tile of *filename* to open it in the viewer. */
async function openViewer(filename: string): Promise<void> {
  (target.querySelector(`button[aria-label="View ${filename}"]`) as HTMLElement).dispatchEvent(new MouseEvent('click', { bubbles: true }));
  await settle();
}

beforeEach(() => {
  layoutState.mode = 'wide';
  projectPick.set(null);
  galleryMemory.forget();
  galleryStore.reset();
  galleryApiMocks.getGallery.mockReset();
  galleryApiMocks.deleteAsset.mockReset();
  galleryApiMocks.moveAssets.mockReset();
  projectApiMocks.listProjects.mockReset().mockResolvedValue([{ name: 'Night Bus', path: '/out/Night Bus' }, { name: 'Zoo', path: '/out/Zoo' }]);
  workspaceApiMocks.getWorkspaceCoreContext.mockReset();
  workspaceApiMocks.getWorkspaceCoreContext.mockRejectedValue(new Error('offline'));
  workspaceMemory.forget();
  toastMocks.addToast.mockReset();
  toastMocks.dismissToast.mockReset();
  routerMocks.params = {};
  routerMocks.replace.mockReset();
  routerMocks.navigate.mockReset();
  // Show the viewer's details panel, where the active asset's prompt and facts appear.
  localStorage.setItem('ziv-viewer-details-v1', 'true');
  MockIntersectionObserver.instances = [];
  globalThis.IntersectionObserver = MockIntersectionObserver as unknown as typeof IntersectionObserver;
});

describe('GalleryPage active detail selection behavior', () => {
  it('keeps active detail separate from batch selection until the checkbox is toggled', async () => {
    const asset = makeAsset();
    galleryApiMocks.getGallery.mockResolvedValue({
      assets: [asset],
      page: 1,
      total_pages: 1,
      total_count: 1,
    });

    mountPage();
    await settle();

    const card = target.querySelector(`button[aria-label="View ${asset.filename}"]`) as HTMLElement | null;
    const checkbox = target.querySelector(`input[aria-label="Select ${asset.filename}"]`) as HTMLInputElement | null;

    expect(card).not.toBeNull();
    expect(checkbox).not.toBeNull();
    expect(checkbox?.checked).toBe(false);

    card!.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    await settle();

    expect(checkbox?.checked).toBe(false);
    expect(target.textContent).toContain(asset.prompt);
  });

});

describe('GalleryPage regressions', () => {
  it('threads filter and sort state through backend gallery loads and pagination', async () => {
    const asset = makeAsset();
    const filteredAsset = makeAsset({ id: 'nested/video-1.mp4', filename: 'video-1.mp4', media_type: 'video' });
    const pagedAsset = makeAsset({ id: 'nested/video-2.mp4', filename: 'video-2.mp4', media_type: 'video' });
    galleryApiMocks.getGallery
      .mockResolvedValueOnce({ assets: [asset], page: 1, total_pages: 2, total_count: 3 })
      .mockResolvedValueOnce({ assets: [filteredAsset], page: 1, total_pages: 1, total_count: 1 })
      .mockResolvedValueOnce({ assets: [filteredAsset], page: 1, total_pages: 2, total_count: 2 })
      .mockResolvedValueOnce({ assets: [pagedAsset], page: 2, total_pages: 2, total_count: 2 });

    mountPage();
    await settle();

    const filterSelect = target.querySelector('[role="combobox"][aria-label="Filter gallery media"]') as HTMLButtonElement | null;
    const sortSelect = target.querySelector('[role="combobox"][aria-label="Sort gallery assets"]') as HTMLButtonElement | null;
    expect(filterSelect).not.toBeNull();
    expect(sortSelect).not.toBeNull();
    expect(galleryApiMocks.getGallery).toHaveBeenNthCalledWith(1, 1, 'all', 'newest', null);

    pickOption(filterSelect!, 'video');
    await settle();
    expect(galleryApiMocks.getGallery).toHaveBeenNthCalledWith(2, 1, 'video', 'newest', null);
    expect(filterSelect!.dataset.value).toBe('video');

    pickOption(sortSelect!, 'oldest');
    await settle();
    expect(galleryApiMocks.getGallery).toHaveBeenNthCalledWith(3, 1, 'video', 'oldest', null);
    expect(sortSelect!.dataset.value).toBe('oldest');

    MockIntersectionObserver.instances.at(-1)?.trigger(true);
    await settle();

    expect(galleryApiMocks.getGallery).toHaveBeenNthCalledWith(4, 2, 'video', 'oldest', null);
    expect(target.textContent).toContain('video-2.mp4');
  });

  it('navigates to workspace reuse through parsed router params', async () => {
    const asset = makeAsset({ reuse_workspace_url: '#/workspace?workflow=img2img&prompt=Reuse%20me&model=zit&seed=9876' });
    galleryApiMocks.getGallery.mockResolvedValue({
      assets: [asset],
      page: 1,
      total_pages: 1,
      total_count: 1,
    });

    mountPage();
    await settle();

    await openViewer(asset.filename);

    const reuseButton = (getViewer()?.querySelector('[data-action="reuse"]') as HTMLButtonElement | null);
    expect(reuseButton).not.toBeNull();

    reuseButton!.dispatchEvent(new MouseEvent('click', { bubbles: true }));

    expect(routerMocks.navigate).toHaveBeenCalledWith('workspace', {
      workflow: 'img2img',
      prompt: 'Reuse me',
      model: 'zit',
      seed: '9876',
      reuse: 'true',
    });
  });

  it('sends an image to the workspace as a first frame with its prompt as the scene through router params', async () => {
    const asset = makeAsset({ file_path: '/outputs/nested/asset.png' });
    galleryApiMocks.getGallery.mockResolvedValue({ assets: [asset], page: 1, total_pages: 1, total_count: 1 });

    mountPage();
    await settle();

    (target.querySelector(`button[aria-label="More actions for ${asset.filename}"]`) as HTMLButtonElement).click();
    flushSync();
    (document.querySelector('[role="menu"] [data-action="reference-video"]') as HTMLButtonElement).click();

    expect(routerMocks.navigate).toHaveBeenCalledWith('workspace', { workflow: 'img2vid', image_path: '/outputs/nested/asset.png', use_as: 'start', scene: asset.prompt });
  });

  it('queues Use as -> Reference and opens the Workspace without params, which would replace the list', async () => {
    const asset = makeAsset({ file_path: '/outputs/nested/asset.png' });
    galleryApiMocks.getGallery.mockResolvedValue({ assets: [asset], page: 1, total_pages: 1, total_count: 1 });
    takeReference();

    mountPage();
    await settle();

    (target.querySelector(`button[aria-label="More actions for ${asset.filename}"]`) as HTMLButtonElement).click();
    flushSync();
    (document.querySelector('[role="menu"] [data-action="reference-reference"]') as HTMLButtonElement).click();

    expect(routerMocks.navigate).toHaveBeenCalledWith('workspace');
    expect(hasPendingReference()).toBe(true);
    expect(takeReference()).toBe('/outputs/nested/asset.png');
  });

  it('shows the Reference and Last frame hints from the remembered Workspace', async () => {
    const asset = makeAsset({ file_path: '/outputs/nested/asset.png' });
    galleryApiMocks.getGallery.mockResolvedValue({ assets: [asset], page: 1, total_pages: 1, total_count: 1 });
    draft.reset();
    const context = makeContext({
      image_model_defaults: { zit: makeImageDefaults({ max_reference_images: 3 }), 'flux-lite': makeImageDefaults() },
      video_model_defaults: { 'ltx-8': makeVideoDefaults({ hidden_controls: ['end_image_path'] }) },
    });
    workspaceMemory.set(context);

    mountPage();
    await settle();

    (target.querySelector(`button[aria-label="More actions for ${asset.filename}"]`) as HTMLButtonElement).click();
    flushSync();
    const reference = document.querySelector('[role="menu"] [data-action="reference-reference"]') as HTMLElement;
    const last = document.querySelector('[role="menu"] [data-action="reference-last-frame"]') as HTMLElement;
    expect(reference.textContent).toContain('as image 1');
    expect(last.getAttribute('aria-disabled')).toBe('true');
    expect(last.textContent).toContain('ltx-8 has no last frame');
    workspaceMemory.forget();
  });

  it('clears the start image and the last frame of the draft when their asset is deleted', async () => {
    const asset = makeAsset({ file_path: '/outputs/nested/asset.png' });
    galleryStore.seed([asset]);
    galleryApiMocks.getGallery.mockResolvedValue({ assets: [asset], page: 1, total_pages: 1, total_count: 1 });
    galleryApiMocks.deleteAsset.mockResolvedValue(undefined);
    draft.reset();
    draft.setStartImage('image', { path: '/outputs/nested/asset.png' }, null);
    draft.setLastFrame({ path: '/outputs/nested/asset.png' });

    mountPage();
    await settle();
    deleteAssetFromCard(target, asset);
    await settle();

    expect(draft.state).toMatchObject({ referenceImagePath: null, endImagePath: null, workflow: 'txt2img' });
    draft.reset();
  });

  it('clears the Video first frame of the draft when its asset is deleted', async () => {
    const asset = makeAsset({ file_path: '/outputs/nested/asset.png' });
    galleryStore.seed([asset]);
    galleryApiMocks.getGallery.mockResolvedValue({ assets: [asset], page: 1, total_pages: 1, total_count: 1 });
    galleryApiMocks.deleteAsset.mockResolvedValue(undefined);
    draft.reset();
    draft.setTab('video');
    draft.setStartImage('video', { path: '/outputs/nested/asset.png' }, null);
    expect(draft.state).toMatchObject({ firstFramePath: '/outputs/nested/asset.png', workflow: 'img2vid' });

    mountPage();
    await settle();
    deleteAssetFromCard(target, asset);
    await settle();

    expect(draft.state).toMatchObject({ firstFramePath: null, workflow: 'txt2vid' });
    draft.reset();
  });

  it('leaves the first frame and the other images of the draft alone when another asset is deleted', async () => {
    const used = makeAsset({ id: 'used.png', filename: 'used.png', file_path: '/outputs/used.png' });
    const other = makeAsset({ id: 'other.png', filename: 'other.png', file_path: '/outputs/other.png' });
    galleryStore.seed([used, other]);
    galleryApiMocks.getGallery.mockResolvedValue({ assets: [used, other], page: 1, total_pages: 1, total_count: 2 });
    galleryApiMocks.deleteAsset.mockResolvedValue(undefined);
    draft.reset();
    draft.setStartImage('image', { path: '/outputs/used.png' }, null);
    draft.setTab('video');
    draft.setStartImage('video', { path: '/outputs/used.png' }, null);
    draft.setLastFrame({ path: '/outputs/used.png' });

    mountPage();
    await settle();
    deleteAssetFromCard(target, other);
    await settle();

    expect(draft.state).toMatchObject({ referenceImagePath: '/outputs/used.png', firstFramePath: '/outputs/used.png', endImagePath: '/outputs/used.png', workflow: 'img2vid' });
    draft.reset();
  });

  it('clears the first frame while the Image tab is open, and the Image tab start image stays', async () => {
    const asset = makeAsset({ file_path: '/outputs/nested/asset.png' });
    galleryStore.seed([asset]);
    galleryApiMocks.getGallery.mockResolvedValue({ assets: [asset], page: 1, total_pages: 1, total_count: 1 });
    galleryApiMocks.deleteAsset.mockResolvedValue(undefined);
    draft.reset();
    draft.setTab('video');
    draft.setStartImage('video', { path: '/outputs/nested/asset.png' }, null);
    draft.setTab('image');

    mountPage();
    await settle();
    deleteAssetFromCard(target, asset);
    await settle();

    expect(draft.state).toMatchObject({ firstFramePath: null, workflow: 'txt2img' });
    draft.setTab('video');
    expect(draft.state.workflow).toBe('txt2vid');
    draft.reset();
  });

  it('sends a video to the Extend workflow from its menu, with its model, and leaves the text to the source', async () => {
    const asset = makeAsset({
      filename: 'walk.mp4', media_type: 'video', workflow: 'txt2vid', file_path: '/outputs/walk.mp4', scene: 'A street at night.', prompt: 'She waves.',
      lora: 'style:0.8', reuse_state: {
        requested_workflow: 'txt2vid', resolved_workflow: 'txt2vid', workflow_available: true, requested_model: 'ltx-8', resolved_model: 'ltx-8', model_available: true, fallback_reasons: [],
      },
      extend: { available: true, reason: null },
    });
    galleryApiMocks.getGallery.mockResolvedValue({ assets: [asset], page: 1, total_pages: 1, total_count: 1 });

    mountPage();
    await settle();

    (target.querySelector(`button[aria-label="More actions for ${asset.filename}"]`) as HTMLButtonElement).click();
    flushSync();
    (document.querySelector('[role="menu"] [data-action="extend"]') as HTMLButtonElement).click();

    expect(routerMocks.navigate).toHaveBeenCalledWith('workspace', {
      workflow: 'extend', source_video: '/outputs/walk.mp4', model: 'ltx-8', lora: 'style:0.8',
    });
    expect(toastMocks.addToast).toHaveBeenCalledWith(expect.any(String), 'success');
  });

  it('extends a video from the viewer, and offers no Extend for an image', async () => {
    const video = makeAsset({ id: 'out/walk.mp4', filename: 'walk.mp4', media_type: 'video', file_path: '/outputs/walk.mp4', prompt: 'She waves.', extend: { available: true, reason: null } });
    const image = makeAsset({ id: 'out/a.png', filename: 'a.png', file_path: '/outputs/a.png' });
    galleryApiMocks.getGallery.mockResolvedValue({ assets: [video, image], page: 1, total_pages: 1, total_count: 2 });

    mountPage();
    await settle();
    await openViewer(video.filename);

    // The gallery's prompt may only be a display fallback; the text comes from the server once it has read the source.
    (getViewer()!.querySelector('[data-action="extend"]') as HTMLButtonElement).click();
    expect(routerMocks.navigate).toHaveBeenCalledWith('workspace', { workflow: 'extend', source_video: '/outputs/walk.mp4', lora: '' });

    (target.querySelector(`button[aria-label="More actions for ${image.filename}"]`) as HTMLButtonElement).click();
    flushSync();
    expect(document.querySelector('[role="menu"] [data-action="extend"]')).toBeNull();
  });

  it('opens an image for editing from the viewer and from its menu, and offers no Edit for a video', async () => {
    const image = makeAsset({
      id: 'out/a.png', filename: 'a.png', file_path: '/outputs/a.png', lora: 'detail:0.8', prompt: 'A calm shoreline',
      reuse_state: { requested_workflow: 'txt2img', resolved_workflow: 'txt2img', workflow_available: true, requested_model: 'klein', resolved_model: 'klein', model_available: true, fallback_reasons: [] },
    });
    const video = makeAsset({ id: 'out/walk.mp4', filename: 'walk.mp4', media_type: 'video', file_path: '/outputs/walk.mp4' });
    galleryApiMocks.getGallery.mockResolvedValue({ assets: [image, video], page: 1, total_pages: 1, total_count: 2 });
    const params = { workflow: 'edit', edit_source_path: '/outputs/a.png', use_as: 'edit', model: 'klein', lora: 'detail:0.8' };

    mountPage();
    await settle();
    (target.querySelector(`button[aria-label="More actions for ${image.filename}"]`) as HTMLButtonElement).click();
    flushSync();
    (document.querySelector('[role="menu"] [data-action="edit"]') as HTMLButtonElement).click();
    expect(routerMocks.navigate).toHaveBeenCalledWith('workspace', params);
    routerMocks.navigate.mockClear();

    await openViewer(image.filename);
    (getViewer()!.querySelector('[data-action="edit"]') as HTMLButtonElement).click();
    expect(routerMocks.navigate).toHaveBeenCalledWith('workspace', params);

    // M does the same from the keyboard; no prompt travels, so the Workspace starts it empty.
    routerMocks.navigate.mockClear();
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'm', bubbles: true }));
    expect(routerMocks.navigate).toHaveBeenCalledWith('workspace', params);
    expect(Object.keys(routerMocks.navigate.mock.calls[0][1] ?? {})).not.toContain('prompt');
  });

  it('removes a deleted asset from the workspace history too', async () => {
    const asset = makeAsset();
    galleryStore.seed([asset]);
    galleryApiMocks.getGallery.mockResolvedValue({ assets: [asset], page: 1, total_pages: 1, total_count: 1 });
    galleryApiMocks.deleteAsset.mockResolvedValue(undefined);
    confirmMocks.requestConfirm.mockResolvedValue(true);

    mountPage();
    await settle();
    deleteAssetFromCard(target, asset);
    await settle();

    expect(galleryStore.assets).toEqual([]);
    vi.restoreAllMocks();
  });

  it('keeps the viewer on later pages after a delete and refills once it closes', async () => {
    const pageOne = [makeAsset({ id: 'p1.png', filename: 'p1.png' })];
    const pageTwo = [makeAsset({ id: 'p2a.png', filename: 'p2a.png' }), makeAsset({ id: 'p2b.png', filename: 'p2b.png' })];
    galleryApiMocks.getGallery
      .mockResolvedValueOnce({ assets: pageOne, page: 1, total_pages: 2, total_count: 3 })
      .mockResolvedValueOnce({ assets: pageTwo, page: 2, total_pages: 2, total_count: 3 })
      .mockResolvedValue({ assets: pageOne, page: 1, total_pages: 2, total_count: 2 });
    galleryApiMocks.deleteAsset.mockResolvedValue(undefined);
    confirmMocks.requestConfirm.mockResolvedValue(true);

    mountPage();
    await settle();
    MockIntersectionObserver.instances.at(-1)?.trigger(true);
    await settle();
    (target.querySelector('button[aria-label="View p2a.png"]') as HTMLButtonElement).click();
    await settle();

    (getViewer()!.querySelector('[data-action="delete"]') as HTMLButtonElement).click();
    await settle();

    expect(getViewer()?.textContent).toContain('p2b.png');
    expect(galleryApiMocks.getGallery).toHaveBeenCalledTimes(2);

    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
    await settle();
    expect(getViewer()).toBeNull();
    expect(galleryApiMocks.getGallery).toHaveBeenCalledTimes(3);
    expect(galleryApiMocks.getGallery).toHaveBeenLastCalledWith(1, 'all', 'newest', null);
    vi.restoreAllMocks();
  });

  it('stays on the open asset when a new output lands at the front of the list', async () => {
    const [a, b, c] = ['a', 'b', 'c'].map((name) => makeAsset({ id: `${name}.png`, filename: `${name}.png` }));
    galleryApiMocks.getGallery.mockResolvedValue({ assets: [a, b, c], page: 1, total_pages: 1, total_count: 3 });

    mountPage();
    await settle();
    await openViewer('b.png');
    const title = () => document.querySelector('#asset-viewer-title')?.textContent;
    expect(title()).toBe('b.png');

    galleryStore.mergeOutputs([makeAsset({ id: 'new.png', filename: 'new.png' })]);
    await settle();

    expect(target.querySelector('button[aria-label="View new.png"]')).not.toBeNull();
    expect(title()).toBe('b.png');
    expect(document.querySelector('.viewer-title small')?.textContent).toContain('3 of 4');
  });

  it('keeps paging in the viewer after a delete without skipping shifted assets', async () => {
    const [a, b, c, d, e, f, g] = ['a', 'b', 'c', 'd', 'e', 'f', 'g'].map((name) => makeAsset({ id: `${name}.png`, filename: `${name}.png` }));
    galleryApiMocks.getGallery
      .mockResolvedValueOnce({ assets: [a, b, c, d, e], page: 1, total_pages: 2, total_count: 7 })
      // After deleting a, the server's page one starts at b and pulls f forward from page two.
      .mockResolvedValueOnce({ assets: [b, c, d, e, f], page: 1, total_pages: 2, total_count: 6 })
      .mockResolvedValueOnce({ assets: [g], page: 2, total_pages: 2, total_count: 6 });
    galleryApiMocks.deleteAsset.mockResolvedValue(undefined);
    confirmMocks.requestConfirm.mockResolvedValue(true);
    const press = async (key: string): Promise<void> => {
      document.dispatchEvent(new KeyboardEvent('keydown', { key, bubbles: true }));
      await settle();
    };

    mountPage();
    await settle();
    (target.querySelector('button[aria-label="View a.png"]') as HTMLButtonElement).click();
    await settle();
    (getViewer()!.querySelector('[data-action="delete"]') as HTMLButtonElement).click();
    await settle();
    for (let step = 0; step < 4; step += 1) await press('ArrowRight');

    expect(galleryApiMocks.getGallery).toHaveBeenNthCalledWith(2, 1, 'all', 'newest', null);
    expect(galleryApiMocks.getGallery).toHaveBeenNthCalledWith(3, 2, 'all', 'newest', null);
    expect(Array.from(document.querySelectorAll('[data-film-index]')).map((el) => el.getAttribute('aria-label'))).toEqual(
      ['b', 'c', 'd', 'e', 'f', 'g'].map((name) => `Show ${name}.png`)
    );
    vi.restoreAllMocks();
  });

  it('shows a recovery action when the current filter has no results', async () => {
    const asset = makeAsset();
    galleryApiMocks.getGallery
      .mockResolvedValueOnce({ assets: [asset], page: 1, total_pages: 1, total_count: 1 })
      .mockResolvedValueOnce({ assets: [], page: 1, total_pages: 1, total_count: 0 })
      .mockResolvedValueOnce({ assets: [asset], page: 1, total_pages: 1, total_count: 1 });

    mountPage();
    await settle();

    const filterSelect = target.querySelector('[role="combobox"][aria-label="Filter gallery media"]') as HTMLButtonElement | null;
    pickOption(filterSelect!, 'video');
    await settle();

    expect(target.textContent).toContain('No matching assets');

    const clearButton = Array.from(target.querySelectorAll('button')).find((button) => button.textContent?.trim() === 'Show all media');
    expect(clearButton).not.toBeUndefined();
    clearButton!.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    await settle();

    expect(galleryApiMocks.getGallery).toHaveBeenNthCalledWith(3, 1, 'all', 'newest', null);
    expect(filterSelect!.dataset.value).toBe('all');
  });

  it('shows a workspace recovery action when the gallery is genuinely empty', async () => {
    galleryApiMocks.getGallery.mockResolvedValue({ assets: [], page: 1, total_pages: 1, total_count: 0 });

    mountPage();
    await settle();

    expect(target.textContent).toContain('No generated assets yet');

    const workspaceButton = Array.from(target.querySelectorAll('button')).find((button) => button.textContent?.trim() === 'Open Workspace');
    expect(workspaceButton).not.toBeUndefined();
    workspaceButton!.dispatchEvent(new MouseEvent('click', { bubbles: true }));

    expect(routerMocks.navigate).toHaveBeenCalledWith('workspace');
  });

  it('disconnects the infinite-scroll observer when the page unmounts', async () => {
    const asset = makeAsset();
    galleryApiMocks.getGallery.mockResolvedValue({
      assets: [asset],
      page: 1,
      total_pages: 2,
      total_count: 2,
    });

    mountPage();
    await settle();

    const observer = MockIntersectionObserver.instances.at(-1);
    expect(observer).toBeDefined();
    expect(observer?.observe).toHaveBeenCalled();

    await page.unmount();

    expect(observer?.disconnect).toHaveBeenCalled();
  });

  it('shows backend reuse reasons in the selected asset panel', async () => {
    const asset = makeAsset({
      reuse_state: {
        requested_workflow: 'img2img',
        resolved_workflow: 'txt2img',
        workflow_available: false,
        requested_model: 'missing-model',
        resolved_model: 'zit',
        model_available: false,
        fallback_reasons: ['workflow_media_mismatch', 'model_not_configured'],
      },
    });
    galleryApiMocks.getGallery.mockResolvedValue({
      assets: [asset],
      page: 1,
      total_pages: 1,
      total_count: 1,
    });

    mountPage();
    await settle();

    await openViewer(asset.filename);

    const notice = getViewer()?.querySelector('[role="note"]');
    expect(notice).not.toBeNull();
    expect(notice?.querySelectorAll('li')).toHaveLength(2);
  });

  it('shows a display prompt without offering reuse when embedded config is missing', async () => {
    const asset = makeAsset({
      model: 'Unavailable',
      prompt: 'Display-only prompt',
      reuse_workspace_url: '#/workspace?workflow=txt2img',
      has_reusable_config: false,
      reuse_state: {
        requested_workflow: 'txt2img',
        resolved_workflow: 'txt2img',
        workflow_available: true,
        requested_model: null,
        resolved_model: null,
        model_available: true,
        fallback_reasons: [],
      },
    });
    galleryApiMocks.getGallery.mockResolvedValue({
      assets: [asset],
      page: 1,
      total_pages: 1,
      total_count: 1,
    });

    mountPage();
    await settle();

    await openViewer(asset.filename);

    expect(getViewer()?.querySelector('[role="note"]')).toBeNull();
    expect(getViewer()?.textContent).toContain('Unavailable');
    expect(getViewer()?.textContent).toContain('Display-only prompt');
    expect((getViewer()?.querySelector('[data-action="reuse"]') as HTMLButtonElement | null)?.disabled).toBe(true);
    expect(routerMocks.navigate).not.toHaveBeenCalled();
  });

});

describe('GalleryPage request ownership (F07)', () => {
  it('commits only the latest replacement request when older loads resolve or reject afterwards', async () => {
    const initial = deferred<GalleryPageResponse>();
    const filtered = deferred<GalleryPageResponse>();
    const currentAsset = makeAsset({ id: 'current.mp4', filename: 'current.mp4', media_type: 'video' });
    galleryApiMocks.getGallery.mockReturnValueOnce(initial.promise).mockReturnValueOnce(filtered.promise);

    mountPage();
    const filterSelect = target.querySelector('[role="combobox"][aria-label="Filter gallery media"]') as HTMLButtonElement;
    pickOption(filterSelect, 'video');

    filtered.resolve({ assets: [currentAsset], page: 1, total_pages: 1, total_count: 1 });
    await settle();
    initial.reject(new Error('stale request failed'));
    await settle();

    expect(target.textContent).toContain('current.mp4');
    expect(target.textContent).not.toContain('stale request failed');
    expect(target.textContent).not.toContain('stale.png');
    expect(target.textContent).toContain('Browsing 1 loaded asset of 1');
  });

  it('invalidates an in-flight old pagination request and permits pagination for the new query', async () => {
    const initialAsset = makeAsset({ id: 'old-page-1.png', filename: 'old-page-1.png' });
    const stalePage = deferred<GalleryPageResponse>();
    const filteredPage = deferred<GalleryPageResponse>();
    const newPage = deferred<GalleryPageResponse>();
    const filteredAsset = makeAsset({ id: 'video-page-1.mp4', filename: 'video-page-1.mp4', media_type: 'video' });
    const pagedAsset = makeAsset({ id: 'video-page-2.mp4', filename: 'video-page-2.mp4', media_type: 'video' });
    galleryApiMocks.getGallery
      .mockResolvedValueOnce({ assets: [initialAsset], page: 1, total_pages: 2, total_count: 2 })
      .mockReturnValueOnce(stalePage.promise)
      .mockReturnValueOnce(filteredPage.promise)
      .mockReturnValueOnce(newPage.promise);

    mountPage();
    await settle();
    MockIntersectionObserver.instances.at(-1)?.trigger(true);
    await settle();

    const filterSelect = target.querySelector('[role="combobox"][aria-label="Filter gallery media"]') as HTMLButtonElement;
    pickOption(filterSelect, 'video');
    filteredPage.resolve({ assets: [filteredAsset], page: 1, total_pages: 2, total_count: 2 });
    await settle();
    stalePage.resolve({ assets: [makeAsset({ id: 'old-page-2.png', filename: 'old-page-2.png' })], page: 2, total_pages: 2, total_count: 2 });
    await settle();

    expect(target.textContent).toContain('video-page-1.mp4');
    expect(target.textContent).not.toContain('old-page-2.png');
    expect(target.textContent).not.toContain('old-page-1.png');

    MockIntersectionObserver.instances.at(-1)?.trigger(true);
    await settle();
    newPage.resolve({ assets: [pagedAsset], page: 2, total_pages: 2, total_count: 2 });
    await settle();

    expect(galleryApiMocks.getGallery).toHaveBeenLastCalledWith(2, 'video', 'newest', null);
    expect(target.textContent).toContain('video-page-2.mp4');
  });

  it('makes an unresolved response inert after unmount', async () => {
    const request = deferred<GalleryPageResponse>();
    galleryApiMocks.getGallery.mockReturnValueOnce(request.promise);

    mountPage();
    await page.unmount();
    request.resolve({ assets: [makeAsset({ filename: 'late.png' })], page: 1, total_pages: 1, total_count: 1 });
    await settle();

    expect(target.textContent).not.toContain('late.png');
  });
});

describe('GalleryPage bulk deletion settlement (F06)', () => {
  beforeEach(() => {
    confirmMocks.requestConfirm.mockResolvedValue(true);
  });

  it('keeps failed IDs selected and preserves selections added while a mixed deletion is in flight', async () => {
    const assetA = makeAsset({ id: 'a.png', filename: 'a.png' });
    const assetB = makeAsset({ id: 'b.png', filename: 'b.png' });
    const assetC = makeAsset({ id: 'c.png', filename: 'c.png' });
    const deleteA = deferred<void>();
    const deleteB = deferred<void>();
    galleryApiMocks.getGallery.mockResolvedValue({ assets: [assetA, assetB, assetC], page: 1, total_pages: 1, total_count: 3 });
    galleryApiMocks.deleteAsset.mockImplementation((id) => id === assetA.id ? deleteA.promise : deleteB.promise);

    mountPage();
    await settle();
    await openViewer(assetA.filename);
    selectAssetForBatch(target, assetA);
    selectAssetForBatch(target, assetB);
    await settle();

    const deleteButton = queryButtonByName(target, 'Delete selected')!;
    deleteButton.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    await settle();
    expect(deleteButton.disabled).toBe(true);
    deleteButton.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    expect(galleryApiMocks.deleteAsset).toHaveBeenCalledTimes(2);

    selectAssetForBatch(target, assetC);
    deleteA.resolve();
    deleteB.reject(new Error('blocked'));
    await settle();

    expect(target.textContent).not.toContain('a.png');
    expect(target.textContent).toContain('b.png');
    expect(target.textContent).toContain('c.png');
    expect(target.textContent).toContain('Browsing 2 loaded assets of 2');
    expect(target.textContent).toContain('2 selected');
    // The viewer moves on to the neighbouring asset instead of closing.
    expect(getViewer()?.textContent).toContain('b.png');
    expect(queryButtonByName(target, 'Delete selected')?.disabled).toBe(false);
    expect(toastMocks.addToast).toHaveBeenCalledWith(expect.any(String), 'warning', {
      action: expect.objectContaining({ label: 'Retry' })
    });
  });

  it('removes only successful originals, moves the viewer past them, and reports a full success', async () => {
    const assetA = makeAsset({ id: 'a.png', filename: 'a.png' });
    const assetB = makeAsset({ id: 'b.png', filename: 'b.png' });
    const assetC = makeAsset({ id: 'c.png', filename: 'c.png' });
    const deleteA = deferred<void>();
    const deleteB = deferred<void>();
    galleryApiMocks.getGallery.mockResolvedValue({ assets: [assetA, assetB, assetC], page: 1, total_pages: 1, total_count: 3 });
    galleryApiMocks.deleteAsset.mockImplementation((id) => id === assetA.id ? deleteA.promise : deleteB.promise);

    mountPage();
    await settle();
    await openViewer(assetA.filename);
    selectAssetForBatch(target, assetA);
    selectAssetForBatch(target, assetB);
    queryButtonByName(target, 'Delete selected')!.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    await settle();
    selectAssetForBatch(target, assetC);
    deleteA.resolve();
    deleteB.resolve();
    await settle();

    expect(target.textContent).not.toContain('a.png');
    expect(target.textContent).not.toContain('b.png');
    expect(target.textContent).toContain('c.png');
    expect(target.textContent).toContain('Browsing 1 loaded asset of 1');
    expect(target.textContent).toContain('1 selected');
    expect(getViewer()?.textContent).toContain('c.png');
    expect(toastMocks.addToast).toHaveBeenCalledWith(expect.any(String), 'success');
  });

  it('retains all failed originals, their active detail, retryability, and an error outcome', async () => {
    const assetA = makeAsset({ id: 'a.png', filename: 'a.png' });
    const assetB = makeAsset({ id: 'b.png', filename: 'b.png', prompt: 'active failed asset' });
    galleryApiMocks.getGallery.mockResolvedValue({ assets: [assetA, assetB], page: 1, total_pages: 1, total_count: 2 });
    galleryApiMocks.deleteAsset.mockRejectedValue(new Error('blocked'));

    mountPage();
    await settle();
    await openViewer(assetB.filename);
    selectAssetForBatch(target, assetA);
    selectAssetForBatch(target, assetB);
    queryButtonByName(target, 'Delete selected')!.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    await settle();

    expect(target.textContent).toContain('a.png');
    expect(target.textContent).toContain('b.png');
    expect(target.textContent).toContain('Browsing 2 loaded assets of 2');
    expect(target.textContent).toContain('2 selected');
    expect(getViewer()?.textContent).toContain('active failed asset');
    expect(queryButtonByName(target, 'Delete selected')?.disabled).toBe(false);
    expect(toastMocks.addToast).toHaveBeenCalledWith(expect.any(String), 'error', {
      action: expect.objectContaining({ label: 'Retry' })
    });
  });

  it('closes its Retry toast when the page unmounts, so Retry cannot act on a page that is gone', async () => {
    const asset = makeAsset({ id: 'a.png', filename: 'a.png' });
    galleryApiMocks.getGallery.mockResolvedValue({ assets: [asset], page: 1, total_pages: 1, total_count: 1 });
    galleryApiMocks.deleteAsset.mockRejectedValue(new Error('blocked'));
    toastMocks.addToast.mockReturnValue('toast-7');

    mountPage();
    await settle();
    selectAssetForBatch(target, asset);
    queryButtonByName(target, 'Delete selected')!.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    await settle();
    expect(toastMocks.dismissToast).not.toHaveBeenCalled();

    await page.unmount();
    expect(toastMocks.dismissToast).toHaveBeenCalledWith('toast-7');
  });
});

describe('GalleryPage replacement and mutation authority (REQ-4 through REQ-7)', () => {
  beforeEach(() => {
    confirmMocks.requestConfirm.mockResolvedValue(true);
  });

  it('uses the same complete reset for filter, sort, and clear-filter transitions', async () => {
    const image = makeAsset({ id: 'image.png', filename: 'image.png' });
    const video = makeAsset({ id: 'video.mp4', filename: 'video.mp4', media_type: 'video' });
    galleryApiMocks.getGallery
      .mockResolvedValueOnce({ assets: [image], page: 1, total_pages: 1, total_count: 1 })
      .mockResolvedValueOnce({ assets: [video], page: 1, total_pages: 1, total_count: 1 })
      .mockResolvedValueOnce({ assets: [video], page: 1, total_pages: 1, total_count: 1 })
      .mockResolvedValueOnce({ assets: [], page: 1, total_pages: 1, total_count: 0 })
      .mockResolvedValueOnce({ assets: [image, video], page: 1, total_pages: 1, total_count: 2 });

    mountPage();
    await settle();
    selectAssetForBatch(target, image);
    await openViewer(image.filename);
    await settle();
    expect(document.querySelector('[role="dialog"]')).not.toBeNull();

    const filter = target.querySelector('[role="combobox"][aria-label="Filter gallery media"]') as HTMLButtonElement;
    const sort = target.querySelector('[role="combobox"][aria-label="Sort gallery assets"]') as HTMLButtonElement;
    routerMocks.replace.mockReset();
    pickOption(filter, 'video');
    await settle();

    expect(galleryApiMocks.getGallery).toHaveBeenNthCalledWith(2, 1, 'video', 'newest', null);
    expect(target.textContent).toContain('0 selected');
    expect(getViewer()).toBeNull();
    expect(document.querySelector('[role="dialog"]')).toBeNull();

    pickOption(sort, 'oldest');
    await settle();
    expect(galleryApiMocks.getGallery).toHaveBeenNthCalledWith(3, 1, 'video', 'oldest', null);

    // The empty filtered view exposes the clear-filter path, which must use the
    // same reset and retain the selected sort order.
    pickOption(filter, 'image');
    await settle();
    queryButtonByName(target, 'Show all media')!.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    await settle();
    expect(galleryApiMocks.getGallery).toHaveBeenNthCalledWith(5, 1, 'all', 'oldest', null);
  });

  it('prevents a stale sort response from resurrecting a successful deletion and refills page one', async () => {
    const deleted = makeAsset({ id: 'deleted.png', filename: 'deleted.png' });
    const survivor = makeAsset({ id: 'survivor.png', filename: 'survivor.png' });
    const deleteRequest = deferred<void>();
    const staleSort = deferred<GalleryPageResponse>();
    const refill = deferred<GalleryPageResponse>();
    galleryApiMocks.getGallery
      .mockResolvedValueOnce({ assets: [deleted, survivor], page: 1, total_pages: 2, total_count: 2 })
      .mockReturnValueOnce(staleSort.promise)
      .mockReturnValueOnce(refill.promise);
    galleryApiMocks.deleteAsset.mockReturnValue(deleteRequest.promise);

    mountPage();
    await settle();
    deleteAssetFromCard(target, deleted);
    await settle();

    const sort = target.querySelector('[role="combobox"][aria-label="Sort gallery assets"]') as HTMLButtonElement;
    pickOption(sort, 'oldest');
    await settle();
    expect(galleryApiMocks.getGallery).toHaveBeenNthCalledWith(2, 1, 'all', 'oldest', null);

    deleteRequest.resolve();
    await settle();
    expect(galleryApiMocks.getGallery).toHaveBeenNthCalledWith(3, 1, 'all', 'oldest', null);

    staleSort.resolve({ assets: [deleted], page: 1, total_pages: 1, total_count: 2 });
    refill.resolve({ assets: [deleted, survivor], page: 1, total_pages: 1, total_count: 2 });
    await settle();

    expect(target.textContent).not.toContain('deleted.png');
    expect(target.textContent).toContain('survivor.png');
    expect(target.textContent).toContain('Browsing 1 loaded asset of 1');
  });

  it('keeps a deleted asset removed and warns when the refresh after the delete fails', async () => {
    const deleted = makeAsset({ id: 'deleted.png', filename: 'deleted.png' });
    const survivor = makeAsset({ id: 'survivor.png', filename: 'survivor.png' });
    galleryApiMocks.getGallery
      .mockResolvedValueOnce({ assets: [deleted, survivor], page: 1, total_pages: 1, total_count: 2 })
      .mockRejectedValueOnce(new Error('refresh unavailable'));
    galleryApiMocks.deleteAsset.mockResolvedValue(undefined);

    mountPage();
    await settle();
    deleteAssetFromCard(target, deleted);
    await settle();

    expect(target.textContent).toContain('survivor.png');
    expect(target.textContent).not.toContain('deleted.png');
    expect(toastMocks.addToast).toHaveBeenCalledWith(expect.any(String), 'warning');
  });

  it('moves the viewer and selected route on only when the viewed asset is deleted', async () => {
    const active = makeAsset({ id: 'active.png', filename: 'active.png', prompt: 'active prompt' });
    const other = makeAsset({ id: 'other.png', filename: 'other.png', prompt: 'other prompt' });
    const nonActiveVictim = makeAsset({ id: 'victim.png', filename: 'victim.png' });
    galleryApiMocks.getGallery.mockResolvedValue({
      assets: [active, other, nonActiveVictim],
      page: 1,
      total_pages: 1,
      total_count: 3,
    });
    galleryApiMocks.deleteAsset.mockResolvedValue(undefined);

    mountPage();
    await settle();
    await openViewer(active.filename);
    await settle();
    routerMocks.replace.mockReset();

    deleteAssetFromCard(target, active);
    await settle();

    // The viewer moves on to the next asset.
    expect(getViewer()?.textContent).toContain('other prompt');
    expect(target.textContent).not.toContain('active.png');

    // A non-active deletion leaves a still-present detail alone.
    await openViewer(other.filename);
    routerMocks.replace.mockReset();
    deleteAssetFromCard(target, nonActiveVictim);
    await settle();
    expect(getViewer()?.textContent).toContain('other prompt');
  });

  it('guards repeated IDs and single/bulk overlap while allowing distinct deletes to settle independently', async () => {
    const assetA = makeAsset({ id: 'a.png', filename: 'a.png' });
    const assetB = makeAsset({ id: 'b.png', filename: 'b.png' });
    const deleteA = deferred<void>();
    const deleteB = deferred<void>();
    galleryApiMocks.getGallery.mockResolvedValue({ assets: [assetA, assetB], page: 1, total_pages: 1, total_count: 2 });
    galleryApiMocks.deleteAsset.mockImplementation((id) => id === assetA.id ? deleteA.promise : deleteB.promise);

    mountPage();
    await settle();
    selectAssetForBatch(target, assetA);
    selectAssetForBatch(target, assetB);
    deleteAssetFromCard(target, assetA);
    await settle();
    deleteAssetFromCard(target, assetA);
    queryButtonByName(target, 'Delete selected')!.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    await settle();

    expect(galleryApiMocks.deleteAsset).toHaveBeenCalledTimes(2);
    expect(galleryApiMocks.deleteAsset).toHaveBeenCalledWith(assetA.id);
    expect(galleryApiMocks.deleteAsset).toHaveBeenCalledWith(assetB.id);

    deleteA.resolve();
    await settle();
    // B remains independently in flight after A has reconciled and requested a refill.
    expect(target.textContent).toContain('b.png');
    deleteB.resolve();
    await settle();

    expect(target.textContent).not.toContain('a.png');
    expect(target.textContent).not.toContain('b.png');
  });

  it('does not call the API or alter active state when deletion is cancelled', async () => {
    const asset = makeAsset({ id: 'cancelled.png', filename: 'cancelled.png', prompt: 'keep me' });
    confirmMocks.requestConfirm.mockResolvedValue(false);
    galleryApiMocks.getGallery.mockResolvedValue({ assets: [asset], page: 1, total_pages: 1, total_count: 1 });

    mountPage();
    await settle();
    await openViewer(asset.filename);
    routerMocks.replace.mockReset();
    deleteAssetFromCard(target, asset);
    await settle();

    expect(galleryApiMocks.deleteAsset).not.toHaveBeenCalled();
    expect(target.textContent).toContain('cancelled.png');
    expect(getViewer()?.textContent).toContain('keep me');
  });

  it('holds pagination at page one during a delete refill, then resumes contiguously from page two', async () => {
    const deleted = makeAsset({ id: 'page-one-deleted.png', filename: 'page-one-deleted.png' });
    const pageTwo = makeAsset({ id: 'page-two.png', filename: 'page-two.png' });
    const pageThree = makeAsset({ id: 'page-three.png', filename: 'page-three.png' });
    const refill = deferred<GalleryPageResponse>();
    const resumedPageTwo = deferred<GalleryPageResponse>();
    galleryApiMocks.getGallery
      .mockResolvedValueOnce({ assets: [deleted], page: 1, total_pages: 3, total_count: 3 })
      .mockResolvedValueOnce({ assets: [pageTwo], page: 2, total_pages: 3, total_count: 3 })
      .mockReturnValueOnce(refill.promise)
      .mockReturnValueOnce(resumedPageTwo.promise);
    galleryApiMocks.deleteAsset.mockResolvedValue(undefined);

    mountPage();
    await settle();
    MockIntersectionObserver.instances.at(-1)?.trigger(true);
    await settle();
    expect(galleryApiMocks.getGallery).toHaveBeenNthCalledWith(2, 2, 'all', 'newest', null);
    expect(target.textContent).toContain('page-two.png');

    deleteAssetFromCard(target, deleted);
    await settle();
    expect(galleryApiMocks.getGallery).toHaveBeenNthCalledWith(3, 1, 'all', 'newest', null);

    // A callback held by a now-disconnected observer must not skip to page 3
    // while the authoritative page-one refill is unresolved.
    MockIntersectionObserver.instances.forEach((observer) => observer.trigger(true));
    await settle();
    expect(galleryApiMocks.getGallery).toHaveBeenCalledTimes(3);

    refill.resolve({ assets: [pageTwo], page: 1, total_pages: 2, total_count: 2 });
    await settle();
    MockIntersectionObserver.instances.at(-1)?.trigger(true);
    await settle();
    expect(galleryApiMocks.getGallery).toHaveBeenNthCalledWith(4, 2, 'all', 'newest', null);

    resumedPageTwo.resolve({ assets: [pageThree], page: 2, total_pages: 2, total_count: 2 });
    await settle();
    expect(target.textContent).toContain('page-two.png');
    expect(target.textContent).toContain('page-three.png');
    expect(target.textContent).not.toContain('page-one-deleted.png');
  });

});

describe('GalleryPage keyboard shortcuts', () => {
  const assets = [
    makeAsset({ id: 'a.png', filename: 'a.png' }),
    makeAsset({ id: 'b.png', filename: 'b.png' }),
    makeAsset({ id: 'c.png', filename: 'c.png' }),
  ];

  async function mountGallery(): Promise<void> {
    galleryApiMocks.getGallery.mockResolvedValue({ assets, page: 1, total_pages: 1, total_count: assets.length });
    mountPage();
    await settle();
  }

  function press(key: string, init: KeyboardEventInit = {}): KeyboardEvent {
    const event = new KeyboardEvent('keydown', { key, bubbles: true, cancelable: true, ...init });
    (document.activeElement ?? document.body).dispatchEvent(event);
    flushSync();
    return event;
  }

  function tileButton(filename: string): HTMLButtonElement {
    return target.querySelector(`button[aria-label="View ${filename}"]`) as HTMLButtonElement;
  }

  function checked(filename: string): boolean {
    return (target.querySelector(`input[aria-label="Select ${filename}"]`) as HTMLInputElement).checked;
  }

  it('moves focus between tiles with the arrow keys', async () => {
    await mountGallery();
    tileButton('a.png').focus();
    press('ArrowRight');
    expect(document.activeElement).toBe(tileButton('b.png'));
    press('ArrowLeft');
    expect(document.activeElement).toBe(tileButton('a.png'));
  });

  it('toggles the focused tile with Space and X and clears the selection with Escape', async () => {
    await mountGallery();
    tileButton('b.png').focus();
    expect(press(' ').defaultPrevented).toBe(true);
    expect(checked('b.png')).toBe(true);
    press('x');
    expect(checked('b.png')).toBe(false);

    press(' ');
    expect(checked('b.png')).toBe(true);
    press('Escape');
    expect(checked('b.png')).toBe(false);
  });

  it('selects every loaded asset with Ctrl+A', async () => {
    await mountGallery();
    expect(press('a', { ctrlKey: true }).defaultPrevented).toBe(true);
    expect(assets.every((asset) => checked(asset.filename))).toBe(true);
  });

  it('asks once for a held Delete key', async () => {
    confirmMocks.requestConfirm.mockResolvedValue(false);
    await mountGallery();

    tileButton('c.png').focus();
    press('Delete');
    press('Delete', { repeat: true });
    await settle();
    expect(confirmMocks.requestConfirm).toHaveBeenCalledTimes(1);
    expect(confirmMocks.requestConfirm).toHaveBeenCalledWith(expect.objectContaining({ question: 'Delete "c.png"?', confirmLabel: 'Delete' }));
    expect(galleryApiMocks.deleteAsset).not.toHaveBeenCalled();
  });

  it('deletes the selection with Delete, else the focused tile', async () => {
    const confirmSpy = confirmMocks.requestConfirm.mockResolvedValue(true);
    galleryApiMocks.deleteAsset.mockResolvedValue();
    await mountGallery();

    tileButton('c.png').focus();
    press('Backspace');
    await settle();
    expect(galleryApiMocks.deleteAsset).toHaveBeenCalledWith('c.png');

    galleryApiMocks.deleteAsset.mockClear();
    selectAssetForBatch(target, assets[0]);
    flushSync();
    press('Delete');
    await settle();
    expect(galleryApiMocks.deleteAsset).toHaveBeenCalledWith('a.png');
    expect(galleryApiMocks.deleteAsset).not.toHaveBeenCalledWith('b.png');
  });
});

describe('GalleryPage memory', () => {
  const assets = [
    makeAsset({ id: 'a.png', filename: 'a.png' }),
    makeAsset({ id: 'b.png', filename: 'b.png' }),
  ];
  const tiles = () => Array.from(target.querySelectorAll('button[aria-label^="View "]:not([aria-label$=" fullscreen"])')).map((b) => b.getAttribute('aria-label'));

  it('draws placeholder tiles on the first visit, then comes back with the loaded assets before the refresh answers', async () => {
    galleryApiMocks.getGallery.mockImplementation(() => new Promise(() => undefined));
    mountPage();
    await settle();
    expect(target.querySelector('[data-testid="gallery-placeholders"]')).not.toBeNull();
    await page.unmount();

    galleryApiMocks.getGallery.mockResolvedValue({ assets, page: 1, total_pages: 1, total_count: assets.length });
    mountPage();
    await settle();
    expect(tiles()).toEqual(['View a.png', 'View b.png']);
    await page.unmount();

    // Back: the grid is there at once, and the refreshed page one adds the new asset without a spinner in between.
    let answer: (page: GalleryPageResponse) => void = () => undefined;
    galleryApiMocks.getGallery.mockImplementation(() => new Promise<GalleryPageResponse>((resolve) => { answer = resolve; }));
    mountPage();
    await settle();
    expect(target.querySelector('[data-testid="gallery-placeholders"]')).toBeNull();
    expect(tiles()).toEqual(['View a.png', 'View b.png']);

    const fresh = makeAsset({ id: 'new.png', filename: 'new.png' });
    answer({ assets: [fresh, ...assets], page: 1, total_pages: 1, total_count: 3 });
    await settle();
    expect(tiles()).toEqual(['View new.png', 'View a.png', 'View b.png']);
  });

  it('forgets a view whose load failed, so the next visit starts from the placeholders', async () => {
    galleryApiMocks.getGallery.mockRejectedValue(new Error('offline'));
    mountPage();
    await settle();
    await page.unmount();

    galleryApiMocks.getGallery.mockImplementation(() => new Promise(() => undefined));
    mountPage();
    await settle();
    expect(target.querySelector('[data-testid="gallery-placeholders"]')).not.toBeNull();
  });
});

describe('GalleryPage with a project picked', () => {
  const reply = (assets: GalleryAsset[]) => ({ assets, page: 1, total_pages: 1, total_count: assets.length });
  const subtitle = () => target.querySelector('.ui-page-sub')?.textContent ?? '';
  const tiles = () => Array.from(target.querySelectorAll('button[aria-label^="View "]:not([aria-label$=" fullscreen"])')).map((b) => b.getAttribute('aria-label'));

  it('asks for the picked project on the first page and on later pages', async () => {
    projectPick.set('Night Bus');
    galleryApiMocks.getGallery
      .mockResolvedValueOnce({ assets: [makeAsset({ id: 'Night Bus/a.png', filename: 'a.png', project: 'Night Bus' })], page: 1, total_pages: 2, total_count: 2 })
      .mockResolvedValueOnce({ assets: [makeAsset({ id: 'Night Bus/b.png', filename: 'b.png', project: 'Night Bus' })], page: 2, total_pages: 2, total_count: 2 });

    mountPage();
    await settle();
    MockIntersectionObserver.instances.at(-1)?.trigger(true);
    await settle();

    expect(galleryApiMocks.getGallery).toHaveBeenNthCalledWith(1, 1, 'all', 'newest', 'Night Bus');
    expect(galleryApiMocks.getGallery).toHaveBeenNthCalledWith(2, 2, 'all', 'newest', 'Night Bus');
    expect(tiles()).toEqual(['View a.png', 'View b.png']);
  });

  it('leads the subtitle with the project name, and keeps the plain text with All files', async () => {
    galleryApiMocks.getGallery.mockResolvedValue(reply([makeAsset()]));
    mountPage();
    await settle();
    expect(subtitle()).toBe('Browsing 1 loaded asset of 1, newest first.');

    projectsStore.choose('Night Bus');
    await settle();
    expect(subtitle()).toBe('Night Bus · Browsing 1 loaded asset of 1, newest first.');

    projectsStore.choose(null);
    await settle();
    expect(subtitle()).toBe('Browsing 1 loaded asset of 1, newest first.');
  });

  it('replaces the view on a pick change: new project, selection cleared, viewer closed, back at the top, filter and sort kept', async () => {
    const mine = makeAsset({ id: 'mine.mp4', filename: 'mine.mp4', media_type: 'video' });
    const theirs = makeAsset({ id: 'Night Bus/theirs.mp4', filename: 'theirs.mp4', media_type: 'video', project: 'Night Bus' });
    galleryApiMocks.getGallery.mockImplementation(async (...args: unknown[]) => reply(args[3] === 'Night Bus' ? [theirs] : [mine]));
    mountPage();
    await settle();
    pickOption(target.querySelector('[role="combobox"][aria-label="Filter gallery media"]') as HTMLButtonElement, 'video');
    pickOption(target.querySelector('[role="combobox"][aria-label="Sort gallery assets"]') as HTMLButtonElement, 'oldest');
    await settle();
    selectAssetForBatch(target, mine);
    await openViewer('mine.mp4');
    const region = target.querySelector('#gallery-scroll-region') as HTMLElement;
    region.scrollTop = 400;
    region.dispatchEvent(new Event('scroll'));
    expect(getViewer()).not.toBeNull();

    projectsStore.choose('Night Bus');
    await settle();

    expect(galleryApiMocks.getGallery).toHaveBeenLastCalledWith(1, 'video', 'oldest', 'Night Bus');
    expect(tiles()).toEqual(['View theirs.mp4']);
    expect(target.textContent).toContain('0 selected');
    expect(getViewer()).toBeNull();
    expect(region.scrollTop).toBe(0);
    expect((target.querySelector('[role="combobox"][aria-label="Filter gallery media"]') as HTMLElement).dataset.value).toBe('video');
    expect((target.querySelector('[role="combobox"][aria-label="Sort gallery assets"]') as HTMLElement).dataset.value).toBe('oldest');
  });

  it('asks the server once for the new project\'s default list when the pick changes under it', async () => {
    galleryApiMocks.getGallery.mockImplementation(async () => reply([makeAsset()]));
    mountPage();
    await settle();
    galleryApiMocks.getGallery.mockClear();

    projectsStore.choose('Night Bus');
    await settle();

    expect(galleryApiMocks.getGallery).toHaveBeenCalledTimes(1);
    expect(galleryApiMocks.getGallery).toHaveBeenCalledWith(1, 'all', 'newest', 'Night Bus');
  });

  it('drops a response still in flight from the old project, for a query other than the default one', async () => {
    const old = deferred<GalleryPageResponse>();
    const fresh = deferred<GalleryPageResponse>();
    const oldAsset = makeAsset({ id: 'old.mp4', filename: 'old.mp4', media_type: 'video' });
    const freshAsset = makeAsset({ id: 'Night Bus/fresh.mp4', filename: 'fresh.mp4', media_type: 'video', project: 'Night Bus' });
    galleryApiMocks.getGallery.mockImplementation((...args: unknown[]) => {
      if (args[1] === 'video') return args[3] === 'Night Bus' ? fresh.promise : old.promise;
      return Promise.resolve(reply([]));
    });
    mountPage();
    await settle();
    pickOption(target.querySelector('[role="combobox"][aria-label="Filter gallery media"]') as HTMLButtonElement, 'video');

    projectsStore.choose('Night Bus');
    await settle();
    fresh.resolve(reply([freshAsset]));
    await settle();
    old.resolve(reply([oldAsset]));
    await settle();

    expect(tiles()).toEqual(['View fresh.mp4']);
    expect(galleryStore.view({ filter: 'video', sort: 'newest' }).assets.map((asset) => asset.id)).toEqual(['Night Bus/fresh.mp4']);
  });

  it('opens at the top with the new project when the pick changed while the page was away', async () => {
    const mine = makeAsset({ id: 'mine.png', filename: 'mine.png' });
    const theirs = makeAsset({ id: 'Night Bus/theirs.png', filename: 'theirs.png', project: 'Night Bus' });
    galleryApiMocks.getGallery.mockImplementation(async (...args: unknown[]) => reply(args[3] === 'Night Bus' ? [theirs] : [mine]));
    mountPage();
    await settle();
    (target.querySelector('#gallery-scroll-region') as HTMLElement).scrollTop = 300;
    target.querySelector('#gallery-scroll-region')!.dispatchEvent(new Event('scroll'));
    await page.unmount();

    projectsStore.choose('Night Bus');
    mountPage();
    await settle();

    expect(tiles()).toEqual(['View theirs.png']);
    expect((target.querySelector('#gallery-scroll-region') as HTMLElement).scrollTop).toBe(0);
  });

  it('says an empty project is empty and where work goes, and lets the media filter message win', async () => {
    galleryApiMocks.getGallery.mockResolvedValue(reply([]));
    mountPage();
    await settle();
    expect(target.textContent).toContain('No generated assets yet');

    projectsStore.choose('Night Bus');
    await settle();
    expect(target.textContent).toContain('Nothing in Night Bus yet');
    expect(target.textContent).toContain('New work saves here. To add existing files, switch to All files and use Move to project.');
    expect(queryButtonByName(target, 'Open Workspace')).not.toBeNull();

    pickOption(target.querySelector('[role="combobox"][aria-label="Filter gallery media"]') as HTMLButtonElement, 'video');
    await settle();
    expect(target.textContent).toContain('No matching assets');
    expect(target.textContent).not.toContain('Nothing in Night Bus yet');
  });

  it('shows the project on a tile in All files only', async () => {
    const inProject = makeAsset({ id: 'Night Bus/a.png', filename: 'a.png', project: 'Night Bus' });
    const loose = makeAsset({ id: 'b.png', filename: 'b.png' });
    galleryApiMocks.getGallery.mockImplementation(async (...args: unknown[]) => reply(args[3] === 'Night Bus' ? [inProject] : [inProject, loose]));
    mountPage();
    await settle();
    const foots = () => Array.from(target.querySelectorAll('.asset-tile-foot p:nth-child(2)')).map((p) => p.textContent?.split(' · ')[0]);
    expect(foots()).toEqual(['Night Bus', 'txt2img']);

    projectsStore.choose('Night Bus');
    await settle();
    expect(foots()).toEqual(['txt2img']);
  });
});

describe('GalleryPage Not in a project', () => {
  const reply = (assets: GalleryAsset[]) => ({ assets, page: 1, total_pages: 1, total_count: assets.length });
  const filterBox = () => target.querySelector('[role="combobox"][aria-label="Filter gallery media"]') as HTMLButtonElement;
  const top = makeAsset({ id: 'top.png', filename: 'top.png' });

  it('is a choice of the media filter with All files, asked of the server as a filter value', async () => {
    galleryApiMocks.getGallery.mockResolvedValue(reply([top]));
    mountPage();
    await settle();

    expect(optionLabels(filterBox())).toContain('Not in a project');
    pickOption(filterBox(), 'no_project');
    await settle();

    expect(galleryApiMocks.getGallery).toHaveBeenLastCalledWith(1, 'no_project', 'newest', null);
    expect(filterBox().dataset.value).toBe('no_project');
  });

  it('is not offered once a project is picked', async () => {
    projectPick.set('Night Bus');
    galleryApiMocks.getGallery.mockResolvedValue(reply([top]));
    mountPage();
    await settle();

    expect(optionLabels(filterBox())).not.toContain('Not in a project');
  });

  it('goes back to All media when a project is picked while it is on', async () => {
    galleryApiMocks.getGallery.mockResolvedValue(reply([top]));
    mountPage();
    await settle();
    pickOption(filterBox(), 'no_project');
    await settle();

    projectsStore.choose('Night Bus');
    await settle();

    expect(galleryApiMocks.getGallery).toHaveBeenLastCalledWith(1, 'all', 'newest', 'Night Bus');
    expect(filterBox().dataset.value).toBe('all');
  });

  it('has its own empty state, which offers to show all media', async () => {
    galleryApiMocks.getGallery.mockResolvedValueOnce(reply([top])).mockResolvedValue(reply([]));
    mountPage();
    await settle();
    pickOption(filterBox(), 'no_project');
    await settle();

    expect(target.textContent).toContain('No matching assets');
    expect(queryButtonByName(target, 'Show all media')).not.toBeNull();
  });
});

describe('GalleryPage with audio', () => {
  const reply = (assets: GalleryAsset[]) => ({ assets, page: 1, total_pages: 1, total_count: assets.length });

  it('offers Audio only in the media filter and asks the server for audio', async () => {
    galleryApiMocks.getGallery.mockResolvedValue(reply([makeAudioAsset()]));
    mountPage();
    await settle();
    const filterSelect = target.querySelector('[role="combobox"][aria-label="Filter gallery media"]') as HTMLButtonElement;

    expect(optionLabels(filterSelect)).toEqual(['All media', 'Images only', 'Videos only', 'Audio only', 'Not in a project']);
    pickOption(filterSelect, 'audio');
    await settle();

    expect(galleryApiMocks.getGallery).toHaveBeenLastCalledWith(1, 'audio', 'newest', null);
    expect(filterSelect.dataset.value).toBe('audio');
  });

  it('lists takes as audio tiles beside the images, and a file without settings as the no-settings tile', async () => {
    galleryApiMocks.getGallery.mockResolvedValue(reply([makeAssetForAudioPage(), makeAudioAsset(), makeSpeechAsset(), makeImportedAudio()]));
    mountPage();
    await settle();

    const covers = Array.from(target.querySelectorAll<HTMLElement>('[data-testid="audio-cover"]'));
    expect(covers.map((cover) => cover.dataset.bare)).toEqual(['false', 'false', 'true']);
    expect(target.querySelectorAll('img')).toHaveLength(1);
    expect(target.textContent).toContain('audio · no settings');
  });

  it('sends a take as the audio track to the Video tab, from 0 s', async () => {
    galleryApiMocks.getGallery.mockResolvedValue(reply([makeAudioAsset()]));
    mountPage();
    await settle();
    (target.querySelector('button[aria-label="More actions for song.flac"]') as HTMLButtonElement).click();
    flushSync();

    (document.querySelector('[role="menu"] [data-action="use-audio-track"]') as HTMLButtonElement).click();

    expect(routerMocks.navigate).toHaveBeenCalledWith('workspace', { workflow: 'txt2vid', audio_track_path: '/outputs/out/song.flac', audio_track_start: '0', use_as: 'audio_track' });
  });

  it('sends a speech take as the voice to the Speech tab, with its text', async () => {
    workspaceMemory.set(withAudio(makeContext()));
    galleryApiMocks.getGallery.mockResolvedValue(reply([makeSpeechAsset()]));
    mountPage();
    await settle();
    (target.querySelector('button[aria-label="More actions for speech.flac"]') as HTMLButtonElement).click();
    flushSync();

    (document.querySelector('[role="menu"] [data-action="use-voice"]') as HTMLButtonElement).click();

    expect(routerMocks.navigate).toHaveBeenCalledWith('workspace', { workflow: 'speech', voice_path: '/outputs/out/speech.flac', voice_text: 'Welcome back to the workshop.', use_as: 'voice' });
  });
});

describe('GalleryPage Use as -> Saved voice', () => {
  const reply = (assets: GalleryAsset[]) => ({ assets, page: 1, total_pages: 1, total_count: assets.length });
  const MORGAN: SavedVoice = { slug: 'harbour', name: 'Harbour', text: 'Welcome back to the workshop.', origin: 'recording', seconds: 21, description: null, created_at: '2026-11-08T11:00:00Z' };

  beforeEach(() => {
    voiceApiMocks.saveRecordingVoice.mockReset();
    workspaceMemory.set(withAudio(makeContext()));
  });

  async function openMenu(asset: GalleryAsset): Promise<void> {
    galleryApiMocks.getGallery.mockResolvedValue(reply([asset]));
    mountPage();
    await settle();
    (target.querySelector(`button[aria-label="More actions for ${asset.filename}"]`) as HTMLButtonElement).click();
    flushSync();
  }

  it('opens the dialog on the page, saves the take by its path, says so in a toast and goes nowhere', async () => {
    voiceApiMocks.saveRecordingVoice.mockResolvedValue(MORGAN);
    await openMenu(makeSpeechAsset());
    (document.querySelector('[role="menu"] [data-action="use-save-voice"]') as HTMLButtonElement).click();
    await settle();

    const name = document.querySelector<HTMLInputElement>('#sv-name')!;
    name.value = 'Harbour';
    flushSync(() => name.dispatchEvent(new Event('input', { bubbles: true })));
    flushSync(() => document.querySelector<HTMLButtonElement>('[data-action="save-voice"]')!.click());
    await settle();

    const form = voiceApiMocks.saveRecordingVoice.mock.calls[0][0];
    expect([form.get('model'), form.get('name'), form.get('voice_path'), form.get('voice_text')]).toEqual(['qwen3_tts', 'Harbour', '/outputs/out/speech.flac', 'Welcome back to the workshop.']);
    expect(toastMocks.addToast).toHaveBeenCalledWith('Saved voice "Harbour"', 'success');
    expect(routerMocks.navigate).not.toHaveBeenCalled();
    expect(document.querySelector('[role="dialog"][aria-label="Save as voice"]')).toBeNull();
  });

  it('does not open the dialog for a take over the limit: the item is greyed', async () => {
    await openMenu(makeSpeechAsset({ duration: 44 }));
    const item = document.querySelector<HTMLButtonElement>('[role="menu"] [data-action="use-save-voice"]')!;
    item.click();
    await settle();

    expect(item.getAttribute('aria-disabled')).toBe('true');
    expect(document.querySelector('[role="dialog"][aria-label="Save as voice"]')).toBeNull();
  });
});

describe('GalleryPage Use as -> Saved voice on a cold load', () => {
  const reply = (assets: GalleryAsset[]) => ({ assets, page: 1, total_pages: 1, total_count: assets.length });

  async function openSavedVoiceMenu(asset: GalleryAsset): Promise<HTMLButtonElement> {
    galleryApiMocks.getGallery.mockResolvedValue(reply([asset]));
    mountPage();
    await settle();
    (target.querySelector(`button[aria-label="More actions for ${asset.filename}"]`) as HTMLButtonElement).click();
    flushSync();
    return document.querySelector<HTMLButtonElement>('[role="menu"] [data-action="use-save-voice"]')!;
  }

  it('reads the Workspace context when an audio take is listed and the memory is empty: a long take is greyed, a short one can be saved', async () => {
    workspaceApiMocks.getWorkspaceCoreContext.mockResolvedValue(withAudio(makeContext()));
    const long = makeSpeechAsset({ id: 'long.flac', filename: 'long.flac', duration: 46 });
    const short = makeSpeechAsset({ id: 'short.flac', filename: 'short.flac', duration: 12 });
    galleryApiMocks.getGallery.mockResolvedValue(reply([long, short]));
    mountPage();
    await settle();

    expect(workspaceApiMocks.getWorkspaceCoreContext).toHaveBeenCalledTimes(1);
    expect(workspaceMemory.get()).not.toBeNull();
    (target.querySelector('button[aria-label="More actions for long.flac"]') as HTMLButtonElement).click();
    flushSync();
    const greyed = document.querySelector<HTMLButtonElement>('[role="menu"] [data-action="use-save-voice"]')!;
    expect(greyed.getAttribute('aria-disabled')).toBe('true');
    expect(document.querySelector('[role="menu"] [data-action="use-voice"]')!.getAttribute('aria-disabled')).toBe('true');
    greyed.click();
    await settle();
    expect(document.querySelector('[role="dialog"][aria-label="Save as voice"]')).toBeNull();
  });

  it('lets a short take be named and saved from the dialog', async () => {
    workspaceApiMocks.getWorkspaceCoreContext.mockResolvedValue(withAudio(makeContext()));
    const item = await openSavedVoiceMenu(makeSpeechAsset({ duration: 12 }));
    expect(item.getAttribute('aria-disabled')).not.toBe('true');
    item.click();
    await settle();

    const name = document.querySelector<HTMLInputElement>('#sv-name')!;
    name.value = 'Harbour';
    flushSync(() => name.dispatchEvent(new Event('input', { bubbles: true })));
    expect(document.querySelector<HTMLButtonElement>('[data-action="save-voice"]')!.disabled).toBe(false);
  });

  it('leaves the item unavailable when the context cannot be read', async () => {
    const item = await openSavedVoiceMenu(makeSpeechAsset({ duration: 12 }));

    expect(workspaceApiMocks.getWorkspaceCoreContext).toHaveBeenCalledTimes(1);
    expect(item.getAttribute('aria-disabled')).toBe('true');
  });

  it('does not read the context for a gallery without audio takes', async () => {
    galleryApiMocks.getGallery.mockResolvedValue(reply([makeAsset()]));
    mountPage();
    await settle();

    expect(workspaceApiMocks.getWorkspaceCoreContext).not.toHaveBeenCalled();
  });
});

function makeAssetForAudioPage(): GalleryAsset {
  return makeAsset({ id: 'img.png', filename: 'img.png' });
}

describe('GalleryPage Move', () => {
  const reply = (assets: GalleryAsset[]) => ({ assets, page: 1, total_pages: 1, total_count: assets.length });
  const inBus = (name: string, overrides: Partial<GalleryAsset> = {}) =>
    makeAsset({ id: `Night Bus/${name}`, filename: name, project: 'Night Bus', file_path: `/outputs/Night Bus/${name}`, ...overrides });
  const top = (name: string, overrides: Partial<GalleryAsset> = {}) =>
    makeAsset({ id: name, filename: name, file_path: `/outputs/${name}`, ...overrides });
  const tileIds = () => Array.from(target.querySelectorAll('.asset-tile')).map((tile) => tile.getAttribute('data-asset-id'));
  const rows = () => Array.from(document.querySelectorAll<HTMLElement>('[role="menu"] [role="menuitemradio"]'));
  const row = (label: string) => rows().find((item) => item.textContent?.trim() === label)!;
  const moved = (from: string, asset: GalleryAsset) => ({ from, asset });
  const lastToastTone = () => toastMocks.addToast.mock.calls.at(-1)?.[1];
  const lastToastMessage = () => String(toastMocks.addToast.mock.calls.at(-1)?.[0]);

  async function openTileMove(asset: GalleryAsset): Promise<void> {
    target.querySelector<HTMLButtonElement>(`button[aria-label="More actions for ${asset.filename}"]`)!.click();
    flushSync();
    document.querySelector<HTMLElement>('[role="menu"] [data-action="move"]')!.click();
    await settle();
  }

  async function moveFromTile(asset: GalleryAsset, label: string): Promise<void> {
    await openTileMove(asset);
    row(label).click();
    await settle();
  }

  it('keeps the tile in place under its new ID when All files is shown, and says where it went', async () => {
    const a = top('a.png');
    const b = top('b.png');
    galleryApiMocks.getGallery.mockResolvedValue(reply([a, b]));
    galleryApiMocks.moveAssets.mockResolvedValue({ moved: [moved('a.png', inBus('a.png'))], failed: [] });
    mountPage();
    await settle();

    await moveFromTile(a, 'Night Bus');

    expect(galleryApiMocks.moveAssets).toHaveBeenCalledWith(['a.png'], 'Night Bus');
    expect(tileIds()).toEqual(['Night Bus/a.png', 'b.png']);
    expect(lastToastTone()).toBe('success');
    expect(lastToastMessage()).toContain('Night Bus');
    // All files shows the tile where it was: no refill is needed.
    expect(galleryApiMocks.getGallery).toHaveBeenCalledTimes(1);
  });

  it('runs a move started while another is running after it, and both land', async () => {
    const [a, b] = [top('a.png'), top('b.png')];
    const first = deferred<MoveResult>();
    galleryApiMocks.getGallery.mockResolvedValue(reply([a, b]));
    galleryApiMocks.moveAssets
      .mockReturnValueOnce(first.promise)
      .mockResolvedValueOnce({ moved: [moved('b.png', inBus('b.png'))], failed: [] });
    mountPage();
    await settle();

    await moveFromTile(a, 'Night Bus');
    await moveFromTile(b, 'Night Bus');
    expect(galleryApiMocks.moveAssets).toHaveBeenCalledTimes(1);
    expect(tileIds()).toEqual(['a.png', 'b.png']);

    first.resolve({ moved: [moved('a.png', inBus('a.png'))], failed: [] });
    await settle();

    expect(galleryApiMocks.moveAssets.mock.calls.map((call) => call[0])).toEqual([['a.png'], ['b.png']]);
    expect(tileIds()).toEqual(['Night Bus/a.png', 'Night Bus/b.png']);
  });

  it('keeps the moved file in the current job\'s outputs under its new ID', async () => {
    const a = top('a.png');
    galleryApiMocks.getGallery.mockResolvedValue(reply([a]));
    const newAsset = inBus('a.png');
    galleryApiMocks.moveAssets.mockResolvedValue({ moved: [moved('a.png', newAsset)], failed: [] });
    const replaceOutputs = vi.spyOn(jobStore, 'replaceOutputs');
    mountPage();
    await settle();

    await moveFromTile(a, 'Night Bus');

    expect(replaceOutputs).toHaveBeenCalledWith([moved('a.png', newAsset)]);
    replaceOutputs.mockRestore();
  });

  it('checks the current location and disables it in the list', async () => {
    const a = inBus('a.png');
    galleryApiMocks.getGallery.mockResolvedValue(reply([a]));
    mountPage();
    await settle();
    await openTileMove(a);

    expect(row('Night Bus').getAttribute('aria-checked')).toBe('true');
    expect(row('Night Bus').getAttribute('aria-disabled')).toBe('true');
    expect(row('No project').getAttribute('aria-checked')).toBe('false');
  });

  it('drops a file that moves out of the picked project, lowers the count, and refills the list', async () => {
    projectPick.set('Night Bus');
    const a = inBus('a.png');
    const b = inBus('b.png');
    galleryApiMocks.getGallery.mockResolvedValueOnce(reply([a, b])).mockResolvedValue(reply([b]));
    galleryApiMocks.moveAssets.mockResolvedValue({ moved: [moved('Night Bus/a.png', top('a.png'))], failed: [] });
    mountPage();
    await settle();

    await moveFromTile(a, 'No project');

    expect(galleryApiMocks.moveAssets).toHaveBeenCalledWith(['Night Bus/a.png'], null);
    expect(tileIds()).toEqual(['Night Bus/b.png']);
    expect(target.querySelector('.ui-page-sub')?.textContent).toContain('Browsing 1 loaded asset of 1');
    expect(galleryApiMocks.getGallery).toHaveBeenCalledTimes(2);
    expect(lastToastTone()).toBe('success');
  });

  it('shows a file again with its original ID after it is moved away and back', async () => {
    const asset = inBus('a.png');
    const other = inBus('z.png');
    projectPick.set('Night Bus');
    galleryApiMocks.getGallery.mockResolvedValue(reply([asset, other]));
    galleryApiMocks.moveAssets.mockResolvedValueOnce({ moved: [moved('Night Bus/a.png', top('a.png'))], failed: [] });
    mountPage();
    await settle();
    // Moving out refills the project view, whose answer no longer holds the file.
    galleryApiMocks.getGallery.mockResolvedValue(reply([other]));
    await moveFromTile(asset, 'No project');
    expect(tileIds()).toEqual(['Night Bus/z.png']);

    // Back in All files, the file is moved into the project again and regains its old ID.
    projectsStore.choose(null);
    galleryApiMocks.getGallery.mockResolvedValue(reply([top('a.png'), top('y.png')]));
    await settle();
    galleryApiMocks.moveAssets.mockResolvedValueOnce({ moved: [moved('a.png', asset)], failed: [] });
    await moveFromTile(top('a.png'), 'Night Bus');
    expect(tileIds()).toEqual(['Night Bus/a.png', 'y.png']);

    // The project's own list holds the file under that ID, and it shows.
    galleryApiMocks.getGallery.mockResolvedValue(reply([asset, other]));
    projectsStore.choose('Night Bus');
    await settle();
    expect(tileIds()).toEqual(['Night Bus/a.png', 'Night Bus/z.png']);
    expect(target.querySelector('.ui-page-sub')?.textContent).toContain('Browsing 2 loaded assets of 2');
  });

  it('points the draft slots that held the file at its new path', async () => {
    const a = top('a.png');
    galleryApiMocks.getGallery.mockResolvedValue(reply([a]));
    galleryApiMocks.moveAssets.mockResolvedValue({ moved: [moved('a.png', inBus('a.png'))], failed: [] });
    draft.reset();
    draft.setStartImage('image', { path: '/outputs/a.png' }, null);
    draft.setLastFrame({ path: '/outputs/a.png' });
    mountPage();
    await settle();

    await moveFromTile(a, 'Night Bus');

    expect(draft.state).toMatchObject({ referenceImagePath: '/outputs/Night Bus/a.png', endImagePath: '/outputs/Night Bus/a.png' });
    draft.reset();
  });

  it('follows the open viewer to the moved file under its new ID, and to the neighbour when it leaves', async () => {
    const a = top('a.png');
    const b = top('b.png');
    galleryApiMocks.getGallery.mockResolvedValue(reply([a, b]));
    galleryApiMocks.moveAssets.mockResolvedValue({ moved: [moved('a.png', inBus('a.png'))], failed: [] });
    mountPage();
    await settle();
    await openViewer('a.png');

    document.querySelector<HTMLButtonElement>('button[data-action="move"]')!.click();
    await settle();
    row('Night Bus').click();
    await settle();

    expect(getViewer()?.textContent).toContain('a.png');
    expect(getViewer()?.textContent).toContain('1 of 2');
    expect(getViewer()?.textContent).toContain('Night Bus');
  });

  it('moves the viewer on to the neighbour when the file leaves the project', async () => {
    projectPick.set('Night Bus');
    const a = inBus('a.png');
    const b = inBus('b.png');
    galleryApiMocks.getGallery.mockResolvedValue(reply([a, b]));
    galleryApiMocks.moveAssets.mockResolvedValue({ moved: [moved('Night Bus/a.png', top('a.png'))], failed: [] });
    mountPage();
    await settle();
    await openViewer('a.png');

    document.querySelector<HTMLButtonElement>('button[data-action="move"]')!.click();
    await settle();
    row('No project').click();
    await settle();

    expect(getViewer()?.textContent).toContain('b.png');
    expect(getViewer()?.textContent).not.toContain('a.png');
  });

  it('reports a failed single move as an error and leaves the tile as it was', async () => {
    const a = top('a.png');
    galleryApiMocks.getGallery.mockResolvedValue(reply([a]));
    galleryApiMocks.moveAssets.mockResolvedValue({ moved: [], failed: [{ id: 'a.png', detail: 'A file named a.png is already there.' }] });
    mountPage();
    await settle();

    await moveFromTile(a, 'Night Bus');

    expect(tileIds()).toEqual(['a.png']);
    expect(lastToastTone()).toBe('error');
    // The server's reason reaches the user.
    expect(lastToastMessage()).toContain('A file named a.png is already there');
  });

  it('reports a request that fails as Move failed', async () => {
    const a = top('a.png');
    galleryApiMocks.getGallery.mockResolvedValue(reply([a]));
    galleryApiMocks.moveAssets.mockRejectedValue(new Error('down'));
    mountPage();
    await settle();

    await moveFromTile(a, 'Night Bus');

    expect(lastToastTone()).toBe('error');
  });

  describe('selection', () => {
    const moveButton = () => queryButtonByName(target, 'Move selected');

    it('disables the header button with nothing selected, and offers the list once files are selected', async () => {
      galleryApiMocks.getGallery.mockResolvedValue(reply([top('a.png'), top('b.png')]));
      mountPage();
      await settle();
      expect(moveButton()?.disabled).toBe(true);
      expect(moveButton()?.getAttribute('aria-haspopup')).toBe('menu');
      // It sits between the count and Delete selected.
      const order = moveButton()!.compareDocumentPosition(queryButtonByName(target, 'Delete selected')!);
      expect(order & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();

      selectAssetForBatch(target, top('a.png'));
      selectAssetForBatch(target, top('b.png'));
      await settle();
      moveButton()!.click();
      await settle();

      expect(document.querySelector('.action-menu-heading')?.textContent).toBe('Move 2 files to');
      expect(row('No project').getAttribute('aria-checked')).toBe('true');
    });

    it('checks no row for a selection that spans locations', async () => {
      galleryApiMocks.getGallery.mockResolvedValue(reply([top('a.png'), inBus('b.png')]));
      mountPage();
      await settle();
      selectAssetForBatch(target, top('a.png'));
      selectAssetForBatch(target, inBus('b.png'));
      await settle();
      moveButton()!.click();
      await settle();

      expect(rows().filter((item) => item.getAttribute('aria-checked') === 'true')).toHaveLength(0);
    });

    it('moves the selection in one call; a refused file stays selected and the toast says so', async () => {
      const [a, b, c] = [top('a.png'), top('b.png'), top('c.png')];
      galleryApiMocks.getGallery.mockResolvedValue(reply([a, b, c]));
      galleryApiMocks.moveAssets.mockResolvedValue({
        moved: [moved('a.png', inBus('a.png')), moved('b.png', inBus('b.png'))],
        failed: [{ id: 'c.png', detail: 'A file named c.png is already there.' }],
      });
      mountPage();
      await settle();
      [a, b, c].forEach((asset) => selectAssetForBatch(target, asset));
      await settle();

      moveButton()!.click();
      await settle();
      row('Night Bus').click();
      await settle();

      expect(galleryApiMocks.moveAssets).toHaveBeenCalledTimes(1);
      expect(galleryApiMocks.moveAssets).toHaveBeenCalledWith(['a.png', 'b.png', 'c.png'], 'Night Bus');
      expect(tileIds()).toEqual(['Night Bus/a.png', 'Night Bus/b.png', 'c.png']);
      expect(target.textContent).toContain('1 selected');
      expect(target.querySelector<HTMLInputElement>('input[aria-label="Select c.png"]')?.checked).toBe(true);
      expect(lastToastTone()).toBe('warning');
      expect(lastToastMessage()).toContain('A file named c.png is already there');
    });

    it('counts a file already in the destination as done: it leaves the selection and nothing is reported as an error', async () => {
      const [a, b] = [top('a.png'), inBus('b.png')];
      galleryApiMocks.getGallery.mockResolvedValue(reply([a, b]));
      // The server leaves a file that is already in the target out of both lists.
      galleryApiMocks.moveAssets.mockResolvedValue({ moved: [moved('a.png', inBus('a.png'))], failed: [] });
      mountPage();
      await settle();
      [a, b].forEach((asset) => selectAssetForBatch(target, asset));
      await settle();

      moveButton()!.click();
      await settle();
      row('Night Bus').click();
      await settle();

      expect(target.textContent).toContain('0 selected');
      expect(lastToastTone()).toBe('success');
    });

    it('does not report an error when every file was already in the destination', async () => {
      const b = inBus('b.png');
      galleryApiMocks.getGallery.mockResolvedValue(reply([b]));
      galleryApiMocks.moveAssets.mockResolvedValue({ moved: [], failed: [] });
      mountPage();
      await settle();
      selectAssetForBatch(target, b);
      await settle();

      moveButton()!.click();
      await settle();
      row('No project').click();
      await settle();

      expect(target.textContent).toContain('0 selected');
      expect(lastToastTone()).not.toBe('error');
    });

    it('shows the short Move label on a tablet, named Move selected', async () => {
      layoutState.mode = 'tall';
      galleryApiMocks.getGallery.mockResolvedValue(reply([top('a.png')]));
      mountPage();
      await settle();

      const button = target.querySelector<HTMLButtonElement>('button[aria-label="Move selected"]')!;
      expect(button.textContent?.trim()).toBe('Move');
      expect(button.disabled).toBe(true);
    });

    it('puts Move on the phone select bar between Select all and Delete, and moves the selection from there', async () => {
      layoutState.mode = 'compact';
      const [a, b] = [top('a.png'), top('b.png')];
      galleryApiMocks.getGallery.mockResolvedValue(reply([a, b]));
      galleryApiMocks.moveAssets.mockResolvedValue({ moved: [moved('a.png', inBus('a.png'))], failed: [] });
      mountPage();
      await settle();
      queryButtonByName(target, 'Select')!.click();
      await settle();

      const bar = target.querySelector<HTMLElement>('[data-testid="gallery-select-bar"]')!;
      expect(Array.from(bar.querySelectorAll('button')).map((el) => el.textContent?.trim())).toEqual(['Select all', 'Move', 'Delete']);
      expect(queryButtonByName(bar, 'Move')?.disabled).toBe(true);

      selectAssetForBatch(target, a);
      await settle();
      queryButtonByName(bar, 'Move')!.click();
      await settle();
      // The phone's list is a sheet, which names itself: no heading.
      expect(document.querySelector('.action-menu-heading')).toBeNull();
      row('Night Bus').click();
      await settle();

      expect(galleryApiMocks.moveAssets).toHaveBeenCalledWith(['a.png'], 'Night Bus');
      expect(lastToastTone()).toBe('success');
    });

    it('reads Moving… and stays disabled while a move runs', async () => {
      const pending = deferred<MoveResult>();
      const a = top('a.png');
      galleryApiMocks.getGallery.mockResolvedValue(reply([a]));
      galleryApiMocks.moveAssets.mockReturnValue(pending.promise);
      mountPage();
      await settle();
      selectAssetForBatch(target, a);
      await settle();
      moveButton()!.click();
      await settle();
      row('Night Bus').click();
      await settle();

      const busy = queryButtonByName(target, 'Moving…')!;
      expect(busy.disabled).toBe(true);
      pending.resolve({ moved: [moved('a.png', inBus('a.png'))], failed: [] });
      await settle();
      expect(queryButtonByName(target, 'Moving…')).toBeNull();
    });

    it('says how many files moved and where when all of them did', async () => {
      const [a, b] = [top('a.png'), top('b.png')];
      galleryApiMocks.getGallery.mockResolvedValue(reply([a, b]));
      galleryApiMocks.moveAssets.mockResolvedValue({ moved: [moved('a.png', inBus('a.png')), moved('b.png', inBus('b.png'))], failed: [] });
      mountPage();
      await settle();
      [a, b].forEach((asset) => selectAssetForBatch(target, asset));
      await settle();

      moveButton()!.click();
      await settle();
      row('Night Bus').click();
      await settle();

      expect(lastToastTone()).toBe('success');
      expect(lastToastMessage()).toContain('Night Bus');
      expect(target.textContent).toContain('0 selected');
    });
  });
});
