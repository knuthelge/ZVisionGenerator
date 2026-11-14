import { flushSync } from 'svelte';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { galleryStore } from '$lib/state/gallery.svelte';
import { draft } from '$lib/state/draft.svelte';
import { jobStore } from '$lib/state/job.svelte';
import { galleryMemory, workspaceMemory } from '$lib/state/pageMemory';
import { projectPick } from '$lib/state/projectPick.svelte';
import { projectsStore } from '$lib/state/projects.svelte';
import { hasPendingReference, takeReference } from '$lib/state/pendingReference.svelte';
import { modeLine } from './selection';
import type { DeleteResult, GalleryAsset, GalleryPage as GalleryPageResponse, MoveResult, RemovedAsset, SavedVoice, WorkspaceContext } from '$lib/types';

const galleryApiMocks = vi.hoisted(() => ({
  getGallery: vi.fn<() => Promise<GalleryPageResponse>>(),
  getGalleryIds: vi.fn<(filter?: string, project?: string | null, starred?: boolean) => Promise<string[]>>(),
  deleteAssets: vi.fn<(ids: string[]) => Promise<DeleteResult>>(),
  downloadAssets: vi.fn<(ids: readonly string[]) => void>(),
  setStarred: vi.fn<(id: string, starred: boolean) => Promise<{ id: string; starred: boolean }>>(),
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
    deleteAssets: galleryApiMocks.deleteAssets,
    downloadAssets: galleryApiMocks.downloadAssets,
    getGalleryIds: galleryApiMocks.getGalleryIds,
    setStarred: galleryApiMocks.setStarred,
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

/** The bulk action button of the Select mode's header (or the phone bar) by its fixed accessible name. */
function bulkButton(name: 'Done' | 'Select all' | 'Move selected' | 'Download selected' | 'Delete selected'): HTMLButtonElement | null {
  return Array.from(document.querySelectorAll<HTMLButtonElement>('#gallery-view button')).find((button) => (button.getAttribute('aria-label') ?? button.textContent?.trim()) === name) ?? null;
}

/** What a bulk Delete or Move reports for one file. */
function removed(asset: GalleryAsset): RemovedAsset {
  return { id: asset.id, media_type: asset.media_type, project: asset.project, starred: asset.starred ?? false, file_path: asset.file_path };
}

/** Turn Select mode on from the header's Select button. */
async function enterSelectMode(): Promise<void> {
  queryButtonByName(target, 'Select')!.click();
  await settle();
}

function selectAssetForBatch(container: ParentNode, asset: GalleryAsset): HTMLInputElement {
  const checkbox = container.querySelector(`input[aria-label="Select ${asset.filename}"]`) as HTMLInputElement | null;
  expect(checkbox).not.toBeNull();
  checkbox!.checked = true;
  checkbox!.dispatchEvent(new Event('change', { bubbles: true }));
  flushSync();
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
  // A tile is named View outside Select mode; inside it, the expand button opens the viewer.
  const opener = target.querySelector(`button[aria-label="View ${filename}"]`) ?? target.querySelector(`button[aria-label="View ${filename} fullscreen"]`);
  (opener as HTMLElement).dispatchEvent(new MouseEvent('click', { bubbles: true }));
  await settle();
}

beforeEach(() => {
  layoutState.mode = 'wide';
  projectPick.set(null);
  galleryMemory.forget();
  galleryStore.reset();
  galleryApiMocks.getGallery.mockReset();
  galleryApiMocks.deleteAssets.mockReset();
  galleryApiMocks.downloadAssets.mockReset();
  galleryApiMocks.getGalleryIds.mockReset();
  galleryApiMocks.setStarred.mockReset().mockImplementation(async (id, starred) => ({ id, starred }));
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

describe('GalleryPage Select mode', () => {
  const tiles = ['a.png', 'b.png', 'c.png', 'd.png', 'e.png', 'f.png'].map((name) => makeAsset({ id: name, filename: name, prompt: `prompt of ${name}` }));
  const reply = (assets: GalleryAsset[], total = assets.length) => ({ assets, page: 1, total_pages: 1, total_count: total });

  const media = (filename: string) => target.querySelector<HTMLButtonElement>(`button[aria-label$=" ${filename}"]:not([aria-label$=" fullscreen"]):not([aria-label^="More"])`)!;
  const checkbox = (filename: string) => target.querySelector<HTMLInputElement>(`input[aria-label="Select ${filename}"]`)!;
  const selectedNames = () => tiles.filter((asset) => checkbox(asset.filename)?.checked).map((asset) => asset.filename);
  const header = () => target.querySelector('header')!;
  const modeLineText = () => target.querySelector('.ui-page-sub')?.textContent ?? '';
  const status = () => target.querySelector('[role="status"]:not([aria-label])')?.textContent ?? '';
  const click = (el: HTMLElement, init: MouseEventInit = {}) => { el.dispatchEvent(new MouseEvent('click', { bubbles: true, ...init })); flushSync(); };
  const press = (key: string, init: KeyboardEventInit = {}) => {
    (document.activeElement ?? document.body).dispatchEvent(new KeyboardEvent('keydown', { key, bubbles: true, cancelable: true, ...init }));
    flushSync();
  };

  async function mountTiles(assets: GalleryAsset[] = tiles): Promise<void> {
    galleryApiMocks.getGallery.mockResolvedValue(reply(assets));
    mountPage();
    await settle();
  }

  it('opens the viewer on a click outside the mode and selects nothing', async () => {
    await mountTiles();
    expect(media('a.png').getAttribute('aria-label')).toBe('View a.png');
    expect(media('a.png').hasAttribute('aria-pressed')).toBe(false);

    click(media('a.png'));
    await settle();

    expect(getViewer()?.textContent).toContain('prompt of a.png');
    expect(selectedNames()).toEqual([]);
    expect(queryButtonByName(target, 'Select')).not.toBeNull();
    expect(bulkButton('Done')).toBeNull();
  });

  it('enters the mode with the tile selected when its checkbox is ticked outside the mode', async () => {
    await mountTiles();

    selectAssetForBatch(target, tiles[2]);
    await settle();

    expect(bulkButton('Done')?.getAttribute('aria-pressed')).toBe('true');
    expect(selectedNames()).toEqual(['c.png']);
    expect(status()).toBe('1 selected');
    expect(media('c.png').getAttribute('aria-label')).toBe('Select c.png');
    expect(media('c.png').getAttribute('aria-pressed')).toBe('true');
    expect(media('a.png').getAttribute('aria-pressed')).toBe('false');
  });

  it('enters the mode from the Select button, focuses Done, and Done leaves it, clears the selection and focuses Select', async () => {
    await mountTiles();
    await enterSelectMode();

    expect(document.activeElement).toBe(bulkButton('Done'));
    expect(status()).toBe('0 selected');
    click(media('a.png'));
    expect(selectedNames()).toEqual(['a.png']);

    click(bulkButton('Done')!);
    await settle();

    expect(selectedNames()).toEqual([]);
    expect(bulkButton('Done')).toBeNull();
    expect(document.activeElement).toBe(queryButtonByName(target, 'Select'));
    expect(media('a.png').getAttribute('aria-label')).toBe('View a.png');
  });

  it('enters the mode with the tile selected from the tile menu, whose Select item leads outside the mode', async () => {
    await mountTiles();
    target.querySelector<HTMLButtonElement>('button[aria-label="More actions for b.png"]')!.click();
    flushSync();
    const items = Array.from(document.querySelectorAll<HTMLElement>('[role="menu"] [data-action]'));
    expect(items[0].dataset.action).toBe('select');

    items[0].click();
    await settle();

    expect(selectedNames()).toEqual(['b.png']);
    expect(bulkButton('Done')).not.toBeNull();
    expect(document.activeElement).toBe(media('b.png'));
  });

  it('leads the tile menu with View inside the mode, and View opens the viewer without changing the selection', async () => {
    await mountTiles();
    await enterSelectMode();
    click(media('a.png'));
    target.querySelector<HTMLButtonElement>('button[aria-label="More actions for b.png"]')!.click();
    flushSync();
    const lead = document.querySelector<HTMLElement>('[role="menu"] [data-action]')!;
    expect(lead.dataset.action).toBe('view');

    lead.click();
    await settle();

    expect(getViewer()?.textContent).toContain('prompt of b.png');
    expect(selectedNames()).toEqual(['a.png']);
  });

  it('enters the mode and toggles the focused tile with Space and X', async () => {
    await mountTiles();
    media('b.png').focus();

    press(' ');
    await settle();
    expect(bulkButton('Done')).not.toBeNull();
    expect(selectedNames()).toEqual(['b.png']);

    press('x');
    expect(selectedNames()).toEqual([]);
    // The mode stays on with nothing selected.
    expect(bulkButton('Done')).not.toBeNull();
  });

  it('enters the mode and selects every matching file with Ctrl+A, loaded or not', async () => {
    await mountTiles(tiles.slice(0, 3));
    const ids = Array.from({ length: 30 }, (_, index) => `file-${index}.png`);
    galleryApiMocks.getGalleryIds.mockResolvedValue(ids);

    press('a', { ctrlKey: true });
    await settle();

    expect(galleryApiMocks.getGalleryIds).toHaveBeenCalledWith('all', null, false);
    expect(bulkButton('Done')).not.toBeNull();
    expect(status()).toBe('30 selected');
  });

  it('toggles a tile with a click in the mode, and the media button names what a click does', async () => {
    await mountTiles();
    await enterSelectMode();

    click(media('d.png'));
    expect(media('d.png').getAttribute('aria-pressed')).toBe('true');
    expect(checkbox('d.png').checked).toBe(true);
    click(media('d.png'));
    expect(media('d.png').getAttribute('aria-pressed')).toBe('false');
    expect(selectedNames()).toEqual([]);
    expect(getViewer()).toBeNull();
  });

  it('selects the run from the anchor with Shift-click, in grid order either way, and never removes', async () => {
    await mountTiles();
    await enterSelectMode();

    click(media('b.png'));
    click(media('e.png'), { shiftKey: true });
    expect(selectedNames()).toEqual(['b.png', 'c.png', 'd.png', 'e.png']);

    click(media('a.png'), { shiftKey: true });
    expect(selectedNames()).toEqual(['a.png', 'b.png', 'c.png', 'd.png', 'e.png']);
  });

  it('opens the viewer on a Shift-click outside the mode, like a click', async () => {
    await mountTiles();

    click(media('c.png'), { shiftKey: true });
    await settle();

    expect(getViewer()?.textContent).toContain('prompt of c.png');
    expect(bulkButton('Done')).toBeNull();
  });

  it('opens the viewer in the mode from the expand button and from Enter', async () => {
    await mountTiles();
    await enterSelectMode();
    click(media('a.png'));

    click(target.querySelector<HTMLButtonElement>('button[aria-label="View b.png fullscreen"]')!);
    await settle();
    expect(getViewer()?.textContent).toContain('prompt of b.png');
    expect(selectedNames()).toEqual(['a.png']);

    document.querySelector<HTMLButtonElement>('[data-testid="asset-viewer"] button[aria-label="Close"]')?.click();
    await settle();
    media('c.png').focus();
    press('Enter');
    await settle();
    expect(getViewer()?.textContent).toContain('prompt of c.png');
    expect(selectedNames()).toEqual(['a.png']);
  });

  it('leaves the mode and clears the selection on Escape', async () => {
    await mountTiles();
    await enterSelectMode();
    click(media('a.png'));
    click(media('b.png'));
    media('a.png').focus();

    press('Escape');
    await settle();

    expect(selectedNames()).toEqual([]);
    expect(bulkButton('Done')).toBeNull();
    expect(queryButtonByName(target, 'Select')).not.toBeNull();
  });

  for (const mode of ['wide', 'compact'] as const) {
    it(`hides the filter, the star toggle and the sort in the mode on ${mode}, and shows them at rest`, async () => {
      layoutState.mode = mode;
      await mountTiles();
      const controls = () => ({
        filter: target.querySelector('[aria-label="Filter gallery media"]'),
        starred: target.querySelector('button[aria-label="Starred"]'),
        sort: target.querySelector('[aria-label="Sort gallery files"]'),
      });
      expect(Object.values(controls()).every((el) => el !== null)).toBe(true);

      await enterSelectMode();

      expect(Object.values(controls()).every((el) => el === null)).toBe(true);
    });
  }

  it('offers Done, Select all, Move, Download and Delete in the header on a wide screen', async () => {
    await mountTiles();
    await enterSelectMode();

    const names = Array.from(header().querySelectorAll('button')).map((el) => el.getAttribute('aria-label') ?? el.textContent?.trim());
    expect(names).toEqual(expect.arrayContaining(['Done', 'Select all', 'Move selected', 'Download selected', 'Delete selected']));
    expect(target.querySelector('[data-testid="gallery-select-bar"]')).toBeNull();
  });

  it('offers Select all, Move, Download and Delete in the bar at the foot on the phone', async () => {
    layoutState.mode = 'compact';
    await mountTiles();
    await enterSelectMode();

    const bar = target.querySelector<HTMLElement>('[data-testid="gallery-select-bar"]')!;
    const names = Array.from(bar.querySelectorAll('button')).map((el) => el.getAttribute('aria-label') ?? el.textContent?.trim());
    expect(names).toEqual(expect.arrayContaining(['Select all', 'Move selected', 'Download selected', 'Delete selected']));
    expect(bulkButton('Done')).not.toBeNull();
  });

  it('describes the view being selected from in the mode line, without a count', async () => {
    await mountTiles();
    pickOption(target.querySelector('[role="combobox"][aria-label="Filter gallery media"]') as HTMLButtonElement, 'video');
    await settle();
    target.querySelector<HTMLButtonElement>('button[aria-label="Starred"]')!.click();
    await settle();
    await enterSelectMode();

    expect(modeLineText()).toBe(modeLine({ project: null, starred: true, filter: 'video', coarse: false }));
  });

  it('selects every matching file with Select all, shows the server\'s count, and sends one request while it is busy', async () => {
    await mountTiles();
    await enterSelectMode();
    const ids = Array.from({ length: 240 }, (_, index) => `file-${index}.png`);
    const answer = deferred<string[]>();
    galleryApiMocks.getGalleryIds.mockReturnValue(answer.promise);

    click(bulkButton('Select all')!);
    await settle();
    click(bulkButton('Select all')!);
    expect(galleryApiMocks.getGalleryIds).toHaveBeenCalledTimes(1);

    answer.resolve(ids);
    await settle();

    expect(status()).toBe('240 selected');
  });

  it('asks for the IDs of the view the mode was entered from: media filter, picked project and the star toggle', async () => {
    projectPick.set('Night Bus');
    galleryApiMocks.getGallery.mockResolvedValue(reply([makeAsset({ id: 'Night Bus/a.png', filename: 'a.png', project: 'Night Bus' })]));
    mountPage();
    await settle();
    pickOption(target.querySelector('[role="combobox"][aria-label="Filter gallery media"]') as HTMLButtonElement, 'image');
    await settle();
    target.querySelector<HTMLButtonElement>('button[aria-label="Starred"]')!.click();
    await settle();
    galleryApiMocks.getGalleryIds.mockResolvedValue(['Night Bus/a.png']);
    await enterSelectMode();

    click(bulkButton('Select all')!);
    await settle();

    expect(galleryApiMocks.getGalleryIds).toHaveBeenCalledWith('image', 'Night Bus', true);
  });

  it('reports a Select all that fails and leaves the selection as it was', async () => {
    await mountTiles();
    await enterSelectMode();
    click(media('a.png'));
    galleryApiMocks.getGalleryIds.mockRejectedValue(new Error('offline'));

    click(bulkButton('Select all')!);
    await settle();

    expect(toastMocks.addToast).toHaveBeenCalledWith(expect.stringContaining('offline'), 'error');
    expect(selectedNames()).toEqual(['a.png']);
  });

  it('keeps the selected files that are not loaded when page one refills, and shows them selected once their page loads', async () => {
    const pageTwo = makeAsset({ id: 'later.png', filename: 'later.png' });
    galleryApiMocks.getGallery
      .mockResolvedValueOnce({ assets: tiles.slice(0, 2), page: 1, total_pages: 2, total_count: 3 })
      .mockResolvedValue({ assets: tiles.slice(0, 2), page: 1, total_pages: 2, total_count: 3 });
    mountPage();
    await settle();
    galleryApiMocks.getGalleryIds.mockResolvedValue(['a.png', 'b.png', 'later.png']);
    press('a', { ctrlKey: true });
    await settle();
    expect(status()).toBe('3 selected');

    // A delete elsewhere refills page one: the unloaded selected file stays selected.
    galleryApiMocks.deleteAssets.mockResolvedValue({ deleted: [], failed: [] });
    deleteAssetFromCard(target, tiles[1]);
    await settle();
    expect(status()).toBe('2 selected');

    galleryApiMocks.getGallery.mockResolvedValue({ assets: [pageTwo], page: 2, total_pages: 2, total_count: 2 });
    MockIntersectionObserver.instances.at(-1)?.trigger(true);
    await settle();
    expect(checkbox('later.png').checked).toBe(true);
  });

  it('leaves the mode when the view changes, here a project pick', async () => {
    await mountTiles();
    await enterSelectMode();
    click(media('a.png'));

    projectsStore.choose('Night Bus');
    await settle();

    expect(bulkButton('Done')).toBeNull();
    expect(queryButtonByName(target, 'Select')).not.toBeNull();
    expect(selectedNames()).toEqual([]);
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
    const sortSelect = target.querySelector('[role="combobox"][aria-label="Sort gallery files"]') as HTMLButtonElement | null;
    expect(filterSelect).not.toBeNull();
    expect(sortSelect).not.toBeNull();
    expect(galleryApiMocks.getGallery).toHaveBeenNthCalledWith(1, 1, 'all', 'newest', null, false);

    pickOption(filterSelect!, 'video');
    await settle();
    expect(galleryApiMocks.getGallery).toHaveBeenNthCalledWith(2, 1, 'video', 'newest', null, false);
    expect(filterSelect!.dataset.value).toBe('video');

    pickOption(sortSelect!, 'oldest');
    await settle();
    expect(galleryApiMocks.getGallery).toHaveBeenNthCalledWith(3, 1, 'video', 'oldest', null, false);
    expect(sortSelect!.dataset.value).toBe('oldest');

    MockIntersectionObserver.instances.at(-1)?.trigger(true);
    await settle();

    expect(galleryApiMocks.getGallery).toHaveBeenNthCalledWith(4, 2, 'video', 'oldest', null, false);
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
    galleryApiMocks.deleteAssets.mockResolvedValue({ deleted: [], failed: [] });
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
    galleryApiMocks.deleteAssets.mockResolvedValue({ deleted: [], failed: [] });
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
    galleryApiMocks.deleteAssets.mockResolvedValue({ deleted: [], failed: [] });
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
    galleryApiMocks.deleteAssets.mockResolvedValue({ deleted: [], failed: [] });
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
    galleryApiMocks.deleteAssets.mockResolvedValue({ deleted: [], failed: [] });
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
    galleryApiMocks.deleteAssets.mockResolvedValue({ deleted: [], failed: [] });
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
    expect(galleryApiMocks.getGallery).toHaveBeenLastCalledWith(1, 'all', 'newest', null, false);
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
    galleryApiMocks.deleteAssets.mockResolvedValue({ deleted: [], failed: [] });
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

    expect(galleryApiMocks.getGallery).toHaveBeenNthCalledWith(2, 1, 'all', 'newest', null, false);
    expect(galleryApiMocks.getGallery).toHaveBeenNthCalledWith(3, 2, 'all', 'newest', null, false);
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

    expect(target.textContent).toContain('No matching files');

    const clearButton = Array.from(target.querySelectorAll('button')).find((button) => button.textContent?.trim() === 'Show all media');
    expect(clearButton).not.toBeUndefined();
    clearButton!.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    await settle();

    expect(galleryApiMocks.getGallery).toHaveBeenNthCalledWith(3, 1, 'all', 'newest', null, false);
    expect(filterSelect!.dataset.value).toBe('all');
  });

  it('shows a workspace recovery action when the gallery is genuinely empty', async () => {
    galleryApiMocks.getGallery.mockResolvedValue({ assets: [], page: 1, total_pages: 1, total_count: 0 });

    mountPage();
    await settle();

    expect(target.textContent).toContain('No generated files yet');

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
    expect(target.textContent).toContain('Browsing 1 loaded file of 1');
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

    expect(galleryApiMocks.getGallery).toHaveBeenLastCalledWith(2, 'video', 'newest', null, false);
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

  const reply = (assets: GalleryAsset[], total = assets.length) => ({ assets, page: 1, total_pages: 1, total_count: total });
  const checkboxOf = (asset: GalleryAsset) => target.querySelector<HTMLInputElement>(`input[aria-label="Select ${asset.filename}"]`);

  it('keeps failed IDs selected and preserves selections added while a mixed deletion is in flight', async () => {
    const assetA = makeAsset({ id: 'a.png', filename: 'a.png' });
    const assetB = makeAsset({ id: 'b.png', filename: 'b.png' });
    const assetC = makeAsset({ id: 'c.png', filename: 'c.png' });
    const answer = deferred<DeleteResult>();
    galleryApiMocks.getGallery.mockResolvedValue(reply([assetA, assetB, assetC]));
    galleryApiMocks.deleteAssets.mockReturnValue(answer.promise);

    mountPage();
    await settle();
    await openViewer(assetA.filename);
    selectAssetForBatch(target, assetA);
    selectAssetForBatch(target, assetB);
    await settle();

    const deleteButton = bulkButton('Delete selected')!;
    deleteButton.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    await settle();
    expect(deleteButton.disabled).toBe(true);
    deleteButton.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    expect(galleryApiMocks.deleteAssets).toHaveBeenCalledTimes(1);

    selectAssetForBatch(target, assetC);
    answer.resolve({ deleted: [removed(assetA)], failed: [{ id: 'b.png', detail: 'blocked' }] });
    await settle();

    expect(target.textContent).not.toContain('a.png');
    expect(target.textContent).toContain('b.png');
    expect(target.textContent).toContain('c.png');
    expect(target.textContent).toContain('2 selected');
    // The viewer moves on to the neighbouring asset instead of closing.
    expect(getViewer()?.textContent).toContain('b.png');
    expect(bulkButton('Delete selected')?.disabled).toBe(false);
    expect(toastMocks.addToast).toHaveBeenCalledWith(expect.any(String), 'warning', {
      action: expect.objectContaining({ label: 'Retry' })
    });
  });

  it('removes only the files the server reports deleted, moves the viewer past them, and leaves the mode on a clean run', async () => {
    const assetA = makeAsset({ id: 'a.png', filename: 'a.png' });
    const assetB = makeAsset({ id: 'b.png', filename: 'b.png' });
    const assetC = makeAsset({ id: 'c.png', filename: 'c.png' });
    galleryApiMocks.getGallery.mockResolvedValue(reply([assetA, assetB, assetC]));
    galleryApiMocks.deleteAssets.mockResolvedValue({ deleted: [removed(assetA), removed(assetB)], failed: [] });

    mountPage();
    await settle();
    await openViewer(assetA.filename);
    selectAssetForBatch(target, assetA);
    selectAssetForBatch(target, assetB);
    bulkButton('Delete selected')!.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    await settle();

    expect(target.textContent).not.toContain('a.png');
    expect(target.textContent).not.toContain('b.png');
    expect(target.textContent).toContain('c.png');
    expect(target.textContent).toContain('Browsing 1 loaded file of 1');
    expect(getViewer()?.textContent).toContain('c.png');
    expect(toastMocks.addToast).toHaveBeenCalledWith(expect.stringContaining('2'), 'success');
    // No file failed: the gallery is back to browsing, with focus on Select.
    expect(bulkButton('Done')).toBeNull();
    expect(document.activeElement).toBe(queryButtonByName(target, 'Select'));
  });

  it('retains all failed files, their active detail, retryability, and an error outcome, and the mode stays on', async () => {
    const assetA = makeAsset({ id: 'a.png', filename: 'a.png' });
    const assetB = makeAsset({ id: 'b.png', filename: 'b.png', prompt: 'active failed asset' });
    galleryApiMocks.getGallery.mockResolvedValue(reply([assetA, assetB]));
    galleryApiMocks.deleteAssets.mockResolvedValue({ deleted: [], failed: [{ id: 'a.png', detail: 'blocked' }, { id: 'b.png', detail: 'blocked' }] });

    mountPage();
    await settle();
    await openViewer(assetB.filename);
    selectAssetForBatch(target, assetA);
    selectAssetForBatch(target, assetB);
    bulkButton('Delete selected')!.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    await settle();

    expect(target.textContent).toContain('a.png');
    expect(target.textContent).toContain('b.png');
    expect(target.textContent).toContain('2 selected');
    expect(getViewer()?.textContent).toContain('active failed asset');
    expect(bulkButton('Delete selected')?.disabled).toBe(false);
    expect(document.activeElement).toBe(bulkButton('Delete selected'));
    expect(toastMocks.addToast).toHaveBeenCalledWith(expect.any(String), 'error', {
      action: expect.objectContaining({ label: 'Retry' })
    });
  });

  it('re-runs the delete for the files that failed when Retry is pressed', async () => {
    const assetA = makeAsset({ id: 'a.png', filename: 'a.png' });
    galleryApiMocks.getGallery.mockResolvedValue(reply([assetA]));
    galleryApiMocks.deleteAssets
      .mockResolvedValueOnce({ deleted: [], failed: [{ id: 'a.png', detail: 'blocked' }] })
      .mockResolvedValueOnce({ deleted: [removed(assetA)], failed: [] });
    toastMocks.addToast.mockReturnValue('toast-1');

    mountPage();
    await settle();
    selectAssetForBatch(target, assetA);
    bulkButton('Delete selected')!.click();
    await settle();
    const retry = toastMocks.addToast.mock.calls.find((call) => call[2]?.action?.label === 'Retry')![2]!.action!;
    retry.run();
    await settle();

    expect(galleryApiMocks.deleteAssets).toHaveBeenCalledTimes(2);
    expect(galleryApiMocks.deleteAssets).toHaveBeenLastCalledWith(['a.png']);
    expect(bulkButton('Done')).toBeNull();
  });

  it('keeps the selection and the mode, with a Retry toast, when the delete request itself fails', async () => {
    const asset = makeAsset({ id: 'a.png', filename: 'a.png' });
    galleryApiMocks.getGallery.mockResolvedValue(reply([asset]));
    galleryApiMocks.deleteAssets.mockRejectedValue(new Error('offline'));

    mountPage();
    await settle();
    selectAssetForBatch(target, asset);
    bulkButton('Delete selected')!.click();
    await settle();

    expect(toastMocks.addToast).toHaveBeenCalledWith('offline', 'error', { action: expect.objectContaining({ label: 'Retry' }) });
    expect(target.textContent).toContain('1 selected');
    expect(bulkButton('Done')).not.toBeNull();
  });

  it('closes its Retry toast when the page unmounts, so Retry cannot act on a page that is gone', async () => {
    const asset = makeAsset({ id: 'a.png', filename: 'a.png' });
    galleryApiMocks.getGallery.mockResolvedValue(reply([asset]));
    galleryApiMocks.deleteAssets.mockResolvedValue({ deleted: [], failed: [{ id: 'a.png', detail: 'blocked' }] });
    toastMocks.addToast.mockReturnValue('toast-7');

    mountPage();
    await settle();
    selectAssetForBatch(target, asset);
    bulkButton('Delete selected')!.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    await settle();
    expect(toastMocks.dismissToast).not.toHaveBeenCalled();

    await page.unmount();
    expect(toastMocks.dismissToast).toHaveBeenCalledWith('toast-7');
  });

  it('asks to confirm the count and sends every selected ID in one request', async () => {
    const assets = ['a', 'b', 'c'].map((name) => makeAsset({ id: `${name}.png`, filename: `${name}.png` }));
    galleryApiMocks.getGallery.mockResolvedValue(reply(assets));
    galleryApiMocks.deleteAssets.mockResolvedValue({ deleted: assets.map(removed), failed: [] });

    mountPage();
    await settle();
    assets.forEach((asset) => selectAssetForBatch(target, asset));
    bulkButton('Delete selected')!.click();
    await settle();

    expect(confirmMocks.requestConfirm).toHaveBeenCalledWith(expect.objectContaining({ question: expect.stringContaining('3'), confirmLabel: 'Delete' }));
    expect(galleryApiMocks.deleteAssets).toHaveBeenCalledTimes(1);
    expect(galleryApiMocks.deleteAssets).toHaveBeenCalledWith(['a.png', 'b.png', 'c.png']);
  });

  describe('with files that are not loaded', () => {
    // 30 files match; the first page holds 12.
    const names = Array.from({ length: 30 }, (_, index) => `f${String(index).padStart(2, '0')}.png`);
    const loaded = names.slice(0, 12).map((name) => makeAsset({ id: name, filename: name }));

    async function mountThirty(): Promise<void> {
      galleryApiMocks.getGallery.mockResolvedValue({ assets: loaded, page: 1, total_pages: 3, total_count: 30 });
      galleryApiMocks.getGalleryIds.mockResolvedValue(names);
      mountPage();
      await settle();
      await enterSelectMode();
      bulkButton('Select all')!.click();
      await settle();
    }

    it('keeps the other 29 selected, and the mode on, when one loaded file is deleted from its menu', async () => {
      await mountThirty();
      expect(target.textContent).toContain('30 selected');
      galleryApiMocks.deleteAssets.mockResolvedValue({ deleted: [], failed: [] });

      deleteAssetFromCard(target, loaded[0]);
      await settle();

      expect(target.textContent).toContain('29 selected');
      expect(bulkButton('Done')).not.toBeNull();
    });

    it('deletes all 30 in one request, unloaded files included, and asks about 30', async () => {
      await mountThirty();
      galleryApiMocks.deleteAssets.mockResolvedValue({ deleted: names.map((name) => ({ id: name, media_type: 'image', project: null, starred: false, file_path: `/outputs/${name}` })), failed: [] });

      bulkButton('Delete selected')!.click();
      await settle();

      expect(confirmMocks.requestConfirm).toHaveBeenCalledWith(expect.objectContaining({ question: expect.stringContaining('30') }));
      expect(galleryApiMocks.deleteAssets).toHaveBeenCalledWith(names);
      expect(bulkButton('Done')).toBeNull();
    });

    it('leaves the two unloaded files that failed selected, with the mode on', async () => {
      await mountThirty();
      const failedNames = names.slice(20, 22);
      const deletedNames = names.filter((name) => !failedNames.includes(name));
      galleryApiMocks.deleteAssets.mockResolvedValue({
        deleted: deletedNames.map((name) => ({ id: name, media_type: 'image' as const, project: null, starred: false, file_path: `/outputs/${name}` })),
        failed: failedNames.map((id) => ({ id, detail: 'Could not delete the file: Permission denied' })),
      });

      bulkButton('Delete selected')!.click();
      await settle();

      expect(target.textContent).toContain('2 selected');
      expect(bulkButton('Done')).not.toBeNull();
      expect(toastMocks.addToast).toHaveBeenCalledWith(expect.stringContaining(String(deletedNames.length)), 'warning', { action: expect.objectContaining({ label: 'Retry' }) });
      // Retry asks again for exactly those two.
      galleryApiMocks.deleteAssets.mockResolvedValue({ deleted: [], failed: [] });
      toastMocks.addToast.mock.calls.at(-1)![2]!.action!.run();
      await settle();
      expect(galleryApiMocks.deleteAssets).toHaveBeenLastCalledWith(failedNames);
    });
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
    await openViewer(image.filename);
    await settle();
    expect(document.querySelector('[role="dialog"]')).not.toBeNull();

    const filter = target.querySelector('[role="combobox"][aria-label="Filter gallery media"]') as HTMLButtonElement;
    const sort = target.querySelector('[role="combobox"][aria-label="Sort gallery files"]') as HTMLButtonElement;
    routerMocks.replace.mockReset();
    pickOption(filter, 'video');
    await settle();

    expect(galleryApiMocks.getGallery).toHaveBeenNthCalledWith(2, 1, 'video', 'newest', null, false);
    expect(getViewer()).toBeNull();
    expect(document.querySelector('[role="dialog"]')).toBeNull();

    pickOption(sort, 'oldest');
    await settle();
    expect(galleryApiMocks.getGallery).toHaveBeenNthCalledWith(3, 1, 'video', 'oldest', null, false);

    // The empty filtered view exposes the clear-filter path, which must use the
    // same reset and retain the selected sort order.
    pickOption(filter, 'image');
    await settle();
    queryButtonByName(target, 'Show all media')!.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    await settle();
    expect(galleryApiMocks.getGallery).toHaveBeenNthCalledWith(5, 1, 'all', 'oldest', null, false);
  });

  it('prevents a stale sort response from resurrecting a successful deletion and refills page one', async () => {
    const deleted = makeAsset({ id: 'deleted.png', filename: 'deleted.png' });
    const survivor = makeAsset({ id: 'survivor.png', filename: 'survivor.png' });
    const deleteRequest = deferred<DeleteResult>();
    const staleSort = deferred<GalleryPageResponse>();
    const refill = deferred<GalleryPageResponse>();
    galleryApiMocks.getGallery
      .mockResolvedValueOnce({ assets: [deleted, survivor], page: 1, total_pages: 2, total_count: 2 })
      .mockReturnValueOnce(staleSort.promise)
      .mockReturnValueOnce(refill.promise);
    galleryApiMocks.deleteAssets.mockReturnValue(deleteRequest.promise);

    mountPage();
    await settle();
    deleteAssetFromCard(target, deleted);
    await settle();

    const sort = target.querySelector('[role="combobox"][aria-label="Sort gallery files"]') as HTMLButtonElement;
    pickOption(sort, 'oldest');
    await settle();
    expect(galleryApiMocks.getGallery).toHaveBeenNthCalledWith(2, 1, 'all', 'oldest', null, false);

    deleteRequest.resolve({ deleted: [removed(deleted)], failed: [] });
    await settle();
    expect(galleryApiMocks.getGallery).toHaveBeenNthCalledWith(3, 1, 'all', 'oldest', null, false);

    staleSort.resolve({ assets: [deleted], page: 1, total_pages: 1, total_count: 2 });
    refill.resolve({ assets: [deleted, survivor], page: 1, total_pages: 1, total_count: 2 });
    await settle();

    expect(target.textContent).not.toContain('deleted.png');
    expect(target.textContent).toContain('survivor.png');
    expect(target.textContent).toContain('Browsing 1 loaded file of 1');
  });

  it('keeps a deleted asset removed and warns when the refresh after the delete fails', async () => {
    const deleted = makeAsset({ id: 'deleted.png', filename: 'deleted.png' });
    const survivor = makeAsset({ id: 'survivor.png', filename: 'survivor.png' });
    galleryApiMocks.getGallery
      .mockResolvedValueOnce({ assets: [deleted, survivor], page: 1, total_pages: 1, total_count: 2 })
      .mockRejectedValueOnce(new Error('refresh unavailable'));
    galleryApiMocks.deleteAssets.mockResolvedValue({ deleted: [], failed: [] });

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
    galleryApiMocks.deleteAssets.mockResolvedValue({ deleted: [], failed: [] });

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

  it('guards repeated IDs and single/bulk overlap: a file whose delete is in flight is left out of a bulk delete', async () => {
    const assetA = makeAsset({ id: 'a.png', filename: 'a.png' });
    const assetB = makeAsset({ id: 'b.png', filename: 'b.png' });
    const deleteA = deferred<DeleteResult>();
    const deleteBulk = deferred<DeleteResult>();
    galleryApiMocks.getGallery.mockResolvedValue({ assets: [assetA, assetB], page: 1, total_pages: 1, total_count: 2 });
    galleryApiMocks.deleteAssets.mockReturnValueOnce(deleteA.promise).mockReturnValueOnce(deleteBulk.promise);

    mountPage();
    await settle();
    selectAssetForBatch(target, assetA);
    selectAssetForBatch(target, assetB);
    deleteAssetFromCard(target, assetA);
    await settle();
    deleteAssetFromCard(target, assetA);
    bulkButton('Delete selected')!.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    await settle();

    expect(galleryApiMocks.deleteAssets).toHaveBeenCalledTimes(2);
    expect(galleryApiMocks.deleteAssets).toHaveBeenNthCalledWith(1, [assetA.id]);
    expect(galleryApiMocks.deleteAssets).toHaveBeenNthCalledWith(2, [assetB.id]);
    expect(confirmMocks.requestConfirm).toHaveBeenLastCalledWith(expect.objectContaining({ question: expect.stringContaining('1') }));

    deleteA.resolve({ deleted: [removed(assetA)], failed: [] });
    await settle();
    // B remains independently in flight after A has reconciled and requested a refill.
    expect(target.textContent).toContain('b.png');
    deleteBulk.resolve({ deleted: [removed(assetB)], failed: [] });
    await settle();

    expect(target.textContent).not.toContain('a.png');
    expect(target.textContent).not.toContain('b.png');
  });

  it('keeps a file and reports an error when the server lists it as failed in a single delete', async () => {
    const asset = makeAsset({ id: 'stuck.png', filename: 'stuck.png' });
    galleryApiMocks.getGallery.mockResolvedValue({ assets: [asset], page: 1, total_pages: 1, total_count: 1 });
    galleryApiMocks.deleteAssets.mockResolvedValue({ deleted: [], failed: [{ id: asset.id, detail: 'Could not delete the file: Permission denied' }] });

    mountPage();
    await settle();
    deleteAssetFromCard(target, asset);
    await settle();

    expect(galleryApiMocks.deleteAssets).toHaveBeenCalledWith([asset.id]);
    expect(toastMocks.addToast).toHaveBeenCalledWith(expect.stringContaining('Permission denied'), 'error');
    expect(target.textContent).toContain('stuck.png');
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

    expect(galleryApiMocks.deleteAssets).not.toHaveBeenCalled();
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
    galleryApiMocks.deleteAssets.mockResolvedValue({ deleted: [], failed: [] });

    mountPage();
    await settle();
    MockIntersectionObserver.instances.at(-1)?.trigger(true);
    await settle();
    expect(galleryApiMocks.getGallery).toHaveBeenNthCalledWith(2, 2, 'all', 'newest', null, false);
    expect(target.textContent).toContain('page-two.png');

    deleteAssetFromCard(target, deleted);
    await settle();
    expect(galleryApiMocks.getGallery).toHaveBeenNthCalledWith(3, 1, 'all', 'newest', null, false);

    // A callback held by a now-disconnected observer must not skip to page 3
    // while the authoritative page-one refill is unresolved.
    MockIntersectionObserver.instances.forEach((observer) => observer.trigger(true));
    await settle();
    expect(galleryApiMocks.getGallery).toHaveBeenCalledTimes(3);

    refill.resolve({ assets: [pageTwo], page: 1, total_pages: 2, total_count: 2 });
    await settle();
    MockIntersectionObserver.instances.at(-1)?.trigger(true);
    await settle();
    expect(galleryApiMocks.getGallery).toHaveBeenNthCalledWith(4, 2, 'all', 'newest', null, false);

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

  it('leaves every tile reachable with Tab, the first before the rest', async () => {
    await mountGallery();
    const tabStops = Array.from(target.querySelectorAll<HTMLElement>('#gallery-scroll-region button, #gallery-scroll-region input')).filter((el) => el.tabIndex >= 0 && !el.hasAttribute('disabled'));
    expect(tabStops[0]).toBe(tileButton('a.png'));
    for (const name of ['a.png', 'b.png', 'c.png']) expect(tabStops).toContain(tileButton(name));
  });

  it('focuses the first tile with Alt+3, also from Select mode, and by key code as Option+3 types another character', async () => {
    await mountGallery();
    const event = press('£', { code: 'Digit3', altKey: true });
    expect(event.defaultPrevented).toBe(true);
    expect(document.activeElement).toBe(tileButton('a.png'));

    tileButton('c.png').focus();
    press(' ');
    expect(checked('c.png')).toBe(true);
    press('£', { code: 'Digit3', altKey: true });
    expect(document.activeElement).toBe(target.querySelector('button[aria-label="Select a.png"]'));
  });

  it('leaves Alt+3 alone while typing in a field', async () => {
    await mountGallery();
    const field = document.createElement('input');
    target.append(field);
    field.focus();
    const event = press('£', { code: 'Digit3', altKey: true });
    expect(event.defaultPrevented).toBe(false);
    expect(document.activeElement).toBe(field);
  });

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

  it('selects every matching asset with Ctrl+A', async () => {
    await mountGallery();
    galleryApiMocks.getGalleryIds.mockResolvedValue(assets.map((asset) => asset.id));
    expect(press('a', { ctrlKey: true }).defaultPrevented).toBe(true);
    await settle();
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
    expect(galleryApiMocks.deleteAssets).not.toHaveBeenCalled();
  });

  it('deletes the selection with Delete, else the focused tile', async () => {
    const confirmSpy = confirmMocks.requestConfirm.mockResolvedValue(true);
    galleryApiMocks.deleteAssets.mockResolvedValue({ deleted: [], failed: [] });
    await mountGallery();

    tileButton('c.png').focus();
    press('Backspace');
    await settle();
    expect(galleryApiMocks.deleteAssets).toHaveBeenCalledWith(['c.png']);

    galleryApiMocks.deleteAssets.mockClear();
    galleryApiMocks.deleteAssets.mockResolvedValue({ deleted: [removed(assets[0])], failed: [] });
    selectAssetForBatch(target, assets[0]);
    flushSync();
    press('Delete');
    await settle();
    expect(galleryApiMocks.deleteAssets).toHaveBeenCalledTimes(1);
    expect(galleryApiMocks.deleteAssets).toHaveBeenCalledWith(['a.png']);
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

    expect(galleryApiMocks.getGallery).toHaveBeenNthCalledWith(1, 1, 'all', 'newest', 'Night Bus', false);
    expect(galleryApiMocks.getGallery).toHaveBeenNthCalledWith(2, 2, 'all', 'newest', 'Night Bus', false);
    expect(tiles()).toEqual(['View a.png', 'View b.png']);
  });

  it('leads the subtitle with the project name, and keeps the plain text with All files', async () => {
    galleryApiMocks.getGallery.mockResolvedValue(reply([makeAsset()]));
    mountPage();
    await settle();
    expect(subtitle()).toBe('Browsing 1 loaded file of 1, newest first.');

    projectsStore.choose('Night Bus');
    await settle();
    expect(subtitle()).toBe('Night Bus · Browsing 1 loaded file of 1, newest first.');

    projectsStore.choose(null);
    await settle();
    expect(subtitle()).toBe('Browsing 1 loaded file of 1, newest first.');
  });

  it('replaces the view on a pick change: new project, selection cleared, viewer closed, back at the top, filter and sort kept', async () => {
    const mine = makeAsset({ id: 'mine.mp4', filename: 'mine.mp4', media_type: 'video' });
    const theirs = makeAsset({ id: 'Night Bus/theirs.mp4', filename: 'theirs.mp4', media_type: 'video', project: 'Night Bus' });
    galleryApiMocks.getGallery.mockImplementation(async (...args: unknown[]) => reply(args[3] === 'Night Bus' ? [theirs] : [mine]));
    mountPage();
    await settle();
    pickOption(target.querySelector('[role="combobox"][aria-label="Filter gallery media"]') as HTMLButtonElement, 'video');
    pickOption(target.querySelector('[role="combobox"][aria-label="Sort gallery files"]') as HTMLButtonElement, 'oldest');
    await settle();
    selectAssetForBatch(target, mine);
    await openViewer('mine.mp4');
    expect(bulkButton('Done')).not.toBeNull();
    const region = target.querySelector('#gallery-scroll-region') as HTMLElement;
    region.scrollTop = 400;
    region.dispatchEvent(new Event('scroll'));
    expect(getViewer()).not.toBeNull();

    projectsStore.choose('Night Bus');
    await settle();

    expect(galleryApiMocks.getGallery).toHaveBeenLastCalledWith(1, 'video', 'oldest', 'Night Bus', false);
    expect(tiles()).toEqual(['View theirs.mp4']);
    // The selection and the mode went with the old view.
    expect(bulkButton('Done')).toBeNull();
    expect(target.querySelector('input[aria-label="Select theirs.mp4"]')).toHaveProperty('checked', false);
    expect(getViewer()).toBeNull();
    expect(region.scrollTop).toBe(0);
    expect((target.querySelector('[role="combobox"][aria-label="Filter gallery media"]') as HTMLElement).dataset.value).toBe('video');
    expect((target.querySelector('[role="combobox"][aria-label="Sort gallery files"]') as HTMLElement).dataset.value).toBe('oldest');
  });

  it('asks the server once for the new project\'s default list when the pick changes under it', async () => {
    galleryApiMocks.getGallery.mockImplementation(async () => reply([makeAsset()]));
    mountPage();
    await settle();
    galleryApiMocks.getGallery.mockClear();

    projectsStore.choose('Night Bus');
    await settle();

    expect(galleryApiMocks.getGallery).toHaveBeenCalledTimes(1);
    expect(galleryApiMocks.getGallery).toHaveBeenCalledWith(1, 'all', 'newest', 'Night Bus', false);
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
    expect(galleryStore.view({ filter: 'video', sort: 'newest', starred: false }).assets.map((asset) => asset.id)).toEqual(['Night Bus/fresh.mp4']);
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
    expect(target.textContent).toContain('No generated files yet');

    projectsStore.choose('Night Bus');
    await settle();
    expect(target.textContent).toContain('Nothing in Night Bus yet');
    expect(target.textContent).toContain('New work saves here. To add existing files, switch to All files and use Move to project.');
    expect(queryButtonByName(target, 'Open Workspace')).not.toBeNull();

    pickOption(target.querySelector('[role="combobox"][aria-label="Filter gallery media"]') as HTMLButtonElement, 'video');
    await settle();
    expect(target.textContent).toContain('No matching files');
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

    expect(galleryApiMocks.getGallery).toHaveBeenLastCalledWith(1, 'no_project', 'newest', null, false);
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

    expect(galleryApiMocks.getGallery).toHaveBeenLastCalledWith(1, 'all', 'newest', 'Night Bus', false);
    expect(filterBox().dataset.value).toBe('all');
  });

  it('has its own empty state, which offers to show all media', async () => {
    galleryApiMocks.getGallery.mockResolvedValueOnce(reply([top])).mockResolvedValue(reply([]));
    mountPage();
    await settle();
    pickOption(filterBox(), 'no_project');
    await settle();

    expect(target.textContent).toContain('No matching files');
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

    expect(galleryApiMocks.getGallery).toHaveBeenLastCalledWith(1, 'audio', 'newest', null, false);
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
  const moved = (from: string, asset: GalleryAsset) => ({ from, from_path: `/outputs/${from}`, asset });
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
    expect(target.querySelector('.ui-page-sub')?.textContent).toContain('Browsing 1 loaded file of 1');
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
    expect(target.querySelector('.ui-page-sub')?.textContent).toContain('Browsing 2 loaded files of 2');
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
    const moveButton = () => bulkButton('Move selected');

    it('disables the header button with nothing selected, and offers the list once files are selected', async () => {
      galleryApiMocks.getGallery.mockResolvedValue(reply([top('a.png'), top('b.png')]));
      mountPage();
      await settle();
      // The button belongs to Select mode.
      expect(moveButton()).toBeNull();
      await enterSelectMode();
      expect(moveButton()?.disabled).toBe(true);
      expect(moveButton()?.getAttribute('aria-haspopup')).toBe('menu');
      // It sits between Select all and Delete.
      const order = moveButton()!.compareDocumentPosition(bulkButton('Delete selected')!);
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

    it('moves the selection in one call; a refused file stays selected, the mode stays on, and the toast says so', async () => {
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
      expect(bulkButton('Done')).not.toBeNull();
      expect(document.activeElement).toBe(moveButton());
      expect(lastToastTone()).toBe('warning');
      expect(lastToastMessage()).toContain('A file named c.png is already there');
    });

    it('moves every selected file in one call, unloaded ones included, and leaves the mode when none failed', async () => {
      const names = Array.from({ length: 30 }, (_, index) => `f${String(index).padStart(2, '0')}.png`);
      const loaded = names.slice(0, 12).map((name) => top(name));
      galleryApiMocks.getGallery.mockResolvedValue({ assets: loaded, page: 1, total_pages: 3, total_count: 30 });
      galleryApiMocks.getGalleryIds.mockResolvedValue(names);
      galleryApiMocks.moveAssets.mockResolvedValue({ moved: names.map((name) => moved(name, inBus(name))), failed: [] });
      mountPage();
      await settle();
      await enterSelectMode();
      bulkButton('Select all')!.click();
      await settle();

      moveButton()!.click();
      await settle();
      expect(document.querySelector('.action-menu-heading')?.textContent).toBe('Move 30 files to');
      row('Night Bus').click();
      await settle();

      expect(galleryApiMocks.moveAssets).toHaveBeenCalledWith(names, 'Night Bus');
      expect(bulkButton('Done')).toBeNull();
      expect(document.activeElement).toBe(queryButtonByName(target, 'Select'));
    });

    it('counts a file already in the destination as done: it leaves the selection, the mode ends and nothing is reported as an error', async () => {
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

      expect(bulkButton('Done')).toBeNull();
      expect(target.querySelector<HTMLInputElement>('input[aria-label="Select b.png"]')?.checked).toBe(false);
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

      expect(bulkButton('Done')).toBeNull();
      expect(lastToastTone()).not.toBe('error');
    });

    it('keeps the mode and the selection when the move request itself fails', async () => {
      const a = top('a.png');
      galleryApiMocks.getGallery.mockResolvedValue(reply([a]));
      galleryApiMocks.moveAssets.mockRejectedValue(new Error('down'));
      mountPage();
      await settle();
      selectAssetForBatch(target, a);
      await settle();

      moveButton()!.click();
      await settle();
      row('Night Bus').click();
      await settle();

      expect(bulkButton('Done')).not.toBeNull();
      expect(target.textContent).toContain('1 selected');
      expect(lastToastTone()).toBe('error');
    });

    it('shows the Move label on a tablet, named Move selected', async () => {
      layoutState.mode = 'tall';
      galleryApiMocks.getGallery.mockResolvedValue(reply([top('a.png')]));
      mountPage();
      await settle();
      await enterSelectMode();

      const button = target.querySelector<HTMLButtonElement>('button[aria-label="Move selected"]')!;
      expect(button.textContent?.trim()).toBe('Move');
      expect(button.disabled).toBe(true);
    });

    it('puts Move on the phone select bar between Select all and Download, and moves the selection from there', async () => {
      layoutState.mode = 'compact';
      const [a, b] = [top('a.png'), top('b.png')];
      galleryApiMocks.getGallery.mockResolvedValue(reply([a, b]));
      galleryApiMocks.moveAssets.mockResolvedValue({ moved: [moved('a.png', inBus('a.png'))], failed: [] });
      mountPage();
      await settle();
      await enterSelectMode();

      const bar = target.querySelector<HTMLElement>('[data-testid="gallery-select-bar"]')!;
      const names = () => Array.from(bar.querySelectorAll('button')).map((el) => el.getAttribute('aria-label') ?? el.textContent?.trim());
      expect(names()).toEqual(['Select all', 'Move selected', 'Download selected', 'Delete selected']);
      const barMove = bar.querySelector<HTMLButtonElement>('button[aria-label="Move selected"]')!;
      expect(barMove.disabled).toBe(true);

      selectAssetForBatch(target, a);
      await settle();
      barMove.click();
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
      expect(bulkButton('Done')).toBeNull();
    });
  });
});

describe('GalleryPage Download', () => {
  const names = Array.from({ length: 30 }, (_, index) => `f${String(index).padStart(2, '0')}.png`);
  const loaded = names.slice(0, 12).map((name) => makeAsset({ id: name, filename: name }));

  async function mountSelectedAll(): Promise<void> {
    galleryApiMocks.getGallery.mockResolvedValue({ assets: loaded, page: 1, total_pages: 3, total_count: 30 });
    galleryApiMocks.getGalleryIds.mockResolvedValue(names);
    mountPage();
    await settle();
    await enterSelectMode();
    bulkButton('Select all')!.click();
    await settle();
  }

  it('is off with nothing selected', async () => {
    galleryApiMocks.getGallery.mockResolvedValue({ assets: loaded, page: 1, total_pages: 1, total_count: 12 });
    mountPage();
    await settle();
    await enterSelectMode();

    expect(bulkButton('Download selected')?.disabled).toBe(true);
  });

  it('downloads every selected ID, unloaded ones included, straight from the click, and keeps the selection and the mode', async () => {
    await mountSelectedAll();

    // No await between the click and the check: the form must be submitted inside the click handler.
    bulkButton('Download selected')!.click();

    expect(galleryApiMocks.downloadAssets).toHaveBeenCalledTimes(1);
    expect(galleryApiMocks.downloadAssets).toHaveBeenCalledWith(names);
    expect(toastMocks.addToast).toHaveBeenCalledWith(expect.stringContaining(String(names.length)), 'success');
    await settle();
    expect(target.textContent).toContain('30 selected');
    expect(bulkButton('Done')).not.toBeNull();
  });

});

describe('GalleryPage stars', () => {
  const reply = (assets: GalleryAsset[]) => ({ assets, page: 1, total_pages: 1, total_count: assets.length });
  const starToggle = () => target.querySelector<HTMLButtonElement>('button[aria-label="Starred"]')!;
  const tileStar = (filename: string) => target.querySelector<HTMLButtonElement>(`button[aria-label="Star ${filename}"]`);
  const tileNames = () => Array.from(target.querySelectorAll('.asset-tile')).map((tile) => tile.getAttribute('aria-label'));
  const filterSelect = () => target.querySelector('[role="combobox"][aria-label="Filter gallery media"]') as HTMLButtonElement;
  const a = makeAsset({ id: 'a.png', filename: 'a.png', prompt: 'prompt of a' });
  const b = makeAsset({ id: 'b.png', filename: 'b.png', prompt: 'prompt of b' });
  const starredA = makeAsset({ id: 'a.png', filename: 'a.png', prompt: 'prompt of a', starred: true });
  const starredB = makeAsset({ id: 'b.png', filename: 'b.png', prompt: 'prompt of b', starred: true });

  it('shows a ★ toggle at rest that requests the starred files', async () => {
    galleryApiMocks.getGallery.mockResolvedValue(reply([a, b]));
    mountPage();
    await settle();
    expect(starToggle().getAttribute('aria-pressed')).toBe('false');

    galleryApiMocks.getGallery.mockResolvedValue(reply([starredA]));
    starToggle().click();
    await settle();

    expect(galleryApiMocks.getGallery).toHaveBeenLastCalledWith(1, 'all', 'newest', null, true);
    expect(starToggle().getAttribute('aria-pressed')).toBe('true');
    expect(tileNames()).toEqual(['a.png']);
  });

  it('combines with the media filter in both orders, and with the picked project', async () => {
    galleryApiMocks.getGallery.mockResolvedValue(reply([a, b]));
    mountPage();
    await settle();

    pickOption(filterSelect(), 'video');
    await settle();
    starToggle().click();
    await settle();
    expect(galleryApiMocks.getGallery).toHaveBeenLastCalledWith(1, 'video', 'newest', null, true);

    pickOption(filterSelect(), 'image');
    await settle();
    expect(galleryApiMocks.getGallery).toHaveBeenLastCalledWith(1, 'image', 'newest', null, true);

    projectsStore.choose('Night Bus');
    await settle();
    expect(galleryApiMocks.getGallery).toHaveBeenLastCalledWith(1, 'image', 'newest', 'Night Bus', true);
    expect(starToggle().getAttribute('aria-pressed')).toBe('true');
  });

  it('pages the starred view with the toggle still on', async () => {
    galleryApiMocks.getGallery
      .mockResolvedValueOnce(reply([a]))
      .mockResolvedValueOnce({ assets: [starredA], page: 1, total_pages: 2, total_count: 2 })
      .mockResolvedValueOnce({ assets: [starredB], page: 2, total_pages: 2, total_count: 2 });
    mountPage();
    await settle();
    starToggle().click();
    await settle();
    MockIntersectionObserver.instances.at(-1)?.trigger(true);
    await settle();

    expect(galleryApiMocks.getGallery).toHaveBeenNthCalledWith(3, 2, 'all', 'newest', null, true);
  });

  it('remembers the toggle across visits', async () => {
    galleryApiMocks.getGallery.mockResolvedValue(reply([starredA]));
    mountPage();
    await settle();
    starToggle().click();
    await settle();
    await page.unmount();

    mountPage();
    await settle();

    expect(galleryApiMocks.getGallery).toHaveBeenLastCalledWith(1, 'all', 'newest', null, true);
    expect(starToggle().getAttribute('aria-pressed')).toBe('true');
  });

  it('shows the starred empty state, and Show all turns the toggle off and keeps the media filter', async () => {
    galleryApiMocks.getGallery.mockResolvedValue(reply([a]));
    mountPage();
    await settle();
    pickOption(filterSelect(), 'image');
    await settle();
    galleryApiMocks.getGallery.mockResolvedValue(reply([]));
    starToggle().click();
    await settle();

    expect(target.textContent).toContain('No starred files');
    expect(queryButtonByName(target, 'Show all media')).toBeNull();

    galleryApiMocks.getGallery.mockResolvedValue(reply([a]));
    queryButtonByName(target, 'Show all')!.click();
    await settle();

    expect(galleryApiMocks.getGallery).toHaveBeenLastCalledWith(1, 'image', 'newest', null, false);
    expect(starToggle().getAttribute('aria-pressed')).toBe('false');
    expect(filterSelect().dataset.value).toBe('image');
    expect(tileNames()).toEqual(['a.png']);
  });

  it('leaves the tile as it was and tells the user when the star cannot be saved', async () => {
    galleryApiMocks.getGallery.mockResolvedValue(reply([a]));
    galleryApiMocks.setStarred.mockRejectedValue(new Error('offline'));
    mountPage();
    await settle();

    tileStar('a.png')!.click();
    await settle();

    expect(tileStar('a.png')!.getAttribute('aria-pressed')).toBe('false');
    expect(toastMocks.addToast).toHaveBeenCalledWith('offline', 'error');
  });

  it('drops a file from the starred view at once when it is unstarred, and lowers the count', async () => {
    galleryApiMocks.getGallery.mockResolvedValueOnce(reply([a])).mockResolvedValueOnce(reply([starredA, starredB]));
    mountPage();
    await settle();
    starToggle().click();
    await settle();
    expect(tileNames()).toEqual(['a.png', 'b.png']);
    // The refill after the unstar never answers: what shows is the store's.
    galleryApiMocks.getGallery.mockImplementation(() => new Promise(() => undefined));

    tileStar('a.png')!.click();
    await settle();

    expect(galleryApiMocks.setStarred).toHaveBeenCalledWith('a.png', false);
    expect(tileNames()).toEqual(['b.png']);
    expect(target.querySelector('.ui-page-sub')?.textContent).toContain('of 1');
  });

  it('stars the focused tile with S, in and out of Select mode, and ignores a held S', async () => {
    galleryApiMocks.getGallery.mockResolvedValue(reply([a, b]));
    mountPage();
    await settle();
    const tileMedia = (name: string) => target.querySelector<HTMLButtonElement>(`button[aria-label$=" ${name}"]:not([aria-label$=" fullscreen"]):not([aria-label^="More"]):not([aria-label^="Star"])`)!;
    const pressS = (init: KeyboardEventInit = {}) => {
      (document.activeElement ?? document.body).dispatchEvent(new KeyboardEvent('keydown', { key: 's', bubbles: true, cancelable: true, ...init }));
    };
    tileMedia('a.png').focus();

    pressS({ repeat: true });
    await settle();
    expect(galleryApiMocks.setStarred).not.toHaveBeenCalled();
    expect(tileStar('a.png')!.getAttribute('aria-pressed')).toBe('false');

    pressS();
    await settle();
    expect(galleryApiMocks.setStarred).toHaveBeenCalledTimes(1);
    expect(galleryApiMocks.setStarred).toHaveBeenCalledWith('a.png', true);
    expect(tileStar('a.png')!.getAttribute('aria-pressed')).toBe('true');
    expect(tileStar('b.png')!.getAttribute('aria-pressed')).toBe('false');

    await enterSelectMode();
    tileMedia('b.png').focus();
    pressS();
    await settle();
    expect(galleryApiMocks.setStarred).toHaveBeenLastCalledWith('b.png', true);
  });

  it('takes a file out of the selection when it is unstarred in the starred view, so Delete acts on the rest', async () => {
    const c = makeAsset({ id: 'c.png', filename: 'c.png', starred: true });
    galleryApiMocks.getGallery.mockResolvedValueOnce(reply([a])).mockResolvedValueOnce(reply([starredA, starredB, c]));
    confirmMocks.requestConfirm.mockResolvedValue(true);
    galleryApiMocks.deleteAssets.mockResolvedValue({ deleted: [removed(starredB), removed(c)], failed: [] });
    mountPage();
    await settle();
    starToggle().click();
    await settle();
    await enterSelectMode();
    for (const name of ['a.png', 'b.png', 'c.png']) (target.querySelector(`button[aria-label="Select ${name}"]`) as HTMLElement).click();
    flushSync();
    expect(target.textContent).toContain('3 selected');
    // The refill after the unstar never answers: what shows is the store's.
    galleryApiMocks.getGallery.mockImplementation(() => new Promise(() => undefined));

    tileStar('a.png')!.click();
    await settle();
    expect(target.textContent).toContain('2 selected');

    bulkButton('Delete selected')!.click();
    await settle();

    expect(confirmMocks.requestConfirm).toHaveBeenLastCalledWith(expect.objectContaining({ question: expect.stringContaining('2') }));
    expect(galleryApiMocks.deleteAssets).toHaveBeenCalledWith(['b.png', 'c.png']);
  });

  it('does not change the selection when a tile star is clicked in Select mode', async () => {
    galleryApiMocks.getGallery.mockResolvedValue(reply([a, b]));
    mountPage();
    await settle();
    await enterSelectMode();
    (target.querySelector('button[aria-label="Select b.png"]') as HTMLElement).click();
    flushSync();

    tileStar('a.png')!.click();
    await settle();

    expect(galleryApiMocks.setStarred).toHaveBeenCalledWith('a.png', true);
    expect(target.querySelector<HTMLInputElement>('input[aria-label="Select a.png"]')!.checked).toBe(false);
    expect(target.querySelector<HTMLInputElement>('input[aria-label="Select b.png"]')!.checked).toBe(true);
    expect(target.textContent).toContain('1 selected');
  });

  it('steps the viewer on to the next file when the button or S unstars one in the starred view, lowers the count and refills once it closes', async () => {
    const starredC = makeAsset({ id: 'c.png', filename: 'c.png', prompt: 'prompt of c', starred: true });
    galleryApiMocks.getGallery.mockResolvedValueOnce(reply([a])).mockResolvedValueOnce(reply([starredA, starredB, starredC]));
    mountPage();
    await settle();
    starToggle().click();
    await settle();
    await openViewer('a.png');
    const title = () => document.querySelector('#asset-viewer-title')?.textContent;
    expect(title()).toBe('a.png');
    expect(document.querySelector('.viewer-title small')?.textContent).toContain('1 of 3');
    galleryApiMocks.getGallery.mockClear();
    galleryApiMocks.getGallery.mockResolvedValue(reply([starredC]));

    document.querySelector<HTMLButtonElement>('[data-testid="asset-viewer"] button[data-action="star"]')!.click();
    await settle();

    expect(galleryApiMocks.setStarred).toHaveBeenLastCalledWith('a.png', false);
    expect(title()).toBe('b.png');
    expect(tileNames()).toEqual(['b.png', 'c.png']);

    document.dispatchEvent(new KeyboardEvent('keydown', { key: 's', bubbles: true }));
    await settle();

    expect(galleryApiMocks.setStarred).toHaveBeenLastCalledWith('b.png', false);
    expect(title()).toBe('c.png');
    expect(document.querySelector('.viewer-title small')?.textContent).toContain('1 of 1');
    expect(target.querySelector('.ui-page-sub')?.textContent).toContain('of 1');
    // The page reloads once the viewer closes, so the list is the server's again.
    expect(galleryApiMocks.getGallery).not.toHaveBeenCalled();
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
    await settle();
    expect(getViewer()).toBeNull();
    expect(galleryApiMocks.getGallery).toHaveBeenCalledTimes(1);
    expect(galleryApiMocks.getGallery).toHaveBeenLastCalledWith(1, 'all', 'newest', null, true);
  });
});
