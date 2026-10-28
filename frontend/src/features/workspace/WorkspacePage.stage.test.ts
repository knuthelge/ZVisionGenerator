// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, mount, unmount } from '../../../node_modules/svelte/src/index-client.js';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { ApiError } from '$lib/api/client';
import { reactiveProps } from '../../test-utils/reactiveProps.svelte';
import { draft } from '$lib/state/draft.svelte';
import { legacySource, sceneSource, sourceInfo } from '$lib/state/extend.fixtures';
import { galleryStore } from '$lib/state/gallery.svelte';
import { requestPromptRun } from '$lib/state/pendingPromptRun';
import { toasts } from '$lib/state/toasts.svelte';
import type { EnhanceFrame, SettingsPreset, JobContext, PromptEnhancerContract, WorkspaceContext, GalleryAsset } from '$lib/types';

import WorkspacePage from './WorkspacePage.svelte';
import { chooseInBrowser, listingWith, placesFor } from '../../test-utils/pathBrowser';
import { lastEventSource } from '../../test-utils/eventSource';
import { makeAsset, makeContext, makeGalleryPage } from '../../test-utils/workspace';
import { answerConfirm, createHost, fakeDebounceTimers, field, holdInspection, mockClipboard, mountWorkspacePage, openEnhanceRun, resetWorkspaceTest, runEnhance, settle, type, waitDebounce } from '../../test-utils/workspaceHarness';
import { browseApiMocks, enhanceApiMocks, extendApiMocks, galleryApiMocks, presetApiMocks, workspaceApiMocks } from '../../test-utils/workspaceMocks';

vi.mock('$lib/state/layoutMode.svelte', async (importOriginal) => {
  const actual = await importOriginal<typeof import('$lib/state/layoutMode.svelte')>();
  const { layoutProbe } = await import('../../test-utils/workspaceMocks');
  return {
    ...actual,
    layout: {
      get mode() { return layoutProbe.compact ? 'compact' : 'wide'; },
      get compact() { return layoutProbe.compact; },
      get coarse() { return layoutProbe.compact; },
    },
  };
});

vi.mock('$lib/api/config', async (importOriginal) => {
  const actual = await importOriginal<typeof import('$lib/api/config')>();
  const { configApiMocks } = await import('../../test-utils/workspaceMocks');
  return { ...actual, updateConfig: configApiMocks.updateConfig };
});

vi.mock('$lib/api/extend', async () => {
  const { extendApiMocks } = await import('../../test-utils/workspaceMocks');
  return { inspectExtendSource: extendApiMocks.inspectExtendSource };
});

vi.mock('$lib/api/videoMemory', async () => {
  const { videoMemoryApiMocks } = await import('../../test-utils/workspaceMocks');
  return { checkVideoMemoryFit: videoMemoryApiMocks.checkVideoMemoryFit };
});

vi.mock('$lib/api/workspace', async (importOriginal) => {
  const actual = await importOriginal<typeof import('$lib/api/workspace')>();
  const { workspaceApiMocks: mocks } = await import('../../test-utils/workspaceMocks');
  return {
    ...actual,
    getWorkspaceContext: mocks.getWorkspaceContext,
    getWorkspaceCoreContext: mocks.getWorkspaceCoreContext,
    submitGenerate: mocks.submitGenerate,
    getJobSnapshot: mocks.getJobSnapshot,
    parseUrlPrefill: mocks.parseUrlPrefill,
    checkReferencePaths: mocks.checkReferencePaths,
    listJobs: mocks.listJobs,
    clearQueue: mocks.clearQueue,
    cancelJob: mocks.cancelJob,
    skipFirstRun: mocks.skipFirstRun,
  };
});

vi.mock('$lib/api/gallery', async (importOriginal) => {
  const actual = await importOriginal<typeof import('$lib/api/gallery')>();
  const { galleryApiMocks } = await import('../../test-utils/workspaceMocks');
  return { ...actual, getGallery: galleryApiMocks.getGallery, deleteAsset: galleryApiMocks.deleteAsset };
});

vi.mock('$lib/api/promptEnhance', async (importOriginal) => {
  const actual = await importOriginal<typeof import('$lib/api/promptEnhance')>();
  const { enhanceApiMocks } = await import('../../test-utils/workspaceMocks');
  return { ...actual, enhancePrompt: enhanceApiMocks.enhancePrompt };
});

vi.mock('$lib/api/browse', async () => (await import('../../test-utils/workspaceMocks')).browseApiMocks);

vi.mock('$lib/api/presets', async () => (await import('../../test-utils/workspaceMocks')).presetApiMocks);

vi.mock('$lib/api/models', async (importOriginal) => {
  const actual = await importOriginal<typeof import('$lib/api/models')>();
  const { modelApiMocks } = await import('../../test-utils/workspaceMocks');
  return { ...actual, getDownloads: modelApiMocks.getDownloads };
});

vi.mock('$lib/api/promptFiles', async () => {
  const { promptFileApiMocks } = await import('../../test-utils/workspaceMocks');
  return {
    inspectPromptFile: promptFileApiMocks.inspectPromptFile,
    writePromptFile: promptFileApiMocks.writePromptFile,
    appendPromptEntry: promptFileApiMocks.appendPromptEntry,
    createPromptFile: promptFileApiMocks.createPromptFile,
  };
});

let target: HTMLDivElement;
let app: Record<string, unknown> | null = null;

// Every test starts from empty stores, reset API mocks, an empty server-side queue and a fresh host element.
beforeEach(() => {
  resetWorkspaceTest();
  target = createHost();
});

afterEach(async () => {
  vi.useRealTimers();
  if (app) { await unmount(app); app = null; }
  target.remove();
  document.body.innerHTML = '';
  vi.restoreAllMocks();
});

/** Mount the Workspace page on the host element and let it settle. */
async function mountWorkspace(context: WorkspaceContext): Promise<void> {
  app = await mountWorkspacePage(context, target);
}

describe('WorkspacePage center pane promotion (REC-UX-001)', () => {
  beforeEach(() => {
    workspaceApiMocks.submitGenerate.mockResolvedValue({
      job_id: 'job-promo',
      workflow: 'txt2img',
      prompt: 'Test prompt',
      model: 'zit',
      runs: 1,
      created_at: '2026-04-30T10:00:00Z',
    });
  });

  it('renders the full authoritative completed output grid and hides the JobCard', async () => {
    const context = makeContext();
    await mountWorkspace(context);

    const form = target.querySelector('form');
    form!.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
    await settle();

    const submitButton = target.querySelector('#ws-submit') as HTMLButtonElement | null;
    // A running job no longer blocks Generate; it adds to the queue.
    expect(submitButton?.disabled).toBe(false);
    expect(submitButton?.textContent).toContain('Add to queue');

    const completedAsset = makeAsset({ id: 'out/first.png', filename: 'first.png' });
    const secondAsset = makeAsset({ id: 'out/second.png', url: '/media/out/second.png', filename: 'second.png' });
    const mockEventSource = lastEventSource();
    mockEventSource.emit('job_completed', { type: 'job_completed', job_id: 'job-promo', total_runs: 2, outputs: [completedAsset, secondAsset] });
    await settle();

    expect(target.textContent).toContain('Completed outputs');
    expect(target.querySelectorAll('.completed-output-grid .asset-tile')).toHaveLength(2);
    expect(target.querySelector(`.completed-output-grid img[src="${completedAsset.url}"]`)).not.toBeNull();
    expect(target.querySelector(`.completed-output-grid img[src="${secondAsset.url}"]`)).not.toBeNull();
    expect(target.textContent).not.toContain('Waiting for worker allocation...');
  });

  it('opens indexed completed outputs in the workspace lightbox and navigates the full list', async () => {
    const context = makeContext();
    await mountWorkspace(context);

    const staleHistoryAsset = makeAsset({
      id: 'out/stale.png',
      url: '/media/out/stale.png',
      thumbnail_url: '/media/out/stale-thumb.png',
      filename: 'stale.png',
      prompt: 'Stale history asset',
    });
    const completedAsset = makeAsset({
      id: 'out/completed.png',
      url: '/media/out/completed.png',
      thumbnail_url: '/media/out/completed-thumb.png',
      filename: 'completed.png',
      prompt: 'Fresh completed asset',
    });
    const secondAsset = makeAsset({
      id: 'out/completed-second.png',
      url: '/media/out/completed-second.png',
      thumbnail_url: '/media/out/completed-second-thumb.png',
      filename: 'completed-second.png',
      prompt: 'Second completed asset',
    });
    galleryStore.seed([staleHistoryAsset]);

    const form = target.querySelector('form');
    form!.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
    await settle();

    const mockEventSource = lastEventSource();
    mockEventSource.emit('job_completed', { type: 'job_completed', job_id: 'job-promo', total_runs: 2, outputs: [completedAsset, secondAsset] });
    await settle();

    const secondButton = target.querySelector(`button[aria-label="View ${secondAsset.filename} fullscreen"]`);
    expect(secondButton).not.toBeNull();
    secondButton!.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    await settle();

    expect(target.querySelector('[data-testid="asset-viewer"]')).not.toBeNull();
    const lightboxImage = target.querySelector('[data-testid="viewer-image"]') as HTMLImageElement | null;
    expect(lightboxImage?.getAttribute('src')).toBe(secondAsset.url);
    expect(document.activeElement?.getAttribute('aria-label')).toBe('Close viewer');

    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowLeft', bubbles: true }));
    await settle();
    expect((target.querySelector('[data-testid="viewer-image"]') as HTMLImageElement | null)?.getAttribute('src')).toBe(completedAsset.url);

    const closeButton = target.querySelector('[data-testid="asset-viewer"] button[aria-label="Close viewer"]');
    closeButton!.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    await settle();
    expect(document.activeElement).toBe(secondButton);
  });
});

describe('WorkspacePage history viewer (REC-UX-002)', () => {

  it('clicking a history row opens the workspace viewer lightbox', async () => {
    const asset = makeAsset({ id: 'out/first.png', url: '/media/out/first.png', filename: 'first.png' });
    galleryApiMocks.getGallery.mockResolvedValue(makeGalleryPage([asset]));
    const context = makeContext({ history_assets: [] });
    await mountWorkspace(context);

    const historyRow = target.querySelector(`button[aria-label="View ${asset.filename}"]`) as HTMLButtonElement | null;
    expect(historyRow).not.toBeNull();

    historyRow!.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    await settle();

    expect(target.querySelector('[data-testid="asset-viewer"]')).not.toBeNull();
  });
});

describe('WorkspacePage asset actions and settings', () => {

  async function mountWithHistory(assets: GalleryAsset[]): Promise<void> {
    galleryApiMocks.getGallery.mockResolvedValue(makeGalleryPage(assets));
    workspaceApiMocks.getWorkspaceCoreContext.mockResolvedValue(makeContext({ history_assets: assets }));
    app = flushSync(() => mount(WorkspacePage, { target }));
    await settle();
  }

  function historyTile(asset: GalleryAsset): HTMLElement {
    return target.querySelector(`#ws-history-shell article[aria-label="${asset.filename}"]`) as HTMLElement;
  }

  function openTileMenu(asset: GalleryAsset): void {
    (historyTile(asset).querySelector(`button[aria-label="More actions for ${asset.filename}"]`) as HTMLButtonElement).click();
    flushSync();
  }

  it('shows history as a filmstrip under the preview that collapses and remembers it', async () => {
    const asset = makeAsset({ id: 'out/first.png', url: '/media/out/first.png', filename: 'first.png' });
    await mountWithHistory([asset]);

    expect(target.querySelector('.workspace-preview #ws-history-shell')).not.toBeNull();
    expect(historyTile(asset)).not.toBeNull();

    (target.querySelector('#ws-history-toggle') as HTMLButtonElement).click();
    await settle();
    expect(historyTile(asset)).toBeNull();
    expect(draft.state.historyCollapsed).toBe(true);
  });

  it('focuses the newest history tile with Alt+3, expanding a collapsed strip', async () => {
    const newest = makeAsset({ id: 'out/newest.png', url: '/media/out/newest.png', filename: 'newest.png' });
    const older = makeAsset({ id: 'out/older.png', url: '/media/out/older.png', filename: 'older.png' });
    draft.update('historyCollapsed', true);
    await mountWithHistory([newest, older]);

    const event = new KeyboardEvent('keydown', { key: '3', code: 'Digit3', altKey: true, bubbles: true, cancelable: true });
    document.dispatchEvent(event);
    await settle();

    expect(event.defaultPrevented).toBe(true);
    expect(draft.state.historyCollapsed).toBe(false);
    expect(document.activeElement?.getAttribute('aria-label')).toBe(`View ${newest.filename}`);
  });

  it('toggles the history strip with Alt+Shift+3, focusing the newest tile when it opens', async () => {
    const newest = makeAsset({ id: 'out/newest.png', url: '/media/out/newest.png', filename: 'newest.png' });
    await mountWithHistory([newest]);
    const press = (): void => { document.dispatchEvent(new KeyboardEvent('keydown', { key: '#', code: 'Digit3', altKey: true, shiftKey: true, bubbles: true, cancelable: true })); };

    press();
    await settle();
    expect(draft.state.historyCollapsed).toBe(true);
    press();
    await settle();
    expect(draft.state.historyCollapsed).toBe(false);
    expect(document.activeElement?.getAttribute('aria-label')).toBe(`View ${newest.filename}`);
    expect(draft.state.sidebarCollapsed).toBe(false);
  });

  it('focuses the history toggle with Alt+3 when there is no history', async () => {
    await mountWithHistory([]);
    document.dispatchEvent(new KeyboardEvent('keydown', { key: '3', code: 'Digit3', altKey: true, bubbles: true, cancelable: true }));
    await settle();
    expect(document.activeElement?.id).toBe('ws-history-toggle');
  });

  it('reuses the settings of an asset in place without leaving the workspace, and forgets the applied preset', async () => {
    draft.setAppliedPreset('look');
    const asset = makeAsset({
      id: 'out/first.png',
      filename: 'first.png',
      has_reusable_config: true,
      reuse_workspace_url: '#/workspace?workflow=txt2img&prompt=reused+prompt&steps=17&seed=99',
    });
    await mountWithHistory([asset]);

    (historyTile(asset).querySelector(`button[aria-label="Reuse settings from ${asset.filename}"]`) as HTMLButtonElement).click();
    await settle();

    expect(draft.state).toMatchObject({ workflow: 'txt2img', prompt: 'reused prompt', steps: 17, seed: 99 });
    expect((target.querySelector('#ws-prompt') as HTMLTextAreaElement).value).toBe('reused prompt');
    expect((target.querySelector('#ws-steps') as HTMLInputElement).value).toBe('17');
    expect(draft.state.appliedPreset).toBeNull();
  });

  it('makes an image the img2img reference and submits its host path', async () => {
    const asset = makeAsset({ id: 'out/first.png', url: '/media/out/first.png', filename: 'first.png', file_path: '/outputs/out/first.png' });
    await mountWithHistory([asset]);

    openTileMenu(asset);
    (document.querySelector('[role="menu"] [data-action="reference-image"]') as HTMLButtonElement).click();
    await settle();

    expect(draft.state.workflow).toBe('img2img');
    expect((target.querySelector('input[name="image_path"]') as HTMLInputElement).value).toBe('/outputs/out/first.png');
    // The reference row previews the known asset.
    expect(target.querySelector(`[data-section="reference"] img[src="${asset.url}"]`)).not.toBeNull();
    expect(new FormData(target.querySelector('form')!).get('image_path')).toBe('/outputs/out/first.png');
  });

  it('deletes an asset from history after confirmation', async () => {
    const first = makeAsset({ id: 'out/first.png', filename: 'first.png' });
    const second = makeAsset({ id: 'out/second.png', filename: 'second.png' });
    await mountWithHistory([first, second]);
    galleryApiMocks.deleteAsset.mockResolvedValue(undefined);

    openTileMenu(second);
    (document.querySelector('[role="menu"] [data-action="delete"]') as HTMLButtonElement).click();
    await settle();
    expect(document.querySelector('[role="alertdialog"]')?.textContent).toContain('Delete "second.png"?');
    await answerConfirm(false);
    expect(galleryApiMocks.deleteAsset).not.toHaveBeenCalled();

    openTileMenu(second);
    (document.querySelector('[role="menu"] [data-action="delete"]') as HTMLButtonElement).click();
    await settle();
    await answerConfirm(true);

    expect(galleryApiMocks.deleteAsset).toHaveBeenCalledWith(second.id);
    expect(galleryStore.assets.map((asset) => asset.id)).toEqual([first.id]);
    expect(historyTile(second)).toBeNull();
  });

  it('marks settings that differ from the model default and resets one row at a time', async () => {
    await mountWithHistory([]);
    const stepsRow = () => (target.querySelector('#ws-steps') as HTMLInputElement).closest('.inspector-row') as HTMLElement;
    const changedCount = () => target.querySelector('[data-testid="changed-count"]')?.textContent ?? '0';

    expect(stepsRow().dataset.changed).toBe('false');
    expect(changedCount()).toBe('0');

    const steps = target.querySelector('#ws-steps') as HTMLInputElement;
    steps.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowUp', shiftKey: true, bubbles: true }));
    await settle();

    expect(draft.state.steps).toBe(38);
    expect(stepsRow().dataset.changed).toBe('true');
    expect(changedCount()).toBe('1');

    (stepsRow().querySelector('button[aria-label="Reset Steps to default"]') as HTMLButtonElement).click();
    await settle();
    expect(draft.state.steps).toBe(28);
    expect(stepsRow().dataset.changed).toBe('false');
  });

  it('changes a number by dragging its label', async () => {
    await mountWithHistory([]);
    const label = target.querySelector('label[for="ws-steps"]') as HTMLElement;

    label.dispatchEvent(new MouseEvent('pointerdown', { clientX: 100, button: 0, bubbles: true }));
    label.dispatchEvent(new MouseEvent('pointermove', { clientX: 120, bubbles: true }));
    label.dispatchEvent(new MouseEvent('pointerup', { clientX: 120, bubbles: true }));
    await settle();

    expect(draft.state.steps).toBe(33);
    expect((target.querySelector('#ws-steps') as HTMLInputElement).value).toBe('33');
  });

  it('settles a typed width on a valid step when the field is committed', async () => {
    await mountWithHistory([]);
    const width = target.querySelector('#ws-width') as HTMLInputElement;
    width.value = '1000';
    width.dispatchEvent(new Event('input', { bubbles: true }));
    width.dispatchEvent(new Event('change', { bubbles: true }));
    await settle();

    expect(width.value).toBe('1008');
    expect(draft.state).toMatchObject({ width: 1008, dimensionMode: 'custom' });
    expect(new FormData(target.querySelector('form')!).get('width')).toBe('1008');
  });

  it('shows video preset dimensions for the chosen ratio and size', async () => {
    workspaceApiMocks.getWorkspaceCoreContext.mockResolvedValue(makeContext({
      video_ratios: ['16:9', '9:16'],
      video_size_options: { '16:9': ['m'], '9:16': ['m'] },
      video_size_dimensions: { '16:9': { m: [704, 448] }, '9:16': { m: [448, 704] } },
    }));
    draft.update('workflow', 'txt2vid');
    app = flushSync(() => mount(WorkspacePage, { target }));
    await settle();

    const ratio916 = Array.from(target.querySelectorAll<HTMLButtonElement>('[role="group"][aria-label="Aspect ratio"] button')).find((b) => b.textContent === '9:16')!;
    ratio916.click();
    await settle();

    expect((target.querySelector('#ws-width') as HTMLInputElement).value).toBe('448');
    expect((target.querySelector('#ws-height') as HTMLInputElement).value).toBe('704');
  });

  it('disables a reference target the current model cannot use', async () => {
    const asset = makeAsset({ id: 'out/first.png', filename: 'first.png', file_path: '/outputs/out/first.png' });
    const context = makeContext({ history_assets: [asset] });
    context.image_model_defaults = { ...context.image_model_defaults, zit: { ...context.image_model_defaults!.zit, supports_img2img: false } };
    galleryApiMocks.getGallery.mockResolvedValue(makeGalleryPage([asset]));
    workspaceApiMocks.getWorkspaceCoreContext.mockResolvedValue(context);
    app = flushSync(() => mount(WorkspacePage, { target }));
    await settle();

    openTileMenu(asset);
    const forImage = document.querySelector('[role="menu"] [data-action="reference-image"]') as HTMLButtonElement;
    expect(forImage.getAttribute('aria-disabled')).toBe('true');
    expect(forImage.title).not.toBe('');
    forImage.click();
    await settle();
    expect(draft.state.workflow).toBe('txt2img');
  });

  it('submits the chosen asset, not an earlier browsed file, after Use as reference', async () => {
    const asset = makeAsset({ id: 'out/first.png', url: '/media/out/first.png', filename: 'first.png', file_path: '/outputs/out/first.png' });
    draft.update('workflow', 'img2img');
    await mountWithHistory([asset]);
    const submitSpy = workspaceApiMocks.submitGenerate.mockResolvedValue({ job_id: 'job-ref' } as JobContext);

    const fileInput = target.querySelector('#ws-image-file') as HTMLInputElement;
    const browsed = new File(['img'], 'browsed.png', { type: 'image/png' });
    Object.defineProperty(fileInput, 'files', { configurable: true, value: [browsed] });
    fileInput.dispatchEvent(new Event('change', { bubbles: true }));
    await settle();

    openTileMenu(asset);
    (document.querySelector('[role="menu"] [data-action="reference-image"]') as HTMLButtonElement).click();
    await settle();
    (target.querySelector('#ws-prompt') as HTMLTextAreaElement).value = 'a prompt';
    (target.querySelector('#ws-prompt') as HTMLTextAreaElement).dispatchEvent(new Event('input', { bubbles: true }));
    await settle();
    target.querySelector('form')!.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
    await settle();

    const submitted = submitSpy.mock.calls.at(-1)![0];
    expect(submitted.has('image_file')).toBe(false);
    expect(submitted.get('image_path')).toBe('/outputs/out/first.png');
  });

  it('carries the asset prompt and keeps the settings when Use as reference switches the workflow', async () => {
    const asset = makeAsset({ id: 'out/first.png', filename: 'first.png', file_path: '/outputs/out/first.png', prompt: 'a red fox' });
    await mountWithHistory([asset]);
    draft.patch({ prompt: 'old prompt', enhancedPrompt: 'old enhanced', steps: 9, runs: 40 });
    await settle();

    openTileMenu(asset);
    (document.querySelector('[role="menu"] [data-action="reference-image"]') as HTMLButtonElement).click();
    await settle();

    expect(draft.state).toMatchObject({ workflow: 'img2img', prompt: 'a red fox', enhancedPrompt: '', steps: 9, runs: 40, referenceImagePath: '/outputs/out/first.png' });
    expect((target.querySelector('#ws-prompt') as HTMLTextAreaElement).value).toBe('a red fox');
  });

  it('resets W × H after a reused custom size back to a submittable preset', async () => {
    await mountWithHistory([]);
    draft.loadFromUrl({ ratio: '2:3', size: 'custom', width: '640', height: '480' }, makeContext());
    await settle();

    const row = (target.querySelector('#ws-width') as HTMLInputElement).closest('.inspector-row') as HTMLElement;
    (row.querySelector('button[aria-label="Reset W × H to default"]') as HTMLButtonElement).click();
    await settle();

    const submitted = new FormData(target.querySelector('form')!);
    expect(submitted.get('size')).toBe('m');
    expect(submitted.has('width')).toBe(false);
  });

  it('clears the reference when its asset is deleted', async () => {
    const asset = makeAsset({ id: 'out/first.png', filename: 'first.png', file_path: '/outputs/out/first.png' });
    await mountWithHistory([asset]);
    galleryApiMocks.deleteAsset.mockResolvedValue(undefined);
    openTileMenu(asset);
    (document.querySelector('[role="menu"] [data-action="reference-image"]') as HTMLButtonElement).click();
    await settle();
    expect(draft.state.referenceImagePath).toBe('/outputs/out/first.png');

    openTileMenu(asset);
    (document.querySelector('[role="menu"] [data-action="delete"]') as HTMLButtonElement).click();
    await settle();
    await answerConfirm(true);

    expect(draft.state.referenceImagePath).toBeNull();
  });

  it('locks the seed to the latest output and unlocks it back to random', async () => {
    const asset = makeAsset({ id: 'out/first.png', filename: 'first.png', seed: 4242 });
    await mountWithHistory([asset]);
    const lock = () => target.querySelector('#ws-seed')!.closest('.inspector-value')!.querySelector('button[aria-pressed]') as HTMLButtonElement;

    lock().click();
    await settle();
    expect(draft.state.seed).toBe(4242);
    expect(lock().getAttribute('aria-pressed')).toBe('true');

    lock().click();
    await settle();
    expect(draft.state.seed).toBeNull();
  });

  describe('stepping through history on the stage', () => {
    const first = () => makeAsset({ id: 'out/first.png', filename: 'first.png' });
    const second = () => makeAsset({ id: 'out/second.png', filename: 'second.png' });
    const third = () => makeAsset({ id: 'out/third.png', filename: 'third.png' });
    const stageTile = () => target.querySelector('.latest-region article') as HTMLElement;
    const position = () => target.querySelector('[data-testid="stage-position"]');
    const arrow = (key: 'ArrowLeft' | 'ArrowRight') => {
      document.dispatchEvent(new KeyboardEvent('keydown', { key, bubbles: true, cancelable: true }));
    };

    it('steps older with → and newer with ←, shows the position, and Latest returns', async () => {
      await mountWithHistory([first(), second(), third()]);
      expect(stageTile().getAttribute('aria-label')).toBe('first.png');
      expect(position()).toBeNull();

      arrow('ArrowRight');
      await settle();
      expect(stageTile().getAttribute('aria-label')).toBe('second.png');
      expect(position()?.textContent).toContain('2');

      arrow('ArrowRight');
      arrow('ArrowRight');
      await settle();
      expect(stageTile().getAttribute('aria-label')).toBe('third.png');

      arrow('ArrowLeft');
      await settle();
      expect(stageTile().getAttribute('aria-label')).toBe('second.png');

      (position()!.querySelector('button') as HTMLButtonElement).click();
      await settle();
      expect(stageTile().getAttribute('aria-label')).toBe('first.png');
      expect(position()).toBeNull();
    });

    it('has Newer and Older buttons that disable at the ends', async () => {
      await mountWithHistory([first(), second()]);
      const newer = () => target.querySelector('button[aria-label="Newer output"]') as HTMLButtonElement;
      const older = () => target.querySelector('button[aria-label="Older output"]') as HTMLButtonElement;
      expect(newer().disabled).toBe(true);
      expect(older().disabled).toBe(false);

      older().click();
      await settle();
      expect(stageTile().getAttribute('aria-label')).toBe('second.png');
      expect(older().disabled).toBe(true);
      expect(newer().disabled).toBe(false);
    });

    it('ignores the arrows while typing in a field', async () => {
      await mountWithHistory([first(), second()]);
      const prompt = target.querySelector('#ws-prompt') as HTMLTextAreaElement;
      prompt.focus();
      prompt.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true, cancelable: true }));
      await settle();
      expect(stageTile().getAttribute('aria-label')).toBe('first.png');
    });
  });
});

describe('WorkspacePage extend', () => {
  const SOURCE = '/clips/walk.mp4';
  const ENHANCER: PromptEnhancerContract = {
    matrix: {
      axes: [
        { key: 'style', label: 'Style', multi: false, video_only: false, audio_only: false, options: [{ slug: 'keep', label: 'Keep', description: '', group: null }], default: ['keep'] },
        { key: 'motion', label: 'Motion', multi: true, video_only: true, audio_only: false, options: [{ slug: 'action', label: 'Action', description: '', group: null }], default: ['action'] },
      ],
      defaults: { style: 'keep', mood: 'keep', theme: 'keep', camera: 'keep', pace: 'keep', details: [], length: 'same', motion: ['action'], sound: [], note: '' },
    },
    model: 'owner/llm',
    revision: null,
    downloaded: true,
    download_size_label: null,
    default_max_words: 300,
    error: null,
  };

  /** A context whose contract lists `extend`, with the control list the server sends for it. */
  function makeExtendContext(options: { withExtend?: boolean } = {}): WorkspaceContext {
    const context = makeContext();
    if (options.withExtend !== false) context.workflow_contract.values = [...context.workflow_contract.values, 'extend'];
    context.workflow_contract.definitions.extend.visible_controls = [
      'workflow', 'model', 'loras', 'prompt_source', 'prompt_scene', 'prompt_inline', 'prompt_enhance', 'prompt_enhance_auto',
      'prompt_file_path', 'prompt_file_option', 'prompt_file_preview', 'prompt_file_edit',
      'source_video', 'source_video_path', 'source_video_clear', 'extend_length', 'runs', 'steps', 'seed', 'audio', 'low_memory',
    ];
    context.prompt_enhancer = ENHANCER;
    return context;
  }

  beforeEach(() => {
    workspaceApiMocks.submitGenerate.mockResolvedValue({
      job_id: 'job-ext', workflow: 'extend', prompt: '', model: 'ltx-8', runs: 1, created_at: '2026-10-11T10:00:00Z',
    });
    browseApiMocks.browsePlaces.mockResolvedValue(placesFor('source_video'));
  });

  async function mountExtend(options: { context?: WorkspaceContext; params?: Record<string, string>; props?: Record<string, unknown> } = {}): Promise<void> {
    workspaceApiMocks.parseUrlPrefill.mockReturnValue(options.params ?? {});
    workspaceApiMocks.getWorkspaceCoreContext.mockResolvedValue(options.context ?? makeExtendContext());
    if (!options.params) draft.update('workflow', 'extend');
    app = flushSync(() => mount(WorkspacePage, { target, props: options.props }));
    await settle();
  }

  const input = (id: string) => target.querySelector(`#${id}`) as HTMLInputElement;
  const button = (id: string) => target.querySelector(`#${id}`) as HTMLButtonElement;
  const text = (selector: string) => (target.querySelector(selector)?.textContent ?? '').replace(/\s+/g, ' ').trim();

  function typePath(value: string, key?: 'Enter' | 'change'): void {
    const el = input('ws-source-path');
    el.value = value;
    el.dispatchEvent(new Event('input', { bubbles: true }));
    if (key === 'Enter') el.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true, cancelable: true }));
    if (key === 'change') el.dispatchEvent(new Event('change', { bubbles: true }));
  }

  /** Pick *path* in the path browser that Browse opens. */
  async function browseTo(path: string): Promise<void> {
    browseApiMocks.listDirectory.mockResolvedValue(listingWith(path));
    button('ws-source-browse').click();
    await settle();
    chooseInBrowser(path.slice(path.lastIndexOf('/') + 1));
    await settle();
  }

  /** Choose a source the way Browse does, and let the server describe it. */
  async function chooseSource(info = sceneSource(), path = SOURCE): Promise<void> {
    extendApiMocks.inspectExtendSource.mockResolvedValue(info);
    await browseTo(path);
  }

  async function submit(): Promise<FormData | undefined> {
    target.querySelector('form')!.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
    await settle();
    return workspaceApiMocks.submitGenerate.mock.calls.at(-1)?.[0] as FormData | undefined;
  }

  const lastInspect = () => extendApiMocks.inspectExtendSource.mock.calls.at(-1)?.[0] as Record<string, unknown>;

  describe('the tab', () => {
    it('tells the top bar which workflows the contract lists', async () => {
      const props = reactiveProps<{ workflows: string[] | null }>({ workflows: null });
      await mountExtend({ props });
      expect(props.workflows).toContain('extend');
    });
  });

  describe('with no source', () => {
    it('offers the path field and Browse for a source video, with Clear disabled', async () => {
      await mountExtend();
      expect(target.querySelector('[data-section="source-video"]')).not.toBeNull();
      expect(input('ws-source-path').name).toBe('source_video');
      expect(button('ws-source-browse').getAttribute('aria-label')).toBe('Browse for a source video');
      expect(button('ws-source-clear').disabled).toBe(true);
      expect(text('[data-section="source-video"] .inspector-summary')).toBe('None');
    });

    it('disables Generate with the reason linked to it, and leaves the length rows waiting', async () => {
      await mountExtend();
      expect(button('ws-submit').disabled).toBe(true);
      expect(text('#ws-generate-reason')).toBe('Choose a video to extend.');
      expect(button('ws-submit').getAttribute('aria-describedby')).toBe('ws-generate-reason');
      expect(input('ws-extend-seconds').disabled).toBe(true);
      expect(text('[data-testid="extend-frames"]')).toBe('—');
      expect(text('[data-testid="extend-total"]')).toBe('—');
    });

    it('has no Size, Duration, Reference image or upscale controls', async () => {
      await mountExtend();
      for (const section of ['size', 'reference', 'upscale']) expect(target.querySelector(`[data-section="${section}"]`)).toBeNull();
      expect(target.querySelector('#ws-duration')).toBeNull();
      expect(target.querySelector('#ws-video-upscale')).toBeNull();
      expect(target.querySelector('[data-section="sampling"]')).not.toBeNull();
    });

    it('refuses to submit, says why and moves focus to the path field', async () => {
      await mountExtend();
      expect(await submit()).toBeUndefined();
      expect(workspaceApiMocks.submitGenerate).not.toHaveBeenCalled();
      expect(toasts.map((toast) => [toast.type, toast.message])).toContainEqual(['error', 'Choose a video to extend.']);
      await settle();
      expect(document.activeElement).toBe(input('ws-source-path'));
    });

    it('disables Generate on the collapsed strip with the reason as its title', async () => {
      draft.update('sidebarCollapsed', true);
      await mountExtend();
      const generate = target.querySelector('[data-testid="sidebar-strip"] button[type="submit"]') as HTMLButtonElement;
      expect(generate.disabled).toBe(true);
      expect(generate.title).toBe('Generate: choose a video to extend');
    });

    it('keeps the placeholders for an empty extension and allows an empty Prompt', async () => {
      await mountExtend();
      expect(field('ws-prompt').placeholder).toBe('What happens next…');
      expect(field('ws-prompt').required).toBe(false);
      expect(field('ws-scene').placeholder).toBe('Who and where, look, style, camera. Kept for extensions.');
    });
  });

  describe('choosing a source', () => {
    it('asks the server about the picked path and shows what it says', async () => {
      await mountExtend();
      await chooseSource(sceneSource());

      expect(browseApiMocks.listDirectory).toHaveBeenCalledWith('source_video', '', expect.anything());
      expect(lastInspect()).toMatchObject({ path: SOURCE, extend_frames: null, model: 'ltx-8' });
      expect(draft.state.extendSourcePath).toBe(SOURCE);
      expect(text('[data-section="source-video"] .inspector-summary')).toBe('walk.mp4');
      expect(text('[data-testid="extend-source-facts"]')).toBe('512 × 288 · 24 fps · 4.04 s · 97 frames · Audio');
      expect(input('ws-source-path').value).toBe(SOURCE);
      expect(button('ws-source-clear').disabled).toBe(false);
      expect(button('ws-submit').disabled).toBe(false);
      expect(target.querySelector('#ws-generate-reason')).toBeNull();
    });

    it('shows the last frame of a clip inside the output folder as its thumbnail', async () => {
      await mountExtend();
      await chooseSource(sceneSource({ media_url: '/media/out/walk.mp4', duration_seconds: 4.04 }));
      expect(target.querySelector('[data-section="source-video"] .reference-thumb video')?.getAttribute('src')).toBe('/media/out/walk.mp4#t=3.99');

      await chooseSource(sceneSource({ media_url: null }), '/elsewhere/walk.mp4');
      expect(target.querySelector('[data-section="source-video"] .reference-thumb video')).toBeNull();
      expect(target.querySelector('[data-section="source-video"] .reference-thumb svg')).not.toBeNull();
    });

    it('lists a rejected source\'s problems, keeps Generate off and points to the Source video section', async () => {
      await mountExtend();
      await chooseSource(sceneSource({ problems: [{ code: 'off_grid', message: 'Its size, 500 × 300, isn\'t a multiple of 32.' }, { code: 'hdr', message: 'It is HDR or wide gamut.' }] }));

      const problems = target.querySelector('[data-testid="extend-source-problems"]');
      expect(problems?.textContent).toContain("This video can't be extended:");
      expect(Array.from(problems!.querySelectorAll('li')).map((item) => item.textContent)).toEqual(["Its size, 500 × 300, isn't a multiple of 32.", 'It is HDR or wide gamut.']);
      expect(button('ws-submit').disabled).toBe(true);
      expect(text('#ws-generate-reason')).toBe("This video can't be extended. See Source video.");
      expect(target.querySelector('#ws-source-help')?.getAttribute('aria-live')).toBe('polite');
    });

    it('reports a path with no video and keeps the reason Choose a video', async () => {
      await mountExtend();
      extendApiMocks.inspectExtendSource.mockRejectedValue(new ApiError('not found', 404, 'Source video not found: /nope.mp4'));
      await browseTo('/nope.mp4');

      expect(text('#ws-source-help')).toBe('No video at this path.');
      expect(input('ws-source-path').getAttribute('aria-invalid')).toBe('true');
      expect(input('ws-source-path').getAttribute('aria-describedby')).toBe('ws-source-help');
      expect(button('ws-submit').disabled).toBe(true);
      expect(text('#ws-generate-reason')).toBe('Choose a video to extend.');
    });

    it('clears the source and its facts but keeps the typed text', async () => {
      await mountExtend();
      await chooseSource(sceneSource());
      type('ws-prompt', 'She turns away.');
      await settle();

      button('ws-source-clear').click();
      await settle();
      expect(draft.state.extendSourcePath).toBeNull();
      expect(draft.extendSource).toBeNull();
      expect(target.querySelector('[data-testid="extend-source-facts"]')).toBeNull();
      expect(input('ws-source-path').value).toBe('');
      expect(field('ws-scene').value).toBe('A woman on a rainy street.');
      expect(field('ws-prompt').value).toBe('She turns away.');
      expect(button('ws-submit').disabled).toBe(true);
    });

    it('shows the moved-file message for a Reuse whose source is gone, and gives the moved path no facts', async () => {
      extendApiMocks.inspectExtendSource.mockRejectedValue(new ApiError('not found', 404, 'gone'));
      await mountExtend({ params: { workflow: 'extend', source_video: '/clips/old.mp4', missing_source_video: 'true', scene: 'A street.', prompt: 'She waves.' } });
      const moved = target.querySelector('[data-testid="extend-source-moved"]');
      expect(moved?.textContent).toContain('The source video has moved or been deleted; choose it again to extend.');
      expect(field('ws-scene').value).toBe('A street.');
      expect(field('ws-prompt').value).toBe('She waves.');
      expect(button('ws-submit').disabled).toBe(true);
      expect(input('ws-source-path').getAttribute('aria-invalid')).toBe('true');
    });
  });

  describe('Scene and Prompt', () => {
    it('fills the old prompt of a legacy source, warns right away and points the Scene placeholder at moving the description', async () => {
      await mountExtend();
      await chooseSource(legacySource());

      expect(field('ws-prompt').value).toBe('A woman stands in the rain and says "Hello".');
      expect(field('ws-scene').value).toBe('');
      expect(field('ws-scene').placeholder).toBe('Move the scene description here from the prompt (who, where, look). Then change the prompt to what happens next.');
      const warning = target.querySelector('[data-testid="original-prompt-warning"]') as HTMLElement;
      expect(warning.id).toBe('ws-prompt-original');
      expect(warning.textContent?.trim()).toBe('Still the original prompt. Change it to what happens next.');
      expect(warning.closest('[aria-live="polite"]')).not.toBeNull();
      expect(field('ws-prompt').getAttribute('aria-describedby')).toBe('ws-prompt-original');
      expect(button('ws-submit').disabled).toBe(false);
    });

    it('hides the warning once the Prompt is edited and still submits either way', async () => {
      await mountExtend();
      await chooseSource(legacySource());

      const unchanged = await submit();
      expect(unchanged?.get('prompt')).toBe('A woman stands in the rain and says "Hello".');

      type('ws-prompt', 'She walks away.');
      await settle();
      expect(target.querySelector('[data-testid="original-prompt-warning"]')).toBeNull();
      expect(field('ws-prompt').getAttribute('aria-describedby')).toBeNull();
      expect((await submit())?.get('prompt')).toBe('She walks away.');
    });

  });

  describe('the source flow', () => {
    it('lets a new source replace both boxes, edits included, and clears the Enhanced pair', async () => {
      await mountExtend();
      await chooseSource(sceneSource());
      type('ws-scene', 'My own scene.');
      type('ws-prompt', 'My own action.');
      draft.patch({ enhancedScene: 'Enhanced scene.', enhancedPrompt: 'Enhanced action.' });
      await settle();

      await chooseSource(sceneSource({ path: '/clips/other.mp4', scene: 'Another scene.', prompt: 'Another action.' }), '/clips/other.mp4');
      expect(field('ws-scene').value).toBe('Another scene.');
      expect(field('ws-prompt').value).toBe('');
      expect(draft.state).toMatchObject({ enhancedScene: '', enhancedPrompt: '', enhancedFrom: null });
    });

    it('replaces the boxes when a different path is typed and checked', async () => {
      await mountExtend();
      await chooseSource(sceneSource());
      type('ws-prompt', 'My own action.');

      extendApiMocks.inspectExtendSource.mockResolvedValue(legacySource({ path: '/clips/other.mp4' }));
      typePath('/clips/other.mp4', 'Enter');
      await settle();
      expect(lastInspect()).toMatchObject({ path: '/clips/other.mp4' });
      expect(field('ws-prompt').value).toBe('A woman stands in the rain and says "Hello".');
      expect(field('ws-scene').value).toBe('');
    });

    it('waits 300 ms after the last keystroke before checking a typed path', async () => {
      fakeDebounceTimers();
      await mountExtend();
      extendApiMocks.inspectExtendSource.mockResolvedValue(sceneSource());
      typePath('/clips/w');
      typePath('/clips/walk.mp4');
      await settle();
      expect(extendApiMocks.inspectExtendSource).not.toHaveBeenCalled();

      await waitDebounce();
      expect(extendApiMocks.inspectExtendSource).toHaveBeenCalledTimes(1);
      expect(lastInspect()).toMatchObject({ path: '/clips/walk.mp4' });
    });

    it('keeps a typed path when the workflow changes before the check', async () => {
      await mountExtend();
      extendApiMocks.inspectExtendSource.mockResolvedValue(sceneSource());
      typePath('/clips/walk.mp4');
      draft.update('workflow', 'txt2vid');
      await settle();
      expect(draft.state.extendSourcePath).toBe('/clips/walk.mp4');

      draft.update('workflow', 'extend');
      await settle();
      expect(lastInspect()).toMatchObject({ path: '/clips/walk.mp4' });
    });

    it('checks the path when the field is left', async () => {
      await mountExtend();
      extendApiMocks.inspectExtendSource.mockResolvedValue(sceneSource());
      typePath(SOURCE, 'change');
      await settle();
      expect(extendApiMocks.inspectExtendSource).toHaveBeenCalledTimes(1);
    });

    it('does not submit the form when Enter is pressed in the path field', async () => {
      await mountExtend();
      extendApiMocks.inspectExtendSource.mockResolvedValue(sceneSource());
      const event = new KeyboardEvent('keydown', { key: 'Enter', bubbles: true, cancelable: true });
      input('ws-source-path').value = SOURCE;
      input('ws-source-path').dispatchEvent(new Event('input', { bubbles: true }));
      input('ws-source-path').dispatchEvent(event);
      expect(event.defaultPrevented).toBe(true);
    });

    it('checks the source again when the length changes, so the memory note follows', async () => {
      fakeDebounceTimers();
      await mountExtend();
      await chooseSource(sceneSource());
      expect(extendApiMocks.inspectExtendSource).toHaveBeenCalledTimes(1);
      extendApiMocks.inspectExtendSource.mockResolvedValue(sceneSource({ extend_frames: 72, memory_fit: { status: 'tight', required_gb: 21, budget_gb: 24 } }));

      input('ws-extend-seconds').value = '3';
      input('ws-extend-seconds').dispatchEvent(new Event('change', { bubbles: true }));
      await settle();
      expect(draft.state.extendFrames).toBe(72);
      await waitDebounce();

      expect(extendApiMocks.inspectExtendSource).toHaveBeenCalledTimes(2);
      expect(lastInspect()).toMatchObject({ path: SOURCE, extend_frames: 72 });
      expect(target.querySelector('[data-testid="extend-memory-note"]')?.getAttribute('data-status')).toBe('tight');
    });

    it('checks a saved source once on load and leaves the text it was reused with', async () => {
      extendApiMocks.inspectExtendSource.mockResolvedValue(sceneSource());
      await mountExtend({ params: { workflow: 'extend', source_video: SOURCE, extend_frames: '48', scene: 'My street.', prompt: 'She waves.' } });
      await settle();
      expect(extendApiMocks.inspectExtendSource).toHaveBeenCalledTimes(1);
      expect(lastInspect()).toMatchObject({ path: SOURCE, extend_frames: 48 });
      expect(field('ws-scene').value).toBe('My street.');
      expect(field('ws-prompt').value).toBe('She waves.');
      expect(button('ws-submit').disabled).toBe(false);
    });

  });

  describe('length', () => {
    it('shows Extend by with its frames and the total at the source frame rate', async () => {
      await mountExtend();
      await chooseSource(sceneSource());
      expect(input('ws-extend-seconds').value).toBe('2');
      expect(input('ws-extend-seconds').disabled).toBe(false);
      expect(text('[data-testid="extend-frames"]')).toBe('+48 frames');
      expect(text('[data-testid="extend-total"]')).toBe('6.04 s · 145 frames');
      expect(text('label[for="ws-extend-seconds"]')).toBe('Extend by (s)');
    });

    it('snaps a typed length to 8-frame steps of the source frame rate', async () => {
      await mountExtend();
      await chooseSource(sceneSource({ fps: 30, frame_count: 90 }));
      input('ws-extend-seconds').value = '2';
      input('ws-extend-seconds').dispatchEvent(new Event('change', { bubbles: true }));
      await settle();
      expect(draft.state.extendFrames).toBe(64);
      expect(input('ws-extend-seconds').value).toBe('2.13');
      expect(text('[data-testid="extend-frames"]')).toBe('+64 frames');
      expect(text('[data-testid="extend-total"]')).toBe('5.13 s · 154 frames');
    });

    it('notes a too-large fit without blocking Generate', async () => {
      await mountExtend();
      await chooseSource(sceneSource({ memory_fit: { status: 'too_large', required_gb: 40, budget_gb: 24 } }));
      expect(target.querySelector('[data-testid="extend-memory-note"]')?.getAttribute('data-status')).toBe('too_large');
      expect(target.querySelector('[data-testid="extend-memory-note"]')?.closest('[aria-live="polite"]')).not.toBeNull();
      expect(button('ws-submit').disabled).toBe(false);
    });
  });

  describe('audio', () => {
    it('disables Audio for a silent source, keeps the preference and submits audio=false', async () => {
      draft.update('audio', true);
      await mountExtend();
      await chooseSource(sourceInfo({ has_audio: false }));

      const audio = input('ws-audio');
      expect(audio.disabled).toBe(true);
      expect(audio.checked).toBe(false);
      expect(draft.state.audio).toBe(true);
      expect(text('[data-section="video"] .inspector-summary')).not.toContain('Audio');
      expect((await submit())?.getAll('audio')).toEqual(['false']);
    });

    it('leaves Audio on for a source with sound', async () => {
      draft.update('audio', true);
      await mountExtend();
      await chooseSource(sourceInfo({ has_audio: true }));
      expect(input('ws-audio').disabled).toBe(false);
      expect(input('ws-audio').checked).toBe(true);
    });
  });

  describe('what Generate submits', () => {
    it('sends the source, the new frames and both text boxes, and none of the size or length fields', async () => {
      await mountExtend();
      await chooseSource(sceneSource());
      type('ws-prompt', 'She waves.');
      await settle();

      const form = await submit();
      expect(form?.get('workflow')).toBe('extend');
      expect(form?.get('mode')).toBe('video');
      expect(form?.get('source_video')).toBe(SOURCE);
      expect(form?.get('extend_frames')).toBe('48');
      expect(form?.get('scene')).toBe('A woman on a rainy street.');
      expect(form?.get('prompt')).toBe('She waves.');
      for (const name of ['ratio', 'size', 'width', 'height', 'frames', 'image_path', 'upscale', 'video_upscale_factor']) expect(form?.has(name)).toBe(false);
    });

    it('records a blank scene as blank, not as the source\'s scene', async () => {
      await mountExtend();
      await chooseSource(sceneSource());
      type('ws-scene', '');
      await settle();
      expect((await submit())?.get('scene')).toBe('');
    });

    it('waits for a source still being checked, then submits, without a reason, a toast or a jump of focus', async () => {
      await mountExtend();
      const release = holdInspection();
      await browseTo(SOURCE);
      expect(button('ws-submit').disabled).toBe(true);
      expect(target.querySelector('#ws-generate-reason')).toBeNull();
      expect(target.querySelector('[data-section="source-video"] .spinner')).not.toBeNull();

      target.querySelector('form')!.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
      await settle();
      expect(workspaceApiMocks.submitGenerate).not.toHaveBeenCalled();
      expect(toasts).toHaveLength(0);
      expect(document.activeElement).not.toBe(input('ws-source-path'));

      release(sceneSource());
      await settle();
      expect(button('ws-submit').disabled).toBe(false);
      expect(workspaceApiMocks.submitGenerate).toHaveBeenCalledTimes(1);
      expect((workspaceApiMocks.submitGenerate.mock.calls[0][0] as FormData).get('source_video')).toBe(SOURCE);
    });

    it('stays off the queue while the source is rejected', async () => {
      await mountExtend();
      await chooseSource(sceneSource({ problems: [{ code: 'hdr', message: 'It is HDR or wide gamut.' }] }));
      expect(await submit()).toBeUndefined();
      expect(toasts.map((toast) => toast.message)).toContain("This video can't be extended. See Source video.");
    });
  });

  describe('with a settings preset', () => {
    const CLIP_PRESET: SettingsPreset = {
      slug: 'clip', name: 'Clip', mode: 'video', model: 'ltx-8',
      settings: { lora: '', ratio: '9:16', size: 's', width: null, height: null, runs: 1, steps: 6, frames: 49, audio: false, low_memory: true, video_upscale_factor: 2, review: true },
    };
    const presetButton = () => target.querySelector('[data-testid="preset-button"]') as HTMLButtonElement;

    async function applyClip(): Promise<void> {
      flushSync(() => presetButton().click());
      await settle();
      flushSync(() => (document.querySelector('[role="menu"][aria-label="Preset"] [data-action="preset:clip"]') as HTMLButtonElement).click());
      await settle();
    }

    it('applies every key of the preset, and the job is submitted with the same fields as without one', async () => {
      presetApiMocks.listPresets.mockResolvedValue([CLIP_PRESET]);
      const context = makeExtendContext();
      context.video_ratios = ['16:9', '9:16'];
      context.video_size_options = { '16:9': ['s', 'm'], '9:16': ['s', 'm'] };
      context.video_size_dimensions = { '16:9': { m: [704, 448], s: [512, 320] }, '9:16': { m: [448, 704], s: [320, 512] } };
      await mountExtend({ context });
      await chooseSource(sceneSource());
      type('ws-prompt', 'She waves.');
      await settle();

      await applyClip();

      expect(draft.state).toMatchObject({ steps: 6, ratio: '9:16', size: 's', audio: false, lowMemory: true, videoUpscaleEnabled: true, videoUpscaleReview: true, appliedPreset: 'clip' });
      const form = await submit();
      expect(form?.get('workflow')).toBe('extend');
      expect(form?.get('source_video')).toBe(SOURCE);
      expect(form?.get('prompt')).toBe('She waves.');
      expect(form?.get('steps')).toBe('6');
      for (const name of ['ratio', 'size', 'width', 'height', 'frames', 'image_path', 'upscale', 'video_upscale_factor']) expect(form?.has(name)).toBe(false);
    });
  });

  describe('Enhance', () => {
    it('is enabled with both boxes empty once a source can be extended, and sends the source and its audio', async () => {
      enhanceApiMocks.enhancePrompt.mockResolvedValue();
      draft.update('audio', true);
      await mountExtend();
      await chooseSource(sourceInfo({ has_audio: false }));

      await openEnhanceRun();
      const run = document.querySelector('#ws-enhance-run') as HTMLButtonElement;
      expect(run.disabled).toBe(false);
      run.click();
      await settle();

      const body = enhanceApiMocks.enhancePrompt.mock.calls[0]?.[0] as { mode: string; context: Record<string, unknown> };
      expect(body.mode).toBe('video');
      expect(body.context).toMatchObject({ source_video: SOURCE, extend_frames: 48, audio: false, reference_image: false });
    });

    it.each([
      ['no source is chosen', undefined, 'Choose a video to extend first.'],
      ['the source is rejected', sceneSource({ problems: [{ code: 'hdr', message: 'It is HDR or wide gamut.' }] }), 'Choose a video that can be extended first.'],
    ])('is disabled, with its own hint, when %s', async (_name, info, hint) => {
      await mountExtend();
      if (info) await chooseSource(info);
      await openEnhanceRun();
      expect((document.querySelector('#ws-enhance-run') as HTMLButtonElement).disabled).toBe(true);
      expect(document.querySelector('.enhance-popover')?.textContent).toContain(hint);
    });

    it('marks the Enhanced pair out of date when the length changes', async () => {
      enhanceApiMocks.enhancePrompt.mockImplementation(async (_body: unknown, ...rest: unknown[]) => {
        (rest[0] as (frame: EnhanceFrame) => void)({ type: 'done', scene: 'A woman on a rainy street.', prompt: 'She waves slowly.', clamped: false });
      });
      await mountExtend();
      await chooseSource(sceneSource());
      await runEnhance();
      expect(draft.state.enhancedFrom).toMatchObject({ mode: 'video', sourceVideoPath: SOURCE, extendFrames: 48 });
      expect(target.textContent).not.toContain('Out of date');

      draft.update('extendFrames', 72);
      await settle();
      expect(target.textContent).toContain('Out of date');
    });

    describe('Action only', () => {
      const rewrite = () => Array.from(document.querySelectorAll('#ws-enhance-panel [aria-labelledby="ws-enhance-rewrite-label"] [role="radio"]')) as HTMLButtonElement[];

      /** The server answers an action-only run with no scene part. */
      function answerActionOnly(): void {
        enhanceApiMocks.enhancePrompt.mockImplementation(async (_body: unknown, ...rest: unknown[]) => {
          const onFrame = rest[0] as (frame: EnhanceFrame) => void;
          onFrame({ type: 'text', text: 'She waves sl', scene: '', prompt: 'She waves sl' });
          onFrame({ type: 'done', scene: '', prompt: 'She waves slowly.', clamped: false });
        });
      }

      it('is the default for an extension and is sent with the request', async () => {
        answerActionOnly();
        await mountExtend();
        await chooseSource(sceneSource());
        type('ws-prompt', 'She waves.');
        await settle();
        await openEnhanceRun();
        expect(rewrite().map((el) => el.getAttribute('aria-checked'))).toEqual(['false', 'true']);
        (document.querySelector('#ws-enhance-run') as HTMLButtonElement).click();
        await settle();
        expect(enhanceApiMocks.enhancePrompt.mock.calls[0]?.[0]).toMatchObject({ scene: 'A woman on a rainy street.', prompt: 'She waves.', settings: { parts: 'action' } });
        expect(draft.state.enhanceSettingsVideo?.partsByWorkflow?.extend).toBeUndefined();
      });

      it('fills only the Action while streaming and keeps the Scene placeholder', async () => {
        let release: () => void = () => undefined;
        const gate = new Promise<void>((resolve) => { release = resolve; });
        enhanceApiMocks.enhancePrompt.mockImplementation(async (_body: unknown, ...rest: unknown[]) => {
          (rest[0] as (frame: EnhanceFrame) => void)({ type: 'text', text: 'She waves sl', scene: '', prompt: 'She waves sl' });
          await gate;
        });
        await mountExtend();
        await chooseSource(sceneSource());
        await runEnhance();
        expect(field('ws-enhanced-prompt').value).toBe('She waves sl');
        expect(field('ws-enhanced-prompt').getAttribute('aria-busy')).toBe('true');
        expect(field('ws-enhanced-scene').value).toBe('');
        expect(field('ws-enhanced-scene').placeholder).toBe('Not enhanced. Generate uses your scene.');
        release();
        await settle();
      });

    });
  });

  describe('restoring the clip text', () => {
    const link = (label: string) => Array.from(target.querySelectorAll('.label-extras button')).find((el) => el.textContent?.trim() === label) as HTMLButtonElement | undefined;
    const status = () => target.querySelector('[role="status"]')?.textContent?.trim();

    it('offers the clip\'s scene only while the Scene differs, and restores it with a status and focus', async () => {
      await mountExtend();
      await chooseSource(sceneSource());
      expect(link("Use the clip's scene")).toBeUndefined();

      type('ws-scene', 'A woman on a sunny street.');
      await settle();
      const restore = link("Use the clip's scene")!;
      expect(restore.title).toBe('Replace your scene with the one recorded in the clip');
      restore.click();
      await settle();

      expect(field('ws-scene').value).toBe('A woman on a rainy street.');
      expect(draft.state.scene).toBe('A woman on a rainy street.');
      expect(status()).toBe('Scene restored from the clip.');
      expect(document.activeElement).toBe(field('ws-scene'));
      expect(link("Use the clip's scene")).toBeUndefined();
    });

    it('clears Out of date when the clip\'s scene is restored', async () => {
      enhanceApiMocks.enhancePrompt.mockImplementation(async (_body: unknown, ...rest: unknown[]) => {
        (rest[0] as (frame: EnhanceFrame) => void)({ type: 'done', scene: '', prompt: 'She waves slowly.', clamped: false });
      });
      await mountExtend();
      await chooseSource(sceneSource());
      type('ws-prompt', 'She waves.');
      await settle();
      await runEnhance();

      type('ws-scene', 'A woman on a sunny street.');
      await settle();
      expect(target.textContent).toContain('Out of date');
      link("Use the clip's scene")!.click();
      await settle();
      expect(target.textContent).not.toContain('Out of date');
    });

    it('offers the clip\'s prompt for a legacy clip, beside Action, and brings back the original-prompt warning', async () => {
      await mountExtend();
      await chooseSource(legacySource());
      expect(link("Use the clip's prompt")).toBeUndefined();
      expect(link("Use the clip's scene")).toBeUndefined();

      type('ws-prompt', 'She walks away.');
      await settle();
      expect(target.querySelector('[data-testid="original-prompt-warning"]')).toBeNull();
      const restore = link("Use the clip's prompt")!;
      expect(restore.title).toBe("Replace your action with the clip's recorded prompt");
      restore.click();
      await settle();

      expect(field('ws-prompt').value).toBe('A woman stands in the rain and says "Hello".');
      expect(target.querySelector('[data-testid="original-prompt-warning"]')).not.toBeNull();
      expect(link("Use the clip's prompt")).toBeUndefined();
    });

    it('offers nothing outside an extension', async () => {
      await mountExtend();
      await chooseSource(sceneSource({ legacy: true }));
      type('ws-scene', 'Something else.');
      type('ws-prompt', 'She looks up.');
      await settle();
      expect(link("Use the clip's scene")).toBeDefined();
      expect(target.querySelector('[data-testid="original-prompt-warning"]')).not.toBeNull();

      draft.update('workflow', 'txt2vid');
      await settle();
      expect(link("Use the clip's scene")).toBeUndefined();
      expect(target.querySelector('[data-testid="original-prompt-warning"]')).toBeNull();
    });

  });

  describe('Copy CLI command', () => {
    let writeText: ReturnType<typeof vi.fn<(text: string) => Promise<void>>>;

    beforeEach(() => {
      writeText = mockClipboard();
    });

    it('copies the extension with the source, the length and the scene', async () => {
      await mountExtend();
      await chooseSource(sceneSource());
      type('ws-prompt', 'She waves.');
      await settle();
      button('ws-copy-cli').click();
      await settle();
      const command = writeText.mock.calls[0]?.[0] ?? '';
      expect(command).toContain('ziv video');
      expect(command).toContain('--extend /clips/walk.mp4 --extend-frames 48');
      expect(command).toContain("--prompt 'She waves.'");
      expect(command).toContain("--scene 'A woman on a rainy street.'");
      for (const flag of ['--ratio', '--size', '--frames', '--upscale', '--width']) expect(command).not.toContain(flag);
    });
  });

  describe('Extend from an asset', () => {
    const clip = (overrides: Partial<GalleryAsset> = {}): GalleryAsset =>
      makeAsset({
        id: 'out/walk.mp4', url: '/media/out/walk.mp4', thumbnail_url: '/media/out/walk.mp4', filename: 'walk.mp4', media_type: 'video', workflow: 'txt2vid',
        file_path: SOURCE, scene: 'A woman on a rainy street.', prompt: 'She looks up.', extend: { available: true, reason: null }, ...overrides,
      });

    async function mountWithClip(asset: GalleryAsset, workflow: 'txt2vid' | 'extend' = 'txt2vid'): Promise<void> {
      extendApiMocks.inspectExtendSource.mockResolvedValue(sceneSource());
      galleryApiMocks.getGallery.mockResolvedValue(makeGalleryPage([asset]));
      workspaceApiMocks.getWorkspaceCoreContext.mockResolvedValue({ ...makeExtendContext(), history_assets: [asset] });
      draft.update('workflow', workflow);
      app = flushSync(() => mount(WorkspacePage, { target }));
      await settle();
    }

    const historyTile = (asset: GalleryAsset) => target.querySelector(`#ws-history-shell article[aria-label="${asset.filename}"]`) as HTMLElement;

    function openMenu(asset: GalleryAsset): void {
      (historyTile(asset).querySelector(`button[aria-label="More actions for ${asset.filename}"]`) as HTMLButtonElement).click();
      flushSync();
    }

    it('lists Extend, then Download and Delete, in a history tile\'s menu', async () => {
      const asset = clip();
      await mountWithClip(asset);
      openMenu(asset);
      expect(Array.from(document.querySelectorAll('[role="menu"] [role="menuitem"]')).map((item) => item.getAttribute('data-action'))).toEqual(['reuse', 'extend', 'download', 'delete']);
    });

    it('turns the Workspace to Extend Video with the clip as source, its scene in the boxes, and checks it', async () => {
      const asset = clip();
      await mountWithClip(asset);
      openMenu(asset);
      (document.querySelector('[role="menu"] [data-action="extend"]') as HTMLButtonElement).click();
      await settle();

      expect(draft.state).toMatchObject({ workflow: 'extend', extendSourcePath: SOURCE, scene: 'A woman on a rainy street.', prompt: '' });
      expect(lastInspect()).toMatchObject({ path: SOURCE });
      expect(input('ws-source-path').value).toBe(SOURCE);
      expect(text('[data-testid="extend-source-facts"]')).toContain('512 × 288');
      expect(toasts.map((toast) => toast.message)).toContain('walk.mp4 is now the source video');
    });

    it('replaces the text of an earlier source when already on Extend Video', async () => {
      const asset = clip();
      await mountWithClip(asset, 'extend');
      draft.patch({ scene: 'Old scene.', prompt: 'Old action.', extendSourcePath: '/clips/old.mp4' });
      await settle();

      openMenu(asset);
      (document.querySelector('[role="menu"] [data-action="extend"]') as HTMLButtonElement).click();
      await settle();
      expect(draft.state).toMatchObject({ extendSourcePath: SOURCE, scene: 'A woman on a rainy street.', prompt: '' });
    });

    it('disables Extend with its reason in the menu when the video cannot be extended', async () => {
      const asset = clip({ extend: { available: false, reason: "Upscaled videos can't be extended yet." } });
      await mountWithClip(asset);
      openMenu(asset);
      const item = document.querySelector('[role="menu"] [data-action="extend"]') as HTMLElement;
      expect(item.getAttribute('aria-disabled')).toBe('true');
      expect(item.getAttribute('title')).toBe("Upscaled videos can't be extended yet.");
      item.click();
      await settle();
      expect(draft.state.workflow).toBe('txt2vid');
    });

    it('extends from the viewer button and the E key, and closes the viewer', async () => {
      const asset = clip();
      await mountWithClip(asset);
      (historyTile(asset).querySelector(`button[aria-label="View ${asset.filename}"]`) as HTMLButtonElement).click();
      await settle();
      const viewerButton = document.querySelector('[data-testid="asset-viewer"] [data-action="extend"]') as HTMLButtonElement;
      expect(viewerButton).not.toBeNull();

      document.dispatchEvent(new KeyboardEvent('keydown', { key: 'e', bubbles: true, cancelable: true }));
      await settle();
      expect(draft.state).toMatchObject({ workflow: 'extend', extendSourcePath: SOURCE });
      expect(document.querySelector('[data-testid="asset-viewer"]')).toBeNull();
    });

    it('has no Extend item for an image or where the server offers none', async () => {
      const image = makeAsset({ id: 'out/a.png', filename: 'a.png', file_path: '/outputs/a.png' });
      const video = clip({ id: 'out/b.mp4', filename: 'b.mp4', extend: null });
      galleryApiMocks.getGallery.mockResolvedValue(makeGalleryPage([image, video]));
      workspaceApiMocks.getWorkspaceCoreContext.mockResolvedValue({ ...makeExtendContext(), history_assets: [image, video] });
      app = flushSync(() => mount(WorkspacePage, { target }));
      await settle();
      for (const asset of [image, video]) {
        openMenu(asset);
        expect(document.querySelector('[role="menu"] [data-action="extend"]')).toBeNull();
        document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
        flushSync();
      }
    });
  });

  describe('a prompt sent from the Prompts page', () => {
    it('queues the run once the saved source has been checked', async () => {
      extendApiMocks.inspectExtendSource.mockResolvedValue(sceneSource());
      draft.update('workflow', 'extend');
      draft.update('extendSourcePath', SOURCE);
      requestPromptRun({ path: '/server/prompts.yaml', optionId: 'portrait:1' });
      await mountExtend();
      await settle();

      expect(workspaceApiMocks.submitGenerate).toHaveBeenCalledTimes(1);
      const form = workspaceApiMocks.submitGenerate.mock.calls[0][0] as FormData;
      expect(form.get('source_video')).toBe(SOURCE);
      expect(form.getAll('prompt_option_id')).toEqual(['portrait:1']);
    });

    it('refuses the run with a reason when no source is chosen', async () => {
      requestPromptRun({ path: '/server/prompts.yaml', optionId: 'portrait:1' });
      await mountExtend();
      await settle();

      expect(workspaceApiMocks.submitGenerate).not.toHaveBeenCalled();
      expect(toasts.map((toast) => toast.message)).toContain('Choose a video to extend.');
    });
  });
});
