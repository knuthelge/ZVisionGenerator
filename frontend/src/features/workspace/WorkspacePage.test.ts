// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, mount, unmount } from '../../../node_modules/svelte/src/index-client.js';
import { createClassComponent } from 'svelte/legacy';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { ApiError } from '$lib/api/client';
import { lastSaveTarget, recentFiles, rememberRecent } from '$lib/state/promptFileStorage';
import { draft } from '$lib/state/draft.svelte';
import { downloadStore } from '$lib/state/downloads.svelte';
import { galleryStore } from '$lib/state/gallery.svelte';
import { jobStore } from '$lib/state/job.svelte';
import { mascotHome } from '$lib/state/mascotHome.svelte';
import { hasPendingJobSettings, requestJobSettings } from '$lib/state/pendingJobSettings.svelte';
import { requestPromptRun } from '$lib/state/pendingPromptRun';
import { router } from '$lib/state/router.svelte';
import { toasts } from '$lib/state/toasts.svelte';
import type { DownloadItem, EnhanceFrame, JobSnapshot, PromptEnhancerContract, RecentPrompt, WorkspaceContext, GalleryAsset } from '$lib/types';
import { workspaceMemory } from '$lib/state/pageMemory';

import WorkspacePage from './WorkspacePage.svelte';
import ControlsSidebar from './ControlsSidebar.svelte';
import { chooseInBrowser, listingWith, placesFor } from '../../test-utils/pathBrowser';
import { lastEventSource } from '../../test-utils/eventSource';
import { optionLabels, optionValues, pickOption } from '../../test-utils/select';
import { jobSnap, makeAsset, makeContext, makeGalleryPage, makeIdeogramDefaults, makeImageDefaults, withEnhancer, withVideoEnhancer } from '../../test-utils/workspace';
import { answerConfirm, choosePrompts, createHost, expectRenderedLabelsToResolveControls, fakeDebounceTimers, field, mockClipboard, mountWorkspacePage, openEnhanceRun, pressKey, resetWorkspaceTest, settle, submitForm, type, waitDebounce } from '../../test-utils/workspaceHarness';
import { browseApiMocks, enhanceApiMocks, galleryApiMocks, layoutProbe, modelApiMocks, promptFileApiMocks, recentPromptsApiMocks, workspaceApiMocks } from '../../test-utils/workspaceMocks';

vi.mock('$lib/state/layoutMode.svelte', async (importOriginal) => {
  const actual = await importOriginal<typeof import('$lib/state/layoutMode.svelte')>();
  const { layoutProbe } = await import('../../test-utils/workspaceMocks');
  return {
    ...actual,
    layout: {
      get mode() { return layoutProbe.compact ? 'compact' : 'wide'; },
      get compact() { return layoutProbe.compact; },
      get coarse() { return layoutProbe.compact || layoutProbe.coarse; },
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

vi.mock('$lib/api/recentPrompts', async () => (await import('../../test-utils/workspaceMocks')).recentPromptsApiMocks);

vi.mock('$lib/api/browse', async () => (await import('../../test-utils/workspaceMocks')).browseApiMocks);

vi.mock('$lib/api/discover', async () => (await import('../../test-utils/workspaceMocks')).discoverApiMocks);

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

describe('WorkspacePage', () => {
  beforeEach(() => {
    workspaceApiMocks.submitGenerate.mockResolvedValue({
      job_id: 'job-123',
      workflow: 'txt2img',
      prompt: 'Test prompt',
      model: 'zit',
      runs: 1,
      created_at: '2026-04-23T10:00:00Z',
    });
    browseApiMocks.browsePlaces.mockResolvedValue(placesFor('source_video'));
    Object.defineProperty(URL, 'createObjectURL', {
      configurable: true,
      value: vi.fn(() => 'blob:test-image'),
    });
    Object.defineProperty(URL, 'revokeObjectURL', {
      configurable: true,
      value: vi.fn(),
    });
    window.location.hash = '#/workspace';
    window.history.replaceState({}, '', '/');
  });

  it('renders a loading-safe shell before workspace authority resolves', async () => {
    draft.update('model', 'stale-model');
    draft.update('prompt', 'stale prompt');
    draft.update('steps', 99);
    workspaceApiMocks.getWorkspaceCoreContext.mockImplementation(
      () => new Promise<WorkspaceContext>(() => undefined)
    );

    app = flushSync(() => mount(WorkspacePage, { target }));
    await settle();

    expect(target.querySelector('#ws-prompt')).toBeNull();
    expect((target.querySelector('#ws-submit') as HTMLButtonElement | null)?.disabled).toBe(true);
    expect((target.querySelector('#ws-model') as HTMLButtonElement | null)?.disabled).toBe(true);
    expect(target.textContent).not.toContain('stale-model');
  });

  it('hydrates workspace authority first, then loads history after mount', async () => {
    const asset = makeAsset({ id: 'out/deferred.png', url: '/media/out/deferred.png', filename: 'deferred.png' });
    galleryApiMocks.getGallery.mockResolvedValue(makeGalleryPage([asset]));

    await mountWorkspace(makeContext({ history_assets: [] }));

    expect(workspaceApiMocks.getWorkspaceCoreContext).toHaveBeenCalledTimes(1);
    expect(workspaceApiMocks.getWorkspaceContext).not.toHaveBeenCalled();
    expect(galleryApiMocks.getGallery).toHaveBeenCalledWith(1, 'all', 'newest');
    expect(target.querySelector(`button[aria-label="View ${asset.filename}"]`)).not.toBeNull();
  });

  it('comes back complete from the context a previous visit loaded, before the fresh one answers', async () => {
    await mountWorkspace(makeContext());
    await unmount(app!);
    app = null;
    workspaceApiMocks.getWorkspaceCoreContext.mockImplementation(
      () => new Promise<WorkspaceContext>(() => undefined)
    );

    app = flushSync(() => mount(WorkspacePage, { target }));
    await settle();

    expect((target.querySelector('#ws-model') as HTMLSelectElement).disabled).toBe(false);
    expect(target.querySelector('#ws-prompt')).not.toBeNull();
  });

  it('keeps the mascot at home in the logo while the latest output shows', async () => {
    const asset = makeAsset({ id: 'out/latest.png', url: '/media/out/latest.png', filename: 'latest.png' });
    galleryApiMocks.getGallery.mockResolvedValue(makeGalleryPage([asset]));

    await mountWorkspace(makeContext({ history_assets: [asset] }));

    const preview = target.querySelector('.workspace-preview') as HTMLElement;
    expect(preview.querySelector('[data-testid="latest-loading"]')).not.toBeNull();
    expect(preview.querySelector('svg.mascot')).toBeNull();
    expect(mascotHome.placement.away).toBe(false);

    const image = preview.querySelector(`img[src="${asset.url}"]`) as HTMLImageElement;
    image.dispatchEvent(new Event('load'));
    await settle();

    expect(preview.querySelector('[data-testid="latest-loading"]')).toBeNull();
    expect(preview.querySelector('svg.mascot')).toBeNull();
    expect(mascotHome.placement.away).toBe(false);
  });

  it('looks for history instead of claiming no assets until the first history check settles', async () => {
    let resolveContext!: (context: WorkspaceContext) => void;
    workspaceApiMocks.getWorkspaceCoreContext.mockImplementation(
      () => new Promise<WorkspaceContext>((resolve) => { resolveContext = resolve; })
    );
    let resolveHistory!: (page: ReturnType<typeof makeGalleryPage>) => void;
    galleryApiMocks.getGallery.mockImplementation(
      () => new Promise((resolve) => { resolveHistory = resolve; })
    );

    app = flushSync(() => mount(WorkspacePage, { target }));
    const preview = () => (target.querySelector('.workspace-preview') as HTMLElement).textContent ?? '';

    // Before the core context arrives.
    expect(preview()).toContain('Looking for your latest work…');
    expect(preview()).not.toContain('No generated assets yet');

    // Context arrived without history; the deferred history fetch is still pending.
    resolveContext(makeContext({ history_assets: [] }));
    await settle();
    expect(preview()).toContain('Looking for your latest work…');
    expect(preview()).not.toContain('No generated assets yet');

    resolveHistory(makeGalleryPage([]));
    await settle();
    expect(preview()).toContain('No generated assets yet');
  });

  it('shows the latest video immediately without waiting for load events', async () => {
    const asset = makeAsset({ id: 'out/latest.mp4', url: '/media/out/latest.mp4', filename: 'latest.mp4', media_type: 'video' });
    galleryApiMocks.getGallery.mockResolvedValue(makeGalleryPage([asset]));

    await mountWorkspace(makeContext({ history_assets: [asset] }));

    // Some browsers ignore preload and fire no load events until play is pressed.
    const preview = target.querySelector('.workspace-preview') as HTMLElement;
    const video = preview.querySelector(`video[src="${asset.url}"]`) as HTMLVideoElement;
    expect(video).not.toBeNull();
    expect(preview.querySelector('[data-testid="latest-loading"]')).toBeNull();
  });

  it('prefills seed from Gallery reuse URL params after backend defaults hydrate', async () => {
    workspaceApiMocks.parseUrlPrefill.mockReturnValue({ workflow: 'txt2img', prompt: 'Reuse prompt', seed: '9876' });

    await mountWorkspace(makeContext());

    const seedInput = target.querySelector('#ws-seed') as HTMLInputElement | null;
    const promptInput = target.querySelector('#ws-prompt') as HTMLTextAreaElement | null;

    expect(seedInput).not.toBeNull();
    expect(seedInput?.value).toBe('9876');
    expect(promptInput?.value).toBe('Reuse prompt');

    const form = target.querySelector('form');
    expect(form).not.toBeNull();

    form!.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
    await settle();

    expect(workspaceApiMocks.submitGenerate).toHaveBeenCalledTimes(1);
    const [submittedFormData] = workspaceApiMocks.submitGenerate.mock.calls[0] ?? [];
    expect(submittedFormData).toBeInstanceOf(FormData);
    expect(submittedFormData.get('seed')).toBe('9876');
    expect(submittedFormData.has('output')).toBe(false);
  });

  it('drops the reuse params from the address once they are in the draft, so a reload keeps the edits', async () => {
    workspaceApiMocks.parseUrlPrefill.mockReturnValue({ workflow: 'txt2img', prompt: 'Reuse prompt', seed: '9876' });
    router.navigate('workspace', { prompt: 'Reuse prompt', seed: '9876' });

    await mountWorkspace(makeContext());

    expect((target.querySelector('#ws-prompt') as HTMLTextAreaElement).value).toBe('Reuse prompt');
    expect(router.page).toBe('workspace');
    expect(router.params).toEqual({});
    expect(window.location.hash).toBe('#/workspace');
  });

  it('applies the reuse params once: an edit made before the server answers survives its context', async () => {
    workspaceApiMocks.parseUrlPrefill.mockReturnValue({ workflow: 'txt2img', prompt: 'Reuse prompt', seed: '9876' });
    workspaceMemory.set(makeContext());
    let answer: (ctx: WorkspaceContext) => void = () => {};
    workspaceApiMocks.getWorkspaceCoreContext.mockReturnValue(new Promise<WorkspaceContext>((resolve) => { answer = resolve; }));

    app = flushSync(() => mount(WorkspacePage, { target }));
    await settle();
    expect((target.querySelector('#ws-prompt') as HTMLTextAreaElement).value).toBe('Reuse prompt');

    type('ws-prompt', 'My edit');
    answer(makeContext());
    await settle();

    expect((target.querySelector('#ws-prompt') as HTMLTextAreaElement).value).toBe('My edit');
    expect(draft.state.seed).toBe(9876);
  });

  it('shows a refused Generate above the button, keeps the preview, and clears it when a setting changes', async () => {
    draft.update('prompt', 'A lighthouse at dusk');
    await mountWorkspace(makeContext());
    toasts.splice(0, toasts.length);
    workspaceApiMocks.submitGenerate.mockRejectedValueOnce(new ApiError('rejected', 422, 'Image to image requires a start image.'));

    target.querySelector('form')!.requestSubmit();
    await settle();

    expect(target.querySelector('[data-testid="generate-error"]')?.textContent).toContain('Image to image requires a start image.');
    expect(target.textContent).not.toContain('The Workspace could not load');
    expect(toasts[0]?.type).toBe('error');
    expect(toasts[0]?.message).toBe('Image to image requires a start image.');

    draft.update('steps', draft.state.steps + 1);
    await settle();
    expect(target.querySelector('[data-testid="generate-error"]')).toBeNull();
  });

  it('collapses the sidebar into a strip, remembers it, and still submits every field', async () => {
    draft.update('prompt', 'A lighthouse at dusk');
    await mountWorkspace(makeContext());

    (target.querySelector('button[aria-label="Collapse sidebar"]') as HTMLButtonElement).click();
    await settle();
    expect(target.querySelector('[data-testid="sidebar-strip"]')).not.toBeNull();
    expect(JSON.parse(localStorage.getItem('ziv-workspace-draft-v1')!).sidebarCollapsed).toBe(true);
    // The fields stay in the form while folded away.
    expect(target.querySelector('#ws-prompt')).not.toBeNull();

    (target.querySelector('[data-testid="sidebar-strip"] button[type="submit"]') as HTMLButtonElement).click();
    await settle();
    expect(workspaceApiMocks.submitGenerate).toHaveBeenCalledTimes(1);
    const [submitted] = workspaceApiMocks.submitGenerate.mock.calls[0] ?? [];
    expect(submitted.get('prompt')).toBe('A lighthouse at dusk');
    expect(submitted.get('steps')).not.toBeNull();

    (target.querySelector('button[aria-label="Expand sidebar"]') as HTMLButtonElement).click();
    await settle();
    expect(target.querySelector('[data-testid="sidebar-strip"]')).toBeNull();
  });

  it('expands the collapsed sidebar to report an empty prompt', async () => {
    draft.update('prompt', '');
    draft.update('sidebarCollapsed', true);
    await mountWorkspace(makeContext());
    const prompt = target.querySelector('#ws-prompt') as HTMLTextAreaElement;
    const report = vi.spyOn(prompt, 'reportValidity');

    target.querySelector('form')!.requestSubmit();
    await settle();

    expect(workspaceApiMocks.submitGenerate).not.toHaveBeenCalled();
    expect(draft.state.sidebarCollapsed).toBe(false);
    expect(report).toHaveBeenCalled();
  });

  it('generates with Ctrl+Enter and re-rolls a locked seed with Ctrl+Shift+Enter', async () => {
    draft.update('prompt', 'a fox');
    await mountWorkspace(makeContext());
    draft.update('seed', 42);
    await settle();

    pressKey('Enter', { ctrlKey: true });
    await settle();
    expect(workspaceApiMocks.submitGenerate.mock.calls.at(-1)?.[0].get('seed')).toBe('42');

    jobStore.clearJob();
    workspaceApiMocks.submitGenerate.mockClear();
    const unmountApp = app!;
    await unmount(unmountApp);
    await mountWorkspace(makeContext());
    draft.update('seed', 42);
    await settle();
    pressKey('Enter', { ctrlKey: true, shiftKey: true });
    await settle();
    const seed = workspaceApiMocks.submitGenerate.mock.calls.at(-1)?.[0].get('seed');
    expect(seed).not.toBe('42');
    expect(Number(seed)).toBeGreaterThanOrEqual(0);
  });

  it('focuses the prompt with Alt+1 and the settings with Alt+2, expanding a collapsed sidebar first', async () => {
    await mountWorkspace(makeContext());
    expect(pressKey('1', { altKey: true, code: 'Digit1' }).defaultPrevented).toBe(true);
    expect(document.activeElement?.id).toBe('ws-prompt');
    pressKey('2', { altKey: true, code: 'Digit2' });
    expect(document.activeElement?.closest('.settings-pane')).not.toBeNull();

    draft.update('sidebarCollapsed', true);
    pressKey('1', { altKey: true, code: 'Digit1' });
    await settle();
    expect(draft.state.sidebarCollapsed).toBe(false);
    expect(document.activeElement?.id).toBe('ws-prompt');

    draft.update('sidebarCollapsed', true);
    pressKey('2', { altKey: true, code: 'Digit2' });
    await settle();
    expect(draft.state.sidebarCollapsed).toBe(false);
    expect(document.activeElement?.closest('.settings-pane')).not.toBeNull();
  });

  it('toggles the sidebar with Alt+Shift+1 or Alt+Shift+2, focusing the pane when it opens', async () => {
    await mountWorkspace(makeContext());
    expect(pressKey('!', { altKey: true, shiftKey: true, code: 'Digit1' }).defaultPrevented).toBe(true);
    expect(draft.state.sidebarCollapsed).toBe(true);
    pressKey('!', { altKey: true, shiftKey: true, code: 'Digit1' });
    await settle();
    expect(draft.state.sidebarCollapsed).toBe(false);
    expect(document.activeElement?.id).toBe('ws-prompt');

    pressKey('@', { altKey: true, shiftKey: true, code: 'Digit2' });
    expect(draft.state.sidebarCollapsed).toBe(true);
    pressKey('@', { altKey: true, shiftKey: true, code: 'Digit2' });
    await settle();
    expect(draft.state.sidebarCollapsed).toBe(false);
    expect(document.activeElement?.closest('.settings-pane')).not.toBeNull();
    expect(draft.state.historyCollapsed).toBe(false);
  });

  function queueContext(queued: JobSnapshot[]): WorkspaceContext {
    workspaceApiMocks.listJobs.mockResolvedValue({ active_job: jobSnap('job-live', 'running'), queued_jobs: queued });
    return makeContext({ active_job: jobSnap('job-live', 'running'), queued_jobs: queued });
  }

  describe('job queue', () => {
    it('keeps the form usable while a job runs and adds new runs to the queue', async () => {
      draft.update('prompt', 'Another run');
      await mountWorkspace(queueContext([]));
      const submit = target.querySelector('#ws-submit') as HTMLButtonElement;
      expect(submit.disabled).toBe(false);
      expect(submit.textContent).toContain('Add to queue');
      expect(target.querySelector('#ws-busy-note')?.textContent).toContain('New runs join the queue');
      expect((target.querySelector('#ws-steps') as HTMLInputElement).disabled).toBe(false);

      const queued = jobSnap('job-2', 'queued', { queue_position: 1, prompt: 'Another run', meta: '2:3 · m · 8 steps' });
      workspaceApiMocks.submitGenerate.mockResolvedValue({ job_id: 'job-2', workflow: 'txt2img', prompt: 'Another run', model: 'zit', runs: 1, created_at: '', queue_position: 1 });
      workspaceApiMocks.listJobs.mockResolvedValue({ active_job: jobSnap('job-live', 'running'), queued_jobs: [queued] });
      target.querySelector('form')!.requestSubmit();
      await settle();

      expect(jobStore.current?.job_id).toBe('job-live');
      expect((target.querySelector('svg.mascot') as SVGSVGElement).dataset.mood).toBe('nodding');
      const panel = target.querySelector('[data-testid="queue-panel"]') as HTMLElement;
      expect(panel.textContent).toContain('Up next · 1');
      expect(panel.textContent).toContain('Another run');
      expect(panel.textContent).toContain('Starts next');
      expect(panel.textContent).toContain('2:3 · m · 8 steps');
    });

    it('puts the mascot on the job card, clear of the live preview', async () => {
      await mountWorkspace(queueContext([]));
      const view = target.querySelector('[data-testid="job-view"]') as HTMLElement;
      expect(view.querySelector('.job-view-card svg.mascot')).not.toBeNull();
      expect(view.querySelector('.job-view-preview svg.mascot')).toBeNull();
      expect(mascotHome.placement.away).toBe(true);
    });

    it('submits once for a held Cmd+Enter', async () => {
      draft.update('prompt', 'Once');
      await mountWorkspace(makeContext());
      pressKey('Enter', { metaKey: true, repeat: true });
      await settle();
      expect(workspaceApiMocks.submitGenerate).not.toHaveBeenCalled();
      pressKey('Enter', { metaKey: true });
      await settle();
      expect(workspaceApiMocks.submitGenerate).toHaveBeenCalledTimes(1);
    });

    it('disables manual enhancement while jobs are active', async () => {
      draft.update('prompt', 'a fox');
      await mountWorkspace(withEnhancer(queueContext([])));
      await openEnhanceRun();
      expect((document.querySelector('#ws-enhance-run') as HTMLButtonElement).disabled).toBe(true);
      expect(document.querySelector('#ws-enhance-panel')?.textContent).toContain('Available when all jobs have finished.');
      pressKey('e', { ctrlKey: true }, target.querySelector('#ws-prompt') as HTMLTextAreaElement);
      await settle();
      expect(enhanceApiMocks.enhancePrompt).not.toHaveBeenCalled();
    });

    it('removes a queued job without stopping the running one', async () => {
      await mountWorkspace(queueContext([jobSnap('job-2', 'queued', { queue_position: 1 })]));
      workspaceApiMocks.listJobs.mockResolvedValue({ active_job: jobSnap('job-live', 'running'), queued_jobs: [] });

      (target.querySelector('[data-job-id="job-2"] button[aria-label="Remove from queue"]') as HTMLButtonElement).click();
      await settle();

      expect(workspaceApiMocks.cancelJob).toHaveBeenCalledWith('job-2');
      expect(target.querySelector('[data-testid="queue-panel"]')).toBeNull();
      expect(jobStore.current?.job_id).toBe('job-live');
      expect(jobStore.current?.status).toBe('running');
    });

    it('asks before clearing the queue and then moves focus to Generate', async () => {
      await mountWorkspace(queueContext([jobSnap('job-2', 'queued', { queue_position: 1 }), jobSnap('job-3', 'queued', { queue_position: 2 })]));
      const clear = () => (Array.from(target.querySelectorAll('button')).find((el) => el.textContent?.trim() === 'Clear queue') as HTMLButtonElement).click();

      clear();
      await settle();
      expect(document.querySelector('[role="alertdialog"]')?.textContent).toContain('Remove 2 queued jobs?');
      await answerConfirm(false);
      expect(workspaceApiMocks.clearQueue).not.toHaveBeenCalled();

      workspaceApiMocks.listJobs.mockResolvedValue({ active_job: jobSnap('job-live', 'running'), queued_jobs: [] });
      clear();
      await settle();
      await answerConfirm(true);
      await settle();

      expect(workspaceApiMocks.clearQueue).toHaveBeenCalledTimes(1);
      expect(target.querySelector('[data-testid="queue-panel"]')).toBeNull();
      expect(document.activeElement?.id).toBe('ws-submit');
      expect(jobStore.current?.status).toBe('running');
    });

    it('loads a queued job\'s settings into the form', async () => {
      const settings = { mode: 'image', workflow: 'txt2img', model: 'zit', prompt_source: 'inline', prompt: 'from the queue', steps: '7', runs: '3', seed: '99' };
      await mountWorkspace(queueContext([jobSnap('job-2', 'queued', { queue_position: 1, settings })]));

      (target.querySelector('[data-job-id="job-2"] button[title="Copy this job\'s settings into the form"]') as HTMLButtonElement).click();
      await settle();

      expect((target.querySelector('#ws-prompt') as HTMLTextAreaElement).value).toBe('from the queue');
      expect(draft.state).toMatchObject({ steps: 7, runs: 3, seed: 99 });
      const submitted = new FormData(target.querySelector('form')!);
      expect(submitted.get('prompt')).toBe('from the queue');
      expect(submitted.get('steps')).toBe('7');
    });

    describe('settings handed over by the Job log', () => {
      const logged = { mode: 'image', workflow: 'txt2img', model: 'zit', prompt_source: 'inline', prompt: 'from the job log', steps: '7', runs: '3', seed: '99' };
      const prompt = () => (target.querySelector('#ws-prompt') as HTMLTextAreaElement).value;

      it('applies settings requested before the page mounted once its context has loaded', async () => {
        draft.update('prompt', 'a saved draft');
        requestJobSettings(logged);
        await mountWorkspace(makeContext());

        expect(prompt()).toBe('from the job log');
        expect(draft.state).toMatchObject({ steps: 7, runs: 3, seed: 99 });
        expect(hasPendingJobSettings()).toBe(false);
        expect(toasts.map((toast) => toast.message)).toContain('Loaded the job\'s settings');
      });

      it('applies settings requested while the page is already mounted', async () => {
        await mountWorkspace(makeContext());
        expect(prompt()).not.toBe('from the job log');

        requestJobSettings(logged);
        await settle();

        expect(prompt()).toBe('from the job log');
        expect(draft.state).toMatchObject({ steps: 7, runs: 3, seed: 99 });
      });

      it('applies a request only once', async () => {
        requestJobSettings(logged);
        await mountWorkspace(makeContext());
        draft.update('prompt', 'typed afterwards');
        await settle();

        expect(prompt()).toBe('typed afterwards');
        expect(toasts.filter((toast) => toast.message === 'Loaded the job\'s settings')).toHaveLength(1);
      });
    });

    it('shows the queue count on the collapsed strip', async () => {
      draft.update('sidebarCollapsed', true);
      await mountWorkspace(queueContext([jobSnap('job-2', 'queued', { queue_position: 1 })]));
      expect(target.querySelector('[data-testid="strip-queue-count"]')?.textContent).toBe('1');
      expect(target.querySelector('[data-testid="sidebar-strip"] button[type="submit"]')?.getAttribute('aria-label')).toBe('Add to queue');
    });
  });

  describe('review before upscaling', () => {
    function reviewSnapshot(extra: Partial<JobSnapshot> = {}): JobSnapshot {
      return jobSnap('job-review', 'paused', {
        paused: true,
        review: { iteration: 3 },
        preview_version: 4,
        runs: 8,
        supported_controls: ['next', 'pause', 'resume', 'repeat', 'approve', 'discard', 'quit'],
        settings: { upscale: '2' },
        last_event: { type: 'job_paused', event_id: 9, review: { iteration: 3 }, preview_version: 4 },
        ...extra,
      });
    }

    function withoutReviewControl(): WorkspaceContext {
      const context = makeContext();
      for (const definition of Object.values(context.workflow_contract.definitions)) {
        definition.visible_controls = definition.visible_controls.filter((id) => id !== 'image_upscale_review');
      }
      return context;
    }

    const reviewToggle = () => target.querySelector<HTMLInputElement>('#ws-upscale-review');
    const summary = () => target.querySelector('[data-section="upscale"] .inspector-summary')?.textContent ?? '';

    it('offers the Review toggle only while upscale is on, and posts the review field only when it is on', async () => {
      await mountWorkspace(makeContext());
      expect(reviewToggle()).toBeNull();

      draft.update('upscaleEnabled', true);
      await settle();
      expect(reviewToggle()).not.toBeNull();
      expect(summary()).toBe('2×');
      expect(new FormData(target.querySelector('form')!).has('review')).toBe(false);

      draft.update('upscaleReview', true);
      await settle();
      expect(summary()).toBe('2× · Review');
      expect(new FormData(target.querySelector('form')!).get('review')).toBe('true');

      draft.update('upscaleEnabled', false);
      await settle();
      expect(reviewToggle()).toBeNull();
      expect(new FormData(target.querySelector('form')!).has('review')).toBe(false);
    });

    it('keeps the Review choice with the other draft settings', async () => {
      await mountWorkspace(makeContext());
      draft.update('upscaleEnabled', true);
      await settle();
      reviewToggle()!.click();
      await settle();
      expect(draft.state.upscaleReview).toBe(true);
      expect(JSON.parse(localStorage.getItem('ziv-workspace-draft-v1') ?? '{}').upscaleReview).toBe(true);
    });

    it('hides the Review toggle when the workflow contract does not list it', async () => {
      draft.update('upscaleEnabled', true);
      await mountWorkspace(withoutReviewControl());
      expect(target.querySelector('#ws-upscale-enabled')).not.toBeNull();
      expect(reviewToggle()).toBeNull();
    });

    it('shows the review over the preview area, leaving the job card and the rest of the page in place', async () => {
      workspaceApiMocks.listJobs.mockResolvedValue({ active_job: reviewSnapshot(), queued_jobs: [] });
      await mountWorkspace(makeContext({ active_job: reviewSnapshot() }));

      const preview = target.querySelector<HTMLElement>('.workspace-preview')!;
      const overlay = target.querySelector<HTMLElement>('[data-testid="review-overlay"]')!;
      expect(preview.contains(overlay)).toBe(true);
      const dialog = overlay.querySelector('[role="dialog"]')!;
      expect(dialog.getAttribute('aria-label')).toBe('Review');
      expect(overlay.querySelector('img')?.getAttribute('src')).toBe('/jobs/job-review/preview?v=4');
      expect(overlay.querySelector('[data-decision="approve"]')).not.toBeNull();
      expect(overlay.querySelector('[data-decision="discard"]')).not.toBeNull();

      // The running-job view is still there under the overlay, with only Cancel.
      const card = preview.querySelector<HTMLElement>('.job-card')!;
      expect(card.contains(overlay)).toBe(false);
      expect(card.querySelector('button[data-control="quit"]')).not.toBeNull();
      for (const control of ['pause-resume', 'next', 'repeat']) expect(card.querySelector(`button[data-control="${control}"]`)).toBeNull();
    });

    it.each([
      ['y', 'approve'],
      ['n', 'discard'],
    ])('sends %s as %s', async (key, action) => {
      workspaceApiMocks.listJobs.mockResolvedValue({ active_job: reviewSnapshot(), queued_jobs: [] });
      const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => ({}) });
      vi.stubGlobal('fetch', fetchMock);
      try {
        await mountWorkspace(makeContext({ active_job: reviewSnapshot() }));
        pressKey(key);
        await settle();
        expect(fetchMock).toHaveBeenCalledWith(`/jobs/job-review/controls/${action}`, expect.objectContaining({ method: 'POST' }));
      } finally {
        vi.unstubAllGlobals();
      }
    });

    it('returns to the job card once the review is over', async () => {
      workspaceApiMocks.listJobs.mockResolvedValue({ active_job: reviewSnapshot(), queued_jobs: [] });
      await mountWorkspace(makeContext({ active_job: reviewSnapshot() }));
      expect(target.querySelector('[data-testid="review-overlay"]')).not.toBeNull();

      const source = lastEventSource();
      source.emit('job_resumed', { type: 'job_resumed', event_id: 10 });
      await settle();
      expect(target.querySelector('[data-testid="review-overlay"]')).toBeNull();
      expect(target.querySelector('.job-card')).not.toBeNull();
    });

    it('does not post review when the draft holds it but the contract no longer lists the control', async () => {
      await mountWorkspace(withoutReviewControl());
      draft.update('upscaleEnabled', true);
      draft.update('upscaleReview', true);
      await settle();
      expect(target.querySelector('#ws-upscale-enabled')).not.toBeNull();
      expect(new FormData(target.querySelector('form')!).has('review')).toBe(false);
      expect(summary()).toBe('2×');
    });

    it('does not decide anything with Y or N for a plain pause or a job that runs without a review', async () => {
      const controls = ['next', 'pause', 'resume', 'repeat', 'approve', 'discard', 'quit'];
      const plain = jobSnap('job-plain', 'paused', { paused: true, review: null, supported_controls: controls, last_event: { type: 'job_paused', event_id: 4 } });
      const running = jobSnap('job-run', 'running', { supported_controls: controls, last_event: { type: 'step_progress', event_id: 4 } });
      const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => ({}) });
      vi.stubGlobal('fetch', fetchMock);
      try {
        for (const job of [plain, running]) {
          workspaceApiMocks.listJobs.mockResolvedValue({ active_job: job, queued_jobs: [] });
          await mountWorkspace(makeContext({ active_job: job }));
          expect(target.querySelector('[data-testid="review-overlay"]')).toBeNull();
          expect(target.querySelector('[data-decision]')).toBeNull();
          for (const key of ['y', 'n']) pressKey(key);
          await settle();
          await unmount(app!);
          app = null;
          jobStore.clearJob();
        }
        const decisions = fetchMock.mock.calls.filter(([url]) => /\/controls\/(approve|discard)$/.test(String(url)));
        expect(decisions).toHaveLength(0);
      } finally {
        vi.unstubAllGlobals();
      }
    });

    describe('for a video', () => {
      const videoToggle = () => target.querySelector<HTMLInputElement>('#ws-video-upscale-review');
      const videoSummary = () => target.querySelector('[data-section="video"] .inspector-summary')?.textContent ?? '';

      function withoutVideoReviewControl(): WorkspaceContext {
        const context = makeContext();
        for (const definition of Object.values(context.workflow_contract.definitions)) {
          definition.visible_controls = definition.visible_controls.filter((id) => id !== 'video_upscale_review');
        }
        return context;
      }

      function videoReviewSnapshot(extra: Partial<JobSnapshot> = {}): JobSnapshot {
        return jobSnap('job-vreview', 'paused', {
          workflow: 'txt2vid', job_type: 'Text to Video', model: 'ltx-8', paused: true, review: { iteration: 2 }, runs: 4,
          supported_controls: ['approve', 'discard', 'quit'], settings: { upscale: '2', audio: ['on', 'false'] },
          last_event: { type: 'job_paused', event_id: 9, review: { iteration: 2 } },
          ...extra,
        });
      }

      beforeEach(() => {
        draft.update('workflow', 'txt2vid');
        draft.update('model', 'ltx-8');
      });

      it('offers the Review toggle under the factor only while upscale is on, and posts the review field only when it is on', async () => {
        await mountWorkspace(makeContext());
        await settle();
        expect(videoToggle()).toBeNull();

        draft.update('videoUpscaleEnabled', true);
        await settle();
        expect(videoToggle()).not.toBeNull();
        expect(new FormData(target.querySelector('form')!).has('review')).toBe(false);

        videoToggle()!.click();
        await settle();
        expect(draft.state.videoUpscaleReview).toBe(true);
        expect(new FormData(target.querySelector('form')!).get('review')).toBe('true');
        expect(videoSummary()).toContain('Review');

        draft.update('videoUpscaleEnabled', false);
        await settle();
        expect(videoToggle()).toBeNull();
        expect(new FormData(target.querySelector('form')!).has('review')).toBe(false);
        expect(videoSummary()).not.toContain('Review');
      });

      it('hides the Review toggle and posts no review when the contract does not list it', async () => {
        draft.update('videoUpscaleEnabled', true);
        draft.update('videoUpscaleReview', true);
        await mountWorkspace(withoutVideoReviewControl());
        await settle();
        expect(target.querySelector('#ws-video-upscale')).not.toBeNull();
        expect(videoToggle()).toBeNull();
        expect(new FormData(target.querySelector('form')!).has('review')).toBe(false);
        expect(videoSummary()).not.toContain('Review');
      });

      it('leaves a kept upscale out of the summary when the workflow offers no video upscale', async () => {
        const context = makeContext();
        for (const definition of Object.values(context.workflow_contract.definitions)) {
          definition.visible_controls = definition.visible_controls.filter((id) => id !== 'video_upscale_enabled');
        }
        draft.update('videoUpscaleEnabled', true);
        await mountWorkspace(context);
        await settle();
        expect(target.querySelector('#ws-video-upscale')).toBeNull();
        expect(videoSummary()).not.toContain('upscale');
      });

      it('keeps the image Review choice apart from the video one', async () => {
        draft.update('upscaleReview', true);
        draft.update('videoUpscaleEnabled', true);
        await mountWorkspace(makeContext());
        await settle();
        expect(videoToggle()!.checked).toBe(false);
        expect(new FormData(target.querySelector('form')!).has('review')).toBe(false);
      });

      it('shows the paused video in the overlay, with Approve and Discard and only Stop on the job card under it', async () => {
        workspaceApiMocks.listJobs.mockResolvedValue({ active_job: videoReviewSnapshot(), queued_jobs: [] });
        await mountWorkspace(makeContext({ active_job: videoReviewSnapshot() }));

        const region = target.querySelector<HTMLElement>('[data-testid="review-overlay"]')!;
        const video = region.querySelector('video')!;
        expect(video.getAttribute('src')).toBe('/jobs/job-vreview/review?v=2-1');
        expect(video.autoplay).toBe(false);
        expect(video.controls).toBe(true);
        expect(video.muted).toBe(false);
        expect(region.querySelector('.review-line')?.textContent).toContain('Review 2 of 4');
        expect(region.querySelector('[data-decision="approve"]')).not.toBeNull();
        expect(region.querySelector('[data-decision="discard"]')).not.toBeNull();
        const card = target.querySelector<HTMLElement>('.workspace-preview .job-card')!;
        expect(region.contains(card)).toBe(false);
        expect(card.querySelector('button[data-control="quit"]')).not.toBeNull();
        for (const control of ['pause-resume', 'next', 'repeat']) expect(card.querySelector(`button[data-control="${control}"]`)).toBeNull();
      });
    });
  });

  it('enhances the prompt with Ctrl+E, even from inside the prompt field', async () => {
    enhanceApiMocks.enhancePrompt.mockResolvedValue();
    draft.update('prompt', 'a fox');
    await mountWorkspace(withEnhancer(makeContext()));
    const prompt = target.querySelector('#ws-prompt') as HTMLTextAreaElement;
    expect(pressKey('e', { ctrlKey: true }, prompt).defaultPrevented).toBe(true);
    await settle();
    expect(enhanceApiMocks.enhancePrompt).toHaveBeenCalledTimes(1);
    expect((enhanceApiMocks.enhancePrompt.mock.calls[0]?.[0] as { prompt: string }).prompt).toBe('a fox');
  });

  it('keeps each rewrite of the same text in the draft history, with how it was written', async () => {
    const rewrites = ['A fox in snow.', 'A fox at dusk.'];
    enhanceApiMocks.enhancePrompt.mockImplementation(async (_body: unknown, ...rest: unknown[]) => {
      (rest[0] as (frame: EnhanceFrame) => void)({ type: 'done', prompt: rewrites.shift()!, clamped: false, model: 'org/fast@r1' });
    });
    draft.update('prompt', 'a fox');
    await mountWorkspace(withEnhancer(makeContext()));

    (await openEnhanceRun()).click();
    await settle();
    // The popover may stay open between runs; open it only when it is closed.
    const again = document.querySelector<HTMLButtonElement>('#ws-enhance-run') ?? (await openEnhanceRun());
    again.click();
    await settle();

    expect(draft.state.enhancedHistory.map((entry) => entry.prompt)).toEqual(['A fox in snow.', 'A fox at dusk.']);
    expect(draft.state.enhancedHistory.map((entry) => entry.from.record?.model)).toEqual(['org/fast@r1', 'org/fast@r1']);
    expect(draft.state.enhancedPrompt).toBe('A fox at dusk.');

    draft.update('prompt', 'a cat');
    rewrites.push('A cat on a sill.');
    const third = document.querySelector<HTMLButtonElement>('#ws-enhance-run') ?? (await openEnhanceRun());
    third.click();
    await settle();
    expect(draft.state.enhancedHistory.map((entry) => entry.prompt)).toEqual(['A cat on a sill.']);
  });

  describe('recent prompts', () => {
    const NOW_S = Date.now() / 1000;
    const recent = (overrides: Partial<RecentPrompt> = {}): RecentPrompt => ({
      format: 'text', prompt: 'A lighthouse at dusk', scene: '', negative_prompt: '', mode: 'image', used_at: NOW_S - 120, ...overrides,
    });
    const rows = () => Array.from(document.querySelectorAll<HTMLElement>('[role="option"]'));
    const listState = () => document.querySelector<HTMLElement>('[role="listbox"][aria-label="Recent prompts"]')?.dataset.state;
    const filterInput = () => document.querySelector('input[aria-label="Filter recent prompts"]') as HTMLInputElement;
    const clearButton = () => Array.from(document.querySelectorAll('button')).find((el) => el.textContent?.trim() === 'Clear recents') as HTMLButtonElement | undefined;
    const historyButton = () => target.querySelector('#ws-recent-prompts') as HTMLButtonElement;
    async function openRecents(): Promise<void> {
      historyButton().click();
      await settle();
    }
    function ideoContext(): WorkspaceContext {
      const defaults = makeIdeogramDefaults();
      return makeContext({ image_models: [{ id: 'ideo', label: 'ideo', type: 'image' }], defaults, current_image_model: 'ideo', image_model_defaults: { ideo: defaults } });
    }

    it('lists the recent prompts newest first, newest selected, and asks the server for the whole list of the current mode', async () => {
      recentPromptsApiMocks.getRecentPrompts.mockResolvedValue({ entries: [recent({ prompt: 'A lighthouse at dusk' }), recent({ prompt: 'A red fox', used_at: NOW_S - 7200 })] });
      await mountWorkspace(makeContext());
      await openRecents();

      expect(recentPromptsApiMocks.getRecentPrompts).toHaveBeenCalledWith('', 'image');
      expect(rows().map((row) => row.textContent)).toEqual([expect.stringContaining('A lighthouse at dusk'), expect.stringContaining('A red fox')]);
      expect(rows().map((row) => row.getAttribute('aria-selected'))).toEqual(['true', 'false']);
      expect(historyButton().getAttribute('aria-expanded')).toBe('true');
    });

    it('exposes the whole prompt of a row on hover, with the scene and the negative prompt on lines of their own', async () => {
      const long = 'A lighthouse on a cliff at dusk, waves breaking below, gulls circling the lamp, warm light on wet rock, a small boat far out at sea, long exposure.';
      recentPromptsApiMocks.getRecentPrompts.mockResolvedValue({ entries: [
        recent({ prompt: long, negative_prompt: 'blurry' }),
        recent({ mode: 'video', scene: 'A night tram.', prompt: 'She turns to the window.' }),
      ] });
      await mountWorkspace(makeContext());
      await openRecents();

      expect(rows()[0].title).toBe(`${long}\nNegative: blurry`);
      expect(rows()[1].title).toBe('She turns to the window.\nScene: A night tram.');
    });

    it('shows the negative prompt of a row, fills the prompt and the negative prompt on a pick, closes, and drops the Enhanced pair', async () => {
      recentPromptsApiMocks.getRecentPrompts.mockResolvedValue({ entries: [recent({ prompt: 'A red fox', negative_prompt: 'blurry' })] });
      draft.patch({ prompt: 'a cat', negativePrompt: '', enhancedPrompt: 'A cat, enhanced.', enhancedFrom: { prompt: 'a cat', scene: '', mode: 'image' } });
      await mountWorkspace(withEnhancer(makeContext()));
      (target.querySelector('#ws-tab-enhanced') as HTMLButtonElement).click();
      await settle();
      await openRecents();
      expect(rows()[0].textContent).toContain('blurry');

      rows()[0].click();
      await settle();

      expect(draft.state).toMatchObject({ prompt: 'A red fox', negativePrompt: 'blurry', enhancedPrompt: '', enhancedScene: '', enhancedFrom: null, jsonPromptEnabled: false });
      expect(document.querySelector('[role="dialog"][aria-label="Recent prompts"]')).toBeNull();
      expect(target.querySelector('#ws-tab-prompt')?.getAttribute('aria-selected')).toBe('true');
      expect(field('ws-prompt').value).toBe('A red fox');
      const submitted = await submitForm(target);
      expect(submitted.get('prompt')).toBe('A red fox');
      expect(submitted.get('negative_prompt')).toBe('blurry');
    });

    it('fills the scene and the action of a video prompt', async () => {
      recentPromptsApiMocks.getRecentPrompts.mockResolvedValue({ entries: [recent({ mode: 'video', scene: 'A night tram.', prompt: 'She turns to the window.' })] });
      draft.update('workflow', 'txt2vid');
      await mountWorkspace(withVideoEnhancer(makeContext()));
      await openRecents();
      expect(recentPromptsApiMocks.getRecentPrompts).toHaveBeenCalledWith('', 'video');
      expect(rows()[0].textContent).toContain('A night tram.');

      rows()[0].click();
      await settle();

      expect(draft.state).toMatchObject({ scene: 'A night tram.', prompt: 'She turns to the window.' });
    });

    it('turns the JSON box on and fills it when a JSON caption is picked on a model that takes one', async () => {
      const caption = '{"high_level_description": "A poster of a lighthouse"}';
      recentPromptsApiMocks.getRecentPrompts.mockResolvedValue({ entries: [recent({ format: 'json', prompt: caption })] });
      draft.update('prompt', 'kept');
      await mountWorkspace(ideoContext());
      await openRecents();
      expect(rows()[0].getAttribute('aria-disabled')).toBeNull();

      rows()[0].click();
      await settle();

      expect(draft.state).toMatchObject({ jsonPromptEnabled: true, jsonPrompt: caption, prompt: 'kept' });
      expect((target.querySelector('#ws-json-prompt') as HTMLTextAreaElement).value).toBe(caption);
      const submitted = await submitForm(target);
      expect(submitted.get('json_prompt')).toBe(caption);
    });

    it('shows a JSON caption as disabled, with the reason, on a model without JSON captions', async () => {
      recentPromptsApiMocks.getRecentPrompts.mockResolvedValue({ entries: [recent({ format: 'json', prompt: '{"a": 1}' })] });
      draft.update('prompt', 'kept');
      await mountWorkspace(makeContext());
      await openRecents();

      expect(rows()[0].getAttribute('aria-disabled')).toBe('true');
      rows()[0].click();
      await settle();
      expect(draft.state.prompt).toBe('kept');
      expect(draft.state.jsonPromptEnabled).toBe(false);
    });

    it('asks the server again, after a pause, with what is typed in the filter', async () => {
      fakeDebounceTimers();
      recentPromptsApiMocks.getRecentPrompts.mockImplementation(async (query) => ({ entries: query === 'fox' ? [recent({ prompt: 'A red fox' })] : [recent(), recent({ prompt: 'A red fox' })] }));
      await mountWorkspace(makeContext());
      await openRecents();
      expect(rows()).toHaveLength(2);

      filterInput().value = 'fox';
      filterInput().dispatchEvent(new Event('input', { bubbles: true }));
      await settle();
      expect(recentPromptsApiMocks.getRecentPrompts).toHaveBeenCalledTimes(1);
      await waitDebounce();

      expect(recentPromptsApiMocks.getRecentPrompts).toHaveBeenLastCalledWith('fox', 'image');
      expect(rows().map((row) => row.textContent)).toEqual([expect.stringContaining('A red fox')]);
    });

    it('picks from the filtered rows when Enter follows the typing before the result is in', async () => {
      fakeDebounceTimers();
      recentPromptsApiMocks.getRecentPrompts.mockImplementation(async (query) => ({ entries: query === 'fox' ? [recent({ prompt: 'A red fox' })] : [recent(), recent({ prompt: 'A red fox' })] }));
      await mountWorkspace(makeContext());
      await openRecents();

      filterInput().value = 'fox';
      filterInput().dispatchEvent(new Event('input', { bubbles: true }));
      pressKey('Enter', {}, filterInput());
      await settle();

      expect(draft.state.prompt).toBe('A red fox');
      expect(document.querySelector('[role="dialog"][aria-label="Recent prompts"]')).toBeNull();
    });

    it('picks from the newest filter when an older answer lands while the newer one is still being typed', async () => {
      fakeDebounceTimers();
      const answers = new Map<string, (value: { entries: RecentPrompt[] }) => void>();
      recentPromptsApiMocks.getRecentPrompts.mockImplementation((query: string) => (query === '' ? Promise.resolve({ entries: [recent({ prompt: 'A lighthouse' })] }) : new Promise((resolve) => { answers.set(query, resolve); })));
      await mountWorkspace(makeContext());
      await openRecents();

      const typeFilter = (value: string) => {
        filterInput().value = value;
        filterInput().dispatchEvent(new Event('input', { bubbles: true }));
      };
      typeFilter('fo');
      await waitDebounce();
      // The fetch for "fo" is out when the next keystroke starts the debounce for "fox".
      typeFilter('fox');
      answers.get('fo')!({ entries: [recent({ prompt: 'A forest' })] });
      await settle();
      pressKey('Enter', {}, filterInput());
      await settle();
      expect(draft.state.prompt).toBe('');
      answers.get('fox')!({ entries: [recent({ prompt: 'A red fox' }), recent({ prompt: 'Another fox' })] });
      await settle();

      expect(draft.state.prompt).toBe('A red fox');
    });

    it('keeps the touch keyboard down when the menu opens from its button on a touch screen, but not after Alt+↑', async () => {
      layoutProbe.coarse = true;
      await mountWorkspace(makeContext());
      await openRecents();
      expect(filterInput()).not.toBe(document.activeElement);
      pressKey('Escape');
      await settle();

      pressKey('ArrowUp', { altKey: true }, field('ws-prompt'));
      await settle();
      expect(filterInput()).toBe(document.activeElement);
    });

    it('says so when nothing is recorded, when nothing matches, and when the list cannot be read', async () => {
      fakeDebounceTimers();
      await mountWorkspace(makeContext());
      await openRecents();
      expect(listState()).toBe('empty');
      // Nothing to clear.
      expect(clearButton()!.disabled).toBe(true);

      recentPromptsApiMocks.getRecentPrompts.mockImplementation(async (query) => ({ entries: query ? [] : [recent()] }));
      pressKey('Escape');
      await settle();
      await openRecents();
      filterInput().value = 'zebra';
      filterInput().dispatchEvent(new Event('input', { bubbles: true }));
      await waitDebounce();
      expect(listState()).toBe('nomatch');
      expect(clearButton()!.disabled).toBe(false);

      recentPromptsApiMocks.getRecentPrompts.mockRejectedValueOnce(new ApiError('Request failed', 500, 'Could not read or write the recent prompts: database is locked'));
      filterInput().value = 'fox';
      filterInput().dispatchEvent(new Event('input', { bubbles: true }));
      await waitDebounce();
      expect(listState()).toBe('error');
      expect(document.querySelector('[role="alert"]')?.textContent).toContain('database is locked');
      expect(clearButton()!.disabled).toBe(true);

      const tryAgain = Array.from(document.querySelectorAll('button')).find((el) => el.textContent?.trim() === 'Try again') as HTMLButtonElement;
      tryAgain.click();
      await settle();
      expect(listState()).toBe('nomatch');
    });

    it('moves over the enabled rows with the arrows and picks with Enter', async () => {
      recentPromptsApiMocks.getRecentPrompts.mockResolvedValue({ entries: [
        recent({ prompt: 'newest' }), recent({ format: 'json', prompt: '{"a": 1}' }), recent({ prompt: 'third' }),
      ] });
      await mountWorkspace(makeContext());
      await openRecents();
      expect(filterInput()).toBe(document.activeElement);

      pressKey('ArrowDown', {}, filterInput());
      await settle();
      // The JSON row is disabled for this model, so the selection skips it.
      expect(rows().map((row) => row.getAttribute('aria-selected'))).toEqual(['false', 'false', 'true']);
      pressKey('ArrowUp', {}, filterInput());
      pressKey('Enter', {}, filterInput());
      await settle();

      expect(draft.state.prompt).toBe('newest');
    });

    it('asks before clearing the recent prompts of this mode, then deletes them and says so', async () => {
      recentPromptsApiMocks.getRecentPrompts.mockResolvedValue({ entries: [recent()] });
      recentPromptsApiMocks.clearRecentPrompts.mockResolvedValue({ deleted_count: 1 });
      await mountWorkspace(makeContext());
      await openRecents();

      clearButton()!.click();
      await settle();
      await answerConfirm(false);
      expect(recentPromptsApiMocks.clearRecentPrompts).not.toHaveBeenCalled();

      await openRecents();
      clearButton()!.click();
      await settle();
      await answerConfirm(true);
      expect(recentPromptsApiMocks.clearRecentPrompts).toHaveBeenCalledExactlyOnceWith('image');
      expect(toasts.map((toast) => toast.type)).toContain('success');
    });

    describe('Alt+↑', () => {
      it('opens the menu from an empty prompt, with the filter focused and Enter bringing back the newest', async () => {
        recentPromptsApiMocks.getRecentPrompts.mockResolvedValue({ entries: [recent({ prompt: 'The last one' })] });
        await mountWorkspace(makeContext());

        const event = pressKey('ArrowUp', { altKey: true }, field('ws-prompt'));
        await settle();
        expect(event.defaultPrevented).toBe(true);
        expect(filterInput()).toBe(document.activeElement);

        pressKey('Enter', {}, filterInput());
        await settle();
        expect(field('ws-prompt').value).toBe('The last one');
      });

      it('leaves the key to the browser while the prompt has text', async () => {
        draft.update('prompt', 'a fox');
        await mountWorkspace(makeContext());
        expect(pressKey('ArrowUp', { altKey: true }, field('ws-prompt')).defaultPrevented).toBe(false);
        await settle();
        expect(document.querySelector('[role="dialog"][aria-label="Recent prompts"]')).toBeNull();
      });

      it('opens from the Action of a video only while the Scene is empty too', async () => {
        draft.update('workflow', 'txt2vid');
        draft.update('scene', 'A night tram.');
        await mountWorkspace(withVideoEnhancer(makeContext()));
        expect(pressKey('ArrowUp', { altKey: true }, field('ws-prompt')).defaultPrevented).toBe(false);
        expect(document.querySelector('[role="dialog"][aria-label="Recent prompts"]')).toBeNull();

        type('ws-scene', '');
        await settle();
        expect(pressKey('ArrowUp', { altKey: true }, field('ws-prompt')).defaultPrevented).toBe(true);
        await settle();
        expect(document.querySelector('[role="dialog"][aria-label="Recent prompts"]')).not.toBeNull();
      });

      it('returns focus to where the menu was opened on Escape, and to the prompt on Tab out of it', async () => {
        await mountWorkspace(makeContext());
        await openRecents();
        pressKey('Escape');
        await settle();
        expect(document.activeElement).toBe(historyButton());

        pressKey('ArrowUp', { altKey: true }, field('ws-prompt'));
        await settle();
        pressKey('Escape');
        await settle();
        expect(document.querySelector('[role="dialog"][aria-label="Recent prompts"]')).toBeNull();
        expect(document.activeElement).toBe(field('ws-prompt'));

        pressKey('ArrowUp', { altKey: true }, field('ws-prompt'));
        await settle();
        // With nothing recorded the filter is the last stop.
        pressKey('Tab', {}, filterInput());
        await settle();
        expect(document.querySelector('[role="dialog"][aria-label="Recent prompts"]')).toBeNull();
        expect(document.activeElement).toBe(field('ws-prompt'));
      });
    });
  });

  describe('status line and rewrite stepper', () => {
    const statusLineEl = () => target.querySelector('[data-testid="prompt-statusline"]') as HTMLElement;
    const lineState = () => statusLineEl().dataset.state;
    const link = (label: string) => Array.from(statusLineEl().querySelectorAll('button')).find((el) => el.textContent?.trim() === label);
    const stepper = () => target.querySelector('[role="group"][aria-label="Rewrites"]');
    const step = (label: string) => stepper()!.querySelector(`button[aria-label="${label}"]`) as HTMLButtonElement;
    const RECORD = { settings: {}, model: 'org/fast@r1', format: 'text' as const };
    const FROM = { prompt: 'a fox', scene: '', mode: 'image' as const };

    /** Two rewrites of "a fox", showing the second; the first was written by another model. */
    function twoRewrites(): void {
      const first = { prompt: 'A fox in snow.', scene: '', from: { ...FROM, record: { ...RECORD, model: 'org/smart@r2' } } };
      const second = { prompt: 'A fox at dusk.', scene: '', from: { ...FROM, record: RECORD } };
      draft.patch({ prompt: 'a fox', enhancedPrompt: second.prompt, enhancedFrom: second.from, enhancedHistory: [first, second] });
    }
    async function showEnhanced(): Promise<void> {
      (target.querySelector('#ws-tab-enhanced') as HTMLButtonElement).click();
      await settle();
    }

    it('says nothing while your own text is sent, and offers Use mine once the Enhanced text is', async () => {
      draft.update('prompt', 'a fox');
      await mountWorkspace(withEnhancer(makeContext()));
      expect(lineState()).toBe('none');
      expect(link('Use mine')).toBeUndefined();

      draft.patch({ enhancedPrompt: 'A red fox in deep snow.', enhancedFrom: FROM });
      await settle();
      expect(lineState()).toBe('used');
      expect(link('Use mine')).toBeDefined();
      expect(target.querySelector('#ws-tab-prompt')?.textContent?.trim()).toBe('Prompt');
      expect(target.querySelector('#ws-tab-enhanced')?.textContent?.trim()).toBe('Enhanced');
    });

    it('drops the Enhanced text with Use mine, so Generate sends the prompt', async () => {
      draft.update('prompt', 'a fox');
      draft.patch({ enhancedPrompt: 'A red fox.', enhancedFrom: FROM });
      await mountWorkspace(withEnhancer(makeContext()));
      await showEnhanced();

      (link('Use mine') as HTMLButtonElement).click();
      await settle();

      expect(draft.state).toMatchObject({ enhancedPrompt: '', enhancedFrom: null });
      expect(target.querySelector('#ws-tab-prompt')?.getAttribute('aria-selected')).toBe('true');
      expect((await submitForm(target)).get('prompt')).toBe('a fox');
    });

    it('offers Re-enhance and Use mine beside Out of date, in the Prompt view too', async () => {
      enhanceApiMocks.enhancePrompt.mockResolvedValue();
      draft.patch({ prompt: 'a cat', enhancedPrompt: 'A red fox.', enhancedFrom: FROM });
      await mountWorkspace(withEnhancer(makeContext()));

      expect(target.querySelector('#ws-tab-prompt')?.getAttribute('aria-selected')).toBe('true');
      expect(lineState()).toBe('stale');
      expect(link('Use mine')).toBeDefined();
      (link('Re-enhance') as HTMLButtonElement).click();
      await settle();
      expect((enhanceApiMocks.enhancePrompt.mock.calls[0]?.[0] as { prompt: string }).prompt).toBe('a cat');
    });

    it('marks the JSON box in the status line', async () => {
      const defaults = makeIdeogramDefaults();
      draft.patch({ jsonPromptEnabled: true, jsonPrompt: '{"a": 1}' });
      await mountWorkspace(makeContext({ image_models: [{ id: 'ideo', label: 'ideo', type: 'image' }], defaults, current_image_model: 'ideo', image_model_defaults: { ideo: defaults } }));
      expect(lineState()).toBe('json');
    });

    it('steps through the rewrites, and Generate sends the one shown with how it was written', async () => {
      twoRewrites();
      await mountWorkspace(withEnhancer(makeContext()));
      expect(stepper()).toBeNull();
      await showEnhanced();

      expect(step('Next rewrite').disabled).toBe(true);
      step('Previous rewrite').click();
      await settle();

      expect(step('Previous rewrite').disabled).toBe(true);
      expect(field('ws-enhanced-prompt').value).toBe('A fox in snow.');
      const submitted = await submitForm(target);
      expect(submitted.get('prompt')).toBe('A fox in snow.');
      expect(JSON.parse(String(submitted.get('enhance_record'))).model).toBe('org/smart@r2');

      step('Next rewrite').click();
      await settle();
      expect(JSON.parse(String((await submitForm(target)).get('enhance_record'))).model).toBe('org/fast@r1');
    });
  });

  describe('Fast/Smart and Text/JSON', () => {
    function ideoEnhancerContext(smart: PromptEnhancerContract['smart'] = null): WorkspaceContext {
      const ideogramDefaults = makeIdeogramDefaults();
      const context = withEnhancer(makeContext({
        image_models: [{ id: 'ideo', label: 'ideo', type: 'image' }],
        defaults: ideogramDefaults,
        current_image_model: 'ideo',
        image_model_defaults: { ideo: ideogramDefaults },
      }));
      context.prompt_enhancer = { ...context.prompt_enhancer!, smart };
      return context;
    }

    it('writes a JSON caption into the Enhanced tab and generates it as the JSON prompt', async () => {
      enhanceApiMocks.enhancePrompt.mockImplementation(async (_body: unknown, ...rest: unknown[]) => {
        const onFrame = rest[0] as (frame: EnhanceFrame) => void;
        onFrame({ type: 'done', prompt: '{"high_level_description":"A fox."}', clamped: false, model: 'org/fast@r1' });
      });
      draft.update('prompt', 'a fox');
      draft.update('enhanceFormat', 'json');
      await mountWorkspace(ideoEnhancerContext());
      const run = await openEnhanceRun();
      expect(document.querySelector('[data-choice="json"]')?.getAttribute('aria-checked')).toBe('true');
      run.click();
      await settle();

      expect(enhanceApiMocks.enhancePrompt.mock.calls[0]?.[0]).toMatchObject({ format: 'json' });
      expect(draft.state.enhancedPrompt).toBe('{\n  "high_level_description": "A fox."\n}');
      const submitted = await submitForm(target);
      expect(submitted.get('json_prompt')).toBe('{\n  "high_level_description": "A fox."\n}');
      expect(submitted.has('prompt')).toBe(false);
      // The asset records the user's prompt and how the caption was written.
      expect(submitted.get('prompt_input')).toBe('a fox');
      expect(JSON.parse(String(submitted.get('enhance_record')))).toMatchObject({ model: 'org/fast@r1', format: 'json' });
    });

    it('offers Write as only for models that take JSON captions', async () => {
      await mountWorkspace(withEnhancer(makeContext()));
      await openEnhanceRun();
      expect(document.querySelector('[data-choice="json"]')).toBeNull();
    });

    it('disables Smart with its reason and runs Fast', async () => {
      enhanceApiMocks.enhancePrompt.mockResolvedValue();
      draft.update('prompt', 'a fox');
      draft.update('enhanceModel', 'smart');
      await mountWorkspace(ideoEnhancerContext({ download_size_label: '19 GB', unavailable: 'Needs about 9 GB of memory; this machine has 6 GB.' }));
      const run = await openEnhanceRun();
      const smart = document.querySelector('[data-choice="smart"]') as HTMLButtonElement;
      expect(smart.getAttribute('aria-disabled')).toBe('true');
      smart.click();
      await settle();
      expect(document.querySelector('[data-choice="fast"]')?.getAttribute('aria-checked')).toBe('true');
      expect(document.getElementById(smart.getAttribute('aria-describedby') ?? '')?.textContent).toBe('Needs about 9 GB of memory; this machine has 6 GB.');
      run.click();
      await settle();
      expect(enhanceApiMocks.enhancePrompt.mock.calls[0]?.[0]).not.toHaveProperty('enhance_model');
    });

    it('runs Smart when chosen and available, and shows its first-use size', async () => {
      enhanceApiMocks.enhancePrompt.mockResolvedValue();
      draft.update('prompt', 'a fox');
      draft.update('enhanceModel', 'smart');
      await mountWorkspace(ideoEnhancerContext({ download_size_label: '19 GB', unavailable: null }));
      const run = await openEnhanceRun();
      expect(document.body.textContent).toContain('First use downloads ≈19 GB.');
      run.click();
      await settle();
      expect(enhanceApiMocks.enhancePrompt.mock.calls[0]?.[0]).toMatchObject({ enhance_model: 'smart' });
    });

    it('sends the enhancer and format with an auto-enhance job', async () => {
      draft.update('prompt', 'a fox');
      draft.update('enhanceAuto', true);
      draft.update('enhanceModel', 'smart');
      draft.update('enhanceFormat', 'json');
      await mountWorkspace(ideoEnhancerContext({ download_size_label: null, unavailable: null }));
      const submitted = await submitForm(target);
      expect(submitted.get('enhance_model')).toBe('smart');
      expect(submitted.get('enhance_format')).toBe('json');
    });

    it('shows a custom model instead of the Fast/Smart choice', async () => {
      await mountWorkspace(ideoEnhancerContext(null));
      await openEnhanceRun();
      expect(document.querySelector('[data-choice="smart"]')).toBeNull();
      expect(document.body.textContent).toContain('Enhancer: owner/llm');
    });
  });

  it('submits the Enhanced prompt instead of the prompt when it has text', async () => {
    draft.update('prompt', 'a fox');
    draft.update('enhancedPrompt', 'A red fox in deep snow, golden light.');
    draft.update('enhancedFrom', { prompt: 'a fox', scene: '', mode: 'image' });
    await mountWorkspace(withEnhancer(makeContext()));

    const enhanced = target.querySelector('#ws-enhanced-prompt') as HTMLTextAreaElement | null;
    expect(enhanced?.value).toBe('A red fox in deep snow, golden light.');
    expect(enhanced?.getAttribute('name')).toBeNull();

    const submitted = await submitForm(target);
    expect(submitted.get('prompt')).toBe('A red fox in deep snow, golden light.');
    expect(submitted.has('enhance_auto')).toBe(false);
  });

  it('marks the Enhanced prompt out of date after the prompt changes', async () => {
    draft.update('prompt', 'a fox');
    draft.update('enhancedPrompt', 'A red fox.');
    draft.update('enhancedFrom', { prompt: 'a fox', scene: '', mode: 'image' });
    await mountWorkspace(withEnhancer(makeContext()));
    expect(target.textContent).not.toContain('Out of date');

    const prompt = target.querySelector('#ws-prompt') as HTMLTextAreaElement;
    prompt.value = 'a cat';
    prompt.dispatchEvent(new Event('input', { bubbles: true }));
    await settle();
    expect(target.textContent).toContain('Out of date');
  });

  it('submits the original prompt plus auto-enhance fields when enhancing each image', async () => {
    draft.update('prompt', 'a fox');
    draft.update('enhancedPrompt', 'Ignored while auto-enhancing.');
    draft.update('enhanceAuto', true);
    draft.update('enhanceSettings', { style: 'photo', mood: 'keep', theme: 'keep', camera: 'keep', pace: 'keep', details: [], length: 'longer', motion: ['action'], sound: [], note: '' });
    await mountWorkspace(withEnhancer(makeContext()));

    expect((target.querySelector('#ws-enhanced-prompt') as HTMLTextAreaElement).disabled).toBe(true);
    const submitted = await submitForm(target);
    expect(submitted.get('prompt')).toBe('a fox');
    expect(submitted.get('enhance_auto')).toBe('true');
    expect(JSON.parse(String(submitted.get('enhance_settings')))).toEqual({ style: 'photo', mood: 'keep', theme: 'keep', details: [], length: 'longer' });
  });

  it('copies the CLI command for what Generate would submit, without submitting', async () => {
    const writeText = mockClipboard();
    draft.update('prompt', 'a fox');
    draft.update('enhancedPrompt', 'A red fox in deep snow.');
    draft.update('enhancedFrom', { prompt: 'a fox', scene: '', mode: 'image' });
    await mountWorkspace(withEnhancer(makeContext()));

    (target.querySelector('#ws-copy-cli') as HTMLButtonElement).click();
    await settle();

    expect(workspaceApiMocks.submitGenerate).not.toHaveBeenCalled();
    expect(writeText).toHaveBeenCalledTimes(1);
    const command = writeText.mock.calls[0]?.[0] ?? '';
    expect(command).toMatch(/^ziv image --model zit /);
    expect(command).toContain("--prompt 'A red fox in deep snow.'");
    expect(command).toContain('--steps 28');
    expect(command).toMatch(/--output \/tmp\/output$/);
  });

  it('resets the options of the current mode only, and offers Reset only when something is picked', async () => {
    const video = { style: 'cinematic', mood: 'keep', theme: 'keep', camera: 'orbit', pace: 'keep', details: [], length: 'same', motion: [], sound: [], note: '' };
    draft.patch({ prompt: 'a fox', enhanceSettings: { ...video, style: 'photo', camera: 'keep' }, enhanceSettingsVideo: video });
    await mountWorkspace(withEnhancer(makeContext()));
    await openEnhanceRun();
    const reset = () => Array.from(document.querySelectorAll('#ws-enhance-panel button')).find((el) => el.textContent?.trim() === 'Reset') as HTMLButtonElement;
    expect(reset().disabled).toBe(false);

    reset().click();
    await settle();
    expect(draft.state.enhanceSettings).toBeNull();
    expect(draft.state.enhanceSettingsVideo).toEqual(video);
    expect(reset().disabled).toBe(true);
  });

  it('never warns that auto-enhance would change nothing: nothing picked is a general rewrite', async () => {
    draft.update('prompt', 'a fox');
    draft.update('enhanceAuto', true);
    draft.update('enhanceSettings', { style: 'keep', mood: 'keep', theme: 'keep', camera: 'keep', pace: 'keep', details: [], length: 'same', motion: [], sound: [], note: '' });
    await mountWorkspace(withEnhancer(makeContext()));
    await openEnhanceRun();
    // The enhancer popover is moved to <body>.
    expect(document.querySelector('#ws-enhance-panel [role="alert"]')).toBeNull();
  });

  describe('scene and prompt pair (video)', () => {
    const isShown = (id: string) => target.querySelector(`#${id}`)?.closest('[hidden]') == null && !(target.querySelector(`#${id}`) as HTMLElement).hidden;

    it('shows the Scene box in the video workflows and not in the image workflows', async () => {
      draft.update('workflow', 'txt2vid');
      await mountWorkspace(makeContext());
      expect(field('ws-scene')).not.toBeNull();
      expect(field('ws-scene').getAttribute('name')).toBe('scene');
      expect(target.querySelector('label[for="ws-scene"]')?.textContent?.trim()).toBe('Scene');

      draft.update('workflow', 'img2vid');
      await settle();
      expect(field('ws-scene')).not.toBeNull();

      for (const workflow of ['txt2img', 'img2img'] as const) {
        draft.update('workflow', workflow);
        await settle();
        expect(target.querySelector('#ws-scene')).toBeNull();
        expect(new FormData(target.querySelector('form')!).has('scene')).toBe(false);
      }
    });

    it('switches Scene and Prompt together with the Your text and Enhanced tabs', async () => {
      draft.update('workflow', 'txt2vid');
      await mountWorkspace(withVideoEnhancer(makeContext()));

      expect(target.querySelector('[role="tablist"]')?.getAttribute('aria-label')).toBe('Text version');
      expect(Array.from(target.querySelectorAll('[role="tab"]')).map((tab) => tab.id)).toEqual(['ws-tab-prompt', 'ws-tab-enhanced']);
      expect(isShown('ws-scene') && isShown('ws-prompt')).toBe(true);
      expect(isShown('ws-enhanced-scene') || isShown('ws-enhanced-prompt')).toBe(false);

      (target.querySelector('#ws-tab-enhanced') as HTMLButtonElement).click();
      await settle();
      expect(isShown('ws-enhanced-scene') && isShown('ws-enhanced-prompt')).toBe(true);
      expect(isShown('ws-scene') || isShown('ws-prompt')).toBe(false);
      expect(target.querySelector('#ws-tab-enhanced')?.getAttribute('aria-selected')).toBe('true');

      // Left and Right switch back, as for the single prompt box.
      pressKey('ArrowLeft', {}, target.querySelector('#ws-tab-enhanced')!);
      await settle();
      expect(isShown('ws-scene') && isShown('ws-prompt')).toBe(true);
    });

    it('lets a scene alone generate, and refuses with a message on the Action box when both boxes are empty', async () => {
      draft.update('workflow', 'txt2vid');
      await mountWorkspace(makeContext());

      target.querySelector('form')!.requestSubmit();
      await settle();
      expect(workspaceApiMocks.submitGenerate).not.toHaveBeenCalled();
      expect(field('ws-prompt').validationMessage).not.toBe('');

      type('ws-scene', 'A castle at dusk.');
      await settle();
      const submitted = await submitForm(target);
      expect(submitted.get('scene')).toBe('A castle at dusk.');
      expect(submitted.get('prompt')).toBe('');
    });

    it('enables Enhance with a scene or an action alone, and explains the disabled button only while both boxes are empty', async () => {
      draft.update('workflow', 'txt2vid');
      await mountWorkspace(withVideoEnhancer(makeContext()));
      expect((await openEnhanceRun()).disabled).toBe(true);
      expect(document.querySelector('#ws-enhance-panel')?.textContent).toContain('Write a scene or an action first.');

      type('ws-scene', 'A castle at dusk.');
      await settle();
      expect((document.querySelector('#ws-enhance-run') as HTMLButtonElement).disabled).toBe(false);
      expect(document.querySelector('#ws-enhance-panel')?.textContent).not.toContain('Write a scene or an action first.');

      type('ws-scene', '');
      type('ws-prompt', 'a knight rides in');
      await settle();
      expect((document.querySelector('#ws-enhance-run') as HTMLButtonElement).disabled).toBe(false);
    });

    it('enhances a video with nothing picked, with or without a scene: a general rewrite', async () => {
      draft.update('workflow', 'txt2vid');
      draft.patch({ prompt: 'a knight rides in', enhanceSettingsVideo: { style: 'keep', mood: 'keep', theme: 'keep', camera: 'keep', pace: 'keep', details: [], length: 'same', motion: [], sound: [], note: '' } });
      await mountWorkspace(withVideoEnhancer(makeContext()));
      expect((await openEnhanceRun()).disabled).toBe(false);

      type('ws-scene', 'A castle at dusk.');
      await settle();
      expect((document.querySelector('#ws-enhance-run') as HTMLButtonElement).disabled).toBe(false);
    });

    it('sends the scene to the enhancer, renders the streamed parts as they are, and keeps the finished pair', async () => {
      let release: () => void = () => undefined;
      const gate = new Promise<void>((resolve) => { release = resolve; });
      enhanceApiMocks.enhancePrompt.mockImplementation(async (_body: unknown, ...rest: unknown[]) => {
        const onFrame = rest[0] as (frame: EnhanceFrame) => void;
        onFrame({ type: 'text', text: 'SCENE: A castle.\nACTION: A kni', scene: 'A castle.', prompt: 'A kni' });
        await gate;
        onFrame({ type: 'done', scene: 'A castle at dusk.', prompt: 'A knight rides in.', clamped: false, warning: 'The enhancer did not follow the format.' });
      });
      draft.update('workflow', 'txt2vid');
      draft.patch({ scene: 'A castle.', prompt: 'a knight' });
      await mountWorkspace(withVideoEnhancer(makeContext()));

      const run = openEnhanceRun();
      (await run).click();
      await settle();
      const body = enhanceApiMocks.enhancePrompt.mock.calls[0]?.[0] as { scene?: string; prompt: string; mode: string };
      expect(body).toMatchObject({ scene: 'A castle.', prompt: 'a knight', mode: 'video' });
      // The tab comes forward and shows the server's split without parsing it again.
      expect(isShown('ws-enhanced-scene')).toBe(true);
      expect(field('ws-enhanced-scene').value).toBe('A castle.');
      expect(field('ws-enhanced-prompt').value).toBe('A kni');
      expect(field('ws-enhanced-scene').readOnly && field('ws-enhanced-prompt').readOnly).toBe(true);

      release();
      await settle();
      expect(draft.state).toMatchObject({ enhancedScene: 'A castle at dusk.', enhancedPrompt: 'A knight rides in.' });
      expect(draft.state.enhancedFrom).toEqual({ scene: 'A castle.', prompt: 'a knight', mode: 'video' });
      expect(field('ws-enhanced-scene').value).toBe('A castle at dusk.');
      expect(field('ws-enhanced-prompt').value).toBe('A knight rides in.');
      expect(target.textContent).toContain('The enhancer did not follow the format.');
      expect(target.textContent).not.toContain('Out of date');
    });

    it('sends no scene when an image prompt is enhanced', async () => {
      enhanceApiMocks.enhancePrompt.mockResolvedValue();
      draft.update('prompt', 'a fox');
      await mountWorkspace(withEnhancer(makeContext()));
      pressKey('e', { ctrlKey: true }, field('ws-prompt'));
      await settle();
      expect(enhanceApiMocks.enhancePrompt.mock.calls[0]?.[0]).not.toHaveProperty('scene');
    });

    it('sends each part enhanced when its Enhanced box has text, with the user text recorded beside them', async () => {
      draft.update('workflow', 'txt2vid');
      draft.patch({ scene: 'A castle.', prompt: 'a knight', enhancedFrom: null, enhancedScene: 'A castle at dusk.', enhancedPrompt: 'A knight rides in.' });
      await mountWorkspace(withVideoEnhancer(makeContext()));

      const submitted = await submitForm(target);
      expect(submitted.get('scene')).toBe('A castle at dusk.');
      expect(submitted.get('prompt')).toBe('A knight rides in.');
      expect(submitted.get('scene_input')).toBe('A castle.');
      expect(submitted.get('prompt_input')).toBe('a knight');
    });

    it('records the user text and how an enhanced image prompt was made', async () => {
      const record = { settings: { style: 'photo' }, model: 'org/fast@r1', format: 'text' as const };
      draft.patch({ prompt: 'a fox in snow', enhancedPrompt: 'A red fox curls up in deep snow.', enhancedFrom: { prompt: 'a fox in snow', scene: '', mode: 'image', record } });
      await mountWorkspace(withEnhancer(makeContext()));

      const submitted = await submitForm(target);
      expect(submitted.get('prompt')).toBe('A red fox curls up in deep snow.');
      expect(submitted.get('prompt_input')).toBe('a fox in snow');
      expect(JSON.parse(String(submitted.get('enhance_record')))).toEqual(record);
    });

    it('records the text an out-of-date rewrite was made from, not the changed Your text', async () => {
      const record = { settings: { style: 'photo' }, model: 'org/fast@r1', format: 'text' as const };
      draft.patch({ prompt: 'a wolf in rain', enhancedPrompt: 'A red fox curls up in deep snow.', enhancedFrom: { prompt: 'a fox in snow', scene: '', mode: 'image', record } });
      await mountWorkspace(withEnhancer(makeContext()));

      const submitted = await submitForm(target);
      expect(submitted.get('prompt')).toBe('A red fox curls up in deep snow.');
      expect(submitted.get('prompt_input')).toBe('a fox in snow');
    });

    it('sends no enhance record once the Enhanced text is edited by hand', async () => {
      const record = { settings: { style: 'photo' }, model: 'org/fast@r1', format: 'text' as const };
      draft.patch({ prompt: 'a fox in snow', enhancedPrompt: 'A red fox curls up in deep snow.', enhancedFrom: { prompt: 'a fox in snow', scene: '', mode: 'image', record } });
      await mountWorkspace(withEnhancer(makeContext()));
      const enhanced = target.querySelector('#ws-enhanced-prompt') as HTMLTextAreaElement;
      enhanced.value = 'A red fox curls up in deep snow, my own words.';
      enhanced.dispatchEvent(new Event('input', { bubbles: true }));
      await settle();

      const submitted = await submitForm(target);
      expect(submitted.get('prompt')).toBe('A red fox curls up in deep snow, my own words.');
      expect(submitted.get('prompt_input')).toBe('a fox in snow');
      expect(submitted.has('enhance_record')).toBe(false);
    });

    it('sends no enhance record without Enhanced text', async () => {
      draft.patch({ prompt: 'a fox in snow', enhancedPrompt: '', enhancedFrom: null });
      await mountWorkspace(withEnhancer(makeContext()));

      const submitted = await submitForm(target);
      expect(submitted.has('enhance_record')).toBe(false);
      expect(submitted.has('prompt_input')).toBe(false);
    });

    describe('Rewrite row', () => {
      const chip = (label: string) => Array.from(document.querySelectorAll('#ws-enhance-panel [role="radiogroup"] [role="radio"]')).find((el) => el.textContent?.trim() === label) as HTMLButtonElement;
      const rewriteChips = () => Array.from(document.querySelectorAll('#ws-enhance-panel [aria-labelledby="ws-enhance-rewrite-label"] [role="radio"]')) as HTMLButtonElement[];

      it('sits in the Output section after the look options, for the video workflows only', async () => {
        draft.update('workflow', 'txt2vid');
        await mountWorkspace(withVideoEnhancer(makeContext()));
        await openEnhanceRun();
        expect(document.querySelector('#ws-enhance-rewrite-label')?.textContent?.trim()).toBe('Rewrite');
        const rewriteLabel = document.querySelector('#ws-enhance-rewrite-label')!;
        const stylePicker = document.querySelector('#ws-enhance-panel #ws-enhance-inline-style')!;
        expect(stylePicker.compareDocumentPosition(rewriteLabel) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
        expect(rewriteLabel.closest('section')?.getAttribute('aria-label')).toBe('Output');
        expect(rewriteChips().map((el) => el.textContent?.trim())).toEqual(['Scene and action', 'Action only']);

        draft.update('workflow', 'txt2img');
        await settle();
        expect(document.querySelector('#ws-enhance-rewrite-label')).toBeNull();
      });

      it('defaults to Scene and action and remembers a choice for Text to Video', async () => {
        draft.update('workflow', 'txt2vid');
        draft.patch({ scene: 'A castle.', prompt: 'a knight' });
        await mountWorkspace(withVideoEnhancer(makeContext()));
        await openEnhanceRun();
        expect(rewriteChips().map((el) => el.getAttribute('aria-checked'))).toEqual(['true', 'false']);

        chip('Action only').click();
        await settle();
        expect(rewriteChips().map((el) => el.getAttribute('aria-checked'))).toEqual(['false', 'true']);
        expect(draft.state.enhanceSettingsVideo?.partsByWorkflow).toEqual({ txt2vid: 'action' });
      });

      it('disables Action only with the reason while the Scene is blank, and the run uses both', async () => {
        enhanceApiMocks.enhancePrompt.mockResolvedValue();
        draft.update('workflow', 'txt2vid');
        draft.patch({ prompt: 'a knight rides in', enhanceSettingsVideo: { style: 'photo', mood: 'keep', theme: 'keep', camera: 'keep', pace: 'keep', details: [], length: 'same', motion: [], sound: [], note: '', partsByWorkflow: { txt2vid: 'action' } } });
        await mountWorkspace(withVideoEnhancer(makeContext()));
        await openEnhanceRun();

        const actionOnly = chip('Action only');
        expect(actionOnly.getAttribute('aria-disabled')).toBe('true');
        expect(chip('Scene and action').getAttribute('aria-checked')).toBe('true');
        // The reason is announced with the disabled chip.
        const reason = document.getElementById(actionOnly.getAttribute('aria-describedby') ?? '');
        expect(reason?.textContent).toContain('Write a scene first');

        (document.querySelector('#ws-enhance-run') as HTMLButtonElement).click();
        await settle();
        expect(enhanceApiMocks.enhancePrompt.mock.calls[0]?.[0]).toMatchObject({ settings: { parts: 'both' } });
        // The remembered choice survives and applies again once there is a scene.
        expect(draft.state.enhanceSettingsVideo?.partsByWorkflow).toEqual({ txt2vid: 'action' });
        type('ws-scene', 'A castle.');
        await settle();
        expect(chip('Action only').disabled).toBe(false);
        expect(chip('Action only').getAttribute('aria-checked')).toBe('true');
      });

      it('sends parts inside enhance_settings with auto-enhance, and none for images', async () => {
        draft.update('workflow', 'txt2vid');
        draft.patch({ scene: 'A castle.', prompt: 'a knight', enhanceAuto: true });
        await mountWorkspace(withVideoEnhancer(makeContext()));
        expect(JSON.parse(String((await submitForm(target)).get('enhance_settings'))).parts).toBe('both');

        draft.patch({ enhanceSettingsVideo: { style: 'keep', mood: 'keep', theme: 'keep', camera: 'keep', pace: 'keep', details: [], length: 'same', motion: [], sound: [], note: '', partsByWorkflow: { txt2vid: 'action' } } });
        await settle();
        expect(JSON.parse(String((await submitForm(target)).get('enhance_settings'))).parts).toBe('action');
      });
    });

    describe('markers', () => {
      const labelRow = (id: string) => target.querySelector(`label[for="${id}"]`)!.closest('.label-row') as HTMLElement;
      /** The text beside the label: clip links and the enhanced marker. */
      const rowText = (id: string) => (labelRow(id).querySelector('.label-extras')?.textContent ?? '').replace(/\s+/g, ' ').trim();
      const useMine = (id: string) => Array.from(labelRow(id).querySelectorAll('button')).find((el) => el.textContent?.trim() === 'Use mine') as HTMLButtonElement | undefined;
      const status = () => target.querySelector('[role="status"]')?.textContent?.trim();
      const lineState = () => (target.querySelector('[data-testid="prompt-statusline"]') as HTMLElement).dataset.state;
      const lineButtons = () => Array.from(target.querySelectorAll('[data-testid="prompt-statusline"] button')).map((el) => el.textContent?.trim());

      it('shows no marker, no Use mine and no pill on Your text when nothing is enhanced', async () => {
        draft.update('workflow', 'txt2vid');
        draft.patch({ scene: 'A castle.', prompt: 'a knight' });
        await mountWorkspace(withVideoEnhancer(makeContext()));
        expect(rowText('ws-scene')).toBe('');
        expect(rowText('ws-prompt')).toBe('');
        expect(target.querySelector('#ws-tab-prompt')?.textContent?.trim()).toBe('Your text');
        expect(target.querySelector('#ws-tab-enhanced')?.textContent?.trim()).toBe('Enhanced');
      });

      it('marks only the part with Enhanced text, in both tabs, and describes the field by it', async () => {
        draft.update('workflow', 'txt2vid');
        draft.patch({ scene: 'A castle.', prompt: 'a knight', enhancedPrompt: 'A knight rides in.', enhancedFrom: null });
        await mountWorkspace(withVideoEnhancer(makeContext()));
        expect(rowText('ws-prompt')).toBe('Uses enhanced Use mine');
        expect(rowText('ws-scene')).not.toContain('Uses enhanced');
        const marker = document.getElementById('ws-action-marker')!;
        expect(marker.textContent?.trim()).toBe('Uses enhanced');
        expect(field('ws-prompt').getAttribute('aria-describedby')).toContain('ws-action-marker');
        expect(field('ws-enhanced-prompt').getAttribute('aria-describedby')).toContain('ws-action-marker');
        expect(field('ws-scene').getAttribute('aria-describedby')).toBeNull();
        // The tabs carry no pills; the label rows say which part is used.
        expect(target.querySelector('#ws-tab-prompt')?.textContent?.trim()).toBe('Your text');
        expect(target.querySelector('#ws-tab-enhanced')?.textContent?.trim()).toBe('Enhanced');

        (target.querySelector('#ws-tab-enhanced') as HTMLButtonElement).click();
        await settle();
        expect(rowText('ws-enhanced-prompt')).toContain('Uses enhanced');
      });

      it('clears one part with Use mine, announces it and sends the user text for that part', async () => {
        draft.update('workflow', 'txt2vid');
        draft.patch({ scene: 'A castle.', prompt: 'a knight', enhancedScene: 'A castle at dusk.', enhancedPrompt: 'A knight rides in.', enhancedFrom: { scene: 'A castle.', prompt: 'a knight', mode: 'video' } });
        await mountWorkspace(withVideoEnhancer(makeContext()));
        expect(rowText('ws-scene')).toContain('Uses enhanced');
        expect(rowText('ws-prompt')).toContain('Uses enhanced');

        useMine('ws-prompt')!.click();
        await settle();
        expect(draft.state).toMatchObject({ enhancedPrompt: '', enhancedScene: 'A castle at dusk.' });
        expect(rowText('ws-prompt')).not.toContain('Uses enhanced');
        expect(rowText('ws-scene')).toContain('Uses enhanced');
        expect(status()).toBe('Action now uses your text.');
        const submitted = await submitForm(target);
        expect(submitted.get('scene')).toBe('A castle at dusk.');
        expect(submitted.get('prompt')).toBe('a knight');

        useMine('ws-scene')!.click();
        await settle();
        expect(status()).toBe('Scene now uses your text.');
        expect(draft.state.enhancedFrom).toBeNull();
        expect(target.querySelector('#ws-tab-enhanced')?.textContent?.trim()).toBe('Enhanced');
      });

      it('shows no marker while auto-enhancing, whatever the boxes hold', async () => {
        draft.update('workflow', 'txt2vid');
        draft.patch({ scene: 'A castle.', prompt: 'a knight', enhancedPrompt: 'Ignored.', enhanceAuto: true });
        await mountWorkspace(withVideoEnhancer(makeContext()));
        expect(rowText('ws-prompt')).not.toContain('Uses enhanced');
        expect(lineState()).toBe('auto');
      });

      it('says Out of date in the status line, not on the part, only while its pair is stale', async () => {
        draft.update('workflow', 'txt2vid');
        draft.patch({ scene: 'A castle.', prompt: 'a knight', enhancedPrompt: 'A knight rides in.', enhancedFrom: { scene: 'A castle.', prompt: 'a knight', mode: 'video' } });
        await mountWorkspace(withVideoEnhancer(makeContext()));
        expect(lineState()).not.toBe('stale');

        type('ws-scene', 'A harbour.');
        await settle();
        expect(rowText('ws-prompt')).toBe('Uses enhanced Use mine');
        // The scene is not enhanced, so what it sends is current.
        expect(rowText('ws-scene')).toBe('');
        expect(lineState()).toBe('stale');
        // The pair has its own Use mine links, so the status line offers Re-enhance only.
        expect(lineButtons()).toEqual(['Re-enhance']);

        type('ws-scene', 'A castle.');
        await settle();
        expect(lineState()).not.toBe('stale');
      });

      it('keeps Out of date and Re-enhance in the status line ahead of a notice', async () => {
        draft.update('workflow', 'txt2vid');
        draft.patch({ scene: 'A castle.', prompt: 'a knight', enhancedScene: 'A castle at dusk.', enhancedPrompt: 'A knight rides in.', enhancedFrom: { scene: 'A castle.', prompt: 'a knight', mode: 'video' } });
        await mountWorkspace(withVideoEnhancer(makeContext()));
        type('ws-scene', 'A harbour.');
        await settle();
        expect(lineState()).toBe('stale');

        useMine('ws-scene')!.click();
        await settle();

        expect(draft.state.enhancedScene).toBe('');
        expect(lineState()).toBe('stale');
        expect(lineButtons()).toContain('Re-enhance');
      });

      it('empties both Enhanced boxes with the two Use mine links, and Enhance again starts from the user text', async () => {
        enhanceApiMocks.enhancePrompt.mockResolvedValue();
        draft.update('workflow', 'txt2vid');
        draft.patch({ scene: 'A castle.', prompt: 'a knight', enhancedScene: 'A castle at dusk.', enhancedPrompt: 'A knight rides in.', enhancedFrom: { scene: 'A castle.', prompt: 'a knight', mode: 'video' } });
        await mountWorkspace(withVideoEnhancer(makeContext()));
        (target.querySelector('#ws-tab-enhanced') as HTMLButtonElement).click();
        await settle();
        useMine('ws-enhanced-scene')!.click();
        await settle();
        useMine('ws-enhanced-prompt')!.click();
        await settle();
        expect(draft.state).toMatchObject({ enhancedScene: '', enhancedPrompt: '', enhancedFrom: null });

        draft.patch({ enhancedScene: 'Old.', enhancedPrompt: 'Old.' });
        await openEnhanceRun();
        (document.querySelector('#ws-enhance-run') as HTMLButtonElement).click();
        await settle();
        expect(enhanceApiMocks.enhancePrompt.mock.calls[0]?.[0]).toMatchObject({ scene: 'A castle.', prompt: 'a knight' });
      });
    });

    describe('Enhanced placeholders', () => {
      it('say what the job will do with auto-enhance, by Rewrite choice', async () => {
        draft.update('workflow', 'txt2vid');
        draft.patch({ scene: 'A castle.', prompt: 'a knight', enhanceAuto: true });
        await mountWorkspace(withVideoEnhancer(makeContext()));
        expect(field('ws-enhanced-scene').placeholder).toBe('Enhanced when the job runs');
        expect(field('ws-enhanced-prompt').placeholder).toBe('Enhanced when the job runs');
        expect(field('ws-enhanced-scene').disabled).toBe(true);

        draft.patch({ enhanceSettingsVideo: { style: 'keep', mood: 'keep', theme: 'keep', camera: 'keep', pace: 'keep', details: [], length: 'same', motion: [], sound: [], note: '', partsByWorkflow: { txt2vid: 'action' } } });
        await settle();
        expect(field('ws-enhanced-scene').placeholder).toBe('Not enhanced. Generate uses your scene.');
        expect(field('ws-enhanced-prompt').placeholder).toBe('Enhanced when the job runs');
      });

    });

    it('fills both tabs from the Reuse params without marking them out of date', async () => {
      workspaceApiMocks.parseUrlPrefill.mockReturnValue({
        workflow: 'txt2vid', scene: 'A castle.', prompt: 'a knight', enhanced_scene: 'A castle at dusk.', enhanced_prompt: 'A knight rides in.',
      });
      await mountWorkspace(withVideoEnhancer(makeContext()));

      expect(field('ws-scene').value).toBe('A castle.');
      expect(field('ws-prompt').value).toBe('a knight');
      expect(field('ws-enhanced-scene').value).toBe('A castle at dusk.');
      expect(field('ws-enhanced-prompt').value).toBe('A knight rides in.');
      expect(target.textContent).not.toContain('Out of date');
    });

    it('focuses the Prompt box, not the Scene box, with Alt+1', async () => {
      draft.update('workflow', 'txt2vid');
      await mountWorkspace(makeContext());
      pressKey('1', { altKey: true, code: 'Digit1' });
      await settle();
      expect(document.activeElement).toBe(field('ws-prompt'));
    });

    describe('prompt-file mode', () => {
      const fileOptions = [
        { id: 'rain:0', set_name: 'rain', source_index: 0, label: 'rain #1', prompt_preview: 'She opens an umbrella', negative_preview: null, scene_preview: 'A rainy street at night.', has_scene: true },
        { id: 'rain:1', set_name: 'rain', source_index: 1, label: 'rain #2', prompt_preview: 'She runs for the tram', negative_preview: null, scene_preview: null, has_scene: false },
      ];

      async function loadFile(): Promise<void> {
        promptFileApiMocks.inspectPromptFile.mockResolvedValue({ path: '/server/rain.yaml', options: fileOptions });
        draft.patch({ promptSource: 'file', promptFilePath: '/server/rain.yaml', promptFileOptionIds: ['rain:0', 'rain:1'] });
        await settle();
      }

      it('shows an editable Default scene that is submitted with the file', async () => {
        draft.update('workflow', 'txt2vid');
        await mountWorkspace(makeContext());
        await loadFile();

        expect(target.querySelector('#ws-prompt')).toBeNull();
        expect(target.querySelector('label[for="ws-scene"]')?.textContent?.trim()).toBe('Default scene');
        type('ws-scene', 'A city in the rain.');
        await settle();
        expect(draft.state.scene).toBe('A city in the rain.');

        const submitted = await submitForm(target);
        expect(submitted.get('prompt_source')).toBe('file');
        expect(submitted.get('scene')).toBe('A city in the rain.');
        expect(submitted.has('prompt')).toBe(false);
      });

      it('keeps the Default scene text when switching between inline and prompt file', async () => {
        draft.update('workflow', 'txt2vid');
        draft.update('scene', 'A city in the rain.');
        await mountWorkspace(makeContext());
        expect(field('ws-scene').value).toBe('A city in the rain.');
        draft.update('promptSource', 'file');
        await settle();
        expect(field('ws-scene').value).toBe('A city in the rain.');
      });

      it('marks the entries that bring their own scene', async () => {
        draft.update('workflow', 'txt2vid');
        await mountWorkspace(makeContext());
        await loadFile();

        const badges = Array.from(target.querySelectorAll('[data-testid="scene-badge"]'));
        expect(badges).toHaveLength(1);
        expect(badges[0]?.getAttribute('title')).toBe('A rainy street at night.');
      });

      it('shows no Default scene and no badge for image workflows', async () => {
        await mountWorkspace(makeContext());
        await loadFile();

        expect(target.querySelector('#ws-scene')).toBeNull();
        expect(target.querySelector('[data-testid="scene-badge"]')).toBeNull();
      });
    });
  });

  it('hides the enhancer when the backend does not advertise it', async () => {
    await mountWorkspace(makeContext());
    expect(target.querySelector('#ws-enhanced-prompt')).toBeNull();
    expect(target.querySelector('#ws-enhance-toggle')).toBeNull();
  });

  it('edits the clip length in seconds and submits it as a frame count the model accepts', async () => {
    const context = makeContext();
    draft.update('workflow', 'txt2vid');
    draft.update('model', 'ltx-8');
    draft.hydrateFromContext(context, 'ltx-8');
    app = flushSync(() => mount(ControlsSidebar, {
      target,
      props: {
        context,
        busy: false,
        imageFile: null,
        onImageFileChange: vi.fn(),
      },
    }));
    await settle();

    const duration = target.querySelector('#ws-duration') as HTMLInputElement;
    const frames = (): string | undefined => target.querySelector<HTMLInputElement>('input[name="frames"]')?.value;
    expect(duration.value).toBe('2');
    expect(frames()).toBe('49');

    // Typed seconds snap to the nearest 8k+1 frame count, with no upper limit.
    duration.value = '20';
    duration.dispatchEvent(new Event('change', { bubbles: true }));
    await settle();
    expect(frames()).toBe('481');
    expect(duration.value).toBe('20');

    duration.value = '2.2';

    duration.dispatchEvent(new Event('change', { bubbles: true }));
    await settle();
    expect(frames()).toBe('57');
    expect(duration.value).toBe('2.33');

    // Arrow keys move one valid frame step.
    duration.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowUp', bubbles: true }));
    await settle();
    expect(frames()).toBe('65');

    // Partial input is left alone while typing; it converts only on commit.
    duration.value = '0';
    duration.dispatchEvent(new Event('input', { bubbles: true }));
    await settle();
    expect(duration.value).toBe('0');
    expect(frames()).toBe('65');

    // A commit that keeps the same frame count still shows the rounded seconds, not the raw step.
    duration.value = '2.68';
    duration.dispatchEvent(new Event('change', { bubbles: true }));
    await settle();
    expect(frames()).toBe('65');
    expect(duration.value).toBe('2.67');

    // Enter commits and submits in the same task, so the frames field must update before any await.
    duration.value = '5';
    duration.dispatchEvent(new Event('change', { bubbles: true }));
    expect(frames()).toBe('121');
    expect(duration.value).toBe('5');

    // Clearing the field and leaving it shows the duration that will still be submitted.
    duration.value = '';
    duration.dispatchEvent(new Event('change', { bubbles: true }));
    expect(duration.value).toBe('5');
    expect(frames()).toBe('121');

    // An arrow key after typing nudges from the typed text, not the last committed value.
    duration.value = '10';
    duration.dispatchEvent(new Event('input', { bubbles: true }));
    duration.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowUp', bubbles: true }));
    await settle();
    expect(frames()).toBe('249');

    // A nudge from typed text that snaps back to the held count replaces the typed text, so a later blur keeps it.
    duration.value = '5';
    duration.dispatchEvent(new Event('change', { bubbles: true }));
    duration.value = '4.8';
    duration.dispatchEvent(new Event('input', { bubbles: true }));
    duration.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowUp', bubbles: true }));
    expect(duration.value).toBe('5');
    duration.dispatchEvent(new Event('change', { bubbles: true }));
    expect(frames()).toBe('121');

    // Any non-nullable field shows the value it still holds when cleared and left.
    const steps = target.querySelector('#ws-steps') as HTMLInputElement;
    const kept = steps.value;
    steps.value = '';
    steps.dispatchEvent(new Event('input', { bubbles: true }));
    steps.dispatchEvent(new Event('change', { bubbles: true }));
    expect(steps.value).toBe(kept);
  });

  it('shows only truthful controls for the active workflow and model capabilities', async () => {
    const context = makeContext();
    draft.update('workflow', 'txt2vid');
    draft.update('model', 'ltx-8');
    draft.hydrateFromContext(context, 'ltx-8');
    app = flushSync(() => mount(ControlsSidebar, {
      target,
      props: {
        context,
        busy: false,
        imageFile: null,
        onImageFileChange: vi.fn(),
      },
    }));
    await settle();

    expect(target.querySelector('#ws-negative-prompt')).toBeNull();
    expect(target.querySelector('input[name="guidance"]')).toBeNull();
    expect(target.querySelector('input[name="audio"]')).not.toBeNull();
    expect(target.querySelector('input[name="low_memory"]')).not.toBeNull();
    expect(target.querySelector('select[name="quantize"]')).toBeNull();
    expect(target.querySelector('#ws-image-file')).toBeNull();
    expect(target.querySelector('input[name="image_path"]')).toBeNull();

    draft.update('workflow', 'img2img');
    draft.update('model', 'flux-lite');
    await settle();

    expect(target.querySelector('#ws-image-file')).not.toBeNull();
    expect(target.querySelector('input[name="image_path"]')).not.toBeNull();
    expect(target.querySelector('input[name="guidance"]')).not.toBeNull();
    expect(target.querySelector('#ws-negative-prompt')).toBeNull();
    expect(target.querySelector('input[name="audio"]')).toBeNull();

    draft.update('model', 'zit');
    await settle();

    expect(target.querySelector('#ws-negative-prompt')).not.toBeNull();
  });

  it('gates reference image and upscale controls by selected model capabilities', async () => {
    draft.update('workflow', 'img2img');

    const ideogramDefaults = makeIdeogramDefaults();
    const permissiveDefaults = makeImageDefaults({
      ratio: '16:9',
      size: 'l',
      width: 1664,
      height: 928,
    });
    const context = makeContext({
      image_models: [
        { id: 'ideo', label: 'ideo', type: 'image' },
        { id: 'zit', label: 'zit', type: 'image' },
      ],
      defaults: ideogramDefaults,
      current_image_model: 'ideo',
      image_model_defaults: {
        ideo: ideogramDefaults,
        zit: permissiveDefaults,
      },
      image_ratios: ['16:9'],
      image_size_options: { '16:9': ['m', 'l', 'xl'] },
      image_size_dimensions: {
        '16:9': {
          m: [1344, 768],
          l: [1664, 928],
          xl: [2112, 1184],
        },
      },
    });

    await mountWorkspace(context);

    expect(target.querySelector('#ws-image-file')).toBeNull();
    expect(target.querySelector('input[name="image_path"]')).toBeNull();
    expect(target.querySelector('#ws-upscale-enabled')).toBeNull();

    const modelSelect = target.querySelector('#ws-model') as HTMLButtonElement | null;
    expect(modelSelect).not.toBeNull();
    pickOption(modelSelect!, 'zit');
    await settle();

    expect(target.querySelector('#ws-image-file')).not.toBeNull();
    expect(target.querySelector('input[name="image_path"]')).not.toBeNull();
    expect(target.querySelector('#ws-upscale-enabled')).not.toBeNull();
  });

  it('shows download and memory-fit status for the selected model and quantize level', async () => {
    const context = makeContext({
      image_models: [
        {
          id: 'zit',
          label: 'zit',
          type: 'image',
          downloaded: true,
          memory_fit: {
            budget_gb: 10.7,
            by_quantize: {
              none: { status: 'too_large', required_gb: 20.8 },
              '8': { status: 'tight', required_gb: 9.5 },
              '4': { status: 'fits', required_gb: 7.1 },
            },
          },
        },
        { id: 'flux-lite', label: 'flux-lite', type: 'image', downloaded: false, memory_fit: null },
      ],
    });

    await mountWorkspace(context);

    const fitBadge = () => target.querySelector('[data-testid="model-memory-fit"] [data-status]');
    expect(fitBadge()?.getAttribute('data-status')).toBe('too_large');
    expect(target.querySelector('[data-testid="model-download-status"]')).toBeNull();

    const quantizeSelect = target.querySelector('[data-testid="quantize-shell"] [role="combobox"]') as HTMLButtonElement;
    pickOption(quantizeSelect, '8');
    await settle();
    expect(fitBadge()?.getAttribute('data-status')).toBe('tight');

    const modelSelect = target.querySelector('#ws-model') as HTMLButtonElement;
    expect(optionLabels(modelSelect)).toEqual(['zit', 'flux-lite (not downloaded)']);
    pickOption(modelSelect, 'flux-lite');
    await settle();
    expect(target.querySelector('[data-testid="model-download-status"]')?.textContent).toContain('Not downloaded');
    expect(fitBadge()).toBeNull();
  });

  describe('model downloads from the Models page', () => {
    const lane = (overrides: Partial<DownloadItem> = {}): DownloadItem => ({
      id: 'd1', kind: 'model', name: 'flux-lite', label: 'flux-lite', state: 'running', message: null, bytes_done: 4.2e9, bytes_total: 10e9, ...overrides,
    });
    const downloadContext = () => makeContext({
      image_models: [
        { id: 'zit', label: 'zit', type: 'image', downloaded: true, memory_fit: null },
        { id: 'flux-lite', label: 'flux-lite', type: 'image', downloaded: false, memory_fit: null },
      ],
    });

    it('shows a running download in the picker option and the status badge instead of "not downloaded"', async () => {
      modelApiMocks.getDownloads.mockResolvedValue({ items: [lane()] });
      await mountWorkspace(downloadContext());

      const modelSelect = target.querySelector('#ws-model') as HTMLButtonElement;
      expect(optionLabels(modelSelect)).toEqual(['zit', 'flux-lite (downloading 42%)']);

      pickOption(modelSelect, 'flux-lite');
      await settle();
      const status = target.querySelector('[data-testid="model-download-status"]');
      expect(status?.textContent).toContain('Downloading 42%');
      expect(status?.textContent).not.toContain('Not downloaded');
    });

    it('says a queued download is queued', async () => {
      modelApiMocks.getDownloads.mockResolvedValue({ items: [lane({ state: 'queued' })] });
      await mountWorkspace(downloadContext());

      const modelSelect = target.querySelector('#ws-model') as HTMLButtonElement;
      expect(optionLabels(modelSelect)[1]).toBe('flux-lite (download queued)');
    });

    it('reads the model status again when a download that was running finishes', async () => {
      modelApiMocks.getDownloads.mockResolvedValue({ items: [lane()] });
      await mountWorkspace(downloadContext());
      expect(workspaceApiMocks.getWorkspaceCoreContext).toHaveBeenCalledTimes(1);

      modelApiMocks.getDownloads.mockResolvedValue({ items: [lane({ state: 'done' })] });
      workspaceApiMocks.getWorkspaceCoreContext.mockResolvedValue(makeContext({
        image_models: [
          { id: 'zit', label: 'zit', type: 'image', downloaded: true, memory_fit: null },
          { id: 'flux-lite', label: 'flux-lite', type: 'image', downloaded: true, memory_fit: null },
        ],
      }));
      await downloadStore.refresh();
      await settle();

      expect(workspaceApiMocks.getWorkspaceCoreContext).toHaveBeenCalledTimes(2);
      const modelSelect = target.querySelector('#ws-model') as HTMLButtonElement;
      expect(optionLabels(modelSelect)).toEqual(['zit', 'flux-lite']);
    });
  });

  it('shows the selected model by its label in the toolbar while the option value stays its id', async () => {
    await mountWorkspace(makeContext({
      image_models: [
        { id: 'klein9b', label: 'FLUX.2 Klein 9B', type: 'image' },
        { id: 'zit', label: 'zit', type: 'image' },
      ],
      current_image_model: 'klein9b',
      image_model_defaults: { klein9b: makeImageDefaults({}), zit: makeImageDefaults({}) },
    }));

    const shell = target.querySelector('[data-testid="model-shell"]');
    expect(shell?.textContent).toContain('FLUX.2 Klein 9B');
    expect(shell?.textContent).not.toContain('klein9b');
    const modelSelect = target.querySelector('#ws-model') as HTMLButtonElement;
    expect(optionValues(modelSelect)).toEqual(['klein9b', 'zit']);
    expect(optionLabels(modelSelect)).toEqual(['FLUX.2 Klein 9B', 'zit']);
    expect(draft.state.model).toBe('klein9b');
  });

  it('keeps a changed setting and applies the new model defaults to the rest when the toolbar model changes', async () => {
    const context = makeContext({
      image_models: [
        { id: 'zit', label: 'zit', type: 'image' },
        { id: 'flux-lite', label: 'flux-lite', type: 'image' },
      ],
      current_image_model: 'zit',
      image_model_defaults: {
        zit: makeImageDefaults({ steps: 8, guidance: 1 }),
        'flux-lite': makeImageDefaults({ steps: 28, guidance: 3.5 }),
      },
    });
    await mountWorkspace(context);
    draft.update('steps', 5);
    await settle();

    const modelSelect = target.querySelector('#ws-model') as HTMLButtonElement;
    pickOption(modelSelect, 'flux-lite');
    await settle();

    expect(draft.state).toMatchObject({ model: 'flux-lite', steps: 5, guidance: 3.5 });
  });

  it('applies ideogram dimension bounds and filters xl presets only for constrained models', async () => {
    const ideogramDefaults = makeIdeogramDefaults();
    const permissiveDefaults = makeImageDefaults({
      ratio: '16:9',
      size: 'xl',
      width: 2112,
      height: 1184,
    });
    const context = makeContext({
      image_models: [
        { id: 'ideo', label: 'ideo', type: 'image' },
        { id: 'zit', label: 'zit', type: 'image' },
      ],
      defaults: ideogramDefaults,
      current_image_model: 'ideo',
      image_model_defaults: {
        ideo: ideogramDefaults,
        zit: permissiveDefaults,
      },
      image_ratios: ['16:9'],
      image_size_options: { '16:9': ['m', 'l', 'xl'] },
      image_size_dimensions: {
        '16:9': {
          m: [1344, 768],
          l: [1664, 928],
          xl: [2112, 1184],
        },
      },
    });

    await mountWorkspace(context);

    const resolutionOptions = (): string[] => Array.from(
      target.querySelectorAll<HTMLButtonElement>('[role="group"][aria-label="Resolution"] button')
    ).map((button) => button.getAttribute('aria-label')?.replace('Resolution ', '') ?? '');
    expect(resolutionOptions()).toEqual(['m', 'l']);

    // Typing a width switches to custom dimensions, bounded by the model's limits.
    const widthInput = target.querySelector('#ws-width') as HTMLInputElement | null;
    const heightInput = target.querySelector('#ws-height') as HTMLInputElement | null;
    expect(widthInput?.name).toBe('');
    widthInput!.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowUp', bubbles: true }));
    await settle();
    expect(widthInput?.value).toBe('1680');
    expect(widthInput?.name).toBe('width');
    expect(heightInput?.name).toBe('height');
    // Bounds are applied when a typed value is committed, not by the browser.
    heightInput!.value = '4000';
    heightInput!.dispatchEvent(new Event('change', { bubbles: true }));
    await settle();
    expect(heightInput?.value).toBe('2048');
    expect(heightInput?.min).toBe('');

    const modelSelect = target.querySelector('#ws-model') as HTMLButtonElement | null;
    expect(modelSelect).not.toBeNull();
    pickOption(modelSelect!, 'zit');
    await settle();

    const ratioButton = target.querySelector('[role="group"][aria-label="Aspect ratio"] button') as HTMLButtonElement | null;
    expect(ratioButton).not.toBeNull();
    ratioButton!.click();
    await settle();

    expect(resolutionOptions()).toEqual(['m', 'l', 'xl']);
  });

  it('submits either prompt or json_prompt based on the structured caption toggle', async () => {
    const ideogramDefaults = makeIdeogramDefaults();
    const permissiveDefaults = makeImageDefaults({ ratio: '16:9', size: 'l', width: 1664, height: 928 });
    const context = makeContext({
      image_models: [
        { id: 'ideo', label: 'ideo', type: 'image' },
        { id: 'zit', label: 'zit', type: 'image' },
      ],
      defaults: ideogramDefaults,
      current_image_model: 'ideo',
      image_model_defaults: {
        ideo: ideogramDefaults,
        zit: permissiveDefaults,
      },
      image_ratios: ['16:9'],
      image_size_options: { '16:9': ['m', 'l'] },
      image_size_dimensions: {
        '16:9': {
          m: [1344, 768],
          l: [1664, 928],
        },
      },
    });

    await mountWorkspace(context);

    const jsonToggle = target.querySelector('#ws-json-prompt-toggle') as HTMLButtonElement | null;
    expect(jsonToggle).not.toBeNull();
    expect(target.querySelector('#ws-prompt')).not.toBeNull();
    expect(target.querySelector('textarea[name="json_prompt"]')).toBeNull();

    const promptInput = target.querySelector('#ws-prompt') as HTMLTextAreaElement | null;
    expect(promptInput).not.toBeNull();
    promptInput!.value = 'plain caption';
    promptInput!.dispatchEvent(new Event('input', { bubbles: true }));
    await settle();

    const form = target.querySelector('form');
    expect(form).not.toBeNull();
    let submittedFormData = new FormData(form!);
    expect(submittedFormData).toBeInstanceOf(FormData);
    expect(submittedFormData.get('prompt')).toBe('plain caption');
    expect(submittedFormData.has('json_prompt')).toBe(false);

    jsonToggle!.click();
    await settle();

    expect(target.querySelector('#ws-prompt')).toBeNull();
    const jsonPromptInput = target.querySelector('textarea[name="json_prompt"]') as HTMLTextAreaElement | null;
    expect(jsonPromptInput).not.toBeNull();
    jsonPromptInput!.value = '{"high_level_description":"json caption"}';
    jsonPromptInput!.dispatchEvent(new Event('input', { bubbles: true }));
    await settle();

    submittedFormData = new FormData(form!);
    expect(submittedFormData).toBeInstanceOf(FormData);
    expect(submittedFormData.get('json_prompt')).toBe('{"high_level_description":"json caption"}');
    expect(submittedFormData.has('prompt')).toBe(false);

    const modelSelect = target.querySelector('#ws-model') as HTMLButtonElement | null;
    expect(modelSelect).not.toBeNull();
    pickOption(modelSelect!, 'zit');
    await settle();

    expect(target.querySelector('#ws-json-prompt-toggle')).toBeNull();
  });

  it('submits first_sigma only when set and hides the control for unsupported models', async () => {
    const ideogramDefaults = makeIdeogramDefaults();
    const permissiveDefaults = makeImageDefaults({ ratio: '16:9', size: 'l', width: 1664, height: 928 });
    const context = makeContext({
      image_models: [
        { id: 'ideo', label: 'ideo', type: 'image' },
        { id: 'zit', label: 'zit', type: 'image' },
      ],
      defaults: ideogramDefaults,
      current_image_model: 'ideo',
      image_model_defaults: {
        ideo: ideogramDefaults,
        zit: permissiveDefaults,
      },
      image_ratios: ['16:9'],
      image_size_options: { '16:9': ['m', 'l'] },
      image_size_dimensions: {
        '16:9': {
          m: [1344, 768],
          l: [1664, 928],
        },
      },
    });

    await mountWorkspace(context);

    const firstSigmaInput = target.querySelector('#ws-first-sigma') as HTMLInputElement | null;
    expect(firstSigmaInput).not.toBeNull();
    expect(firstSigmaInput?.placeholder).toBe('1.004');
    expect(target.querySelector('input[name="first_sigma"]')).toBeNull();

    const form = target.querySelector('form');
    expect(form).not.toBeNull();
    let submittedFormData = new FormData(form!);
    expect(submittedFormData).toBeInstanceOf(FormData);
    expect(submittedFormData.has('first_sigma')).toBe(false);

    firstSigmaInput!.value = '1.005';
    firstSigmaInput!.dispatchEvent(new Event('input', { bubbles: true }));
    await settle();

    expect(target.querySelector('input[name="first_sigma"]')).not.toBeNull();

    submittedFormData = new FormData(form!);
    expect(submittedFormData).toBeInstanceOf(FormData);
    expect(submittedFormData.get('first_sigma')).toBe('1.005');

    const modelSelect = target.querySelector('#ws-model') as HTMLButtonElement | null;
    expect(modelSelect).not.toBeNull();
    pickOption(modelSelect!, 'zit');
    await settle();

    expect(target.querySelector('#ws-first-sigma')).toBeNull();
  });

  it('serializes post-processing hidden fields when enabled controls are active', async () => {
    const context = makeContext();
    await mountWorkspace(context);

    draft.update('postprocessSharpenEnabled', true);
    draft.update('postprocessSharpenAmount', 0.75);
    draft.update('postprocessContrastEnabled', false);
    draft.update('postprocessSaturationEnabled', true);
    draft.update('postprocessSaturationAmount', 1.2);
    await settle();

    const form = target.querySelector('form');
    expect(form).not.toBeNull();
    form!.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
    await settle();

    expect(workspaceApiMocks.submitGenerate).toHaveBeenCalledTimes(1);
    const [submittedFormData] = workspaceApiMocks.submitGenerate.mock.calls[0] ?? [];
    expect(submittedFormData).toBeInstanceOf(FormData);
    expect(submittedFormData.get('sharpen_enabled')).toBe('true');
    expect(submittedFormData.get('sharpen_amount')).toBe('0.75');
    expect(submittedFormData.get('contrast_enabled')).toBe('false');
    expect(submittedFormData.has('contrast_amount')).toBe(false);
    expect(submittedFormData.get('saturation_enabled')).toBe('true');
    expect(submittedFormData.get('saturation_amount')).toBe('1.2');
  });

  it('sends no sharpen amount when it is left on auto, so the backend uses the config', async () => {
    const context = makeContext();
    await mountWorkspace(context);

    draft.update('postprocessSharpenEnabled', true);
    draft.update('postprocessSharpenAmount', null);
    await settle();

    target.querySelector('form')!.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
    await settle();

    const [submittedFormData] = workspaceApiMocks.submitGenerate.mock.calls[0] ?? [];
    expect(submittedFormData.get('sharpen_enabled')).toBe('true');
    expect(submittedFormData.has('sharpen_amount')).toBe(false);
  });

  it('renders video upscale controls for txt2vid/img2vid and serializes upscale fields', async () => {
    const context = makeContext();
    await mountWorkspace(context);

    for (const workflow of ['txt2vid', 'img2vid'] as const) {
      draft.update('workflow', workflow);
      draft.update('model', 'ltx-8');
      await settle();

      draft.update('videoUpscaleEnabled', true);
      await settle();

      expect(target.querySelector('#ws-video-upscale')).not.toBeNull();
      expect(target.querySelector('input[name="video_upscale_factor"]')).not.toBeNull();

      const factorSelect = target.querySelector('#ws-video-upscale-factor') as HTMLButtonElement | null;
      expect(factorSelect).not.toBeNull();
      expect(optionValues(factorSelect!)).toEqual(['2']);

      const form = target.querySelector('form');
      expect(form).not.toBeNull();
      const serializedFormData = new FormData(form!);
      expect(serializedFormData.get('workflow')).toBe(workflow);
      expect(serializedFormData.get('upscale')).toBe('2');
      expect(serializedFormData.get('video_upscale_factor')).toBe('2');
    }
  });

  it('clears a stale generation error when a later submit starts successfully', async () => {
    workspaceApiMocks.submitGenerate
      .mockRejectedValueOnce(new Error('backend rejected first attempt'))
      .mockResolvedValueOnce({
        job_id: 'job-retry',
        workflow: 'txt2img',
        prompt: 'Retry prompt',
        model: 'zit',
        runs: 1,
        created_at: '2026-04-23T10:01:00Z',
      });

    const context = makeContext();
    await mountWorkspace(context);

    const form = target.querySelector('form');
    expect(form).not.toBeNull();

    form!.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
    await settle();

    expect(target.textContent).toContain('backend rejected first attempt');
    expect(jobStore.current).toBeNull();

    form!.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
    await settle();

    expect(workspaceApiMocks.submitGenerate).toHaveBeenCalledTimes(2);
    expect(target.textContent).not.toContain('backend rejected first attempt');
    expect(jobStore.current?.job_id).toBe('job-retry');
    expect(target.querySelector('.job-card')).not.toBeNull();
  });

  it('keeps rendered workspace labels associated with controls across workflow states', async () => {
    const context = makeContext();
    await mountWorkspace(context);

    expectRenderedLabelsToResolveControls(target);

    draft.update('workflow', 'img2img');
    draft.update('model', 'zit');
    await settle();

    expectRenderedLabelsToResolveControls(target);

    draft.update('workflow', 'txt2vid');
    draft.update('model', 'ltx-8');
    await settle();

    expectRenderedLabelsToResolveControls(target);
  });

  it('adds a LoRA from the Add LoRA menu and closes the menu on Escape or an outside click', async () => {
    draft.update('workflow', 'txt2img');
    draft.update('model', 'zit');
    const context = makeContext();
    context.loras = [
      { name: 'film-grain', path: '/loras/film-grain.safetensors' },
      { name: 'krea-flat', path: '/loras/krea-flat.safetensors' },
    ];
    await mountWorkspace(context);
    await settle();

    const addButton = Array.from(target.querySelectorAll('button')).find((b) => b.textContent?.includes('Add LoRA')) as HTMLButtonElement;
    const menu = () => document.querySelector('[role="menu"][aria-label="Available LoRAs"]');

    flushSync(() => addButton.click());
    expect(menu()).not.toBeNull();
    flushSync(() => document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true })));
    expect(menu()).toBeNull();

    flushSync(() => addButton.click());
    flushSync(() => document.body.dispatchEvent(new Event('pointerdown', { bubbles: true })));
    expect(menu()).toBeNull();

    flushSync(() => addButton.click());
    const item = menu()?.querySelector('[data-action="krea-flat"]') as HTMLButtonElement;
    flushSync(() => item.click());
    expect(menu()).toBeNull();
    expect(draft.state.loraString).toBe('krea-flat:1');
  });

  describe('Add LoRA menu grouping', () => {
    const picker = () => document.querySelector('[role="menu"][aria-label="Available LoRAs"]') as HTMLElement | null;
    const openPicker = () => {
      const addButton = Array.from(target.querySelectorAll('button')).find((b) => b.textContent?.includes('Add LoRA')) as HTMLButtonElement;
      flushSync(() => addButton.click());
    };
    const actions = () => Array.from(picker()?.querySelectorAll('[data-action]') ?? []).map((el) => el.getAttribute('data-action'));

    function loraContext(loras: WorkspaceContext['loras'], assets: GalleryAsset[] = []): WorkspaceContext {
      galleryApiMocks.getGallery.mockResolvedValue(makeGalleryPage(assets));
      const context = makeContext();
      context.image_model_defaults = { ...context.image_model_defaults, zit: makeImageDefaults({ family: 'zimage' }) };
      context.loras = loras;
      return context;
    }

    beforeEach(() => {
      draft.update('workflow', 'txt2img');
      draft.update('model', 'zit');
    });

    it('lists the model matches first, recent ones leading, and folds the others, unknown LoRAs included', async () => {
      const context = loraContext(
        [
          { name: 'a-match', path: '/l/a', family: 'zimage', size_variant: null },
          { name: 'b-recent', path: '/l/b', family: 'zimage', size_variant: null },
          { name: 'c-unknown', path: '/l/c', family: null, size_variant: null },
          { name: 'd-ltx', path: '/l/d', family: 'ltx', size_variant: null },
        ],
        [makeAsset({ lora: 'b-recent:1', details: { model_family: 'zimage' } })],
      );
      await mountWorkspace(context);
      await settle();
      openPicker();

      expect(actions()).toEqual(['b-recent', 'a-match', 'other-models', 'discover-loras']);
      const fold = picker()?.querySelector('[data-action="other-models"]') as HTMLElement;
      expect(fold.getAttribute('aria-expanded')).toBe('false');

      flushSync(() => fold.click());
      expect(picker()).not.toBeNull();
      expect(fold.getAttribute('aria-expanded')).toBe('true');
      expect(picker()?.querySelector('[data-action="d-ltx"]')?.textContent).toContain('LTX');
      expect(picker()?.querySelector('[data-action="c-unknown"]')).not.toBeNull();
    });
  });

  it('renders model and quantize selects enabled for txt2img with zit model', async () => {
    draft.update('workflow', 'txt2img');
    draft.update('model', 'zit');

    const context = makeContext();
    await mountWorkspace(context);

    const modelSelect = target.querySelector('#ws-model') as HTMLButtonElement | null;
    expect(modelSelect, 'Model select must be rendered').not.toBeNull();
    expect(modelSelect!.disabled, 'Model select must be enabled after context loads').toBe(false);

    // Wait for the quantize block (requires async context to load and supportsQuantize=true).
    await settle();
    const quantizeSelect = target.querySelector('[data-testid="quantize-shell"] [role="combobox"]') as HTMLButtonElement | null;
    expect(quantizeSelect, 'Quantize select must be rendered for zit model (supports_quantize: true)').not.toBeNull();
    expect(quantizeSelect!.disabled, 'Quantize select must be enabled after context loads').toBe(false);
  });

  it('browses the host for a reference image and submits its path instead of an uploaded file', async () => {
    draft.update('workflow', 'img2img');
    browseApiMocks.browsePlaces.mockResolvedValue(placesFor('reference_image'));
    browseApiMocks.listDirectory.mockResolvedValue(listingWith('/pictures/cat.png'));
    await mountWorkspace(makeContext());

    (target.querySelector('#ws-image-browse') as HTMLButtonElement).click();
    await settle();
    expect(browseApiMocks.listDirectory).toHaveBeenCalledWith('reference_image', '', expect.anything());
    chooseInBrowser('cat.png');
    await settle();

    expect(draft.state.referenceImagePath).toBe('/pictures/cat.png');
    expect((target.querySelector('#ws-image-path') as HTMLInputElement).value).toBe('/pictures/cat.png');
  });

  it('does not submit stale reference image fields after switching to a non-reference workflow', async () => {
    draft.update('workflow', 'img2img');
    draft.update('referenceImagePath', '/tmp/stale-reference.png');

    const context = makeContext();
    await mountWorkspace(context);

    const fileInput = target.querySelector('#ws-image-file') as HTMLInputElement | null;
    expect(fileInput).not.toBeNull();

    const imageFile = new File(['fake-image'], 'reference.png', { type: 'image/png' });
    Object.defineProperty(fileInput!, 'files', {
      configurable: true,
      value: [imageFile],
    });
    fileInput!.dispatchEvent(new Event('change', { bubbles: true }));
    await settle();

    draft.update('workflow', 'txt2img');
    await settle();

    expect(target.querySelector('#ws-image-file')).toBeNull();
    expect(target.querySelector('input[name="image_path"]')).toBeNull();

    const form = target.querySelector('form');
    expect(form).not.toBeNull();
    form!.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
    await settle();

    expect(workspaceApiMocks.submitGenerate).toHaveBeenCalledTimes(1);
    const [submittedFormData] = workspaceApiMocks.submitGenerate.mock.calls[0] ?? [];
    expect(submittedFormData).toBeInstanceOf(FormData);
    expect(submittedFormData.get('workflow')).toBe('txt2img');
    expect(submittedFormData.get('mode')).toBe('image');
    expect(submittedFormData.has('image_file')).toBe(false);
    expect(submittedFormData.has('image_path')).toBe(false);
  });

  it('leaves busy mode when a job started in the current session is cancelled', async () => {
    const context = makeContext();
    await mountWorkspace(context);

    draft.update('prompt', 'Cancel this run');
    await settle();

    const form = target.querySelector('form');
    const submitButton = target.querySelector('#ws-submit') as HTMLButtonElement | null;
    expect(form).not.toBeNull();
    expect(submitButton).not.toBeNull();
    expect(submitButton?.disabled).toBe(false);

    form!.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
    await settle();

    expect(workspaceApiMocks.submitGenerate).toHaveBeenCalledTimes(1);
    // A running job no longer blocks Generate; it adds to the queue.
    expect(submitButton?.disabled).toBe(false);
    expect(submitButton?.textContent).toContain('Add to queue');

    const mockEventSource = lastEventSource();
    mockEventSource.emit('job_cancelled', { type: 'job_cancelled', job_id: 'job-123' });
    await settle();

    expect(jobStore.current?.status).toBe('cancelled');
    expect(submitButton?.disabled).toBe(false);
    expect(submitButton?.textContent).not.toContain('Add to queue');
  });

  it('refreshes model download and memory status after a job ends', async () => {
    const notDownloaded = makeContext({
      image_models: [{ id: 'zit', label: 'zit', type: 'image', downloaded: false, memory_fit: null }],
    });
    await mountWorkspace(notDownloaded);
    expect(target.querySelector('[data-testid="model-download-status"]')).not.toBeNull();

    workspaceApiMocks.getWorkspaceCoreContext.mockResolvedValue(makeContext({
      image_models: [{
        id: 'zit',
        label: 'zit',
        type: 'image',
        downloaded: true,
        memory_fit: { budget_gb: 10.7, by_quantize: { none: { status: 'fits', required_gb: 7.1 } } },
      }],
    }));
    draft.update('prompt', 'Download on first use');
    await settle();
    target.querySelector('form')!.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
    await settle();

    const mockEventSource = lastEventSource();
    mockEventSource.emit('job_failed', { type: 'job_failed', job_id: 'job-123', message: 'boom' });
    await settle();

    expect(target.querySelector('[data-testid="model-download-status"]')).toBeNull();
    expect(target.querySelector('[data-testid="model-memory-fit"] [data-status]')?.getAttribute('data-status')).toBe('fits');
  });

  it('adds successful batch outputs to the history pane immediately', async () => {
    await mountWorkspace(makeContext());
    draft.update('prompt', 'Show each batch result');
    await settle();

    const form = target.querySelector('form');
    expect(form).not.toBeNull();
    form!.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
    await settle();

    const firstAsset = makeAsset({
      id: 'batch/first.png',
      url: '/media/batch/first.png',
      filename: 'first.png',
    });
    const secondAsset = makeAsset({
      id: 'batch/second.png',
      url: '/media/batch/second.png',
      filename: 'second.png',
    });
    const source = lastEventSource();
    source.emit('generation_finished', {
      type: 'generation_finished',
      job_id: 'job-123',
      status: 'success',
      run_index: 0,
      asset: firstAsset,
    });
    await settle();
    source.emit('generation_finished', {
      type: 'generation_finished',
      job_id: 'job-123',
      status: 'success',
      run_index: 1,
      asset: secondAsset,
    });
    await settle();

    expect(target.querySelector('button[aria-label="View first.png"]')).not.toBeNull();
    expect(target.querySelector('button[aria-label="View second.png"]')).not.toBeNull();
    expect(galleryStore.assets.map((item) => item.id)).toEqual([secondAsset.id, firstAsset.id]);

    // A job that ends badly leaves the outputs it made in the history.
    source.emit('job_failed', { type: 'job_failed', job_id: 'job-123' });
    await settle();
    expect(target.querySelector('.job-card[data-tone="failed"]')).not.toBeNull();
    expect(target.querySelector('button[aria-label="View first.png"]')).not.toBeNull();
    expect(target.querySelector('button[aria-label="View second.png"]')).not.toBeNull();
  });

  it('restores an active job from the backend workspace snapshot on mount', async () => {
    const context = makeContext({
      active_job: {
        id: 'job-live',
        job_id: 'job-live',
        workflow: 'txt2vid',
        job_type: 'Text to Video',
        status: 'running',
        created_at: '2026-04-23T10:00:00Z',
        completed_at: null,
        event_count: 3,
        last_event: {
          type: 'step_progress',
          current_step: 2,
          total_steps: 12,
          elapsed_secs: 4,
          eta_secs: 18,
        },
        supported_controls: [],
        paused: false,
        result_path: null,
        prompt: 'Recovered active job',
        model: 'ltx-8',
        runs: 1,
      },
    });

    await mountWorkspace(context);

    expect(workspaceApiMocks.getJobSnapshot).not.toHaveBeenCalled();
    expect(jobStore.current?.job_id).toBe('job-live');
    expect(jobStore.current?.currentStep).toBe(2);
    expect(jobStore.current?.totalSteps).toBe(12);
    expect(jobStore.current?.remaining).toBe(18);

    const submitButton = target.querySelector('#ws-submit') as HTMLButtonElement | null;
    // A running job no longer blocks Generate; it adds to the queue.
    expect(submitButton?.disabled).toBe(false);
    expect(submitButton?.textContent).toContain('Add to queue');

    const activeCardText = target.textContent ?? '';
    const activeJobCard = target.querySelector('article');
    expect(activeCardText).toContain('Recovered active job');
    expect(activeJobCard).not.toBeNull();
    expect(activeJobCard!.querySelectorAll('button')).toHaveLength(0);
  });

  it('rebinds completion ownership across a workspace remount without opening another EventSource', async () => {
    const context = makeContext({
      active_job: {
        id: 'job-remount',
        job_id: 'job-remount',
        workflow: 'txt2img',
        job_type: 'Text to Image',
        status: 'running',
        created_at: '2026-04-30T10:00:00Z',
        completed_at: null,
        event_count: 1,
        last_event: { type: 'job_submitted' },
        supported_controls: [],
        paused: false,
        result_path: null,
        prompt: 'Remain active while navigating',
        model: 'zit',
        runs: 1,
      },
    });

    await mountWorkspace(context);
    const MockEventSource = globalThis.EventSource as unknown as {
      instances: Array<{ emit: (type: string, data: unknown) => void; closeCalls: number }>;
      lastInstance: { emit: (type: string, data: unknown) => void; closeCalls: number };
    };
    const source = MockEventSource.lastInstance;
    const constructionCount = MockEventSource.instances.length;
    // A running job no longer blocks Generate; it adds to the queue.
    expect((target.querySelector('#ws-submit') as HTMLButtonElement | null)?.disabled).toBe(false);
    expect((target.querySelector('#ws-submit') as HTMLButtonElement | null)?.textContent).toContain('Add to queue');

    await unmount(app!);
    app = null;
    target.replaceChildren();

    await mountWorkspace(context);
    expect(MockEventSource.instances).toHaveLength(constructionCount);
    // A running job no longer blocks Generate; it adds to the queue.
    expect((target.querySelector('#ws-submit') as HTMLButtonElement | null)?.disabled).toBe(false);
    expect((target.querySelector('#ws-submit') as HTMLButtonElement | null)?.textContent).toContain('Add to queue');

    galleryApiMocks.getGallery.mockClear();
    source.emit('job_completed', { type: 'job_completed', job_id: 'job-remount', total_runs: 1, outputs: [] });
    await settle();

    expect(jobStore.current?.status).toBe('completed');
    expect(source.closeCalls).toBe(1);
    expect(galleryApiMocks.getGallery).toHaveBeenCalledOnce();
    expect((target.querySelector('#ws-submit') as HTMLButtonElement | null)?.disabled).toBe(false);
    expect((target.querySelector('#ws-submit') as HTMLButtonElement | null)?.textContent).not.toContain('Add to queue');
  });

  it('submits explicit false values for video toggles when they are switched off', async () => {
    draft.update('workflow', 'txt2vid');

    const context = makeContext();
    await mountWorkspace(context);

    // Set toggles to false AFTER context has loaded and hydrateFromContext has run,
    // which otherwise resets audio/lowMemory to the context defaults (true).
    // This simulates a user toggling the controls off after the workspace loads.
    draft.update('audio', false);
    draft.update('lowMemory', false);
    await settle();

    const form = target.querySelector('form');
    expect(form).not.toBeNull();
    form!.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
    await settle();

    expect(workspaceApiMocks.submitGenerate).toHaveBeenCalledTimes(1);
    const [submittedFormData] = workspaceApiMocks.submitGenerate.mock.calls[0] ?? [];
    expect(submittedFormData).toBeInstanceOf(FormData);
    expect(submittedFormData.get('audio')).toBe('false');
    expect(submittedFormData.get('low_memory')).toBe('false');
  });

  it('submits normalized prompt-file fields and omits inline prompt fields in file mode', async () => {
    promptFileApiMocks.inspectPromptFile.mockResolvedValue({
      path: '/server/prompts.yaml',
      options: [
        {
          id: 'portrait:0',
          set_name: 'portrait',
          source_index: 0,
          label: 'portrait #1 · first option',
          prompt_preview: 'first option',
          negative_preview: null,
        },
        {
          id: 'portrait:1',
          set_name: 'portrait',
          source_index: 1,
          label: 'portrait #2 · second option',
          prompt_preview: 'second option',
          negative_preview: 'muddy',
        },
      ],
    });

    const context = makeContext();
    await mountWorkspace(context);

    draft.update('prompt', 'stale inline prompt');
    draft.update('negativePrompt', 'stale negative');
    await settle();

    const promptSource = target.querySelector('[aria-label="Prompt source"] [data-value="file"]') as HTMLButtonElement | null;
    expect(promptSource).not.toBeNull();
    promptSource!.click();
    await settle();

    const submitButton = target.querySelector('#ws-submit') as HTMLButtonElement | null;
    expect(submitButton?.disabled).toBe(true);

    const pathInput = target.querySelector('#ws-prompts-file') as HTMLInputElement | null;
    expect(pathInput).not.toBeNull();
    pathInput!.value = '~/prompts.yaml';
    pathInput!.dispatchEvent(new Event('input', { bubbles: true }));
    pathInput!.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
    await settle();

    expect(pathInput!.value).toBe('/server/prompts.yaml');
    const hiddenPath = target.querySelector('input[name="prompts_file"]') as HTMLInputElement | null;
    expect(hiddenPath?.value).toBe('/server/prompts.yaml');

    await choosePrompts(target, ['portrait:0', 'portrait:1']);
    expect(submitButton?.disabled).toBe(false);
    expect(draft.state.promptFileOptionIds).toEqual(['portrait:0', 'portrait:1']);

    // The dialog clamps long prompts until expanded, and shows the negative prompt only when expanded.
    (target.querySelector('[data-action="choose-prompts"]') as HTMLButtonElement).click();
    await settle();
    expect(document.body.textContent).not.toContain('muddy');
    const showMore = document.querySelector('[aria-label="Show more for portrait #2"]') as HTMLButtonElement;
    expect(showMore.getAttribute('aria-expanded')).toBe('false');
    showMore.click();
    await settle();
    expect(showMore.getAttribute('aria-expanded')).toBe('true');
    expect(document.body.textContent).toContain('muddy');
    // Cancel keeps the confirmed selection.
    (document.querySelector('[data-testid="prompt-chooser"] input[value="portrait:0"]') as HTMLInputElement).click();
    (document.querySelector('[data-action="cancel-prompts"]') as HTMLButtonElement).click();
    await settle();
    expect(draft.state.promptFileOptionIds).toEqual(['portrait:0', 'portrait:1']);

    const form = target.querySelector('form');
    expect(form).not.toBeNull();
    form!.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
    await settle();

    expect(workspaceApiMocks.submitGenerate).toHaveBeenCalledTimes(1);
    const [submittedFormData] = workspaceApiMocks.submitGenerate.mock.calls[0] ?? [];
    expect(submittedFormData).toBeInstanceOf(FormData);
    expect(submittedFormData.get('prompt_source')).toBe('file');
    expect(submittedFormData.get('prompts_file')).toBe('/server/prompts.yaml');
    expect(submittedFormData.getAll('prompt_option_id')).toEqual(['portrait:0', 'portrait:1']);
    // The compose summary lists the selected prompts once each.
    expect(target.textContent?.match(/first option/g)).toHaveLength(1);
    expect(target.textContent?.match(/second option/g)).toHaveLength(1);
    expect(target.textContent).not.toContain('muddy');
    expect(target.textContent).not.toContain('Prompt Preview');
    expect(submittedFormData.has('prompt')).toBe(false);
    expect(submittedFormData.has('negative_prompt')).toBe(false);
  });

  it('selects all prompts, unchecks individual prompts, and disables generation when cleared', async () => {
    promptFileApiMocks.inspectPromptFile.mockResolvedValue({
      path: '/server/prompts.yaml',
      options: [0, 1, 2].map((index) => ({
        id: `portrait:${index}`, set_name: 'portrait', source_index: index,
        label: `Prompt ${index + 1}`, prompt_preview: `Prompt ${index + 1}`, negative_preview: null,
      })),
    });
    await mountWorkspace(makeContext());
    draft.update('promptSource', 'file');
    draft.update('promptFilePath', '/server/prompts.yaml');
    await settle();
    const chooser = (): HTMLElement => document.querySelector('[data-testid="prompt-chooser"]') as HTMLElement;
    const chooserButton = (action: string) => chooser().querySelector(`[data-action="${action}"]`) as HTMLButtonElement;
    const confirm = async (): Promise<void> => {
      (document.querySelector('[data-action="confirm-prompts"]') as HTMLButtonElement).click();
      await settle();
    };
    const open = async (): Promise<void> => {
      (target.querySelector('[data-action="choose-prompts"]') as HTMLButtonElement).click();
      await settle();
    };

    await open();
    chooserButton('select-all').click();
    await settle();
    expect((document.querySelector('[data-action="confirm-prompts"]') as HTMLElement).dataset.count).toBe('3');
    await confirm();
    expect(draft.state.promptFileOptionIds).toEqual(['portrait:0', 'portrait:1', 'portrait:2']);
    expect(target.querySelectorAll('input[name="prompt_option_id"]')).toHaveLength(3);

    await open();
    (chooser().querySelector('input[value="portrait:1"]') as HTMLInputElement).click();
    await confirm();
    expect(draft.state.promptFileOptionIds).toEqual(['portrait:0', 'portrait:2']);
    expect(new FormData(target.querySelector('form')!).getAll('prompt_option_id')).toEqual(['portrait:0', 'portrait:2']);

    await open();
    chooserButton('select-none').click();
    await confirm();
    expect(draft.state.promptFileOptionIds).toEqual([]);
    expect(target.querySelectorAll('input[name="prompt_option_id"]')).toHaveLength(0);
    expect((target.querySelector('#ws-submit') as HTMLButtonElement).disabled).toBe(true);
  });

  it('copies a prompts-file command from the loaded file, reporting only a partial prompt choice', async () => {
    const writeText = mockClipboard();
    promptFileApiMocks.inspectPromptFile.mockResolvedValue({
      path: '/server/prompts.yaml',
      options: [0, 1].map((index) => ({
        id: `portrait:${index}`, set_name: 'portrait', source_index: index,
        label: `Prompt ${index + 1}`, prompt_preview: `Prompt ${index + 1}`, negative_preview: null,
      })),
    });
    await mountWorkspace(makeContext());
    draft.update('promptSource', 'file');
    draft.update('promptFilePath', '/server/prompts.yaml');
    await settle();
    const inspections = promptFileApiMocks.inspectPromptFile.mock.calls.length;
    // Each copy starts from an empty toast queue, so toasts[0] is the toast it raised.
    const copy = async (): Promise<void> => {
      toasts.splice(0, toasts.length);
      (target.querySelector('#ws-copy-cli') as HTMLButtonElement).click();
      await settle();
    };

    // Nothing chosen: Generate would refuse, so nothing is copied.
    draft.update('promptFileOptionIds', []);
    await settle();
    await copy();
    expect(writeText).not.toHaveBeenCalled();
    expect(toasts[0]?.type).toBe('error');

    draft.update('promptFileOptionIds', ['portrait:0', 'portrait:1']);
    await settle();
    await copy();
    expect(writeText).toHaveBeenLastCalledWith(expect.stringContaining('--prompts-file /server/prompts.yaml'));
    expect(toasts[0]?.type).toBe('success');

    draft.update('promptFileOptionIds', ['portrait:0']);
    await settle();
    await copy();
    expect(toasts[0]?.type).toBe('info');
    // The count comes from the file the field already loaded, so the clipboard write stays inside the click.
    expect(promptFileApiMocks.inspectPromptFile).toHaveBeenCalledTimes(inspections);
  });

  it('filters the prompt chooser by prompt text', async () => {
    promptFileApiMocks.inspectPromptFile.mockResolvedValue({
      path: '/server/prompts.yaml',
      options: ['a red fox', 'a blue whale', 'a red barn'].map((text, index) => ({
        id: `scene:${index}`, set_name: 'scene', source_index: index, label: text, prompt_preview: text, negative_preview: null,
      })),
    });
    await mountWorkspace(makeContext());
    draft.update('promptSource', 'file');
    draft.update('promptFilePath', '/server/prompts.yaml');
    await settle();

    (target.querySelector('[data-action="choose-prompts"]') as HTMLButtonElement).click();
    await settle();
    const filter = document.querySelector('[data-testid="prompt-chooser"] input[type="search"]') as HTMLInputElement;
    filter.value = 'red';
    filter.dispatchEvent(new Event('input', { bubbles: true }));
    await settle();
    expect(Array.from(document.querySelectorAll<HTMLInputElement>('[data-testid="prompt-chooser"] input[type="checkbox"]')).map((box) => box.value)).toEqual(['scene:0', 'scene:2']);

    // All selects only the matching prompts.
    (document.querySelector('[data-testid="prompt-chooser"] [data-action="select-all"]') as HTMLButtonElement).click();
    await settle();
    (document.querySelector('[data-action="confirm-prompts"]') as HTMLButtonElement).click();
    await settle();
    expect(draft.state.promptFileOptionIds).toEqual(['scene:0', 'scene:2']);
  });

  it('opens the prompt file on the Prompts page from Edit', async () => {
    promptFileApiMocks.inspectPromptFile.mockResolvedValueOnce({ path: '/server/prompts.yaml', options: [] });

    await mountWorkspace(makeContext());

    const promptSource = target.querySelector('[aria-label="Prompt source"] [data-value="file"]') as HTMLButtonElement | null;
    promptSource!.click();
    await settle();

    const pathInput = target.querySelector('#ws-prompts-file') as HTMLInputElement | null;
    pathInput!.value = '~/prompts.yaml';
    pathInput!.dispatchEvent(new Event('input', { bubbles: true }));
    pathInput!.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
    await settle();

    (target.querySelector('[data-action="edit-prompts"]') as HTMLButtonElement).click();
    await settle();

    expect(router.page).toBe('prompts');
    expect(router.params).toEqual({ path: '/server/prompts.yaml' });
  });

  it('invalidates prompt-file options when the visible path is manually changed, and keeps a rejected path visible', async () => {
    promptFileApiMocks.inspectPromptFile.mockResolvedValueOnce({
      path: '/server/prompts.yaml',
      options: [
        {
          id: 'portrait:0',
          set_name: 'portrait',
          source_index: 0,
          label: 'portrait #1 · first option',
          prompt_preview: 'first option',
          negative_preview: null,
        },
      ],
    });

    const context = makeContext();
    await mountWorkspace(context);

    const promptSource = target.querySelector('[aria-label="Prompt source"] [data-value="file"]') as HTMLButtonElement | null;
    promptSource!.click();
    await settle();

    const pathInput = target.querySelector('#ws-prompts-file') as HTMLInputElement | null;
    pathInput!.value = '~/prompts.yaml';
    pathInput!.dispatchEvent(new Event('input', { bubbles: true }));
    pathInput!.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
    await settle();

    await choosePrompts(target, ['portrait:0']);
    expect((target.querySelector('#ws-submit') as HTMLButtonElement | null)?.disabled).toBe(false);

    pathInput!.value = '/server/other-prompts.yaml';
    pathInput!.dispatchEvent(new Event('input', { bubbles: true }));
    await settle();

    const hiddenPath = target.querySelector('input[name="prompts_file"]') as HTMLInputElement | null;
    expect(hiddenPath?.value).toBe('/server/other-prompts.yaml');
    expect(target.querySelectorAll('input[name="prompt_option_id"]')).toHaveLength(0);
    expect((target.querySelector('#ws-submit') as HTMLButtonElement | null)?.disabled).toBe(true);

    promptFileApiMocks.inspectPromptFile.mockRejectedValueOnce(new Error('POST /api/prompt-files/inspect → 422: missing file'));
    pathInput!.value = '/missing/prompts.yaml';
    pathInput!.dispatchEvent(new Event('input', { bubbles: true }));
    pathInput!.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
    await settle();

    expect(pathInput!.value).toBe('/missing/prompts.yaml');
    expect(hiddenPath?.value).toBe('/missing/prompts.yaml');
    expect(target.querySelectorAll('input[name="prompt_option_id"]')).toHaveLength(0);
    expect((target.querySelector('#ws-submit') as HTMLButtonElement | null)?.disabled).toBe(true);
  });

  it('queues a prompt sent from the Prompts page without changing the prompt selection', async () => {
    requestPromptRun({ path: '/server/prompts.yaml', optionId: 'portrait:1' });

    await mountWorkspace(makeContext());
    await settle();

    expect(workspaceApiMocks.submitGenerate).toHaveBeenCalledTimes(1);
    const form = workspaceApiMocks.submitGenerate.mock.calls[0][0];
    expect(form.get('prompt_source')).toBe('file');
    expect(form.get('prompts_file')).toBe('/server/prompts.yaml');
    expect(form.getAll('prompt_option_id')).toEqual(['portrait:1']);
    expect(form.has('prompt')).toBe(false);
    expect(draft.state.promptSource).toBe('inline');
    expect(draft.state.promptFileOptionIds).toEqual([]);
  });

  it('revokes the previous reference image blob URL when the file changes', async () => {
    const revokeObjectURL = vi.fn();
    const createObjectURL = vi
      .fn<(file: Blob | MediaSource) => string>()
      .mockReturnValueOnce('blob:first-image')
      .mockReturnValueOnce('blob:second-image');
    Object.defineProperty(URL, 'createObjectURL', { configurable: true, value: createObjectURL });
    Object.defineProperty(URL, 'revokeObjectURL', { configurable: true, value: revokeObjectURL });

    const context = makeContext();
    draft.update('workflow', 'img2img');
    draft.hydrateFromContext(context, 'zit');

    const firstFile = new File(['first-image'], 'first.png', { type: 'image/png' });
    const secondFile = new File(['second-image'], 'second.png', { type: 'image/png' });
    const legacyApp = createClassComponent({
      component: ControlsSidebar,
      target,
      props: { context, busy: false, imageFile: firstFile, onImageFileChange: vi.fn() },
    });
    await settle();

    legacyApp.$set({ imageFile: secondFile });
    await settle();

    expect(createObjectURL).toHaveBeenNthCalledWith(1, firstFile);
    expect(createObjectURL).toHaveBeenNthCalledWith(2, secondFile);
    expect(revokeObjectURL).toHaveBeenCalledWith('blob:first-image');

    legacyApp.$destroy();
  });

  describe('save to prompt file', () => {
    const saveButton = () => target.querySelector('#ws-save-prompt') as HTMLButtonElement | null;

    async function openSaveMenu(): Promise<void> {
      saveButton()!.click();
      await settle();
    }

    async function chooseTarget(path: string): Promise<void> {
      (document.querySelector(`[role="menu"] [data-action="target-${path}"]`) as HTMLButtonElement).click();
      await settle();
    }

    beforeEach(() => {
      promptFileApiMocks.appendPromptEntry.mockImplementation(async (data: { path: string }) => ({
        path: data.path, revision: 'r1', entry_id: 's0.e1', set_name: 'prompts',
      }));
    });

    it('appends the typed prompt to a recent file, remembers it and offers to open it in Prompts', async () => {
      rememberRecent('/p/a.yaml');
      draft.update('prompt', 'a fox in snow');
      await mountWorkspace(makeContext());

      await openSaveMenu();
      await chooseTarget('/p/a.yaml');

      expect(promptFileApiMocks.appendPromptEntry).toHaveBeenCalledWith({ path: '/p/a.yaml', prompt: 'a fox in snow' });
      expect(recentFiles()[0]).toBe('/p/a.yaml');
      expect(lastSaveTarget()).toBe('/p/a.yaml');
      const toast = toasts.find((item) => item.type === 'success');
      expect(typeof toast?.action?.run).toBe('function');
      toast!.action!.run();
      expect(router.page).toBe('prompts');
      expect(router.params.path).toBe('/p/a.yaml');
    });

    it('saves the Enhanced text when the Enhanced tab is in view', async () => {
      rememberRecent('/p/a.yaml');
      draft.update('prompt', 'a fox');
      draft.update('enhancedPrompt', 'A red fox in deep snow.');
      await mountWorkspace(withEnhancer(makeContext()));
      (target.querySelector('#ws-tab-enhanced') as HTMLButtonElement).click();
      await settle();

      await openSaveMenu();
      await chooseTarget('/p/a.yaml');

      expect(promptFileApiMocks.appendPromptEntry).toHaveBeenCalledWith({ path: '/p/a.yaml', prompt: 'A red fox in deep snow.' });
    });

    it('is disabled on the Enhanced tab while nothing is enhanced', async () => {
      draft.update('prompt', 'a fox');
      await mountWorkspace(withEnhancer(makeContext()));
      (target.querySelector('#ws-tab-enhanced') as HTMLButtonElement).click();
      await settle();

      expect(saveButton()!.disabled).toBe(true);
    });

    it('is absent in the Prompt file source', async () => {
      await mountWorkspace(makeContext());
      expect(saveButton()).not.toBeNull();

      draft.update('promptSource', 'file');
      await settle();
      expect(saveButton()).toBeNull();
    });

    it('is absent in JSON prompt mode', async () => {
      const ideogramDefaults = makeIdeogramDefaults();
      draft.update('jsonPromptEnabled', true);
      await mountWorkspace(makeContext({
        image_models: [{ id: 'ideo', label: 'ideo', type: 'image' }],
        defaults: ideogramDefaults,
        current_image_model: 'ideo',
        image_model_defaults: { ideo: ideogramDefaults },
      }));

      expect(target.querySelector('#ws-json-prompt-toggle')).not.toBeNull();
      expect(target.querySelector('textarea[name="json_prompt"]')).not.toBeNull();
      expect(saveButton()).toBeNull();
    });

    it('saves to a file chosen with Browse…', async () => {
      draft.update('prompt', 'a fox');
      browseApiMocks.browsePlaces.mockResolvedValue(placesFor('prompt_file'));
      browseApiMocks.listDirectory.mockResolvedValue(listingWith('/p/b.yaml'));
      await mountWorkspace(makeContext());

      await openSaveMenu();
      (document.querySelector('[role="menu"] [data-action="browse"]') as HTMLButtonElement).click();
      await settle();
      chooseInBrowser('b.yaml');
      await settle();

      expect(promptFileApiMocks.appendPromptEntry).toHaveBeenCalledWith({ path: '/p/b.yaml', prompt: 'a fox' });
      expect(lastSaveTarget()).toBe('/p/b.yaml');
    });

    it('saves the enhanced action of a video with your scene when only the action was enhanced', async () => {
      rememberRecent('/p/a.yaml');
      draft.update('workflow', 'txt2vid');
      draft.update('scene', 'A rainy street.');
      draft.update('prompt', 'She walks.');
      draft.update('enhancedPrompt', 'She walks slowly under a red umbrella.');
      await mountWorkspace(withVideoEnhancer(makeContext()));
      (target.querySelector('#ws-tab-enhanced') as HTMLButtonElement).click();
      await settle();

      await openSaveMenu();
      await chooseTarget('/p/a.yaml');

      expect(promptFileApiMocks.appendPromptEntry).toHaveBeenCalledWith({
        path: '/p/a.yaml', scene: 'A rainy street.', prompt: 'She walks slowly under a red umbrella.',
      });
    });

    it('leaves the negative prompt out for a workflow that does not show it', async () => {
      rememberRecent('/p/a.yaml');
      draft.update('model', 'flux-lite');
      draft.update('prompt', 'a fox');
      await mountWorkspace(makeContext());
      draft.update('negativePrompt', 'blurry');
      await settle();
      expect(target.querySelector('#ws-negative-prompt')).toBeNull();

      await openSaveMenu();
      await chooseTarget('/p/a.yaml');

      expect(promptFileApiMocks.appendPromptEntry).toHaveBeenCalledWith({ path: '/p/a.yaml', prompt: 'a fox' });
    });

    it('creates a new file from New file… and then appends the prompt to it', async () => {
      rememberRecent('/p/a.yaml');
      draft.update('prompt', 'a fox');
      promptFileApiMocks.createPromptFile.mockResolvedValue({ path: '/p/new.yaml' });
      await mountWorkspace(makeContext());

      await openSaveMenu();
      (document.querySelector('[role="menu"] [data-action="new"]') as HTMLButtonElement).click();
      await settle();
      const name = document.querySelector('#prompts-new-name') as HTMLInputElement;
      name.value = 'new';
      name.dispatchEvent(new Event('input', { bubbles: true }));
      await settle();
      (Array.from(document.querySelectorAll('button')).find((b) => b.textContent?.trim() === 'Create') as HTMLButtonElement).click();
      await settle();

      expect(promptFileApiMocks.createPromptFile).toHaveBeenCalledWith('/p/', 'new');
      expect(promptFileApiMocks.appendPromptEntry).toHaveBeenCalledWith({ path: '/p/new.yaml', prompt: 'a fox' });
      expect(promptFileApiMocks.createPromptFile.mock.invocationCallOrder[0]).toBeLessThan(promptFileApiMocks.appendPromptEntry.mock.invocationCallOrder[0]);
      expect(lastSaveTarget()).toBe('/p/new.yaml');
      expect(recentFiles()[0]).toBe('/p/new.yaml');
    });

    it('toasts the server error and remembers nothing when the save is rejected', async () => {
      rememberRecent('/p/a.yaml');
      draft.update('prompt', 'a fox $bill');
      const detail = 'server-detail-for-rejected-save';
      promptFileApiMocks.appendPromptEntry.mockRejectedValue(new ApiError('rejected', 422, detail));
      await mountWorkspace(makeContext());

      await openSaveMenu();
      await chooseTarget('/p/a.yaml');

      const toast = toasts.find((item) => item.type === 'error');
      expect(toast).toBeDefined();
      expect(toast?.message).toBe(detail);
      expect(toasts.some((item) => item.type === 'success')).toBe(false);
      expect(lastSaveTarget()).toBeNull();
      expect(saveButton()!.disabled).toBe(false);
    });
  });
});
