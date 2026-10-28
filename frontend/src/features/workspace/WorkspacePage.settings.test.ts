// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, mount, unmount } from '../../../node_modules/svelte/src/index-client.js';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { ApiError } from '$lib/api/client';
import { draft } from '$lib/state/draft.svelte';
import { downloadStore } from '$lib/state/downloads.svelte';
import { requestJobSettings } from '$lib/state/pendingJobSettings.svelte';
import { toasts } from '$lib/state/toasts.svelte';
import type { SettingsPreset, JobContext, WorkspaceContext, GalleryAsset } from '$lib/types';

import WorkspacePage from './WorkspacePage.svelte';
import { placesFor } from '../../test-utils/pathBrowser';
import { pickOption } from '../../test-utils/select';
import { makeAsset, makeContext, makeGalleryPage, makeImageDefaults, makeVideoDefaults, withEnhancer } from '../../test-utils/workspace';
import { answerConfirm, createHost, fakeDebounceTimers, mountWorkspacePage, pressKey, resetWorkspaceTest, settle, waitDebounce } from '../../test-utils/workspaceHarness';
import { browseApiMocks, configApiMocks, enhanceApiMocks, galleryApiMocks, layoutProbe, modelApiMocks, presetApiMocks, videoMemoryApiMocks, workspaceApiMocks } from '../../test-utils/workspaceMocks';

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

describe('WorkspacePage clip memory note', () => {
  const fit = (status: string, required_gb = 21) => ({ memory_fit: { status, required_gb, budget_gb: 24 }, width: 704, height: 448, frames: 49 });
  const note = () => target.querySelector('[data-testid="clip-memory-note"]');
  const lastRequest = () => videoMemoryApiMocks.checkVideoMemoryFit.mock.calls.at(-1)?.[0];

  async function mountVideo(workflow: 'txt2vid' | 'img2vid' | 'extend' = 'txt2vid'): Promise<void> {
    const context = makeContext();
    context.workflow_contract.values = [...context.workflow_contract.values, 'extend'];
    workspaceApiMocks.getWorkspaceCoreContext.mockResolvedValue(context);
    draft.update('workflow', workflow);
    fakeDebounceTimers();
    app = flushSync(() => mount(WorkspacePage, { target }));
    await settle();
    await waitDebounce();
  }


  it('notes a tight clip under Duration without blocking Generate', async () => {
    videoMemoryApiMocks.checkVideoMemoryFit.mockResolvedValue(fit('tight'));

    await mountVideo();

    expect(note()?.getAttribute('data-status')).toBe('tight');
    expect(note()?.closest('[aria-live="polite"]')).not.toBeNull();
    expect((target.querySelector('#ws-submit') as HTMLButtonElement).disabled).toBe(false);
  });

  it('asks again when the length changes, and follows the answer', async () => {
    videoMemoryApiMocks.checkVideoMemoryFit.mockResolvedValue(fit('fits'));
    await mountVideo();
    expect(note()).toBeNull();
    expect(lastRequest()).toEqual({ model: 'ltx-8', ratio: expect.any(String), size: expect.any(String), frames: draft.state.frameCount, low_memory: draft.state.lowMemory, upscale: null });
    videoMemoryApiMocks.checkVideoMemoryFit.mockResolvedValue(fit('tight'));

    draft.update('frameCount', 241);
    await settle();
    expect(videoMemoryApiMocks.checkVideoMemoryFit).toHaveBeenCalledTimes(1);
    await waitDebounce();

    expect(lastRequest()).toMatchObject({ frames: 241 });
    expect(note()?.getAttribute('data-status')).toBe('tight');
  });

  it('drops the note at once when the workflow leaves video', async () => {
    videoMemoryApiMocks.checkVideoMemoryFit.mockResolvedValue(fit('tight'));
    await mountVideo();
    expect(note()).not.toBeNull();

    draft.update('workflow', 'txt2img');
    await settle();

    expect(note()).toBeNull();
  });


  describe('first-run dialog', () => {
    const firstRunContext = (overrides: Partial<WorkspaceContext> = {}) => makeContext({
      image_models: [
        { id: 'zit', label: 'Z-Image Turbo', type: 'image', downloaded: false, memory_fit: null, download_size_label: '33 GB' },
        { id: 'klein4b', label: 'FLUX.2 Klein 4B', type: 'image', downloaded: false, memory_fit: null, download_size_label: '16 GB' },
      ],
      current_image_model: 'zit',
      image_model_defaults: { zit: makeImageDefaults({}), klein4b: makeImageDefaults({ steps: 4 }) },
      first_run: { skipped: false, recommended_model: 'klein4b' },
      ...overrides,
    });
    const dialog = () => document.querySelector('[data-testid="first-run-dialog"]');
    const press = async (action: string) => {
      (document.querySelector(`[data-action="${action}"]`) as HTMLButtonElement).click();
      await settle();
    };

    it('opens with the recommended model chosen when nothing is downloaded', async () => {
      await mountWorkspace(firstRunContext());

      expect(dialog()).not.toBeNull();
      expect((document.querySelector('[data-testid="first-run-model-klein4b"]') as HTMLInputElement).checked).toBe(true);
    });

    it('stays closed while an image model is already downloading', async () => {
      modelApiMocks.getDownloads.mockResolvedValue({ items: [{ id: 'd1', kind: 'model', name: 'klein4b', label: 'FLUX.2 Klein 4B', state: 'running', message: null }] });

      await mountWorkspace(firstRunContext());

      expect(dialog()).toBeNull();
    });

    it('queues the chosen model, saves it as the default, selects it and closes', async () => {
      const request = vi.spyOn(downloadStore, 'request').mockResolvedValue('queued');
      await mountWorkspace(firstRunContext());

      (document.querySelector('[data-testid="first-run-model-zit"]') as HTMLInputElement).click();
      await settle();
      await press('first-run-download');

      expect(request).toHaveBeenCalledWith('model', 'zit', 'Z-Image Turbo');
      expect(configApiMocks.updateConfig).toHaveBeenCalledWith({ 'ui.default_models.image': 'zit' });
      expect(draft.state.model).toBe('zit');
      expect(dialog()).toBeNull();
    });

    it('selects the downloaded model in the draft', async () => {
      vi.spyOn(downloadStore, 'request').mockResolvedValue('present');
      await mountWorkspace(firstRunContext());

      await press('first-run-download');

      expect(configApiMocks.updateConfig).toHaveBeenCalledWith({ 'ui.default_models.image': 'klein4b' });
      expect(draft.state.model).toBe('klein4b');
    });

    it.each(['declined', 'failed'] as const)('stays open and saves nothing when the download is %s', async (result) => {
      vi.spyOn(downloadStore, 'request').mockResolvedValue(result);
      await mountWorkspace(firstRunContext());

      await press('first-run-download');

      expect(dialog()).not.toBeNull();
      expect(configApiMocks.updateConfig).not.toHaveBeenCalled();
    });

    it('remembers a skip on the server and closes', async () => {
      await mountWorkspace(firstRunContext());

      await press('first-run-skip');

      expect(workspaceApiMocks.skipFirstRun).toHaveBeenCalledOnce();
      expect(dialog()).toBeNull();
    });

    it('does not reopen when the queued download fails and the lane empties', async () => {
      vi.spyOn(downloadStore, 'request').mockResolvedValue('queued');
      await mountWorkspace(firstRunContext());
      await press('first-run-download');
      expect(dialog()).toBeNull();

      // The download ends without a model on disk: the lane is empty and every model is still missing.
      modelApiMocks.getDownloads.mockResolvedValue({ items: [] });
      await downloadStore.refresh();
      await settle();

      expect(dialog()).toBeNull();
    });

    it('still closes, and says so, when remembering the skip fails', async () => {
      workspaceApiMocks.skipFirstRun.mockRejectedValueOnce(new Error('disk full'));
      await mountWorkspace(firstRunContext());

      await press('first-run-skip');

      expect(dialog()).toBeNull();
      expect(toasts.some((toast) => toast.type === 'error')).toBe(true);
    });

    it('selects the model and says so when saving the default fails', async () => {
      vi.spyOn(downloadStore, 'request').mockResolvedValue('queued');
      configApiMocks.updateConfig.mockRejectedValueOnce(new Error('read-only config'));
      await mountWorkspace(firstRunContext());

      await press('first-run-download');

      expect(dialog()).toBeNull();
      expect(draft.state.model).toBe('klein4b');
      expect(toasts.some((toast) => toast.type === 'error')).toBe(true);
    });

    it('closes on Escape without remembering anything, and does not reopen after the page is mounted again', async () => {
      const context = firstRunContext();
      await mountWorkspace(context);

      document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
      await settle();
      expect(dialog()).toBeNull();
      expect(workspaceApiMocks.skipFirstRun).not.toHaveBeenCalled();

      await unmount(app!);
      app = null;
      await mountWorkspace(context);

      expect(dialog()).toBeNull();
    });
  });
});

describe('WorkspacePage settings presets', () => {
  const IMAGE_SETTINGS: SettingsPreset['settings'] = {
    quantize: null, lora: 'style-a:0.8', ratio: '16:9', size: 'l', width: null, height: null, runs: 2, steps: 12, guidance: 3.5, image_strength: 0.4, scheduler: 'beta', first_sigma: null,
    sharpen_enabled: true, sharpen_amount: null, contrast_enabled: false, contrast_amount: null, saturation_enabled: false, saturation_amount: null,
    upscale: null, upscale_denoise: null, upscale_steps: null, upscale_guidance: null, upscale_sharpen: null, review: null,
  };
  const LOOK: SettingsPreset = { slug: 'look', name: 'Look', mode: 'image', model: 'flux-lite', settings: IMAGE_SETTINGS };
  const PORTRAIT: SettingsPreset = { slug: 'portrait', name: 'Portrait', mode: 'image', model: null, settings: { ...IMAGE_SETTINGS, steps: 20 } };
  const CLIP: SettingsPreset = {
    slug: 'clip', name: 'Clip', mode: 'video', model: 'ltx-8',
    settings: { lora: '', ratio: '16:9', size: 'm', width: null, height: null, runs: 1, steps: 8, frames: 49, audio: false, low_memory: true, video_upscale_factor: null, review: null },
  };

  function presetContext(): WorkspaceContext {
    const context = makeContext();
    context.loras = [{ name: 'style-a', path: '/loras/style-a.safetensors' }, { name: 'detail-b', path: '/loras/detail-b.safetensors' }];
    return context;
  }

  const presetButton = () => target.querySelector('[data-testid="preset-button"]') as HTMLButtonElement;
  const presetMenu = () => document.querySelector('[role="menu"][aria-label="Preset"]') as HTMLElement | null;
  const menuActions = () => Array.from(presetMenu()?.querySelectorAll('[data-action]') ?? []).map((el) => el.getAttribute('data-action'));
  const menuItem = (action: string) => presetMenu()?.querySelector(`[data-action="${action}"]`) as HTMLButtonElement;

  async function openMenu(): Promise<void> {
    flushSync(() => presetButton().click());
    await settle();
  }

  async function choose(action: string): Promise<void> {
    await openMenu();
    flushSync(() => menuItem(action).click());
    await settle();
  }

  /** Mount with presets saved; the page reads them when it mounts. */
  async function mountWithPresets(presets: SettingsPreset[] = [LOOK, PORTRAIT, CLIP], context: WorkspaceContext = presetContext()): Promise<void> {
    presetApiMocks.listPresets.mockResolvedValue(presets);
    await mountWorkspace(context);
    await settle();
  }

  function typeName(value: string): void {
    const input = document.querySelector('#preset-name') as HTMLInputElement;
    input.value = value;
    input.dispatchEvent(new Event('input', { bubbles: true }));
    flushSync();
  }

  const nameDialogError = () => document.querySelector('[role="dialog"] [role="alert"]');
  const confirmName = () => document.querySelector('[data-action="confirm-preset-name"]') as HTMLButtonElement;
  const dot = () => presetButton().dataset.changed === 'true';

  beforeEach(() => {
    workspaceApiMocks.submitGenerate.mockResolvedValue({ job_id: 'job-1', workflow: 'txt2img', prompt: 'p', model: 'zit', runs: 1, created_at: '2026-04-23T10:00:00Z' });
    window.location.hash = '#/workspace';
    window.history.replaceState({}, '', '/');
  });

  describe('the toolbar control', () => {
    it('comes first, before the Model select, and says none until a preset is applied', async () => {
      await mountWithPresets();

      expect(presetButton()).not.toBeNull();
      expect(presetButton().compareDocumentPosition(target.querySelector('#ws-model') as Node) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
      expect(presetButton().getAttribute('aria-haspopup')).toBe('menu');
      expect(presetButton().dataset.applied).toBe('');
      expect(dot()).toBe(false);
    });

    it('is disabled until the workspace has loaded', async () => {
      workspaceApiMocks.getWorkspaceCoreContext.mockImplementation(() => new Promise<WorkspaceContext>(() => undefined));
      app = flushSync(() => mount(WorkspacePage, { target }));
      await settle();

      expect(presetButton().disabled).toBe(true);
    });

    it('names the applied preset and shows the changed dot only while the form differs from it', async () => {
      await mountWithPresets();
      await choose('preset:look');

      expect(presetButton().dataset.applied).toBe('look');
      expect(dot()).toBe(false);
      draft.update('steps', 99);
      flushSync();
      expect(dot()).toBe(true);
      draft.update('steps', 12);
      flushSync();
      expect(dot()).toBe(false);
    });
  });

  describe('the menu', () => {
    it('lists the current mode\'s presets with only Save as when none is applied', async () => {
      await mountWithPresets();
      await openMenu();

      expect(menuActions()).toEqual(['preset:look', 'preset:portrait', 'preset-save-as']);
      expect(presetMenu()?.querySelectorAll('[role="menuitemradio"][aria-checked="true"]')).toHaveLength(0);
    });

    it('offers only Save as when the mode has no presets, and lists video presets in a video workflow', async () => {
      await mountWithPresets([CLIP], presetContext());
      await openMenu();
      expect(menuActions()).toEqual(['preset-save-as']);

      await unmount(app!);
      app = null;
      document.body.innerHTML = '';
      target = document.createElement('div');
      document.body.appendChild(target);
      draft.update('workflow', 'txt2vid');
      await mountWithPresets([CLIP, LOOK]);
      await openMenu();
      expect(menuActions()).toEqual(['preset:clip', 'preset-save-as']);
    });

    it('checks the applied preset and adds Rename and Delete, but Update only once the form differs', async () => {
      await mountWithPresets();
      await choose('preset:look');
      await openMenu();

      expect(menuActions()).toEqual(['preset:look', 'preset:portrait', 'preset-save-as', 'preset-rename', 'preset-delete']);
      expect(menuItem('preset:look').getAttribute('aria-checked')).toBe('true');
      expect(menuItem('preset:portrait').getAttribute('aria-checked')).toBe('false');

      draft.update('steps', 30);
      flushSync();
      await settle();
      expect(menuActions()).toContain('preset-update');
      expect(menuActions().indexOf('preset-update')).toBeLessThan(menuActions().indexOf('preset-save-as'));
    });

    it('reads the list again each time it opens', async () => {
      await mountWithPresets();
      const reads = presetApiMocks.listPresets.mock.calls.length;
      await openMenu();
      flushSync(() => document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true })));
      await openMenu();

      expect(presetApiMocks.listPresets.mock.calls.length).toBeGreaterThanOrEqual(reads + 2);
    });
  });

  describe('applying a preset', () => {
    it('sets the preset\'s settings and model, and keeps the prompt and seed', async () => {
      draft.update('prompt', 'A lighthouse at dusk');
      draft.update('seed', 77);
      await mountWithPresets();

      await choose('preset:look');

      expect(draft.state).toMatchObject({ model: 'flux-lite', steps: 12, runs: 2, referenceImageStrength: 0.4, ratio: '16:9', size: 'l', loraString: 'style-a:0.8', prompt: 'A lighthouse at dusk', seed: 77 });
      expect(draft.state.appliedPreset).toBe('look');
      expect((target.querySelector('#ws-prompt') as HTMLTextAreaElement).value).toBe('A lighthouse at dusk');
    });

    it('forgets the applied preset when a job\'s settings are loaded', async () => {
      draft.update('prompt', 'A lighthouse');
      await mountWithPresets();
      await choose('preset:look');
      expect(draft.state.appliedPreset).toBe('look');

      requestJobSettings({ mode: 'image', workflow: 'txt2img', model: 'zit', prompt_source: 'inline', prompt: 'from the job log', steps: '7' });
      await settle();

      expect(draft.state.appliedPreset).toBeNull();
      expect(presetButton().dataset.applied).toBe('');
    });
  });

  describe('Save as', () => {
    it('saves the form under the typed name, applies it and closes', async () => {
      await mountWithPresets();
      draft.update('steps', 14);
      flushSync();
      presetApiMocks.createPreset.mockImplementation(async (body) => ({ slug: 'my-new-look', ...(body as Omit<SettingsPreset, 'slug'>) }));

      await choose('preset-save-as');
      typeName('  My new look ');
      flushSync(() => confirmName().click());
      await settle();

      expect(presetApiMocks.createPreset).toHaveBeenCalledTimes(1);
      const [body] = presetApiMocks.createPreset.mock.calls[0] as [SettingsPreset];
      expect(body).toMatchObject({ name: 'My new look', mode: 'image', model: 'zit', settings: { steps: 14 } });
      expect(Object.keys(body.settings)).toHaveLength(24);
      expect(draft.state.appliedPreset).toBe('my-new-look');
      expect(document.querySelector('[role="dialog"]')).toBeNull();
      expect(presetButton().dataset.applied).toBe('my-new-look');
    });

    it('shows the server\'s reason in the dialog and keeps it open', async () => {
      await mountWithPresets();
      presetApiMocks.createPreset.mockRejectedValue(new ApiError('POST 409', 409, 'That name belongs to a video preset.'));
      await choose('preset-save-as');
      typeName('Fresh');

      flushSync(() => confirmName().click());
      await settle();

      expect(nameDialogError()?.textContent).toContain('That name belongs to a video preset.');
      expect(draft.state.appliedPreset).toBeNull();
    });
  });

  describe('Update', () => {
    it('writes the form back under the applied preset\'s slug', async () => {
      await mountWithPresets();
      await choose('preset:look');
      draft.update('steps', 30);
      flushSync();
      presetApiMocks.updatePreset.mockImplementation(async (mode, slug, body) => ({ slug, mode, ...(body as Omit<SettingsPreset, 'slug' | 'mode'>) }));

      await choose('preset-update');

      const [mode, slug, body] = presetApiMocks.updatePreset.mock.calls[0] as [string, string, SettingsPreset];
      expect([mode, slug]).toEqual(['image', 'look']);
      expect(body).toMatchObject({ name: 'Look', model: 'flux-lite', settings: { steps: 30 } });
      expect(dot()).toBe(false);
    });
  });

  describe('Rename', () => {
    it('starts from the current name and moves the applied slug to the returned one', async () => {
      await mountWithPresets();
      await choose('preset:look');
      presetApiMocks.renamePreset.mockResolvedValue({ ...LOOK, slug: 'better-look', name: 'Better look' });

      await choose('preset-rename');
      expect((document.querySelector('#preset-name') as HTMLInputElement).value).toBe('Look');
      typeName('Better look');
      flushSync(() => confirmName().click());
      await settle();

      expect(presetApiMocks.renamePreset).toHaveBeenCalledWith('image', 'look', 'Better look');
      expect(draft.state.appliedPreset).toBe('better-look');
      expect(presetButton().dataset.applied).toBe('better-look');
    });
  });

  describe('Delete', () => {
    it('asks first, and does nothing when declined', async () => {
      await mountWithPresets();
      await choose('preset:look');

      await choose('preset-delete');
      expect(document.querySelector('[role="alertdialog"]')).not.toBeNull();
      await answerConfirm(false);

      expect(presetApiMocks.deletePreset).not.toHaveBeenCalled();
      expect(draft.state.appliedPreset).toBe('look');
    });

    it('deletes the preset and clears the applied slug when confirmed', async () => {
      await mountWithPresets();
      await choose('preset:look');
      presetApiMocks.deletePreset.mockResolvedValue({ status: 'ok' });

      await choose('preset-delete');
      presetApiMocks.listPresets.mockResolvedValue([PORTRAIT, CLIP]);
      await answerConfirm(true);

      expect(presetApiMocks.deletePreset).toHaveBeenCalledWith('image', 'look');
      expect(draft.state.appliedPreset).toBeNull();
      expect(presetButton().dataset.applied).toBe('');
      await openMenu();
      expect(menuActions()).toEqual(['preset:portrait', 'preset-save-as']);
    });

    it('says so, and keeps the preset, when the server refuses', async () => {
      await mountWithPresets();
      await choose('preset:look');
      presetApiMocks.deletePreset.mockRejectedValue(new ApiError('DELETE 500', 500, 'Disk is read-only.'));
      toasts.splice(0, toasts.length);

      await choose('preset-delete');
      await answerConfirm(true);

      expect(toasts.some((toast) => toast.type === 'error' && toast.message.includes('Disk is read-only.'))).toBe(true);
      expect(draft.state.appliedPreset).toBe('look');
    });
  });

  describe('on a phone', () => {
    const chips = () => Array.from(target.querySelectorAll('.strip-chips [data-chip]')) as HTMLButtonElement[];
    const presetChip = () => target.querySelector('[data-chip="preset"]') as HTMLButtonElement;
    const sheetMenu = () => document.querySelector('[data-testid="action-sheet"]') as HTMLElement | null;

    beforeEach(() => {
      layoutProbe.compact = true;
      // jsdom has no Web Animations; the sheets' transitions need it.
      Element.prototype.animate = vi.fn(() => ({ cancel: vi.fn(), finished: Promise.resolve(), onfinish: null })) as unknown as Element['animate'];
    });
    afterEach(() => { delete (Element.prototype as Partial<Element>).animate; });

    it('puts the Preset chip first, saying none, and opens the preset menu as an action sheet', async () => {
      await mountWithPresets();

      expect(chips()[0].dataset.chip).toBe('preset');
      expect(presetChip().dataset.applied).toBe('');
      flushSync(() => presetChip().click());
      await settle();

      expect(sheetMenu()).not.toBeNull();
      expect(sheetMenu()?.querySelector('[data-action="preset:look"]')).not.toBeNull();
      expect(sheetMenu()?.querySelector('[data-action="preset-save-as"]')).not.toBeNull();
      // The chip opens a menu, never a settings popover.
      expect(document.querySelector('.pop-list')).toBeNull();
    });
  });
});

describe('WorkspacePage last frame', () => {
  const END_CONTROLS = ['end_image', 'end_image_clear', 'end_image_path'];

  /** A context with two video models: MiniMax H3 (has a last frame) and LTX (hides the last-frame controls). */
  function lastFrameContext(assets: GalleryAsset[] = [], withEnhancer = false): WorkspaceContext {
    const context = makeContext({
      history_assets: assets,
      video_models: [{ id: 'h3', label: 'MiniMax H3', type: 'video' }, { id: 'ltx-8', label: 'ltx-8', type: 'video' }],
      current_video_model: 'h3',
      video_model_defaults: {
        h3: makeVideoDefaults({ hidden_controls: ['low_memory'] }),
        'ltx-8': makeVideoDefaults({ hidden_controls: END_CONTROLS }),
      },
    });
    const { definitions } = context.workflow_contract;
    definitions.img2vid.visible_controls = [...definitions.img2vid.visible_controls, 'end_image', 'end_image_path', 'end_image_clear'];
    for (const workflow of ['txt2img', 'img2img', 'txt2vid'] as const) definitions[workflow].clear_fields = [...definitions[workflow].clear_fields, 'end_image_path'];
    if (withEnhancer) {
      definitions.img2vid.visible_controls.push('prompt_enhance', 'prompt_enhance_auto');
      context.prompt_enhancer = {
        matrix: {
          axes: [{ key: 'style', label: 'Style', multi: false, video_only: false, audio_only: false, options: [{ slug: 'keep', label: 'Keep', description: '', group: null }], default: ['keep'] }],
          defaults: { style: 'keep', mood: 'keep', theme: 'keep', camera: 'keep', pace: 'keep', details: [], length: 'same', motion: [], sound: [], note: '' },
        },
        model: 'owner/llm', revision: null, downloaded: true, download_size_label: null, default_max_words: 300, error: null,
      };
    }
    return context;
  }

  async function mountLastFrame(assets: GalleryAsset[] = [], workflow: 'img2vid' | 'txt2vid' = 'img2vid', withEnhancer = false): Promise<void> {
    draft.update('workflow', workflow);
    galleryApiMocks.getGallery.mockResolvedValue(makeGalleryPage(assets));
    await mountWorkspace(lastFrameContext(assets, withEnhancer));
    draft.update('model', 'h3');
    await settle();
  }

  beforeEach(() => {
    workspaceApiMocks.submitGenerate.mockResolvedValue({ job_id: 'job-1' } as JobContext);
    browseApiMocks.browsePlaces.mockResolvedValue(placesFor('reference_image'));
    Object.defineProperty(URL, 'createObjectURL', { configurable: true, value: vi.fn(() => 'blob:test-image') });
    Object.defineProperty(URL, 'revokeObjectURL', { configurable: true, value: vi.fn() });
    window.location.hash = '#/workspace';
  });

  const endPath = () => target.querySelector('input[name="end_image_path"]') as HTMLInputElement | null;
  const submit = async (): Promise<FormData> => {
    (target.querySelector('#ws-prompt') as HTMLTextAreaElement).value = 'a prompt';
    (target.querySelector('#ws-prompt') as HTMLTextAreaElement).dispatchEvent(new Event('input', { bubbles: true }));
    await settle();
    target.querySelector('form')!.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
    await settle();
    return workspaceApiMocks.submitGenerate.mock.calls.at(-1)![0];
  };
  async function upload(inputId: string, name: string): Promise<File> {
    const input = target.querySelector(`#${inputId}`) as HTMLInputElement;
    const file = new File(['img'], name, { type: 'image/png' });
    Object.defineProperty(input, 'files', { configurable: true, value: [file] });
    input.dispatchEvent(new Event('change', { bubbles: true }));
    await settle();
    return file;
  }

  it('offers a Last frame slot beside the first frame for a model that has one', async () => {
    await mountLastFrame();

    expect(endPath()).not.toBeNull();
    expect(target.querySelector('#ws-end-image-file')).not.toBeNull();
    expect(target.querySelector('#ws-image-label')?.textContent).toContain('First frame');
    expect(target.querySelector('#ws-end-image-label')?.textContent).toContain('Last frame');
  });

  it('leaves the reference picker alone, and offers no slot, for a model without a last frame', async () => {
    await mountLastFrame();
    pickOption(target.querySelector('#ws-model') as HTMLButtonElement, 'ltx-8');
    await settle();

    expect(endPath()).toBeNull();
    expect(target.querySelector('#ws-end-image-file')).toBeNull();
    expect(target.querySelector('#ws-image-label')?.textContent).toContain('Image');
    expect(target.querySelector('input[name="image_path"]')).not.toBeNull();
  });

  it('submits a typed last-frame path with the first frame', async () => {
    await mountLastFrame();
    draft.patch({ referenceImagePath: '/in/start.png', endImagePath: '/in/end.png' });
    await settle();

    const submitted = await submit();

    expect(submitted.get('image_path')).toBe('/in/start.png');
    expect(submitted.get('end_image_path')).toBe('/in/end.png');
    expect(submitted.has('end_image_file')).toBe(false);
  });

  it('submits an uploaded last frame as end_image_file, apart from the first frame', async () => {
    await mountLastFrame();
    const file = await upload('ws-end-image-file', 'end.png');

    const submitted = await submit();

    expect(submitted.get('end_image_file')).toBe(file);
    expect(submitted.has('image_file')).toBe(false);
  });

  it('keeps an uploaded first and last frame when the page is left and comes back', async () => {
    await mountLastFrame();
    const first = await upload('ws-image-file', 'first.png');
    const last = await upload('ws-end-image-file', 'last.png');
    await unmount(app!);
    app = null;

    await mountLastFrame();
    const submitted = await submit();

    expect(submitted.get('image_file')).toBe(first);
    expect(submitted.get('end_image_file')).toBe(last);
  });

  it('shows and submits the paths a Use as chose in the Gallery, not the upload of the slot it set', async () => {
    await mountLastFrame();
    const last = await upload('ws-end-image-file', 'last.png');
    await upload('ws-image-file', 'first.png');
    await unmount(app!);
    app = null;

    // The Gallery's Use as navigates back with the chosen path in the address.
    workspaceApiMocks.parseUrlPrefill.mockReturnValue({ workflow: 'img2vid', image_path: '/in/new-start.png', use_as: 'start' });
    await mountLastFrame();
    const submitted = await submit();

    expect(draft.state.referenceImagePath).toBe('/in/new-start.png');
    expect(submitted.get('image_path')).toBe('/in/new-start.png');
    expect(submitted.has('image_file')).toBe(false);
    // The last frame was not the slot it set, so its upload stays.
    expect(submitted.get('end_image_file')).toBe(last);
  });

  it('keeps the first frame uploaded before leaving when a Gallery Use as sets the last frame', async () => {
    await mountLastFrame();
    const first = await upload('ws-image-file', 'first.png');
    await unmount(app!);
    app = null;

    workspaceApiMocks.parseUrlPrefill.mockReturnValue({ workflow: 'img2vid', end_image_path: '/in/new-end.png', use_as: 'last' });
    await mountLastFrame();
    const submitted = await submit();

    expect(submitted.get('image_file')).toBe(first);
    expect(submitted.get('end_image_path')).toBe('/in/new-end.png');
    expect(submitted.has('end_image_file')).toBe(false);
  });

  it('drops the frames uploaded before leaving when a Gallery Reuse of a clip without a last frame comes back', async () => {
    await mountLastFrame();
    await upload('ws-image-file', 'first.png');
    await upload('ws-end-image-file', 'last.png');
    await unmount(app!);
    app = null;

    workspaceApiMocks.parseUrlPrefill.mockReturnValue({ workflow: 'img2vid', model: 'h3', image_path: '/in/clip-start.png', prompt: 'a clip' });
    await mountLastFrame();
    const submitted = await submit();

    expect(submitted.get('image_path')).toBe('/in/clip-start.png');
    expect(submitted.has('image_file')).toBe(false);
    expect(submitted.get('end_image_path')).toBe('');
    expect(submitted.has('end_image_file')).toBe(false);
  });

  it('drops the uploaded frames with Reset', async () => {
    await mountLastFrame();
    await upload('ws-image-file', 'first.png');
    await upload('ws-end-image-file', 'last.png');

    (target.querySelector('#ws-reset') as HTMLButtonElement).click();
    await settle();
    await unmount(app!);
    app = null;
    await mountLastFrame();
    const submitted = await submit();

    expect(submitted.has('image_file')).toBe(false);
    expect(submitted.has('end_image_file')).toBe(false);
  });

  it('shows the pair in the section summary', async () => {
    await mountLastFrame();
    draft.patch({ referenceImagePath: '/in/start.png', endImagePath: '/in/end.png' });
    await settle();

    expect(target.querySelector('[data-section="reference"] .inspector-summary')?.textContent).toBe('start.png → end.png');
  });

  it('clears a chosen path and a pending upload when the model or the workflow leaves the last frame behind', async () => {
    await mountLastFrame();
    draft.patch({ referenceImagePath: '/in/start.png', endImagePath: '/in/end.png' });
    await upload('ws-end-image-file', 'end.png');

    pickOption(target.querySelector('#ws-model') as HTMLButtonElement, 'ltx-8');
    await settle();
    let submitted = await submit();

    expect(draft.state.endImagePath).toBeNull();
    expect(submitted.has('end_image_path')).toBe(false);
    expect(submitted.has('end_image_file')).toBe(false);
    expect(submitted.get('image_path')).toBe('/in/start.png');

    pickOption(target.querySelector('#ws-model') as HTMLButtonElement, 'h3');
    await settle();
    draft.update('endImagePath', '/in/end.png');
    await upload('ws-end-image-file', 'end.png');

    draft.update('workflow', 'txt2vid');
    await settle();
    submitted = await submit();

    expect(endPath()).toBeNull();
    expect(submitted.has('end_image_path')).toBe(false);
    expect(submitted.has('end_image_file')).toBe(false);
  });

  it('clears only the last frame with its Clear button', async () => {
    await mountLastFrame();
    draft.patch({ referenceImagePath: '/in/start.png', endImagePath: '/in/end.png' });
    await settle();

    (target.querySelector('button[aria-label="Clear last frame"]') as HTMLButtonElement).click();
    await settle();

    expect(draft.state.endImagePath).toBeNull();
    expect(draft.state.referenceImagePath).toBe('/in/start.png');
  });

  describe('from the history strip', () => {
    const asset = makeAsset({ id: 'out/end.png', url: '/media/out/end.png', filename: 'end.png', file_path: '/outputs/out/end.png', prompt: 'a red fox' });
    const menuFor = async (): Promise<NodeListOf<Element>> => {
      const tile = target.querySelector(`#ws-history-shell article[aria-label="${asset.filename}"]`) as HTMLElement;
      (tile.querySelector(`button[aria-label="More actions for ${asset.filename}"]`) as HTMLButtonElement).click();
      flushSync();
      return document.querySelectorAll('[role="menu"] [data-action^="reference-"]');
    };

    it('offers For last frame and sets only the last frame, leaving the text and the first frame', async () => {
      await mountLastFrame([asset]);
      draft.patch({ referenceImagePath: '/in/start.png', prompt: 'my action' });
      await settle();

      const actions = Array.from(await menuFor()).map((el) => el.getAttribute('data-action'));
      (document.querySelector('[role="menu"] [data-action="reference-last-frame"]') as HTMLButtonElement).click();
      await settle();

      expect(actions).toEqual(['reference-image', 'reference-video', 'reference-last-frame']);
      expect(draft.state).toMatchObject({ workflow: 'img2vid', endImagePath: '/outputs/out/end.png', referenceImagePath: '/in/start.png', prompt: 'my action', scene: '' });
      expect(toasts.some((toast) => toast.message === 'end.png is now the last frame')).toBe(true);
      expect(target.querySelector(`[data-section="reference"] img[src="${asset.url}"]`)).not.toBeNull();
    });

    it('keeps a first frame uploaded from this device when Use as sets the last frame, and the other way round', async () => {
      await mountLastFrame([asset]);
      const first = await upload('ws-image-file', 'first.png');

      await menuFor();
      (document.querySelector('[role="menu"] [data-action="reference-last-frame"]') as HTMLButtonElement).click();
      await settle();
      const afterLast = await submit();

      expect(afterLast.get('image_file')).toBe(first);
      expect(afterLast.get('end_image_path')).toBe('/outputs/out/end.png');
      expect(afterLast.has('end_image_file')).toBe(false);

      const last = await upload('ws-end-image-file', 'last.png');
      await menuFor();
      (document.querySelector('[role="menu"] [data-action="reference-video"]') as HTMLButtonElement).click();
      await settle();
      const afterFirst = await submit();

      expect(afterFirst.get('image_path')).toBe('/outputs/out/end.png');
      expect(afterFirst.has('image_file')).toBe(false);
      expect(afterFirst.get('end_image_file')).toBe(last);
    });

    it('does not offer it for a model without a last frame, or outside Image to Video', async () => {
      await mountLastFrame([asset], 'txt2vid');

      expect(Array.from(await menuFor()).map((el) => el.getAttribute('data-action'))).toEqual(['reference-image', 'reference-video']);
    });

    it('clears the last frame when its asset is deleted', async () => {
      await mountLastFrame([asset]);
      galleryApiMocks.deleteAsset.mockResolvedValue(undefined);
      await menuFor();
      (document.querySelector('[role="menu"] [data-action="reference-last-frame"]') as HTMLButtonElement).click();
      await settle();
      expect(draft.state.endImagePath).toBe('/outputs/out/end.png');

      await menuFor();
      (document.querySelector('[role="menu"] [data-action="delete"]') as HTMLButtonElement).click();
      await settle();
      await answerConfirm(true);

      expect(draft.state.endImagePath).toBeNull();
    });
  });
});

describe('WorkspacePage references', () => {
  /** Text to image and Image to image list the controls; "zit" takes 3 references and "flux-lite" none. */
  function referencesContext(): WorkspaceContext {
    const context = makeContext({
      image_model_defaults: { zit: makeImageDefaults({ max_reference_images: 3 }), 'flux-lite': makeImageDefaults() },
    });
    for (const workflow of ['txt2img', 'img2img'] as const) context.workflow_contract.definitions[workflow].visible_controls.push('references', 'references_clear');
    return context;
  }

  beforeEach(() => {
    workspaceApiMocks.submitGenerate.mockResolvedValue({ job_id: 'job-1' } as JobContext);
    browseApiMocks.browsePlaces.mockResolvedValue(placesFor('reference_image'));
    Object.defineProperty(URL, 'createObjectURL', { configurable: true, value: vi.fn(() => 'blob:test-image') });
    Object.defineProperty(URL, 'revokeObjectURL', { configurable: true, value: vi.fn() });
    window.location.hash = '#/workspace';
  });

  const section = () => target.querySelector('[data-section="references"]');
  const summary = () => section()?.querySelector('.inspector-summary')?.textContent;
  const rowLabels = () => Array.from(section()?.querySelectorAll('.inspector-label-text') ?? []).map((el) => el.textContent);
  const image = (name: string) => new File(['img'], name, { type: 'image/png' });

  async function mountReferences(): Promise<void> {
    await mountWorkspace(referencesContext());
    await settle();
  }

  async function addFiles(...files: File[]): Promise<void> {
    const input = target.querySelector('#ws-ref-add-file') as HTMLInputElement;
    Object.defineProperty(input, 'files', { configurable: true, value: files });
    input.dispatchEvent(new Event('change', { bubbles: true }));
    await settle();
  }

  async function submit(): Promise<FormData> {
    (target.querySelector('#ws-prompt') as HTMLTextAreaElement).value = 'the woman from image 1';
    (target.querySelector('#ws-prompt') as HTMLTextAreaElement).dispatchEvent(new Event('input', { bubbles: true }));
    await settle();
    target.querySelector('form')!.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
    await settle();
    return workspaceApiMocks.submitGenerate.mock.calls.at(-1)![0];
  }

  it('offers an empty Add reference row on a model that takes references', async () => {
    await mountReferences();

    expect(summary()).not.toMatch(/\d/);
    expect(rowLabels()).toHaveLength(1);
    expect(section()?.querySelector('#ws-ref-add-file')).not.toBeNull();
  });

  it('takes what fits, tells how many did not in one toast, and shows no add row when full', async () => {
    await mountReferences();

    await addFiles(image('a.png'), image('b.png'), image('c.png'), image('d.png'));

    expect(summary()).toMatch(/3.*3/);
    expect(section()?.querySelector('#ws-ref-add-file')).toBeNull();
    expect(toasts).toHaveLength(1);
    expect(toasts[0]).toMatchObject({ type: 'info', count: 1 });
  });

  it('submits one repeated reference field, a file or a path per row, in row order', async () => {
    await mountReferences();
    draft.update('referenceImagePaths', ['/in/a.png']);
    await settle();
    const second = image('second.png');
    await addFiles(second);
    draft.patch({ referenceImagePaths: ['/in/a.png', '', '/in/c.png'] });
    await settle();

    const submitted = await submit();

    expect(submitted.getAll('reference')).toEqual(['/in/a.png', second, '/in/c.png']);
  });

  it('tells the enhancer how many references are set, uploads included', async () => {
    enhanceApiMocks.enhancePrompt.mockResolvedValue();
    draft.update('prompt', 'the woman from image 1');
    await mountWorkspace(withEnhancer(referencesContext()));
    draft.update('referenceImagePaths', ['/in/a.png']);
    await settle();
    await addFiles(image('b.png'));

    pressKey('e', { ctrlKey: true }, target.querySelector('#ws-prompt') as HTMLTextAreaElement);
    await settle();

    const body = enhanceApiMocks.enhancePrompt.mock.calls[0]?.[0] as { context: Record<string, unknown> };
    expect(body.context).toMatchObject({ references: 2, reference_image: false });
  });

  it('removes a row, renumbers the rest and keeps focus in the list', async () => {
    await mountReferences();
    draft.update('referenceImagePaths', ['/in/a.png', '/in/b.png', '/in/c.png']);
    await settle();

    (target.querySelector('button[aria-label="Remove image 2"]') as HTMLButtonElement).click();
    await settle();

    expect(draft.state.referenceImagePaths).toEqual(['/in/a.png', '/in/c.png']);
    expect(document.activeElement?.id).toBe('ws-ref-2-clear');
    expect(section()?.textContent).toContain('c.png');
  });

  it('keeps a row whose file is gone, says so, and blocks Generate until it is removed', async () => {
    await mountReferences();
    draft.patch({ referenceImagePaths: ['/in/a.png', '/in/gone.png'], missingReferencePaths: ['/in/gone.png'] });
    await settle();

    const generate = target.querySelector('#ws-submit') as HTMLButtonElement;
    const group = target.querySelector('#ws-ref-2-label')?.closest('[role="group"]') as HTMLElement;
    const hint = target.querySelector(`#${group.getAttribute('aria-describedby')}`);
    expect(hint?.textContent).toBeTruthy();
    expect(target.querySelector('#ws-ref-1-label')?.closest('[role="group"]')?.getAttribute('aria-describedby')).toBeNull();
    expect(generate.disabled).toBe(true);
    expect(target.querySelector('#ws-generate-reason')?.textContent).toContain('2');

    (target.querySelector('button[aria-label="Remove image 2"]') as HTMLButtonElement).click();
    await settle();

    expect(generate.disabled).toBe(false);
    expect(draft.state.missingReferencePaths).toEqual([]);
  });

  it('keeps the picked uploads when the page is left and comes back, each in its own row', async () => {
    await mountReferences();
    draft.update('referenceImagePaths', ['/in/a.png']);
    await settle();
    const second = image('second.png');
    await addFiles(second);
    await unmount(app!);
    app = null;

    await mountReferences();

    expect(draft.state.referenceImagePaths).toEqual(['/in/a.png', '']);
    expect(target.querySelector('#ws-submit')).toHaveProperty('disabled', false);
    expect((await submit()).getAll('reference')).toEqual(['/in/a.png', second]);
  });

  it('shows and submits the references a Reuse chose in the Gallery, not the files uploaded before leaving', async () => {
    await mountReferences();
    draft.update('referenceImagePaths', ['/in/a.png']);
    await settle();
    await addFiles(image('second.png'));
    await unmount(app!);
    app = null;

    workspaceApiMocks.parseUrlPrefill.mockReturnValue({ workflow: 'txt2img', reference_path: ['/in/new.png', '/in/other.png'] } as never);
    await mountReferences();

    expect(draft.state.referenceImagePaths).toEqual(['/in/new.png', '/in/other.png']);
    expect((await submit()).getAll('reference')).toEqual(['/in/new.png', '/in/other.png']);
  });

  it('keeps an upload that lost its file as a missing row in its own place, blocks Generate, and lets an upload replace it', async () => {
    // An upload has no path; leaving the page or reloading drops its file, and the draft that comes back still has the row.
    draft.patch({ referenceImagePaths: ['/in/a.png', '', '/in/c.png'] });
    await mountReferences();

    const generate = target.querySelector('#ws-submit') as HTMLButtonElement;
    expect(rowLabels()).toHaveLength(3);
    expect(draft.state.referenceImagePaths).toEqual(['/in/a.png', '', '/in/c.png']);
    expect(generate.disabled).toBe(true);

    const file = image('again.png');
    const input = target.querySelector('#ws-ref-2-file') as HTMLInputElement;
    Object.defineProperty(input, 'files', { configurable: true, value: [file] });
    input.dispatchEvent(new Event('change', { bubbles: true }));
    await settle();

    expect(generate.disabled).toBe(false);
    expect((await submit()).getAll('reference')).toEqual(['/in/a.png', file, '/in/c.png']);
  });

  it('asks the server which reference files are gone when it loads, and flags those rows in place', async () => {
    draft.patch({ referenceImagePaths: ['/in/a.png', '/in/gone.png', '', '/in/c.png'], missingReferencePaths: [] });
    workspaceApiMocks.checkReferencePaths.mockResolvedValue(['/in/gone.png']);
    await mountWorkspace({ ...referencesContext(), image_model_defaults: { zit: makeImageDefaults({ max_reference_images: 4 }) } });
    await settle();

    expect(workspaceApiMocks.checkReferencePaths).toHaveBeenCalledWith(['/in/a.png', '/in/gone.png', '/in/c.png']);
    expect(draft.state.referenceImagePaths).toEqual(['/in/a.png', '/in/gone.png', '', '/in/c.png']);
    expect(draft.state.missingReferencePaths).toEqual(['/in/gone.png']);
    expect((target.querySelector('#ws-submit') as HTMLButtonElement).disabled).toBe(true);
  });

  it('lets the server clear a flag left over from before', async () => {
    draft.patch({ referenceImagePaths: ['/in/a.png', '/in/back.png'], missingReferencePaths: ['/in/back.png'] });
    workspaceApiMocks.checkReferencePaths.mockResolvedValue([]);
    await mountReferences();

    expect(draft.state.missingReferencePaths).toEqual([]);
  });

  it('keeps a reference row whose asset is deleted from the history strip, flagged missing', async () => {
    const asset = makeAsset({ id: 'out/ref.png', url: '/media/out/ref.png', filename: 'ref.png', file_path: '/outputs/out/ref.png' });
    galleryApiMocks.getGallery.mockResolvedValue(makeGalleryPage([asset]));
    galleryApiMocks.deleteAsset.mockResolvedValue(undefined);
    workspaceApiMocks.checkReferencePaths.mockResolvedValue([]);
    draft.patch({ referenceImagePaths: ['/in/a.png', '/outputs/out/ref.png', '/in/c.png'] });
    await mountWorkspace({ ...referencesContext(), history_assets: [asset] });
    await settle();

    const tile = target.querySelector(`#ws-history-shell article[aria-label="${asset.filename}"]`) as HTMLElement;
    (tile.querySelector(`button[aria-label="More actions for ${asset.filename}"]`) as HTMLButtonElement).click();
    flushSync();
    (document.querySelector('[role="menu"] [data-action="delete"]') as HTMLButtonElement).click();
    await settle();
    workspaceApiMocks.checkReferencePaths.mockResolvedValue(['/outputs/out/ref.png']);
    await answerConfirm(true);
    await settle();

    expect(draft.state.referenceImagePaths).toEqual(['/in/a.png', '/outputs/out/ref.png', '/in/c.png']);
    expect(draft.state.missingReferencePaths).toEqual(['/outputs/out/ref.png']);
    expect((target.querySelector('#ws-submit') as HTMLButtonElement).disabled).toBe(true);
  });

  it('hides the references and sends none on a model without them, and brings them back with the model that takes them', async () => {
    await mountReferences();
    draft.update('referenceImagePaths', ['/in/a.png']);
    await settle();
    const upload = image('b.png');
    await addFiles(upload);

    pickOption(target.querySelector('#ws-model') as HTMLButtonElement, 'flux-lite');
    await settle();
    expect(section()).toBeNull();
    expect((await submit()).has('reference')).toBe(false);
    expect(draft.state.referenceImagePaths).toEqual(['/in/a.png', '']);

    pickOption(target.querySelector('#ws-model') as HTMLButtonElement, 'zit');
    await settle();

    expect(rowLabels()).toContain('Image 2');
    expect((await submit()).getAll('reference')).toEqual(['/in/a.png', upload]);
  });

  it('shows and sends the first rows a lower cap allows, and a missing row beyond them does not block Generate', async () => {
    await mountReferences();
    draft.patch({ referenceImagePaths: ['/in/a.png', '/in/b.png', '/in/c.png', '/in/gone.png'], missingReferencePaths: ['/in/gone.png'] });
    await settle();

    expect(summary()).toMatch(/3.*3/);
    expect(target.querySelector('#ws-submit')).toHaveProperty('disabled', false);
    expect((await submit()).getAll('reference')).toEqual(['/in/a.png', '/in/b.png', '/in/c.png']);
    expect(draft.state.referenceImagePaths).toHaveLength(4);
  });

  it('keeps the references, uploads included, while a video workflow is open', async () => {
    await mountReferences();
    draft.update('referenceImagePaths', ['/in/b.png']);
    await settle();
    const upload = image('a.png');
    await addFiles(upload);

    draft.update('workflow', 'txt2vid');
    await settle();
    expect(section()).toBeNull();
    expect((await submit()).has('reference')).toBe(false);
    draft.update('workflow', 'txt2img');
    await settle();

    expect(draft.state.referenceImagePaths).toEqual(['/in/b.png', '']);
    expect((await submit()).getAll('reference')).toEqual(['/in/b.png', upload]);
  });

  it('clears the references with Reset', async () => {
    await mountReferences();
    await addFiles(image('a.png'));

    (target.querySelector('#ws-reset') as HTMLButtonElement).click();
    await settle();

    expect(draft.state.referenceImagePaths).toEqual([]);
    expect(summary()).not.toMatch(/\d/);
  });

  describe('Reuse from the history strip', () => {
    const reuse = async (asset: GalleryAsset): Promise<void> => {
      const tile = target.querySelector(`#ws-history-shell article[aria-label="${asset.filename}"]`) as HTMLElement;
      (tile.querySelector(`button[aria-label="More actions for ${asset.filename}"]`) as HTMLButtonElement).click();
      flushSync();
      (document.querySelector('[role="menu"] [data-action="reuse"]') as HTMLButtonElement).click();
      await settle();
    };
    const assetWith = (url: string): GalleryAsset => makeAsset({ id: 'out/fox.png', url: '/media/out/fox.png', filename: 'fox.png', file_path: '/outputs/out/fox.png', has_reusable_config: true, reuse_workspace_url: url });

    async function mountWith(asset: GalleryAsset): Promise<void> {
      galleryApiMocks.getGallery.mockResolvedValue(makeGalleryPage([asset]));
      await mountWorkspace({ ...referencesContext(), history_assets: [asset] });
      await settle();
    }

    it('brings back the references in order, re-checks them, keeps a missing one as its own row and blocks Generate', async () => {
      await mountWith(assetWith('#/workspace?workflow=txt2img&model=zit&reference_path=%2Fin%2Fa.png&reference_path=%2Fin%2Fgone.png'));
      workspaceApiMocks.checkReferencePaths.mockClear();
      workspaceApiMocks.checkReferencePaths.mockResolvedValue(['/in/gone.png']);

      await reuse(makeAsset({ filename: 'fox.png' }));

      expect(workspaceApiMocks.checkReferencePaths).toHaveBeenCalledWith(['/in/a.png', '/in/gone.png']);
      expect(draft.state.referenceImagePaths).toEqual(['/in/a.png', '/in/gone.png']);
      expect(draft.state.missingReferencePaths).toEqual(['/in/gone.png']);
      expect((target.querySelector('#ws-submit') as HTMLButtonElement).disabled).toBe(true);
    });

    it('clears the references already chosen and their uploads when the reused image had none', async () => {
      await mountWith(assetWith('#/workspace?workflow=txt2img&model=zit'));
      await addFiles(image('a.png'));
      draft.update('referenceImagePaths', ['', '/in/b.png']);

      await reuse(makeAsset({ filename: 'fox.png' }));

      expect(draft.state.referenceImagePaths).toEqual([]);
      expect((await submit()).has('reference')).toBe(false);
    });
  });

  it('keeps the reference uploads when a Reuse lands on a video workflow, which has no References section', async () => {
    const clip = makeAsset({ id: 'out/clip.mp4', url: '/media/out/clip.mp4', filename: 'clip.mp4', media_type: 'video', file_path: '/outputs/out/clip.mp4', has_reusable_config: true, reuse_workspace_url: '#/workspace?workflow=txt2vid&prompt=a%20clip' });
    galleryApiMocks.getGallery.mockResolvedValue(makeGalleryPage([clip]));
    await mountWorkspace({ ...referencesContext(), history_assets: [clip] });
    await settle();
    const upload = image('a.png');
    await addFiles(upload);

    const tile = target.querySelector('#ws-history-shell article[aria-label="clip.mp4"]') as HTMLElement;
    (tile.querySelector('button[aria-label="More actions for clip.mp4"]') as HTMLButtonElement).click();
    flushSync();
    (document.querySelector('[role="menu"] [data-action="reuse"]') as HTMLButtonElement).click();
    await settle();
    expect(draft.state.workflow).toBe('txt2vid');

    draft.update('workflow', 'txt2img');
    await settle();

    expect((await submit()).getAll('reference')).toEqual([upload]);
  });

  describe('dropping files', () => {
    /** Fire a drag event carrying *files* at *node*. */
    function drag(node: Element, type: 'dragover' | 'drop' | 'dragleave', files: File[] = []): Event {
      const event = new Event(type, { bubbles: true, cancelable: true });
      Object.defineProperty(event, 'dataTransfer', { value: { types: ['Files'], files, dropEffect: 'none' } });
      node.dispatchEvent(event);
      return event;
    }
    const addRow = () => (target.querySelector('#ws-ref-add-label') as HTMLElement).closest('[role="group"]') as HTMLElement;

    it('appends the images dropped on the add row in order and skips what is not an image', async () => {
      await mountReferences();
      const first = image('one.png');
      const second = image('two.png');

      drag(addRow(), 'dragover');
      const dropped = drag(addRow(), 'drop', [first, new File(['x'], 'notes.txt', { type: 'text/plain' }), second]);
      await settle();

      expect(dropped.defaultPrevented).toBe(true);
      expect(summary()).toMatch(/2.*3/);
      expect((await submit()).getAll('reference')).toEqual([first, second]);
    });

    it('replaces only the row a file is dropped on, with the first file', async () => {
      await mountReferences();
      draft.update('referenceImagePaths', ['/in/a.png', '/in/b.png']);
      await settle();
      const row = (target.querySelector('#ws-ref-2-label') as HTMLElement).closest('[role="group"]') as HTMLElement;
      const dropped = image('new.png');

      drag(row, 'dragover');
      drag(row, 'drop', [dropped, image('other.png')]);
      await settle();

      expect((await submit()).getAll('reference')).toEqual(['/in/a.png', dropped]);
    });
  });
});

describe('WorkspacePage Video summary', () => {
  it('leaves Low memory out of the summary when the model has no such control', async () => {
    Object.defineProperty(URL, 'createObjectURL', { configurable: true, value: vi.fn(() => 'blob:test-image') });
    draft.update('workflow', 'txt2vid');
    const context = makeContext({ video_model_defaults: { 'ltx-8': makeVideoDefaults({ hidden_controls: ['low_memory'], low_memory: true }) } });
    await mountWorkspace(context);
    draft.update('lowMemory', true);
    await settle();

    const summary = target.querySelector('[data-section="video"] .inspector-summary')?.textContent ?? '';
    expect(target.querySelector('#ws-low-memory')).toBeNull();
    expect(summary).toContain('Audio');
    expect(summary).not.toContain('Low memory');
  });

  it('keeps Low memory in the summary when the model has the control', async () => {
    draft.update('workflow', 'txt2vid');
    await mountWorkspace(makeContext());
    draft.update('lowMemory', true);
    await settle();

    expect(target.querySelector('[data-section="video"] .inspector-summary')?.textContent).toContain('Low memory');
  });
});
