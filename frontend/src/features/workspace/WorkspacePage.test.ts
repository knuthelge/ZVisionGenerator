// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, mount, unmount } from '../../../node_modules/svelte/src/index-client.js';
import { createClassComponent } from 'svelte/legacy';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { ApiError } from '$lib/api/client';
import { reactiveProps } from '../../test-utils/reactiveProps.svelte';
import { draft } from '$lib/state/draft.svelte';
import { clipMemory } from '$lib/state/clipMemory.svelte';
import { extendInspection } from '$lib/state/extendInspection.svelte';
import { legacySource, sceneSource, sourceInfo } from '$lib/state/extend.fixtures';
import { historyStore } from '$lib/state/history.svelte';
import { jobStore } from '$lib/state/job.svelte';
import { requestPromptRun } from '$lib/state/pendingPromptRun';
import { router } from '$lib/state/router.svelte';
import { toasts } from '$lib/state/toasts.svelte';
import type { EnhanceFrame, ImageModelDefaults, JobContext, JobSnapshot, PromptEnhancerContract, VideoModelDefaults, WorkspaceContext, GalleryAsset, GalleryPage } from '$lib/types';

const workspaceApiMocks = vi.hoisted(() => ({
  getWorkspaceContext: vi.fn<() => Promise<WorkspaceContext>>(),
  getWorkspaceCoreContext: vi.fn<() => Promise<WorkspaceContext>>(),
  submitGenerate: vi.fn<(formData: FormData) => Promise<JobContext>>(),
  getJobSnapshot: vi.fn<(jobId: string) => Promise<JobSnapshot>>(),
  getHistory: vi.fn<(page?: number) => Promise<GalleryPage>>(),
  parseUrlPrefill: vi.fn<() => Record<string, string>>(),
  listJobs: vi.fn(async () => ({ active_job: null as JobSnapshot | null, queued_jobs: [] as JobSnapshot[] })),
  clearQueue: vi.fn(async () => ({ cancelled: [] as string[] })),
  cancelJob: vi.fn(async () => undefined),
}));

const browseApiMocks = vi.hoisted(() => ({
  listDirectory: vi.fn(),
  browsePlaces: vi.fn(),
  createFolder: vi.fn(),
}));

const promptFileApiMocks = vi.hoisted(() => ({
  inspectPromptFile: vi.fn(),
  writePromptFile: vi.fn(),
}));

const extendApiMocks = vi.hoisted(() => ({
  inspectExtendSource: vi.fn(),
}));

vi.mock('$lib/api/extend', () => ({ inspectExtendSource: extendApiMocks.inspectExtendSource }));

const videoMemoryApiMocks = vi.hoisted(() => ({
  checkVideoMemoryFit: vi.fn<(body: Record<string, unknown>) => Promise<{ memory_fit: { status: string; required_gb: number; budget_gb: number } | null; width: number; height: number; frames: number }>>(),
}));

vi.mock('$lib/api/videoMemory', () => ({ checkVideoMemoryFit: videoMemoryApiMocks.checkVideoMemoryFit }));

vi.mock('$lib/api/workspace', async (importOriginal) => {
  const actual = await importOriginal<typeof import('$lib/api/workspace')>();
  return {
    ...actual,
    getWorkspaceContext: workspaceApiMocks.getWorkspaceContext,
    getWorkspaceCoreContext: workspaceApiMocks.getWorkspaceCoreContext,
    submitGenerate: workspaceApiMocks.submitGenerate,
    getJobSnapshot: workspaceApiMocks.getJobSnapshot,
    getHistory: workspaceApiMocks.getHistory,
    parseUrlPrefill: workspaceApiMocks.parseUrlPrefill,
    listJobs: workspaceApiMocks.listJobs,
    clearQueue: workspaceApiMocks.clearQueue,
    cancelJob: workspaceApiMocks.cancelJob,
  };
});


const galleryApiMocks = vi.hoisted(() => ({
  deleteAsset: vi.fn<(assetId: string) => Promise<void>>(),
}));

vi.mock('$lib/api/gallery', async (importOriginal) => {
  const actual = await importOriginal<typeof import('$lib/api/gallery')>();
  return { ...actual, deleteAsset: galleryApiMocks.deleteAsset };
});

const enhanceApiMocks = vi.hoisted(() => ({
  enhancePrompt: vi.fn<(body: unknown) => Promise<void>>(),
}));

vi.mock('$lib/api/promptEnhance', async (importOriginal) => {
  const actual = await importOriginal<typeof import('$lib/api/promptEnhance')>();
  return { ...actual, enhancePrompt: enhanceApiMocks.enhancePrompt };
});

vi.mock('$lib/api/browse', () => browseApiMocks);

vi.mock('$lib/api/promptFiles', () => ({
  inspectPromptFile: promptFileApiMocks.inspectPromptFile,
  writePromptFile: promptFileApiMocks.writePromptFile,
}));

import WorkspacePage from './WorkspacePage.svelte';
import ControlsSidebar from './ControlsSidebar.svelte';
import { chooseInBrowser, listingWith, placesFor } from '../../test-utils/pathBrowser';

let target: HTMLDivElement;
let app: Record<string, unknown> | null = null;

// Every test starts from empty stores, reset API mocks, an empty server-side queue and a fresh host element.
beforeEach(() => {
  localStorage.clear();
  sessionStorage.clear();
  draft.reset();
  extendInspection.reset();
  clipMemory.reset();
  historyStore.seedHistory([]);
  jobStore.clearJob();
  toasts.splice(0, toasts.length);
  for (const mock of [
    workspaceApiMocks.getWorkspaceContext, workspaceApiMocks.getWorkspaceCoreContext, workspaceApiMocks.submitGenerate, workspaceApiMocks.getJobSnapshot,
    workspaceApiMocks.getHistory, workspaceApiMocks.parseUrlPrefill, workspaceApiMocks.listJobs, browseApiMocks.listDirectory, browseApiMocks.browsePlaces,
    promptFileApiMocks.inspectPromptFile, promptFileApiMocks.writePromptFile, extendApiMocks.inspectExtendSource, enhanceApiMocks.enhancePrompt,
    videoMemoryApiMocks.checkVideoMemoryFit, galleryApiMocks.deleteAsset,
  ]) mock.mockReset();
  workspaceApiMocks.clearQueue.mockClear();
  workspaceApiMocks.cancelJob.mockClear();
  workspaceApiMocks.listJobs.mockResolvedValue({ active_job: null, queued_jobs: [] });
  workspaceApiMocks.parseUrlPrefill.mockReturnValue({});
  workspaceApiMocks.getHistory.mockResolvedValue(makeGalleryPage());
  target = document.createElement('div');
  document.body.appendChild(target);
});

afterEach(async () => {
  if (app) { await unmount(app); app = null; }
  target.remove();
  document.body.innerHTML = '';
  vi.restoreAllMocks();
});

function makeImageDefaults(overrides: Partial<ImageModelDefaults> = {}): ImageModelDefaults {
  return {
    ratio: '2:3',
    size: 'm',
    steps: 28,
    guidance: 6.2,
    width: 832,
    height: 1216,
    scheduler: 'beta',
    supports_negative_prompt: true,
    supports_quantize: true,
    quantize: null,
    image_strength: 0.5,
    postprocess: { sharpen: 0.8, contrast: false, saturation: false },
    upscale: {
      enabled: false,
      factor: null,
      denoise: null,
      steps: null,
      guidance: null,
      sharpen: true,
      save_pre: false,
    },
    supports_img2img: true,
    supports_upscale: true,
    supports_json_prompt: false,
    supports_first_sigma: false,
    supports_scheduler: true,
    dimension_min: 16,
    dimension_max: null,
    dimension_step: 16,
    ...overrides,
  };
}

function makeIdeogramDefaults(overrides: Partial<ImageModelDefaults> = {}): ImageModelDefaults {
  return makeImageDefaults({
    ratio: '16:9',
    size: 'l',
    steps: 20,
    guidance: 7,
    width: 1664,
    height: 928,
    supports_negative_prompt: false,
    supports_img2img: false,
    supports_upscale: false,
    supports_json_prompt: true,
    supports_first_sigma: true,
    supports_scheduler: true,
    dimension_min: 256,
    dimension_max: 2048,
    dimension_step: 16,
    ...overrides,
  });
}

function makeVideoDefaults(overrides: Partial<VideoModelDefaults> = {}): VideoModelDefaults {
  return {
    ratio: '16:9',
    size: 'm',
    steps: 8,
    width: 704,
    height: 448,
    frame_count: 49,
    audio: true,
    low_memory: true,
    supports_i2v: true,
    supports_quantize: false,
    quantize: null,
    max_steps: 8,
    fps: 24,
    frame_alignment: 8,
    upscale: {
      enabled: false,
      factor: 2,
      steps: null,
    },
    ...overrides,
  };
}

function makeContext(overrides: Partial<WorkspaceContext> = {}): WorkspaceContext {
  const imageDefaults = makeImageDefaults();
  const alternateImageDefaults = makeImageDefaults({
    ratio: '16:9',
    size: 'l',
    steps: 12,
    guidance: 3.5,
    width: 1216,
    height: 832,
    supports_negative_prompt: false,
    supports_quantize: false,
  });
  const videoDefaults = makeVideoDefaults();

  return {
    image_models: [
      { id: 'zit', label: 'zit', type: 'image' },
      { id: 'flux-lite', label: 'flux-lite', type: 'image' },
    ],
    video_models: [{ id: 'ltx-8', label: 'ltx-8', type: 'video' }],
    loras: [],
    history_assets: [],
    active_job: null,
    defaults: imageDefaults,
    video_defaults: videoDefaults,
    image_model_defaults: {
      zit: imageDefaults,
      'flux-lite': alternateImageDefaults,
    },
    video_model_defaults: {
      'ltx-8': videoDefaults,
    },
    current_image_model: 'zit',
    current_video_model: 'ltx-8',
    config: {
      gallery_page_size: 20,
      startup_view: 'workspace',
    },
    output_dir: '/tmp/output',
    quantize_options: [4, 8],
    image_ratios: ['2:3', '16:9'],
    video_ratios: ['16:9'],
    image_size_options: { '2:3': ['m'], '16:9': ['l'] },
    image_size_dimensions: {},
    video_size_options: { '16:9': ['m'] },
    scheduler_options: ['beta'],
    workflow_contract: {
      values: ['txt2img', 'img2img', 'txt2vid', 'img2vid'],
      definitions: {
        txt2img: {
          mode: 'image',
          model_kind: 'image',
          visible_controls: [
            'workflow', 'model', 'quantize', 'loras', 'prompt_source', 'prompt_inline', 'negative_prompt',
            'prompt_file_path', 'prompt_file_option', 'prompt_file_preview', 'prompt_file_edit',
            'ratio', 'size', 'custom_dimensions', 'runs', 'steps', 'guidance', 'seed', 'scheduler',
            'postprocess_sharpen', 'postprocess_contrast', 'postprocess_saturation',
            'image_upscale_enabled', 'image_upscale_factor', 'image_upscale_denoise', 'image_upscale_steps',
            'image_upscale_guidance', 'image_upscale_sharpen', 'image_upscale_review'
          ],
          supports_reference_image: false,
          requires_reference_image: false,
          requires_source_video: false,
          clear_fields: ['image_path', 'image_strength', 'frames', 'audio', 'low_memory'],
        },
        img2img: {
          mode: 'image',
          model_kind: 'image',
          visible_controls: [
            'workflow', 'model', 'quantize', 'loras', 'prompt_source', 'prompt_inline', 'negative_prompt',
            'prompt_file_path', 'prompt_file_option', 'prompt_file_preview', 'prompt_file_edit',
            'reference_image', 'reference_image_path', 'reference_image_clear',
            'ratio', 'size', 'custom_dimensions', 'runs', 'steps', 'guidance', 'image_strength', 'seed', 'scheduler',
            'postprocess_sharpen', 'postprocess_contrast', 'postprocess_saturation',
            'image_upscale_enabled', 'image_upscale_factor', 'image_upscale_denoise', 'image_upscale_steps',
            'image_upscale_guidance', 'image_upscale_sharpen', 'image_upscale_review'
          ],
          supports_reference_image: true,
          requires_reference_image: true,
          requires_source_video: false,
          clear_fields: ['frames', 'audio', 'low_memory'],
        },
        txt2vid: {
          mode: 'video',
          model_kind: 'video',
          visible_controls: [
            'workflow', 'model', 'loras', 'prompt_source', 'prompt_inline', 'prompt_scene',
            'prompt_file_path', 'prompt_file_option', 'prompt_file_preview', 'prompt_file_edit',
            'ratio', 'size', 'custom_dimensions', 'runs', 'frame_count', 'steps', 'seed', 'audio', 'low_memory',
            'video_upscale_enabled', 'video_upscale_factor', 'video_upscale_review'
          ],
          supports_reference_image: false,
          requires_reference_image: false,
          requires_source_video: false,
          clear_fields: ['negative_prompt', 'guidance', 'image_path', 'image_strength', 'quantize'],
        },
        img2vid: {
          mode: 'video',
          model_kind: 'video',
          visible_controls: [
            'workflow', 'model', 'loras', 'prompt_source', 'prompt_inline', 'prompt_scene',
            'prompt_file_path', 'prompt_file_option', 'prompt_file_preview', 'prompt_file_edit',
            'reference_image', 'reference_image_path', 'reference_image_clear',
            'ratio', 'size', 'custom_dimensions', 'runs', 'frame_count', 'steps', 'seed', 'audio', 'low_memory',
            'video_upscale_enabled', 'video_upscale_factor', 'video_upscale_review'
          ],
          supports_reference_image: true,
          requires_reference_image: true,
          requires_source_video: false,
          clear_fields: ['negative_prompt', 'guidance', 'quantize'],
        },
        extend: {
          mode: 'video',
          model_kind: 'video',
          visible_controls: ['workflow', 'model', 'loras', 'prompt_source', 'prompt_scene', 'prompt_inline', 'source_video', 'source_video_path', 'source_video_clear', 'extend_length', 'runs', 'steps', 'seed', 'audio', 'low_memory'],
          supports_reference_image: false,
          requires_reference_image: false,
          requires_source_video: true,
          clear_fields: ['negative_prompt', 'guidance', 'image_path', 'image_strength', 'quantize', 'ratio', 'size', 'width', 'height', 'frames', 'video_upscale_enabled', 'video_upscale_factor'],
        },
      },
      field_precedence: {
        defaults: ['cli', 'model_variant', 'model_family', 'global'],
        dimensions: 'explicit_width_height_overrides_ratio_size',
      },
    },
    prompt_sources: ['inline', 'file'],
    default_prompt_source: 'inline',
    prompt_file: {
      accepted_extensions: ['.yaml', '.yml'],
      selection_required: true,
      trust_boundary: {
        scope: 'server_host_only',
        manual_entry: 'submitted_value_kept_until_backend_validation',
        picker: 'in_app_browser',
        read_write: 'yaml_files_only',
      },
      help: {
        path: 'Prompt file path.',
        option_required: 'Select an active prompt option before generating.',
        option_optional: 'Select an active prompt option from the file.',
        empty_options: 'This prompt file has no active prompt options.',
        stale_selection: 'The previously selected prompt option is no longer active.',
        loaded: 'Prompt file loaded.',
        ignored_negative_video: 'Negative prompt entries are ignored for video workflows.',
        ignored_negative_unsupported: 'The current image model ignores negative prompt entries.',
        ignored_scene_image: 'Scene entries are ignored for image workflows.',
      },
    },
    ...overrides,
  };
}

async function settle(): Promise<void> {
  await new Promise((resolve) => setTimeout(resolve, 0));
  await new Promise((resolve) => setTimeout(resolve, 0));
  await Promise.resolve();
  await Promise.resolve();
  flushSync();
}

/** Answer the open confirmation dialog. */
async function answerConfirm(confirmed: boolean): Promise<void> {
  const action = confirmed ? 'confirm' : 'cancel';
  (document.querySelector(`[role="alertdialog"] [data-action="${action}"]`) as HTMLButtonElement).click();
  await settle();
}

/** Pick prompt-file prompts through the Choose prompts dialog and confirm. */
async function choosePrompts(container: ParentNode, ids: string[]): Promise<void> {
  (container.querySelector('[data-action="choose-prompts"]') as HTMLButtonElement).click();
  await settle();
  const chooser = document.querySelector('[data-testid="prompt-chooser"]') as HTMLElement;
  for (const box of Array.from(chooser.querySelectorAll<HTMLInputElement>('input[type="checkbox"]'))) {
    if (box.checked !== ids.includes(box.value)) box.click();
  }
  await settle();
  (document.querySelector('[data-action="confirm-prompts"]') as HTMLButtonElement).click();
  await settle();
}

function expectRenderedLabelsToResolveControls(container: ParentNode): void {
  const explicitLabels = Array.from(container.querySelectorAll('label[for]')) as HTMLLabelElement[];
  expect(explicitLabels.length).toBeGreaterThan(0);
  for (const label of explicitLabels) {
    const control = label.control ?? container.querySelector(`[id="${label.htmlFor}"]`);
    expect(
      control,
      `Expected label "${label.textContent?.trim() ?? label.htmlFor}" to resolve control "${label.htmlFor}"`
    ).not.toBeNull();
  }

  const wrappedLabels = Array.from(container.querySelectorAll('label:not([for])')) as HTMLLabelElement[];
  for (const label of wrappedLabels) {
    expect(
      label.querySelector('input, select, textarea'),
      `Expected wrapped label "${label.textContent?.trim() ?? '(unnamed label)'}" to contain a control`
    ).not.toBeNull();
  }
}


/** Mount the Workspace page on the host element and let it settle. */
async function mountWorkspace(context: WorkspaceContext): Promise<void> {
  workspaceApiMocks.getWorkspaceCoreContext.mockResolvedValue(context);
  app = flushSync(() => mount(WorkspacePage, { target }));
  await settle();
}

const field = (id: string) => document.querySelector(`#${id}`) as HTMLTextAreaElement;

function type(id: string, value: string): void {
  const el = field(id);
  el.value = value;
  el.dispatchEvent(new Event('input', { bubbles: true }));
}

/** Wait out the 300 ms input debounce of the memory check and the source check. */
async function waitDebounce(): Promise<void> {
  await new Promise((resolve) => setTimeout(resolve, 340));
  await settle();
}

/** Open the Enhance popover and return its Enhance button. */
async function openEnhanceRun(): Promise<HTMLButtonElement> {
  (document.querySelector('#ws-enhance-toggle') as HTMLButtonElement).click();
  await settle();
  return document.querySelector('#ws-enhance-run') as HTMLButtonElement;
}

/** Enhance with the current settings through the popover. */
async function runEnhance(): Promise<void> {
  (await openEnhanceRun()).click();
  await settle();
}

/** Make the next source check wait until the returned function is called with the server's answer. */
function holdInspection(): (info: ReturnType<typeof sceneSource>) => void {
  let release: (info: ReturnType<typeof sceneSource>) => void = () => undefined;
  extendApiMocks.inspectExtendSource.mockReturnValue(new Promise((resolve) => { release = resolve; }));
  return (info) => release(info);
}

/** Replace the clipboard with a spy and return it. */
function mockClipboard(): ReturnType<typeof vi.fn<(text: string) => Promise<void>>> {
  const writeText = vi.fn<(text: string) => Promise<void>>(async () => undefined);
  Object.defineProperty(navigator, 'clipboard', { value: { writeText }, configurable: true });
  return writeText;
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
    expect((target.querySelector('#ws-model') as HTMLSelectElement | null)?.disabled).toBe(true);
    expect(target.textContent).not.toContain('stale-model');
  });

  it('hydrates workspace authority first, then loads history after mount', async () => {
    const asset = makeAsset({ id: 'out/deferred.png', url: '/media/out/deferred.png', filename: 'deferred.png' });
    workspaceApiMocks.getHistory.mockResolvedValue(makeGalleryPage([asset]));

    await mountWorkspace(makeContext({ history_assets: [] }));

    expect(workspaceApiMocks.getWorkspaceCoreContext).toHaveBeenCalledTimes(1);
    expect(workspaceApiMocks.getWorkspaceContext).not.toHaveBeenCalled();
    expect(workspaceApiMocks.getHistory).toHaveBeenCalledWith(1);
    expect(target.querySelector(`button[aria-label="View ${asset.filename}"]`)).not.toBeNull();
  });

  it('keeps the mascot on stage until the latest output paints, then docks it', async () => {
    const asset = makeAsset({ id: 'out/latest.png', url: '/media/out/latest.png', filename: 'latest.png' });
    workspaceApiMocks.getHistory.mockResolvedValue(makeGalleryPage([asset]));

    await mountWorkspace(makeContext({ history_assets: [asset] }));

    const preview = target.querySelector('.workspace-preview') as HTMLElement;
    expect(preview.querySelector('[data-testid="latest-loading"] svg.mascot')).not.toBeNull();
    expect(preview.querySelector('[data-testid="mascot-dock"]')).toBeNull();

    const image = preview.querySelector(`img[src="${asset.url}"]`) as HTMLImageElement;
    image.dispatchEvent(new Event('load'));
    await settle();

    expect(preview.querySelector('[data-testid="latest-loading"]')).toBeNull();
    expect(preview.querySelector('[data-testid="mascot-dock"] svg.mascot')).not.toBeNull();
  });

  it('looks for history instead of claiming no assets until the first history check settles', async () => {
    let resolveContext!: (context: WorkspaceContext) => void;
    workspaceApiMocks.getWorkspaceCoreContext.mockImplementation(
      () => new Promise<WorkspaceContext>((resolve) => { resolveContext = resolve; })
    );
    let resolveHistory!: (page: ReturnType<typeof makeGalleryPage>) => void;
    workspaceApiMocks.getHistory.mockImplementation(
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
    workspaceApiMocks.getHistory.mockResolvedValue(makeGalleryPage([asset]));

    await mountWorkspace(makeContext({ history_assets: [asset] }));

    // Some browsers ignore preload and fire no load events until play is pressed.
    const preview = target.querySelector('.workspace-preview') as HTMLElement;
    const video = preview.querySelector(`video[src="${asset.url}"]`) as HTMLVideoElement;
    expect(video).not.toBeNull();
    expect(preview.querySelector('[data-testid="latest-loading"]')).toBeNull();
    expect(preview.querySelector('[data-testid="mascot-dock"] svg.mascot')).not.toBeNull();
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

  it('shows a refused Generate above the button, keeps the preview, and clears it when a setting changes', async () => {
    draft.update('prompt', 'A lighthouse at dusk');
    await mountWorkspace(makeContext());
    toasts.splice(0, toasts.length);
    workspaceApiMocks.submitGenerate.mockRejectedValueOnce(new ApiError('rejected', 422, 'Image-to-image requires a reference image.'));

    target.querySelector('form')!.requestSubmit();
    await settle();

    expect(target.querySelector('[data-testid="generate-error"]')?.textContent).toContain('Image-to-image requires a reference image.');
    expect(target.textContent).not.toContain('The Workspace could not load');
    expect(toasts[0]?.type).toBe('error');
    expect(toasts[0]?.message).toBe('Image-to-image requires a reference image.');

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

  function withEnhancer(context: WorkspaceContext): WorkspaceContext {
    context.workflow_contract.definitions.txt2img.visible_controls.push('prompt_enhance', 'prompt_enhance_auto');
    context.prompt_enhancer = {
      matrix: {
        axes: [
          { key: 'style', label: 'Style', multi: false, video_only: false, audio_only: false, options: [{ slug: 'keep', label: 'Keep' }, { slug: 'photo', label: 'Photographic' }], default: ['keep'] },
          { key: 'details', label: 'Details', multi: true, video_only: false, audio_only: false, options: [{ slug: 'lighting', label: 'Lighting' }], default: ['lighting'] },
          { key: 'length', label: 'Length', multi: false, video_only: false, audio_only: false, options: [{ slug: 'same', label: 'Same' }, { slug: 'longer', label: 'Longer' }], default: ['same'] },
          { key: 'motion', label: 'Motion', multi: true, video_only: true, audio_only: false, options: [{ slug: 'action', label: 'Action' }], default: ['action'] },
        ],
        defaults: { style: 'keep', mood: 'keep', details: ['lighting'], length: 'same', motion: ['action'], sound: [], note: '' },
      },
      model: 'owner/llm',
      revision: null,
      downloaded: true,
      download_size_label: null,
      default_max_words: 300,
      error: null,
    };
    return context;
  }

  async function submitForm(): Promise<FormData> {
    const form = target.querySelector('form');
    form!.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
    await settle();
    const [submitted] = workspaceApiMocks.submitGenerate.mock.calls.at(-1) ?? [];
    return submitted as FormData;
  }

  function pressKey(key: string, init: KeyboardEventInit = {}, from: EventTarget = document): KeyboardEvent {
    const event = new KeyboardEvent('keydown', { key, bubbles: true, cancelable: true, ...init });
    from.dispatchEvent(event);
    return event;
  }

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

  it('focuses the prompt with Alt+1 and the settings with Alt+2', async () => {
    await mountWorkspace(makeContext());
    expect(pressKey('1', { altKey: true, code: 'Digit1' }).defaultPrevented).toBe(true);
    expect(document.activeElement?.id).toBe('ws-prompt');
    pressKey('2', { altKey: true, code: 'Digit2' });
    expect(document.activeElement?.closest('.settings-pane')).not.toBeNull();
  });

  it('expands a collapsed sidebar with Alt+1 and Alt+2 before focusing', async () => {
    await mountWorkspace(makeContext());
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

  function jobSnap(jobId: string, status: JobSnapshot['status'], extra: Partial<JobSnapshot> = {}): JobSnapshot {
    return {
      id: jobId, job_id: jobId, workflow: 'txt2img', job_type: 'Text to Image', status, created_at: '2026-10-04T10:00:00Z', completed_at: null,
      event_count: 1, last_event: { type: 'job_started' }, supported_controls: ['quit'], paused: false, result_path: null,
      prompt: `prompt of ${jobId}`, model: 'zit', runs: 1, ...extra,
    };
  }

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
      expect(overlay.querySelector('.review-line')?.textContent).toContain('Review 3 of 8');
      expect(overlay.querySelector('[data-decision="approve"]')).not.toBeNull();
      expect(overlay.querySelector('[data-decision="discard"]')).not.toBeNull();
      expect([...overlay.querySelectorAll('button')].map((b) => b.getAttribute('aria-label'))).toEqual(['Approve', 'Discard']);

      // The running-job view is still there under the overlay, with only Cancel.
      const card = preview.querySelector<HTMLElement>('.job-card')!;
      expect(card.contains(overlay)).toBe(false);
      expect(card.querySelector('button[data-control="quit"]')).not.toBeNull();
      for (const control of ['pause-resume', 'next', 'repeat']) expect(card.querySelector(`button[data-control="${control}"]`)).toBeNull();
    });

    it('sends the decision from the buttons, once', async () => {
      workspaceApiMocks.listJobs.mockResolvedValue({ active_job: reviewSnapshot(), queued_jobs: [] });
      const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => ({}) });
      vi.stubGlobal('fetch', fetchMock);
      try {
        await mountWorkspace(makeContext({ active_job: reviewSnapshot() }));
        target.querySelector<HTMLButtonElement>('[data-decision="approve"]')!.click();
        await settle();
        expect(fetchMock).toHaveBeenCalledWith('/jobs/job-review/controls/approve', expect.objectContaining({ method: 'POST' }));
        // Both buttons wait for the server to resume the job.
        expect(target.querySelector<HTMLButtonElement>('[data-decision="discard"]')!.disabled).toBe(true);
        target.querySelector<HTMLButtonElement>('[data-decision="discard"]')!.click();
        pressKey('n');
        await settle();
        expect(fetchMock).toHaveBeenCalledTimes(1);
      } finally {
        vi.unstubAllGlobals();
      }
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

      const source = (globalThis.EventSource as unknown as { lastInstance: { emit: (type: string, data: unknown) => void } }).lastInstance;
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

    it.each(['y', 'n'])('does not decide a plain pause with %s', async (key) => {
      const plain = jobSnap('job-plain', 'paused', {
        paused: true,
        review: null,
        supported_controls: ['next', 'pause', 'resume', 'repeat', 'approve', 'discard', 'quit'],
        last_event: { type: 'job_paused', event_id: 4 },
      });
      workspaceApiMocks.listJobs.mockResolvedValue({ active_job: plain, queued_jobs: [] });
      const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => ({}) });
      vi.stubGlobal('fetch', fetchMock);
      try {
        await mountWorkspace(makeContext({ active_job: plain }));
        expect(target.querySelector('[data-testid="review-overlay"]')).toBeNull();
        expect(target.querySelector('[data-decision]')).toBeNull();
        pressKey(key);
        await settle();
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
        expect(video.getAttribute('src')).toBe('/jobs/job-vreview/review?v=2');
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

      it('mutes the video and drops the native controls when the job has Audio off', async () => {
        const snapshot = videoReviewSnapshot({ settings: { upscale: '2', audio: 'false' } });
        workspaceApiMocks.listJobs.mockResolvedValue({ active_job: snapshot, queued_jobs: [] });
        await mountWorkspace(makeContext({ active_job: snapshot }));

        const video = target.querySelector<HTMLVideoElement>('[data-testid="review-overlay"] video')!;
        expect(video.muted).toBe(true);
        expect(video.controls).toBe(false);
        expect(target.querySelector('button[data-playback]')).not.toBeNull();
      });

      it.each([
        ['y', 'approve'],
        ['n', 'discard'],
      ])('sends %s as %s', async (key, action) => {
        workspaceApiMocks.listJobs.mockResolvedValue({ active_job: videoReviewSnapshot(), queued_jobs: [] });
        const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => ({}) });
        vi.stubGlobal('fetch', fetchMock);
        try {
          await mountWorkspace(makeContext({ active_job: videoReviewSnapshot() }));
          pressKey(key);
          await settle();
          expect(fetchMock).toHaveBeenCalledWith(`/jobs/job-vreview/controls/${action}`, expect.objectContaining({ method: 'POST' }));
        } finally {
          vi.unstubAllGlobals();
        }
      });
    });

    it('does not decide anything with Y or N while a job runs without a review', async () => {
      const running = jobSnap('job-run', 'running', {
        supported_controls: ['next', 'pause', 'resume', 'repeat', 'approve', 'discard', 'quit'],
        last_event: { type: 'step_progress', event_id: 4 },
      });
      workspaceApiMocks.listJobs.mockResolvedValue({ active_job: running, queued_jobs: [] });
      const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => ({}) });
      vi.stubGlobal('fetch', fetchMock);
      try {
        await mountWorkspace(makeContext({ active_job: running }));
        pressKey('y');
        await settle();
        const decisions = fetchMock.mock.calls.filter(([url]) => /\/controls\/(approve|discard)$/.test(String(url)));
        expect(decisions).toHaveLength(0);
      } finally {
        vi.unstubAllGlobals();
      }
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
      const submitted = await submitForm();
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
      expect(smart.disabled).toBe(true);
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
      const submitted = await submitForm();
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
    draft.update('enhancedFrom', { prompt: 'a fox', mode: 'image' });
    await mountWorkspace(withEnhancer(makeContext()));

    const enhanced = target.querySelector('#ws-enhanced-prompt') as HTMLTextAreaElement | null;
    expect(enhanced?.value).toBe('A red fox in deep snow, golden light.');
    expect(enhanced?.getAttribute('name')).toBeNull();

    const submitted = await submitForm();
    expect(submitted.get('prompt')).toBe('A red fox in deep snow, golden light.');
    expect(submitted.has('enhance_auto')).toBe(false);
  });

  it('marks the Enhanced prompt out of date after the prompt changes', async () => {
    draft.update('prompt', 'a fox');
    draft.update('enhancedPrompt', 'A red fox.');
    draft.update('enhancedFrom', { prompt: 'a fox', mode: 'image' });
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
    draft.update('enhanceSettings', { style: 'photo', mood: 'keep', details: [], length: 'longer', motion: ['action'], sound: [], note: '' });
    await mountWorkspace(withEnhancer(makeContext()));

    expect((target.querySelector('#ws-enhanced-prompt') as HTMLTextAreaElement).disabled).toBe(true);
    const submitted = await submitForm();
    expect(submitted.get('prompt')).toBe('a fox');
    expect(submitted.get('enhance_auto')).toBe('true');
    expect(JSON.parse(String(submitted.get('enhance_settings')))).toEqual({ style: 'photo', mood: 'keep', details: [], length: 'longer' });
  });

  it('copies the CLI command for what Generate would submit, without submitting', async () => {
    const writeText = mockClipboard();
    draft.update('prompt', 'a fox');
    draft.update('enhancedPrompt', 'A red fox in deep snow.');
    draft.update('enhancedFrom', { prompt: 'a fox', mode: 'image' });
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

  it('copies nothing while the prompt is empty', async () => {
    const writeText = mockClipboard();
    await mountWorkspace(makeContext());

    (target.querySelector('#ws-copy-cli') as HTMLButtonElement).click();
    await settle();

    expect(writeText).not.toHaveBeenCalled();
  });

  it('warns inline when auto-enhance would change nothing', async () => {
    draft.update('prompt', 'a fox');
    draft.update('enhanceAuto', true);
    draft.update('enhanceSettings', { style: 'keep', mood: 'keep', details: [], length: 'same', motion: [], sound: [], note: '' });
    await mountWorkspace(withEnhancer(makeContext()));
    await openEnhanceRun();
    // The enhancer popover is moved to <body>.
    expect(document.querySelector('#ws-enhance-panel [role="alert"]')).not.toBeNull();
  });

  describe('scene and prompt pair (video)', () => {
    function withVideoEnhancer(context: WorkspaceContext): WorkspaceContext {
      withEnhancer(context);
      for (const workflow of ['txt2vid', 'img2vid'] as const) {
        context.workflow_contract.definitions[workflow].visible_controls.push('prompt_enhance', 'prompt_enhance_auto');
      }
      return context;
    }

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
      const submitted = await submitForm();
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

    it('lets a video prompt without a scene be enhanced with the default settings, because the split is the change', async () => {
      draft.update('workflow', 'txt2vid');
      draft.patch({ prompt: 'a knight rides in', enhanceSettings: { style: 'keep', mood: 'keep', details: [], length: 'same', motion: [], sound: [], note: '' } });
      await mountWorkspace(withVideoEnhancer(makeContext()));
      expect((await openEnhanceRun()).disabled).toBe(false);

      // With a scene of its own the same settings change nothing.
      type('ws-scene', 'A castle at dusk.');
      await settle();
      expect((document.querySelector('#ws-enhance-run') as HTMLButtonElement).disabled).toBe(true);
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

    it.each([
      ['neither part', {}, 'A castle.', 'a knight'],
      ['both parts', { enhancedScene: 'A castle at dusk.', enhancedPrompt: 'A knight rides in.' }, 'A castle at dusk.', 'A knight rides in.'],
    ])('sends each part enhanced when its Enhanced box has text: %s', async (_name, enhanced, scene, prompt) => {
      draft.update('workflow', 'txt2vid');
      draft.patch({ scene: 'A castle.', prompt: 'a knight', enhancedFrom: null, ...enhanced });
      await mountWorkspace(withVideoEnhancer(makeContext()));

      const submitted = await submitForm();
      expect(submitted.get('scene')).toBe(scene);
      expect(submitted.get('prompt')).toBe(prompt);
      // The user's own text is recorded only when some part ran enhanced.
      expect(submitted.has('scene_input')).toBe(Object.keys(enhanced).length > 0);
      expect(submitted.has('prompt_input')).toBe(Object.keys(enhanced).length > 0);
      if (Object.keys(enhanced).length > 0) {
        expect(submitted.get('scene_input')).toBe('A castle.');
        expect(submitted.get('prompt_input')).toBe('a knight');
      }
    });

    it('records the user text and how an enhanced image prompt was made', async () => {
      const record = { settings: { style: 'photo' }, model: 'org/fast@r1', format: 'text' as const };
      draft.patch({ prompt: 'a fox in snow', enhancedPrompt: 'A red fox curls up in deep snow.', enhancedFrom: { prompt: 'a fox in snow', mode: 'image', record } });
      await mountWorkspace(withEnhancer(makeContext()));

      const submitted = await submitForm();
      expect(submitted.get('prompt')).toBe('A red fox curls up in deep snow.');
      expect(submitted.get('prompt_input')).toBe('a fox in snow');
      expect(JSON.parse(String(submitted.get('enhance_record')))).toEqual(record);
    });

    it('sends no enhance record without Enhanced text', async () => {
      draft.patch({ prompt: 'a fox in snow', enhancedPrompt: '', enhancedFrom: null });
      await mountWorkspace(withEnhancer(makeContext()));

      const submitted = await submitForm();
      expect(submitted.has('enhance_record')).toBe(false);
      expect(submitted.has('prompt_input')).toBe(false);
    });

    describe('Rewrite row', () => {
      const chip = (label: string) => Array.from(document.querySelectorAll('#ws-enhance-panel [role="radiogroup"] [role="radio"]')).find((el) => el.textContent?.trim() === label) as HTMLButtonElement;
      const rewriteChips = () => Array.from(document.querySelectorAll('#ws-enhance-panel [aria-labelledby="ws-enhance-rewrite-label"] [role="radio"]')) as HTMLButtonElement[];

      it('comes first in the popover, for the video workflows only', async () => {
        draft.update('workflow', 'txt2vid');
        await mountWorkspace(withVideoEnhancer(makeContext()));
        await openEnhanceRun();
        expect(document.querySelector('#ws-enhance-rewrite-label')?.textContent?.trim()).toBe('Rewrite');
        const rewriteLabel = document.querySelector('#ws-enhance-rewrite-label')!;
        const styleLabel = document.querySelector('#ws-enhance-panel [id$="-style-label"]')!;
        expect(rewriteLabel.compareDocumentPosition(styleLabel) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
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
        expect(draft.state.enhanceSettings?.partsByWorkflow).toEqual({ txt2vid: 'action' });
      });

      it('disables Action only with the reason while the Scene is blank, and the run uses both', async () => {
        enhanceApiMocks.enhancePrompt.mockResolvedValue();
        draft.update('workflow', 'txt2vid');
        draft.patch({ prompt: 'a knight rides in', enhanceSettings: { style: 'photo', mood: 'keep', details: [], length: 'same', motion: [], sound: [], note: '', partsByWorkflow: { txt2vid: 'action' } } });
        await mountWorkspace(withVideoEnhancer(makeContext()));
        await openEnhanceRun();

        const actionOnly = chip('Action only');
        expect(actionOnly.disabled).toBe(true);
        expect(chip('Scene and action').getAttribute('aria-checked')).toBe('true');
        // The reason is announced with the disabled chip.
        const reason = document.getElementById(actionOnly.getAttribute('aria-describedby') ?? '');
        expect(reason?.textContent).toContain('Write a scene first');

        (document.querySelector('#ws-enhance-run') as HTMLButtonElement).click();
        await settle();
        expect(enhanceApiMocks.enhancePrompt.mock.calls[0]?.[0]).toMatchObject({ settings: { parts: 'both' } });
        // The remembered choice survives and applies again once there is a scene.
        expect(draft.state.enhanceSettings?.partsByWorkflow).toEqual({ txt2vid: 'action' });
        type('ws-scene', 'A castle.');
        await settle();
        expect(chip('Action only').disabled).toBe(false);
        expect(chip('Action only').getAttribute('aria-checked')).toBe('true');
      });

      it('sends parts with an on-demand enhancement and keeps the memory out of the payload', async () => {
        enhanceApiMocks.enhancePrompt.mockResolvedValue();
        draft.update('workflow', 'txt2vid');
        draft.patch({ scene: 'A castle.', prompt: 'a knight' });
        await mountWorkspace(withVideoEnhancer(makeContext()));
        await openEnhanceRun();
        chip('Action only').click();
        await settle();
        (document.querySelector('#ws-enhance-run') as HTMLButtonElement).click();
        await settle();
        const body = enhanceApiMocks.enhancePrompt.mock.calls[0]?.[0] as { settings: Record<string, unknown> };
        expect(body.settings.parts).toBe('action');
        expect(body.settings).not.toHaveProperty('partsByWorkflow');
      });

      it('sends parts inside enhance_settings with auto-enhance, and none for images', async () => {
        draft.update('workflow', 'txt2vid');
        draft.patch({ scene: 'A castle.', prompt: 'a knight', enhanceAuto: true });
        await mountWorkspace(withVideoEnhancer(makeContext()));
        expect(JSON.parse(String((await submitForm()).get('enhance_settings'))).parts).toBe('both');

        draft.patch({ enhanceSettings: { style: 'keep', mood: 'keep', details: [], length: 'same', motion: [], sound: [], note: '', partsByWorkflow: { txt2vid: 'action' } } });
        await settle();
        expect(JSON.parse(String((await submitForm()).get('enhance_settings'))).parts).toBe('action');
      });
    });

    describe('markers', () => {
      const labelRow = (id: string) => target.querySelector(`label[for="${id}"]`)!.closest('.label-row') as HTMLElement;
      /** The text beside the label: clip links and the enhanced marker. */
      const rowText = (id: string) => (labelRow(id).querySelector('.label-extras')?.textContent ?? '').replace(/\s+/g, ' ').trim();
      const useMine = (id: string) => Array.from(labelRow(id).querySelectorAll('button')).find((el) => el.textContent?.trim() === 'Use mine') as HTMLButtonElement | undefined;
      const status = () => target.querySelector('[role="status"]')?.textContent?.trim();

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
        // The pill moves to the Enhanced tab only.
        expect(target.querySelector('#ws-tab-prompt')?.textContent?.trim()).toBe('Your text');
        expect(target.querySelector('#ws-tab-enhanced')?.textContent).toContain('used');

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
        const submitted = await submitForm();
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
        expect(target.querySelector('#ws-tab-enhanced')?.textContent).toContain('auto');
      });

      it('marks a part out of date only while its pair is stale, and keeps the tab badge', async () => {
        draft.update('workflow', 'txt2vid');
        draft.patch({ scene: 'A castle.', prompt: 'a knight', enhancedPrompt: 'A knight rides in.', enhancedFrom: { scene: 'A castle.', prompt: 'a knight', mode: 'video' } });
        await mountWorkspace(withVideoEnhancer(makeContext()));
        expect(rowText('ws-prompt')).not.toContain('Out of date');

        type('ws-scene', 'A harbour.');
        await settle();
        expect(rowText('ws-prompt')).toContain('Uses enhanced Out of date Use mine');
        // The scene is not enhanced, so what it sends is current.
        expect(rowText('ws-scene')).toBe('');
        expect(target.querySelector('#ws-tab-enhanced')?.textContent).toContain('Out of date');

        type('ws-scene', 'A castle.');
        await settle();
        expect(rowText('ws-prompt')).not.toContain('Out of date');
      });

      it('empties both Enhanced boxes with Clear, and Enhance again starts from the user text', async () => {
        enhanceApiMocks.enhancePrompt.mockResolvedValue();
        draft.update('workflow', 'txt2vid');
        draft.patch({ scene: 'A castle.', prompt: 'a knight', enhancedScene: 'A castle at dusk.', enhancedPrompt: 'A knight rides in.', enhancedFrom: { scene: 'A castle.', prompt: 'a knight', mode: 'video' } });
        await mountWorkspace(withVideoEnhancer(makeContext()));
        (target.querySelector('#ws-tab-enhanced') as HTMLButtonElement).click();
        await settle();
        const clear = Array.from(target.querySelectorAll('.prompt-link')).find((el) => el.textContent?.trim() === 'Clear') as HTMLButtonElement;
        clear.click();
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

        draft.patch({ enhanceSettings: { style: 'keep', mood: 'keep', details: [], length: 'same', motion: [], sound: [], note: '', partsByWorkflow: { txt2vid: 'action' } } });
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

    it('copies a scene alone as --scene without --prompt', async () => {
      const writeText = mockClipboard();
      draft.update('workflow', 'txt2vid');
      draft.patch({ scene: 'A castle.', prompt: '' });
      await mountWorkspace(makeContext());

      (target.querySelector('#ws-copy-cli') as HTMLButtonElement).click();
      await settle();
      expect(writeText).toHaveBeenCalledTimes(1);
      expect(writeText.mock.calls[0]?.[0]).toContain("--scene 'A castle.'");
      expect(writeText.mock.calls[0]?.[0]).not.toContain('--prompt');
    });

    it('focuses the Prompt box, not the Scene box, with Alt+1', async () => {
      draft.update('workflow', 'txt2vid');
      await mountWorkspace(makeContext());
      pressKey('1', { altKey: true, code: 'Digit1' });
      await settle();
      expect(document.activeElement).toBe(field('ws-prompt'));
    });

    it('does not warn that auto-enhance changes nothing for a video with a blank scene, but does once a scene is set', async () => {
      draft.update('workflow', 'txt2vid');
      draft.patch({ prompt: 'a knight', enhanceAuto: true, enhanceSettings: { style: 'keep', mood: 'keep', details: [], length: 'same', motion: [], sound: [], note: '' } });
      await mountWorkspace(withVideoEnhancer(makeContext()));
      await openEnhanceRun();
      expect(document.querySelector('#ws-enhance-panel [role="alert"]')).toBeNull();

      type('ws-scene', 'A castle.');
      await settle();
      expect(document.querySelector('#ws-enhance-panel [role="alert"]')).not.toBeNull();
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

        const submitted = await submitForm();
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

  it('uses backend visible_controls instead of workflow-name literals for sidebar visibility', async () => {
    const context = makeContext({
      workflow_contract: {
        ...makeContext().workflow_contract,
        definitions: {
          ...makeContext().workflow_contract.definitions,
          txt2vid: {
            ...makeContext().workflow_contract.definitions.txt2vid,
            visible_controls: ['workflow', 'model', 'prompt_inline', 'ratio', 'size', 'custom_dimensions', 'runs', 'steps', 'seed'],
          },
        },
      },
    });
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

    expect(target.querySelector('input[name="audio"]')).toBeNull();
    expect(target.querySelector('input[name="low_memory"]')).toBeNull();
    expect(target.querySelector('input[name="frames"]')).toBeNull();
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

  it('shows scheduler for txt2img and hides it for txt2vid', async () => {
    const context = makeContext();
    await mountWorkspace(context);

    expect(target.querySelector('#ws-scheduler')).not.toBeNull();

    draft.update('workflow', 'txt2vid');
    draft.update('model', 'ltx-8');
    await settle();

    expect(target.querySelector('#ws-scheduler')).toBeNull();
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

    const modelSelect = target.querySelector('#ws-model') as HTMLSelectElement | null;
    expect(modelSelect).not.toBeNull();
    modelSelect!.value = 'zit';
    modelSelect!.dispatchEvent(new Event('change', { bubbles: true }));
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

    const quantizeSelect = target.querySelector('select[name="quantize"]') as HTMLSelectElement;
    quantizeSelect.value = '8';
    quantizeSelect.dispatchEvent(new Event('change', { bubbles: true }));
    await settle();
    expect(fitBadge()?.getAttribute('data-status')).toBe('tight');

    const modelSelect = target.querySelector('#ws-model') as HTMLSelectElement;
    expect(Array.from(modelSelect.options).map((option) => option.textContent)).toEqual(['zit', 'flux-lite (not downloaded)']);
    modelSelect.value = 'flux-lite';
    modelSelect.dispatchEvent(new Event('change', { bubbles: true }));
    await settle();
    expect(target.querySelector('[data-testid="model-download-status"]')?.textContent).toContain('Not downloaded');
    expect(fitBadge()).toBeNull();
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

    const modelSelect = target.querySelector('#ws-model') as HTMLSelectElement | null;
    expect(modelSelect).not.toBeNull();
    modelSelect!.value = 'zit';
    modelSelect!.dispatchEvent(new Event('change', { bubbles: true }));
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

    const modelSelect = target.querySelector('#ws-model') as HTMLSelectElement | null;
    expect(modelSelect).not.toBeNull();
    modelSelect!.value = 'zit';
    modelSelect!.dispatchEvent(new Event('change', { bubbles: true }));
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

    const modelSelect = target.querySelector('#ws-model') as HTMLSelectElement | null;
    expect(modelSelect).not.toBeNull();
    modelSelect!.value = 'zit';
    modelSelect!.dispatchEvent(new Event('change', { bubbles: true }));
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

      const factorSelect = target.querySelector('#ws-video-upscale-factor') as HTMLSelectElement | null;
      expect(factorSelect).not.toBeNull();
      expect(Array.from(factorSelect!.options).map((option) => option.value)).toEqual(['2']);

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

  it('renders toolbar selector controls for txt2img with zit model', async () => {
    draft.update('workflow', 'txt2img');
    draft.update('model', 'zit');

    const context = makeContext();
    await mountWorkspace(context);

    // Model select is always rendered.
    expect(target.querySelector('#ws-model')).not.toBeNull();

    // The quantize block is gated on {#if supportsQuantize} which requires the
    // async context to load. An extra settle() lets the onMount promise chain
    // (getWorkspaceContext → context = ctx → reactive re-render) complete.
    await settle();
    expect(target.querySelector('select[name="quantize"]'), 'Quantize select must be rendered for zit model (supports_quantize: true)').not.toBeNull();
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

  it('renders model and quantize selects enabled for txt2img with zit model', async () => {
    draft.update('workflow', 'txt2img');
    draft.update('model', 'zit');

    const context = makeContext();
    await mountWorkspace(context);

    const modelSelect = target.querySelector('#ws-model') as HTMLSelectElement | null;
    expect(modelSelect, 'Model select must be rendered').not.toBeNull();
    expect(modelSelect!.disabled, 'Model select must be enabled after context loads').toBe(false);

    // Wait for the quantize block (requires async context to load and supportsQuantize=true).
    await settle();
    const quantizeSelect = target.querySelector('select[name="quantize"]') as HTMLSelectElement | null;
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

    const mockEventSource = (globalThis.EventSource as unknown as {
      lastInstance: { emit: (type: string, data: unknown) => void };
    }).lastInstance;
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

    const mockEventSource = (globalThis.EventSource as unknown as {
      lastInstance: { emit: (type: string, data: unknown) => void };
    }).lastInstance;
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
    const source = (globalThis.EventSource as unknown as {
      lastInstance: { emit: (type: string, data: unknown) => void };
    }).lastInstance;
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
    expect(historyStore.assets.map((item) => item.id)).toEqual([secondAsset.id, firstAsset.id]);
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

    workspaceApiMocks.getHistory.mockClear();
    source.emit('job_completed', { type: 'job_completed', job_id: 'job-remount', total_runs: 1, outputs: [] });
    await settle();

    expect(jobStore.current?.status).toBe('completed');
    expect(source.closeCalls).toBe(1);
    expect(workspaceApiMocks.getHistory).toHaveBeenCalledOnce();
    expect((target.querySelector('#ws-submit') as HTMLButtonElement | null)?.disabled).toBe(false);
    expect((target.querySelector('#ws-submit') as HTMLButtonElement | null)?.textContent).not.toContain('Add to queue');
  });

  it('reconnects the active job across workspace remounts from stored continuity state', async () => {
    const context = makeContext({ active_job: null });
    const runningSnapshot: JobSnapshot = {
      id: 'job-reconnect',
      job_id: 'job-reconnect',
      workflow: 'txt2img',
      job_type: 'Text to Image',
      status: 'running',
      created_at: '2026-04-30T10:00:00Z',
      completed_at: null,
      event_count: 2,
      last_event: {
        type: 'step_progress',
        current_step: 3,
        total_steps: 12,
        elapsed_secs: 5,
        eta_secs: 14,
      },
      supported_controls: ['next', 'pause', 'resume', 'repeat', 'quit'],
      paused: false,
      result_path: null,
      prompt: 'Resume me after remount',
      model: 'zit',
      runs: 2,
    };

    workspaceApiMocks.getJobSnapshot.mockResolvedValue(runningSnapshot);
    sessionStorage.setItem('ziv-active-job-id-v1', 'job-reconnect');

    await mountWorkspace(context);

    expect(workspaceApiMocks.getJobSnapshot).toHaveBeenCalledWith('job-reconnect');
    expect(target.textContent).toContain('Resume me after remount');
    // A running job no longer blocks Generate; it adds to the queue.
    expect((target.querySelector('#ws-submit') as HTMLButtonElement | null)?.disabled).toBe(false);
    expect((target.querySelector('#ws-submit') as HTMLButtonElement | null)?.textContent).toContain('Add to queue');

    await unmount(app!);
    app = null;

    jobStore.clearJob();
    sessionStorage.setItem('ziv-active-job-id-v1', 'job-reconnect');
    workspaceApiMocks.getJobSnapshot.mockClear();

    await mountWorkspace(context);

    expect(workspaceApiMocks.getJobSnapshot).toHaveBeenCalledWith('job-reconnect');
    expect(target.textContent).toContain('Resume me after remount');
    // A running job no longer blocks Generate; it adds to the queue.
    expect((target.querySelector('#ws-submit') as HTMLButtonElement | null)?.disabled).toBe(false);
    expect((target.querySelector('#ws-submit') as HTMLButtonElement | null)?.textContent).toContain('Add to queue');
  });

  it('hides image-only controls (negative prompt, quantize) on a txt2vid URL reuse landing', async () => {
    // Simulate a reuse landing: draft is left in txt2vid state (persisted to
    // localStorage by router.navigate + URL-param application on prior navigation).
    // WorkspacePage.onMount calls draft.loadDraft() which reads this, then
    // hydrateFromContext confirms video defaults — image-only controls must be absent.
    draft.update('workflow', 'txt2vid');

    const context = makeContext();
    await mountWorkspace(context);

    // Video workflow: image-only controls must be absent.
    expect(target.querySelector('#ws-negative-prompt')).toBeNull();
    expect(target.querySelector('select[name="quantize"]')).toBeNull();
    expect(target.querySelector('input[name="guidance"]')).toBeNull();
    // Video controls must be present.
    expect(target.querySelector('input[name="audio"]')).not.toBeNull();
    expect(target.querySelector('input[name="low_memory"]')).not.toBeNull();
    // Reference image must be absent (txt2vid, not img2vid).
    expect(target.querySelector('#ws-image-file')).toBeNull();
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

  it('keeps the rejected prompt-file path visible when a manual reload fails', async () => {
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

    promptFileApiMocks.inspectPromptFile.mockRejectedValueOnce(new Error('POST /api/prompt-files/inspect → 422: missing file'));
    pathInput!.value = '/missing/prompts.yaml';
    pathInput!.dispatchEvent(new Event('input', { bubbles: true }));
    pathInput!.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
    await settle();

    const hiddenPath = target.querySelector('input[name="prompts_file"]') as HTMLInputElement | null;
    expect(pathInput!.value).toBe('/missing/prompts.yaml');
    expect(hiddenPath?.value).toBe('/missing/prompts.yaml');
    expect(target.querySelectorAll('input[name="prompt_option_id"]')).toHaveLength(0);
    expect((target.querySelector('#ws-submit') as HTMLButtonElement | null)?.disabled).toBe(true);
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

  it('invalidates prompt-file options when the visible path is manually changed', async () => {
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

  it('revokes the reference image blob URL on teardown to prevent memory leaks', async () => {
    const revokeObjectURL = vi.fn();
    Object.defineProperty(URL, 'revokeObjectURL', { configurable: true, value: revokeObjectURL });

    const context = makeContext();
    draft.update('workflow', 'img2img');
    draft.hydrateFromContext(context, 'zit');

    const imageFile = new File(['img-data'], 'ref.png', { type: 'image/png' });
    app = flushSync(() => mount(ControlsSidebar, {
      target,
      props: { context, busy: false, imageFile, onImageFileChange: vi.fn() },
    }));
    await settle();

    // The effect must have created a blob URL for the image file.
    expect(URL.createObjectURL).toHaveBeenCalledWith(imageFile);

    // Unmounting triggers the $effect cleanup, which must revoke the URL.
    await unmount(app!);
    app = null;

    expect(revokeObjectURL).toHaveBeenCalledWith('blob:test-image');
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
});

function makeAsset(overrides: Partial<GalleryAsset> = {}): GalleryAsset {
  return {
    id: 'output/result.png',
    url: '/media/output/result.png',
    thumbnail_url: '/media/output/result.png',
    filename: 'result.png',
    created_at: '2026-04-30T12:00:00Z',
    workflow: 'txt2img',
    prompt: 'Test completed output',
    model: 'zit',
    media_type: 'image',
    reuse_workspace_url: '#/workspace?workflow=txt2img',
    ...overrides,
  };
}

function makeGalleryPage(assets: GalleryAsset[] = []): GalleryPage {
  return {
    assets,
    page: 1,
    total_pages: 1,
    total_count: assets.length,
  };
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
    const mockEventSource = (globalThis.EventSource as unknown as {
      lastInstance: { emit: (type: string, data: unknown) => void };
    }).lastInstance;
    mockEventSource.emit('job_completed', { type: 'job_completed', job_id: 'job-promo', total_runs: 2, outputs: [completedAsset, secondAsset] });
    await settle();

    expect(target.textContent).toContain('Completed outputs');
    expect(target.querySelectorAll('.completed-output-grid .asset-tile')).toHaveLength(2);
    expect(target.querySelector(`.completed-output-grid img[src="${completedAsset.url}"]`)).not.toBeNull();
    expect(target.querySelector(`.completed-output-grid img[src="${secondAsset.url}"]`)).not.toBeNull();
    expect(target.textContent).not.toContain('Waiting for worker allocation...');
  });

  it.each([
    ['job_failed', 'failed'],
    ['job_cancelled', 'stopped'],
  ] as const)('adds generation_finished assets to the history before %s and keeps them afterward', async (terminal, tone) => {
    await mountWorkspace(makeContext());
    target.querySelector('form')!.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
    await settle();
    const source = (globalThis.EventSource as unknown as {
      lastInstance: { emit: (type: string, data: unknown) => void };
    }).lastInstance;
    const output = makeAsset({ id: `out/${terminal}.png`, filename: `${terminal}.png` });

    source.emit('generation_finished', {
      type: 'generation_finished', job_id: 'job-promo', status: 'success', run_index: 0, asset: output,
    });
    await settle();
    expect(target.querySelector(`.job-card img[alt="${output.filename}"]`)).toBeNull();
    expect(target.querySelector(`.history-strip img[src="${output.thumbnail_url || output.url}"]`)).not.toBeNull();

    source.emit(terminal, { type: terminal, job_id: 'job-promo' });
    await settle();
    expect(target.querySelector(`.job-card[data-tone="${tone}"]`)).not.toBeNull();
    expect(target.querySelector(`.history-strip img[src="${output.thumbnail_url || output.url}"]`)).not.toBeNull();
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
    historyStore.seedHistory([staleHistoryAsset]);

    const form = target.querySelector('form');
    form!.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
    await settle();

    const mockEventSource = (globalThis.EventSource as unknown as {
      lastInstance: { emit: (type: string, data: unknown) => void };
    }).lastInstance;
    mockEventSource.emit('job_completed', { type: 'job_completed', job_id: 'job-promo', total_runs: 2, outputs: [completedAsset, secondAsset] });
    await settle();

    const secondButton = target.querySelector(`button[aria-label="View ${secondAsset.filename} fullscreen"]`);
    expect(secondButton).not.toBeNull();
    secondButton!.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    await settle();

    expect(target.querySelector('[data-testid="asset-viewer"]')).not.toBeNull();
    const lightboxImage = target.querySelector('[data-testid="asset-viewer"] img') as HTMLImageElement | null;
    expect(lightboxImage?.getAttribute('src')).toBe(secondAsset.url);
    expect(document.activeElement?.getAttribute('aria-label')).toBe('Close viewer');

    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowLeft', bubbles: true }));
    await settle();
    expect((target.querySelector('[data-testid="asset-viewer"] img') as HTMLImageElement | null)?.getAttribute('src')).toBe(completedAsset.url);

    const closeButton = target.querySelector('[data-testid="asset-viewer"] button[aria-label="Close viewer"]');
    closeButton!.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    await settle();
    expect(document.activeElement).toBe(secondButton);
  });
});

describe('WorkspacePage history viewer (REC-UX-002)', () => {

  it('clicking a history row opens the workspace viewer lightbox', async () => {
    const asset = makeAsset({ id: 'out/first.png', url: '/media/out/first.png', filename: 'first.png' });
    workspaceApiMocks.getHistory.mockResolvedValue(makeGalleryPage([asset]));
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
    workspaceApiMocks.getHistory.mockResolvedValue(makeGalleryPage(assets));
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

  it('reuses the settings of an asset in place without leaving the workspace', async () => {
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
    expect(historyStore.assets.map((asset) => asset.id)).toEqual([first.id]);
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

  it('keeps changed settings when the workspace is opened again', async () => {
    await mountWithHistory([]);
    const steps = target.querySelector('#ws-steps') as HTMLInputElement;
    steps.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowDown', bubbles: true }));
    await settle();
    expect(draft.state.steps).toBe(27);

    await unmount(app!);
    app = null;
    await mountWithHistory([]);

    expect((target.querySelector('#ws-steps') as HTMLInputElement).value).toBe('27');
  });

  it('allows batches of up to 100', async () => {
    await mountWithHistory([]);
    const runs = target.querySelector('#ws-runs') as HTMLInputElement;
    runs.value = '250';
    runs.dispatchEvent(new Event('input', { bubbles: true }));
    runs.dispatchEvent(new Event('change', { bubbles: true }));
    await settle();
    expect(draft.state.runs).toBe(100);
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

  it('returns to the preset size when reusing an asset after a custom size', async () => {
    const asset = makeAsset({
      id: 'out/first.png',
      filename: 'first.png',
      has_reusable_config: true,
      reuse_workspace_url: '#/workspace?workflow=txt2img&prompt=p&ratio=2%3A3&size=m&width=832&height=1216',
    });
    await mountWithHistory([asset]);
    draft.patch({ dimensionMode: 'custom', width: 1000, height: 1000 });
    await settle();

    (historyTile(asset).querySelector(`button[aria-label="Reuse settings from ${asset.filename}"]`) as HTMLButtonElement).click();
    await settle();

    const submitted = new FormData(target.querySelector('form')!);
    expect(submitted.get('ratio')).toBe('2:3');
    expect(submitted.get('size')).toBe('m');
    expect(submitted.has('width')).toBe(false);
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
    workspaceApiMocks.getHistory.mockResolvedValue(makeGalleryPage([asset]));
    workspaceApiMocks.getWorkspaceCoreContext.mockResolvedValue(context);
    app = flushSync(() => mount(WorkspacePage, { target }));
    await settle();

    openTileMenu(asset);
    const forImage = document.querySelector('[role="menu"] [data-action="reference-image"]') as HTMLButtonElement;
    expect(forImage.getAttribute('aria-disabled')).toBe('true');
    expect(forImage.title).toContain("can't use a reference image");
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

  it('carries the asset prompt when Use as reference keeps the workflow', async () => {
    const asset = makeAsset({ id: 'out/first.png', filename: 'first.png', file_path: '/outputs/out/first.png', prompt: 'a red fox' });
    await mountWithHistory([asset]);
    draft.update('workflow', 'img2img');
    draft.patch({ prompt: 'old prompt', enhancedPrompt: 'old enhanced', steps: 9 });
    await settle();

    openTileMenu(asset);
    (document.querySelector('[role="menu"] [data-action="reference-image"]') as HTMLButtonElement).click();
    await settle();

    expect(draft.state).toMatchObject({ workflow: 'img2img', prompt: 'a red fox', enhancedPrompt: '', steps: 9, referenceImagePath: '/outputs/out/first.png' });
    expect((target.querySelector('#ws-prompt') as HTMLTextAreaElement).value).toBe('a red fox');
  });

  it('keeps settings when Use as reference switches the workflow', async () => {
    const asset = makeAsset({ id: 'out/first.png', filename: 'first.png', file_path: '/outputs/out/first.png' });
    await mountWithHistory([asset]);
    draft.patch({ steps: 9, runs: 40 });
    await settle();

    openTileMenu(asset);
    (document.querySelector('[role="menu"] [data-action="reference-image"]') as HTMLButtonElement).click();
    await settle();

    expect(draft.state).toMatchObject({ workflow: 'img2img', prompt: asset.prompt, steps: 9, runs: 40, referenceImagePath: '/outputs/out/first.png' });
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

  it('never submits a preset the model is too small for after a ratio change', async () => {
    const constrained = makeIdeogramDefaults({ ratio: '1:1', size: 'l', dimension_max: 1024 });
    workspaceApiMocks.getWorkspaceCoreContext.mockResolvedValue(makeContext({
      image_models: [{ id: 'ideo', label: 'ideo', type: 'image' }],
      defaults: constrained,
      current_image_model: 'ideo',
      image_model_defaults: { ideo: constrained },
      image_ratios: ['1:1', '16:9'],
      image_size_options: { '1:1': ['m', 'l'], '16:9': ['m', 'l'] },
      image_size_dimensions: { '1:1': { m: [768, 768], l: [1024, 1024] }, '16:9': { m: [1024, 576], l: [1344, 768] } },
    }));
    app = flushSync(() => mount(WorkspacePage, { target }));
    await settle();
    expect(new FormData(target.querySelector('form')!).get('size')).toBe('l');

    const ratio169 = Array.from(target.querySelectorAll<HTMLButtonElement>('[role="group"][aria-label="Aspect ratio"] button')).find((b) => b.textContent === '16:9')!;
    ratio169.click();
    await settle();

    expect(new FormData(target.querySelector('form')!).get('size')).toBe('m');
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
        { key: 'style', label: 'Style', multi: false, video_only: false, audio_only: false, options: [{ slug: 'keep', label: 'Keep' }], default: ['keep'] },
        { key: 'motion', label: 'Motion', multi: true, video_only: true, audio_only: false, options: [{ slug: 'action', label: 'Action' }], default: ['action'] },
      ],
      defaults: { style: 'keep', mood: 'keep', details: [], length: 'same', motion: ['action'], sound: [], note: '' },
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

    it('keeps edits when the same source is checked again', async () => {
      await mountExtend();
      await chooseSource(sceneSource());
      type('ws-scene', 'My own scene.');
      type('ws-prompt', 'My own action.');
      await settle();

      extendApiMocks.inspectExtendSource.mockResolvedValue(sceneSource({ scene: 'Changed on disk.' }));
      typePath(SOURCE, 'Enter');
      await settle();
      expect(extendApiMocks.inspectExtendSource).toHaveBeenCalledTimes(2);
      expect(field('ws-scene').value).toBe('My own scene.');
      expect(field('ws-prompt').value).toBe('My own action.');
    });

    it('checks the source again when the length changes, so the memory note follows', async () => {
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

    it.each([['fits', { status: 'fits', required_gb: 10, budget_gb: 24 }], ['unknown', null]])('shows no memory note when the fit is %s', async (_name, fit) => {
      await mountExtend();
      await chooseSource(sceneSource({ memory_fit: fit }));
      expect(target.querySelector('[data-testid="extend-memory-note"]')).toBeNull();
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
      const extras = (id: string) => (target.querySelector(`label[for="${id}"]`)!.closest('.label-row')!.querySelector('.label-extras')?.textContent ?? '').replace(/\s+/g, ' ').trim();

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
        expect(draft.state.enhanceSettings?.partsByWorkflow?.extend).toBeUndefined();
      });

      it('leaves the Enhanced scene empty, even when an older server returns one, so an edited Scene applies', async () => {
        enhanceApiMocks.enhancePrompt.mockImplementation(async (_body: unknown, ...rest: unknown[]) => {
          (rest[0] as (frame: EnhanceFrame) => void)({ type: 'done', scene: 'A woman on a rainy street.', prompt: 'She waves slowly.', clamped: false });
        });
        await mountExtend();
        await chooseSource(sceneSource());
        type('ws-prompt', 'She waves.');
        await settle();
        await runEnhance();

        expect(draft.state).toMatchObject({ enhancedScene: '', enhancedPrompt: 'She waves slowly.' });
        expect(field('ws-enhanced-scene').value).toBe('');
        expect(field('ws-enhanced-scene').placeholder).toBe('Not enhanced. Generate uses your scene.');
        expect(extras('ws-prompt')).toBe('Uses enhanced Use mine');
        expect(extras('ws-scene')).toBe('');

        type('ws-scene', 'A woman on a sunny street.');
        await settle();
        const form = await submit();
        expect(form?.get('scene')).toBe('A woman on a sunny street.');
        expect(form?.get('prompt')).toBe('She waves slowly.');
        expect(form?.get('scene_input')).toBe('A woman on a sunny street.');
        expect(form?.get('prompt_input')).toBe('She waves.');
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

    it("sends an empty --scene where the CLI would otherwise fill it in from the source", async () => {
      await mountExtend();
      await chooseSource(sceneSource());
      type('ws-scene', '');
      await settle();
      button('ws-copy-cli').click();
      await settle();
      expect(writeText.mock.calls[0]?.[0]).toContain("--scene ''");
    });

    it('does not copy without a source and says why', async () => {
      await mountExtend();
      button('ws-copy-cli').click();
      await settle();
      expect(writeText).not.toHaveBeenCalled();
      expect(toasts.map((toast) => toast.message)).toContain('Choose a video to extend.');
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
      workspaceApiMocks.getHistory.mockResolvedValue(makeGalleryPage([asset]));
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
      expect(Array.from(document.querySelectorAll('[role="menu"] [role="menuitem"]')).map((item) => item.getAttribute('data-action'))).toEqual(['extend', 'download', 'delete']);
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
      workspaceApiMocks.getHistory.mockResolvedValue(makeGalleryPage([image, video]));
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


describe('WorkspacePage clip memory note', () => {
  const fit = (status: string, required_gb = 21) => ({ memory_fit: { status, required_gb, budget_gb: 24 }, width: 704, height: 448, frames: 49 });
  const note = () => target.querySelector('[data-testid="clip-memory-note"]');
  const lastRequest = () => videoMemoryApiMocks.checkVideoMemoryFit.mock.calls.at(-1)?.[0];

  async function mountVideo(workflow: 'txt2vid' | 'img2vid' | 'extend' = 'txt2vid'): Promise<void> {
    const context = makeContext();
    context.workflow_contract.values = [...context.workflow_contract.values, 'extend'];
    workspaceApiMocks.getWorkspaceCoreContext.mockResolvedValue(context);
    draft.update('workflow', workflow);
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

  it("asks about the form's model, size, length, Low memory and upscale", async () => {
    videoMemoryApiMocks.checkVideoMemoryFit.mockResolvedValue(fit('fits'));

    await mountVideo();

    expect(lastRequest()).toEqual({ model: 'ltx-8', ratio: expect.any(String), size: expect.any(String), frames: draft.state.frameCount, low_memory: draft.state.lowMemory, upscale: null });
  });

  it('asks again when the length changes, and follows the answer', async () => {
    videoMemoryApiMocks.checkVideoMemoryFit.mockResolvedValue(fit('fits'));
    await mountVideo();
    expect(note()).toBeNull();
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

});
