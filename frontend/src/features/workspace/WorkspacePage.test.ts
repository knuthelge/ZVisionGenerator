// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, mount, unmount } from '../../../node_modules/svelte/src/index-client.js';
import { createClassComponent } from 'svelte/legacy';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { draft } from '$lib/state/draft.svelte';
import { historyStore } from '$lib/state/history.svelte';
import { jobStore } from '$lib/state/job.svelte';
import type { ImageModelDefaults, JobContext, JobSnapshot, VideoModelDefaults, WorkspaceContext, GalleryAsset, GalleryPage } from '$lib/types';

const workspaceApiMocks = vi.hoisted(() => ({
  getWorkspaceContext: vi.fn<() => Promise<WorkspaceContext>>(),
  getWorkspaceCoreContext: vi.fn<() => Promise<WorkspaceContext>>(),
  submitGenerate: vi.fn<(formData: FormData) => Promise<JobContext>>(),
  getJobSnapshot: vi.fn<(jobId: string) => Promise<JobSnapshot>>(),
  getHistory: vi.fn<(page?: number) => Promise<GalleryPage>>(),
  parseUrlPrefill: vi.fn<() => Record<string, string>>(),
}));

const promptFileApiMocks = vi.hoisted(() => ({
  openPathPicker: vi.fn(),
  inspectPromptFile: vi.fn(),
  readPromptFile: vi.fn(),
  writePromptFile: vi.fn(),
}));

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
  };
});

const galleryApiMocks = vi.hoisted(() => ({
  deleteAsset: vi.fn<(assetId: string) => Promise<void>>(),
}));

vi.mock('$lib/api/gallery', async (importOriginal) => {
  const actual = await importOriginal<typeof import('$lib/api/gallery')>();
  return { ...actual, deleteAsset: galleryApiMocks.deleteAsset };
});

vi.mock('$lib/api/promptFiles', () => ({
  openPathPicker: promptFileApiMocks.openPathPicker,
  inspectPromptFile: promptFileApiMocks.inspectPromptFile,
  readPromptFile: promptFileApiMocks.readPromptFile,
  writePromptFile: promptFileApiMocks.writePromptFile,
}));

import WorkspacePage from './WorkspacePage.svelte';
import ControlsSidebar from './ControlsSidebar.svelte';

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
            'image_upscale_guidance', 'image_upscale_sharpen'
          ],
          supports_reference_image: false,
          requires_reference_image: false,
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
            'image_upscale_guidance', 'image_upscale_sharpen'
          ],
          supports_reference_image: true,
          requires_reference_image: true,
          clear_fields: ['frames', 'audio', 'low_memory'],
        },
        txt2vid: {
          mode: 'video',
          model_kind: 'video',
          visible_controls: [
            'workflow', 'model', 'loras', 'prompt_source', 'prompt_inline',
            'prompt_file_path', 'prompt_file_option', 'prompt_file_preview', 'prompt_file_edit',
            'ratio', 'size', 'custom_dimensions', 'runs', 'frame_count', 'steps', 'seed', 'audio', 'low_memory',
            'video_upscale_enabled', 'video_upscale_factor'
          ],
          supports_reference_image: false,
          requires_reference_image: false,
          clear_fields: ['negative_prompt', 'guidance', 'image_path', 'image_strength', 'quantize'],
        },
        img2vid: {
          mode: 'video',
          model_kind: 'video',
          visible_controls: [
            'workflow', 'model', 'loras', 'prompt_source', 'prompt_inline',
            'prompt_file_path', 'prompt_file_option', 'prompt_file_preview', 'prompt_file_edit',
            'reference_image', 'reference_image_path', 'reference_image_clear',
            'ratio', 'size', 'custom_dimensions', 'runs', 'frame_count', 'steps', 'seed', 'audio', 'low_memory',
            'video_upscale_enabled', 'video_upscale_factor'
          ],
          supports_reference_image: true,
          requires_reference_image: true,
          clear_fields: ['negative_prompt', 'guidance', 'quantize'],
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
      browse_kind: 'existing_file',
      selection_required: true,
      trust_boundary: {
        scope: 'server_host_only',
        manual_entry: 'submitted_value_kept_until_backend_validation',
        picker: 'server_host_native_picker',
        read_write: 'existing_yaml_files_only',
      },
      help: {
        path: 'Prompt file path.',
        editor: 'Prompt file editor help.',
        option_required: 'Select an active prompt option before generating.',
        option_optional: 'Select an active prompt option from the file.',
        empty_options: 'This prompt file has no active prompt options.',
        stale_selection: 'The previously selected prompt option is no longer active.',
        loaded: 'Prompt file loaded.',
        saved: 'Prompt file saved.',
        ignored_negative_video: 'Negative prompt entries are ignored for video workflows.',
        ignored_negative_unsupported: 'The current image model ignores negative prompt entries.',
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


describe('WorkspacePage', () => {
  let target: HTMLDivElement;
  let app: Record<string, unknown> | null = null;

  beforeEach(() => {
    localStorage.clear();
    sessionStorage.clear();
    draft.reset();
    historyStore.seedHistory([]);
    jobStore.clearJob();
    workspaceApiMocks.getWorkspaceContext.mockReset();
    workspaceApiMocks.getWorkspaceCoreContext.mockReset();
    workspaceApiMocks.submitGenerate.mockReset();
    workspaceApiMocks.getJobSnapshot.mockReset();
    workspaceApiMocks.getHistory.mockReset();
    workspaceApiMocks.parseUrlPrefill.mockReset();
    promptFileApiMocks.openPathPicker.mockReset();
    promptFileApiMocks.inspectPromptFile.mockReset();
    promptFileApiMocks.readPromptFile.mockReset();
    promptFileApiMocks.writePromptFile.mockReset();
    // Default: no URL prefill params (plain workspace navigation)
    workspaceApiMocks.parseUrlPrefill.mockReturnValue({});
    workspaceApiMocks.submitGenerate.mockResolvedValue({
      job_id: 'job-123',
      workflow: 'txt2img',
      prompt: 'Test prompt',
      model: 'zit',
      runs: 1,
      created_at: '2026-04-23T10:00:00Z',
    });
    workspaceApiMocks.getHistory.mockResolvedValue(makeGalleryPage());
    promptFileApiMocks.openPathPicker.mockResolvedValue({ status: 'cancelled', path: null, message: null });
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
    target = document.createElement('div');
    document.body.appendChild(target);
  });

  afterEach(async () => {
    if (app) {
      await unmount(app);
      app = null;
    }
    target.remove();
    document.body.innerHTML = '';
  });

  async function mountWorkspace(context: WorkspaceContext): Promise<void> {
    workspaceApiMocks.getWorkspaceCoreContext.mockResolvedValue(context);
    app = flushSync(() => mount(WorkspacePage, { target }));
    await settle();
  }

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
    expect(video.classList.contains('latest-media')).toBe(false);
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

  function withEnhancer(context: WorkspaceContext): WorkspaceContext {
    context.workflow_contract.definitions.txt2img.visible_controls.push('prompt_enhance', 'prompt_enhance_auto');
    context.prompt_enhancer = {
      matrix: {
        axes: [
          { key: 'style', label: 'Style', multi: false, video_only: false, options: [{ slug: 'keep', label: 'Keep' }, { slug: 'photo', label: 'Photographic' }], default: ['keep'] },
          { key: 'details', label: 'Details', multi: true, video_only: false, options: [{ slug: 'lighting', label: 'Lighting' }], default: ['lighting'] },
          { key: 'length', label: 'Length', multi: false, video_only: false, options: [{ slug: 'same', label: 'Same' }, { slug: 'longer', label: 'Longer' }], default: ['same'] },
          { key: 'motion', label: 'Motion', multi: true, video_only: true, options: [{ slug: 'action', label: 'Action' }], default: ['action'] },
        ],
        defaults: { style: 'keep', details: ['lighting'], length: 'same', motion: ['action'] },
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
    draft.update('enhanceSettings', { style: 'photo', details: [], length: 'longer', motion: ['action'] });
    await mountWorkspace(withEnhancer(makeContext()));

    expect((target.querySelector('#ws-enhanced-prompt') as HTMLTextAreaElement).disabled).toBe(true);
    const submitted = await submitForm();
    expect(submitted.get('prompt')).toBe('a fox');
    expect(submitted.get('enhance_auto')).toBe('true');
    expect(JSON.parse(String(submitted.get('enhance_settings')))).toEqual({ style: 'photo', details: [], length: 'longer' });
  });

  it('warns inline when auto-enhance would change nothing', async () => {
    draft.update('prompt', 'a fox');
    draft.update('enhanceAuto', true);
    draft.update('enhanceSettings', { style: 'keep', details: [], length: 'same', motion: [] });
    await mountWorkspace(withEnhancer(makeContext()));
    (target.querySelector('#ws-enhance-toggle') as HTMLButtonElement).click();
    await settle();
    // The enhancer popover is moved to <body>.
    expect(document.querySelector('#ws-enhance-panel [role="alert"]')?.textContent).toContain('Pick a style, a detail, or a length.');
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
    expect(target.textContent).toContain('Retry prompt');
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
    expect(submitButton?.disabled).toBe(true);

    const mockEventSource = (globalThis.EventSource as unknown as {
      lastInstance: { emit: (type: string, data: unknown) => void };
    }).lastInstance;
    mockEventSource.emit('job_cancelled', { type: 'job_cancelled', job_id: 'job-123' });
    await settle();

    expect(jobStore.current?.status).toBe('cancelled');
    expect(submitButton?.disabled).toBe(false);
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
    expect(submitButton?.disabled).toBe(true);

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
    expect((target.querySelector('#ws-submit') as HTMLButtonElement | null)?.disabled).toBe(true);

    await unmount(app!);
    app = null;
    target.replaceChildren();

    await mountWorkspace(context);
    expect(MockEventSource.instances).toHaveLength(constructionCount);
    expect((target.querySelector('#ws-submit') as HTMLButtonElement | null)?.disabled).toBe(true);

    workspaceApiMocks.getHistory.mockClear();
    source.emit('job_completed', { type: 'job_completed', job_id: 'job-remount', total_runs: 1, outputs: [] });
    await settle();

    expect(jobStore.current?.status).toBe('completed');
    expect(source.closeCalls).toBe(1);
    expect(workspaceApiMocks.getHistory).toHaveBeenCalledOnce();
    expect((target.querySelector('#ws-submit') as HTMLButtonElement | null)?.disabled).toBe(false);
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
    expect((target.querySelector('#ws-submit') as HTMLButtonElement | null)?.disabled).toBe(true);

    await unmount(app!);
    app = null;

    jobStore.clearJob();
    sessionStorage.setItem('ziv-active-job-id-v1', 'job-reconnect');
    workspaceApiMocks.getJobSnapshot.mockClear();

    await mountWorkspace(context);

    expect(workspaceApiMocks.getJobSnapshot).toHaveBeenCalledWith('job-reconnect');
    expect(target.textContent).toContain('Resume me after remount');
    expect((target.querySelector('#ws-submit') as HTMLButtonElement | null)?.disabled).toBe(true);
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

    const promptSource = target.querySelector('[data-prompt-source="file"]') as HTMLButtonElement | null;
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
    const detail = document.querySelector('[id="prompt-detail-portrait:1"]') as HTMLElement;
    expect(detail.classList.contains('line-clamp-2')).toBe(true);
    expect(document.body.textContent).not.toContain('muddy');
    const showMore = document.querySelector('[aria-label="Show more for portrait #2"]') as HTMLButtonElement;
    showMore.click();
    await settle();
    expect(detail.classList.contains('line-clamp-2')).toBe(false);
    expect(detail.classList.contains('block')).toBe(true);
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

    const promptSource = target.querySelector('[data-prompt-source="file"]') as HTMLButtonElement | null;
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

  it('renders prompt-file path and editor guidance from the backend contract', async () => {
    const context = makeContext({
      prompt_file: {
        accepted_extensions: ['.yaml', '.yml'],
        browse_kind: 'existing_file',
        selection_required: true,
        trust_boundary: {
          scope: 'server_host_only',
          manual_entry: 'submitted_value_kept_until_backend_validation',
          picker: 'server_host_native_picker',
          read_write: 'existing_yaml_files_only',
        },
        help: {
          path: 'Backend-owned prompt path guidance.',
          editor: 'Backend-owned prompt editor guidance.',
          option_required: 'Select an active prompt option before generating.',
          option_optional: 'Select an active prompt option from the file.',
          empty_options: 'This prompt file has no active prompt options.',
          stale_selection: 'The previously selected prompt option is no longer active.',
          loaded: 'Prompt file loaded.',
          saved: 'Prompt file saved.',
          ignored_negative_video: 'Negative prompt entries are ignored for video workflows.',
          ignored_negative_unsupported: 'The current image model ignores negative prompt entries.',
        },
      },
    });
    promptFileApiMocks.inspectPromptFile.mockResolvedValueOnce({
      path: '/server/prompts.yaml',
      options: [],
    });
    promptFileApiMocks.readPromptFile.mockResolvedValueOnce({
      path: '/server/prompts.yaml',
      raw_text: 'prompts: []\n',
      options: [],
    });

    await mountWorkspace(context);

    const promptSource = target.querySelector('[data-prompt-source="file"]') as HTMLButtonElement | null;
    promptSource!.click();
    await settle();

    expect(target.querySelector('#ws-prompts-file')).not.toBeNull();

    const pathInput = target.querySelector('#ws-prompts-file') as HTMLInputElement | null;
    pathInput!.value = '~/prompts.yaml';
    pathInput!.dispatchEvent(new Event('input', { bubbles: true }));
    pathInput!.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
    await settle();

    const editButton = Array.from(target.querySelectorAll('button')).find((button) => button.textContent?.includes('Edit YAML')) as HTMLButtonElement | undefined;
    editButton!.click();
    await settle();

    expect(target.querySelector('#ws-prompt-file-editor')).not.toBeNull();
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

    const promptSource = target.querySelector('[data-prompt-source="file"]') as HTMLButtonElement | null;
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

  it('reloads prompt-file editor content every time the same file is opened', async () => {
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
      ],
    });
    promptFileApiMocks.readPromptFile
      .mockResolvedValueOnce({
        path: '/server/prompts.yaml',
        options: [],
        raw_text: 'portrait:\n  - prompt: first disk version\n',
      })
      .mockResolvedValueOnce({
        path: '/server/prompts.yaml',
        options: [],
        raw_text: 'portrait:\n  - prompt: second disk version\n',
      });

    const context = makeContext();
    await mountWorkspace(context);

    const promptSource = target.querySelector('[data-prompt-source="file"]') as HTMLButtonElement | null;
    promptSource!.click();
    await settle();

    const pathInput = target.querySelector('#ws-prompts-file') as HTMLInputElement | null;
    pathInput!.value = '~/prompts.yaml';
    pathInput!.dispatchEvent(new Event('input', { bubbles: true }));
    pathInput!.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
    await settle();

    const editButton = Array.from(target.querySelectorAll('button')).find((button) => button.textContent?.trim() === 'Edit YAML');
    expect(editButton).not.toBeUndefined();
    editButton!.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    await settle();
    expect((target.querySelector('#ws-prompt-file-editor') as HTMLTextAreaElement | null)?.value).toContain('first disk version');

    const cancelButton = Array.from(target.querySelectorAll('button')).find((button) => button.textContent?.trim() === 'Cancel');
    expect(cancelButton).not.toBeUndefined();
    cancelButton!.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    await settle();

    editButton!.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    await settle();

    expect(promptFileApiMocks.readPromptFile).toHaveBeenCalledTimes(2);
    expect((target.querySelector('#ws-prompt-file-editor') as HTMLTextAreaElement | null)?.value).toContain('second disk version');
  });

  it('clears a stale prompt-file selection after saving edited yaml', async () => {
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
      ],
    });
    promptFileApiMocks.readPromptFile.mockResolvedValue({
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
      raw_text: 'portrait:\n  - prompt: first option\n',
    });
    promptFileApiMocks.writePromptFile.mockResolvedValue({
      path: '/server/prompts.yaml',
      options: [
        {
          id: 'portrait:9',
          set_name: 'portrait',
          source_index: 9,
          label: 'portrait #10 · replacement option',
          prompt_preview: 'replacement option',
          negative_preview: null,
        },
      ],
    });

    const context = makeContext();
    await mountWorkspace(context);

    const promptSource = target.querySelector('[data-prompt-source="file"]') as HTMLButtonElement | null;
    promptSource!.click();
    await settle();

    const pathInput = target.querySelector('#ws-prompts-file') as HTMLInputElement | null;
    pathInput!.value = '~/prompts.yaml';
    pathInput!.dispatchEvent(new Event('input', { bubbles: true }));
    pathInput!.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
    await settle();

    await choosePrompts(target, ['portrait:0']);

    const editButton = Array.from(target.querySelectorAll('button')).find((button) => button.textContent?.trim() === 'Edit YAML');
    expect(editButton).not.toBeUndefined();
    editButton!.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    await settle();

    const editor = target.querySelector('#ws-prompt-file-editor') as HTMLTextAreaElement | null;
    expect(editor?.value).toContain('first option');
    editor!.value = 'portrait:\n  - prompt: replacement option\n';
    editor!.dispatchEvent(new Event('input', { bubbles: true }));
    const saveButton = Array.from(target.querySelectorAll('button')).find((button) => button.textContent?.trim() === 'Save File');
    expect(saveButton).not.toBeUndefined();
    saveButton!.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    await settle();

    expect(promptFileApiMocks.writePromptFile).toHaveBeenCalledWith('/server/prompts.yaml', 'portrait:\n  - prompt: replacement option\n');
    expect(target.querySelectorAll('input[name="prompt_option_id"]')).toHaveLength(0);
    expect((target.querySelector('#ws-submit') as HTMLButtonElement | null)?.disabled).toBe(true);
    expect(target.textContent).toContain('no longer active');
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
  let target: HTMLDivElement;
  let app: Record<string, unknown> | null = null;

  beforeEach(() => {
    localStorage.clear();
    sessionStorage.clear();
    draft.reset();
    historyStore.seedHistory([]);
    jobStore.clearJob();
    workspaceApiMocks.getWorkspaceContext.mockReset();
    workspaceApiMocks.getWorkspaceCoreContext.mockReset();
    workspaceApiMocks.submitGenerate.mockReset();
    workspaceApiMocks.getHistory.mockReset();
    workspaceApiMocks.parseUrlPrefill.mockReturnValue({});
    workspaceApiMocks.getHistory.mockResolvedValue(makeGalleryPage());
    workspaceApiMocks.submitGenerate.mockResolvedValue({
      job_id: 'job-promo',
      workflow: 'txt2img',
      prompt: 'Test prompt',
      model: 'zit',
      runs: 1,
      created_at: '2026-04-30T10:00:00Z',
    });
    target = document.createElement('div');
    document.body.appendChild(target);
  });

  afterEach(async () => {
    if (app) { await unmount(app); app = null; }
    target.remove();
    document.body.innerHTML = '';
  });

  async function mountWorkspace(context: WorkspaceContext): Promise<void> {
    workspaceApiMocks.getWorkspaceCoreContext.mockResolvedValue(context);
    app = flushSync(() => mount(WorkspacePage, { target }));
    await settle();
  }

  it('renders the full authoritative completed output grid and hides the JobCard', async () => {
    const context = makeContext();
    await mountWorkspace(context);

    const form = target.querySelector('form');
    form!.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
    await settle();

    // While running, submit button is disabled
    const submitButton = target.querySelector('#ws-submit') as HTMLButtonElement | null;
    expect(submitButton?.disabled).toBe(true);

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
    ['job_failed', 'Job failed.'],
    ['job_cancelled', 'Job stopped.'],
  ] as const)('shows generation_finished assets before %s and preserves them afterward', async (terminal, terminalMessage) => {
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
    expect(target.textContent).toContain('Outputs · 1');
    expect(target.querySelector(`.job-card img[alt="${output.filename}"]`)).not.toBeNull();

    source.emit(terminal, { type: terminal, job_id: 'job-promo' });
    await settle();
    expect(target.textContent).toContain(terminalMessage);
    expect(target.querySelector(`.job-card img[alt="${output.filename}"]`)).not.toBeNull();
  });

  it('keeps many running outputs reachable inside the constrained-height preview scroller', async () => {
    target.style.height = '160px';
    await mountWorkspace(makeContext());
    target.querySelector('form')!.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
    await settle();
    const source = (globalThis.EventSource as unknown as {
      lastInstance: { emit: (type: string, data: unknown) => void };
    }).lastInstance;
    const outputs = Array.from({ length: 12 }, (_, index) => makeAsset({
      id: `out/running-${index}.png`, filename: `running-${index}.png`, url: `/media/out/running-${index}.png`,
    }));

    for (const [index, output] of outputs.entries()) {
      source.emit('generation_finished', {
        type: 'generation_finished', job_id: 'job-promo', status: 'success', run_index: index, asset: output,
      });
    }
    await settle();

    const scroller = target.querySelector('.workspace-preview div.h-full.w-full.overflow-y-auto') as HTMLDivElement | null;
    expect(scroller).not.toBeNull();
    expect(scroller?.classList.contains('overflow-y-auto')).toBe(true);
    expect(target.querySelectorAll('.job-card button[aria-label^="View "]')).toHaveLength(outputs.length);
    expect(target.querySelector(`.job-card img[alt="${outputs.at(-1)!.filename}"]`)).not.toBeNull();
  });

  it('opens running previews in the shared modal and keeps it open as the batch completes', async () => {
    await mountWorkspace(makeContext());
    target.querySelector('form')!.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
    await settle();
    const source = (globalThis.EventSource as unknown as { lastInstance: { emit: (type: string, data: unknown) => void } }).lastInstance;
    const first = makeAsset({ id: 'out/first.png', filename: 'first.png', url: '/media/first.png' });
    const second = makeAsset({ id: 'out/second.png', filename: 'second.png', url: '/media/second.png' });
    source.emit('generation_finished', { type: 'generation_finished', status: 'success', asset: first });
    await settle();
    const trigger = target.querySelector('.job-card button[aria-label="View first.png fullscreen"]') as HTMLButtonElement;
    trigger.click();
    await settle();
    expect(target.querySelector('[data-testid="asset-viewer"] img')?.getAttribute('src')).toBe(first.url);
    source.emit('generation_finished', { type: 'generation_finished', status: 'success', asset: second });
    await settle();
    (target.querySelector('[aria-label="Next asset"]') as HTMLButtonElement).click();
    await settle();
    expect(target.querySelector('[data-testid="asset-viewer"] img')?.getAttribute('src')).toBe(second.url);
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
    await settle();
    expect(target.querySelector('[data-testid="asset-viewer"]')).toBeNull();
    expect(document.activeElement).toBe(trigger);
    trigger.click();
    await settle();
    source.emit('job_completed', { type: 'job_completed', outputs: [first, second] });
    await settle();
    expect(target.querySelector('[data-testid="asset-viewer"] img')?.getAttribute('src')).toBe(first.url);
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
  let target: HTMLDivElement;
  let app: Record<string, unknown> | null = null;

  beforeEach(() => {
    localStorage.clear();
    sessionStorage.clear();
    draft.reset();
    historyStore.seedHistory([]);
    jobStore.clearJob();
    workspaceApiMocks.getWorkspaceContext.mockReset();
    workspaceApiMocks.getWorkspaceCoreContext.mockReset();
    workspaceApiMocks.getHistory.mockReset();
    workspaceApiMocks.parseUrlPrefill.mockReturnValue({});
    workspaceApiMocks.getHistory.mockResolvedValue(makeGalleryPage());
    target = document.createElement('div');
    document.body.appendChild(target);
  });

  afterEach(async () => {
    if (app) { await unmount(app); app = null; }
    target.remove();
    document.body.innerHTML = '';
  });

  async function mountWorkspace(context: WorkspaceContext): Promise<void> {
    workspaceApiMocks.getWorkspaceCoreContext.mockResolvedValue(context);
    app = flushSync(() => mount(WorkspacePage, { target }));
    await settle();
  }

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
  let target: HTMLDivElement;
  let app: Record<string, unknown> | null = null;

  beforeEach(() => {
    localStorage.clear();
    sessionStorage.clear();
    draft.reset();
    historyStore.seedHistory([]);
    jobStore.clearJob();
    workspaceApiMocks.getWorkspaceCoreContext.mockReset();
    workspaceApiMocks.getHistory.mockReset();
    workspaceApiMocks.parseUrlPrefill.mockReturnValue({});
    workspaceApiMocks.getHistory.mockResolvedValue(makeGalleryPage());
    galleryApiMocks.deleteAsset.mockReset();
    target = document.createElement('div');
    document.body.appendChild(target);
  });

  afterEach(async () => {
    if (app) { await unmount(app); app = null; }
    target.remove();
    document.body.innerHTML = '';
    vi.restoreAllMocks();
  });

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
    const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValueOnce(false).mockReturnValueOnce(true);
    galleryApiMocks.deleteAsset.mockResolvedValue(undefined);

    openTileMenu(second);
    (document.querySelector('[role="menu"] [data-action="delete"]') as HTMLButtonElement).click();
    await settle();
    expect(galleryApiMocks.deleteAsset).not.toHaveBeenCalled();

    openTileMenu(second);
    (document.querySelector('[role="menu"] [data-action="delete"]') as HTMLButtonElement).click();
    await settle();

    expect(confirmSpy).toHaveBeenCalledTimes(2);
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

  it('keeps settings when Use as reference switches the workflow', async () => {
    const asset = makeAsset({ id: 'out/first.png', filename: 'first.png', file_path: '/outputs/out/first.png' });
    await mountWithHistory([asset]);
    draft.patch({ steps: 9, runs: 40 });
    await settle();

    openTileMenu(asset);
    (document.querySelector('[role="menu"] [data-action="reference-image"]') as HTMLButtonElement).click();
    await settle();

    expect(draft.state).toMatchObject({ workflow: 'img2img', steps: 9, runs: 40, referenceImagePath: '/outputs/out/first.png' });
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
    vi.spyOn(window, 'confirm').mockReturnValue(true);
    galleryApiMocks.deleteAsset.mockResolvedValue(undefined);
    openTileMenu(asset);
    (document.querySelector('[role="menu"] [data-action="reference-image"]') as HTMLButtonElement).click();
    await settle();
    expect(draft.state.referenceImagePath).toBe('/outputs/out/first.png');

    openTileMenu(asset);
    (document.querySelector('[role="menu"] [data-action="delete"]') as HTMLButtonElement).click();
    await settle();

    expect(draft.state.referenceImagePath).toBeNull();
  });

  it('locks the seed to the latest output and unlocks it back to random', async () => {
    const asset = makeAsset({ id: 'out/first.png', filename: 'first.png', seed: 4242 });
    await mountWithHistory([asset]);
    const lock = () => target.querySelector('#ws-seed')!.parentElement!.querySelector('button[aria-pressed]') as HTMLButtonElement;

    lock().click();
    await settle();
    expect(draft.state.seed).toBe(4242);
    expect(lock().getAttribute('aria-pressed')).toBe('true');

    lock().click();
    await settle();
    expect(draft.state.seed).toBeNull();
  });
});

