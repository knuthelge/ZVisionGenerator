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
    expect(target.querySelector('label[aria-label="Enable upscale"]')).toBeNull();

    const modelSelect = target.querySelector('#ws-model') as HTMLSelectElement | null;
    expect(modelSelect).not.toBeNull();
    modelSelect!.value = 'zit';
    modelSelect!.dispatchEvent(new Event('change', { bubbles: true }));
    await settle();

    expect(target.querySelector('#ws-image-file')).not.toBeNull();
    expect(target.querySelector('input[name="image_path"]')).not.toBeNull();
    expect(target.querySelector('label[aria-label="Enable upscale"]')).not.toBeNull();
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

    const sizeSelect = target.querySelector('#ws-size') as HTMLSelectElement | null;
    expect(sizeSelect).not.toBeNull();
    expect(Array.from(sizeSelect!.options).map((option) => option.value)).toEqual(['m', 'l']);

    const customDimensionsButton = Array.from(target.querySelectorAll('button')).find(
      (button) => button.textContent?.trim() === 'Custom W/H'
    ) as HTMLButtonElement | undefined;
    expect(customDimensionsButton).not.toBeUndefined();
    customDimensionsButton!.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    await settle();

    const widthInput = target.querySelector('#ws-width') as HTMLInputElement | null;
    const heightInput = target.querySelector('#ws-height') as HTMLInputElement | null;
    expect(widthInput?.min).toBe('256');
    expect(widthInput?.max).toBe('2048');
    expect(widthInput?.step).toBe('16');
    expect(heightInput?.min).toBe('256');
    expect(heightInput?.max).toBe('2048');
    expect(heightInput?.step).toBe('16');

    const modelSelect = target.querySelector('#ws-model') as HTMLSelectElement | null;
    expect(modelSelect).not.toBeNull();
    modelSelect!.value = 'zit';
    modelSelect!.dispatchEvent(new Event('change', { bubbles: true }));
    await settle();

    const ratioModeButton = Array.from(target.querySelectorAll('button')).find(
      (button) => button.textContent?.trim() === 'Ratio'
    ) as HTMLButtonElement | undefined;
    expect(ratioModeButton).not.toBeUndefined();
    ratioModeButton!.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    await settle();

    const permissiveSizeSelect = target.querySelector('#ws-size') as HTMLSelectElement | null;
    expect(permissiveSizeSelect).not.toBeNull();
    expect(Array.from(permissiveSizeSelect!.options).map((option) => option.value)).toEqual(['m', 'l', 'xl']);
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

    const jsonToggle = target.querySelector('#ws-json-prompt-toggle') as HTMLInputElement | null;
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

    jsonToggle!.checked = true;
    jsonToggle!.dispatchEvent(new Event('change', { bubbles: true }));
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

    const promptSource = target.querySelector('#ws-prompt-source') as HTMLSelectElement | null;
    expect(promptSource).not.toBeNull();
    promptSource!.value = 'file';
    promptSource!.dispatchEvent(new Event('change', { bubbles: true }));
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

    const optionSelect = target.querySelector('#ws-prompt-option') as HTMLSelectElement | null;
    expect(optionSelect).not.toBeNull();
    optionSelect!.value = 'portrait:1';
    optionSelect!.dispatchEvent(new Event('change', { bubbles: true }));
    await settle();

    expect(submitButton?.disabled).toBe(false);

    const form = target.querySelector('form');
    expect(form).not.toBeNull();
    form!.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
    await settle();

    expect(workspaceApiMocks.submitGenerate).toHaveBeenCalledTimes(1);
    const [submittedFormData] = workspaceApiMocks.submitGenerate.mock.calls[0] ?? [];
    expect(submittedFormData).toBeInstanceOf(FormData);
    expect(submittedFormData.get('prompt_source')).toBe('file');
    expect(submittedFormData.get('prompts_file')).toBe('/server/prompts.yaml');
    expect(submittedFormData.get('prompt_option_id')).toBe('portrait:1');
    expect(submittedFormData.has('prompt')).toBe(false);
    expect(submittedFormData.has('negative_prompt')).toBe(false);
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

    const promptSource = target.querySelector('#ws-prompt-source') as HTMLSelectElement | null;
    promptSource!.value = 'file';
    promptSource!.dispatchEvent(new Event('change', { bubbles: true }));
    await settle();

    const pathInput = target.querySelector('#ws-prompts-file') as HTMLInputElement | null;
    pathInput!.value = '~/prompts.yaml';
    pathInput!.dispatchEvent(new Event('input', { bubbles: true }));
    pathInput!.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
    await settle();

    const optionSelect = target.querySelector('#ws-prompt-option') as HTMLSelectElement | null;
    optionSelect!.value = 'portrait:0';
    optionSelect!.dispatchEvent(new Event('change', { bubbles: true }));
    await settle();

    promptFileApiMocks.inspectPromptFile.mockRejectedValueOnce(new Error('POST /api/prompt-files/inspect → 422: missing file'));
    pathInput!.value = '/missing/prompts.yaml';
    pathInput!.dispatchEvent(new Event('input', { bubbles: true }));
    pathInput!.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
    await settle();

    const hiddenPath = target.querySelector('input[name="prompts_file"]') as HTMLInputElement | null;
    expect(pathInput!.value).toBe('/missing/prompts.yaml');
    expect(hiddenPath?.value).toBe('/missing/prompts.yaml');
    expect((target.querySelector('#ws-prompt-option') as HTMLSelectElement | null)?.value).toBe('');
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

    const promptSource = target.querySelector('#ws-prompt-source') as HTMLSelectElement | null;
    promptSource!.value = 'file';
    promptSource!.dispatchEvent(new Event('change', { bubbles: true }));
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

    const promptSource = target.querySelector('#ws-prompt-source') as HTMLSelectElement | null;
    promptSource!.value = 'file';
    promptSource!.dispatchEvent(new Event('change', { bubbles: true }));
    await settle();

    const pathInput = target.querySelector('#ws-prompts-file') as HTMLInputElement | null;
    pathInput!.value = '~/prompts.yaml';
    pathInput!.dispatchEvent(new Event('input', { bubbles: true }));
    pathInput!.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
    await settle();

    const optionSelect = target.querySelector('#ws-prompt-option') as HTMLSelectElement | null;
    optionSelect!.value = 'portrait:0';
    optionSelect!.dispatchEvent(new Event('change', { bubbles: true }));
    await settle();
    expect((target.querySelector('#ws-submit') as HTMLButtonElement | null)?.disabled).toBe(false);

    pathInput!.value = '/server/other-prompts.yaml';
    pathInput!.dispatchEvent(new Event('input', { bubbles: true }));
    await settle();

    const hiddenPath = target.querySelector('input[name="prompts_file"]') as HTMLInputElement | null;
    expect(hiddenPath?.value).toBe('/server/other-prompts.yaml');
    expect((target.querySelector('#ws-prompt-option') as HTMLSelectElement | null)?.value).toBe('');
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

    const promptSource = target.querySelector('#ws-prompt-source') as HTMLSelectElement | null;
    promptSource!.value = 'file';
    promptSource!.dispatchEvent(new Event('change', { bubbles: true }));
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

    const promptSource = target.querySelector('#ws-prompt-source') as HTMLSelectElement | null;
    promptSource!.value = 'file';
    promptSource!.dispatchEvent(new Event('change', { bubbles: true }));
    await settle();

    const pathInput = target.querySelector('#ws-prompts-file') as HTMLInputElement | null;
    pathInput!.value = '~/prompts.yaml';
    pathInput!.dispatchEvent(new Event('input', { bubbles: true }));
    pathInput!.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
    await settle();

    const optionSelect = target.querySelector('#ws-prompt-option') as HTMLSelectElement | null;
    optionSelect!.value = 'portrait:0';
    optionSelect!.dispatchEvent(new Event('change', { bubbles: true }));
    await settle();

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
    expect((target.querySelector('#ws-prompt-option') as HTMLSelectElement | null)?.value).toBe('');
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

  it('promotes the first completed job output into the center pane and hides the JobCard', async () => {
    const context = makeContext();
    await mountWorkspace(context);

    const form = target.querySelector('form');
    form!.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
    await settle();

    // While running, submit button is disabled
    const submitButton = target.querySelector('#ws-submit') as HTMLButtonElement | null;
    expect(submitButton?.disabled).toBe(true);

    const completedAsset = makeAsset();
    const mockEventSource = (globalThis.EventSource as unknown as {
      lastInstance: { emit: (type: string, data: unknown) => void };
    }).lastInstance;
    mockEventSource.emit('job_completed', { type: 'job_completed', job_id: 'job-promo', total_runs: 1, outputs: [completedAsset] });
    await settle();

    // Completed output image is visible in center pane
    const img = target.querySelector(`img[src="${completedAsset.url}"]`);
    expect(img).not.toBeNull();

    // Open fullscreen button is present
    const openBtn = Array.from(target.querySelectorAll('button')).find(
      (b) => b.textContent?.trim() === 'Open fullscreen'
    );
    expect(openBtn).not.toBeUndefined();
  });

  it('opens the workspace lightbox from a completed output open-fullscreen button', async () => {
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
    historyStore.seedHistory([staleHistoryAsset]);

    const form = target.querySelector('form');
    form!.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
    await settle();

    const mockEventSource = (globalThis.EventSource as unknown as {
      lastInstance: { emit: (type: string, data: unknown) => void };
    }).lastInstance;
    mockEventSource.emit('job_completed', { type: 'job_completed', job_id: 'job-promo', total_runs: 1, outputs: [completedAsset] });
    await settle();

    const openBtn = Array.from(target.querySelectorAll('button')).find(
      (b) => b.textContent?.trim() === 'Open fullscreen'
    );
    expect(openBtn).not.toBeUndefined();
    openBtn!.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    await settle();

    expect(target.querySelector('[data-testid="lightbox"]')).not.toBeNull();
    const lightboxImage = target.querySelector('[data-testid="lightbox"] img') as HTMLImageElement | null;
    expect(lightboxImage?.getAttribute('src')).toBe(completedAsset.url);
    expect(lightboxImage?.getAttribute('alt')).toBe(completedAsset.filename);
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

    expect(target.querySelector('[data-testid="lightbox"]')).not.toBeNull();
  });
});