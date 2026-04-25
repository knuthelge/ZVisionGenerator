// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, mount, unmount } from '../../../node_modules/svelte/src/index-client.js';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { draft } from '$lib/state/draft.svelte';
import { historyStore } from '$lib/state/history.svelte';
import { jobStore } from '$lib/state/job.svelte';
import type { ImageModelDefaults, JobContext, VideoModelDefaults, WorkspaceContext } from '$lib/types';

const workspaceApiMocks = vi.hoisted(() => ({
  getWorkspaceContext: vi.fn<() => Promise<WorkspaceContext>>(),
  submitGenerate: vi.fn<(formData: FormData) => Promise<JobContext>>(),
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
    submitGenerate: workspaceApiMocks.submitGenerate,
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
    ...overrides,
  };
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
      visible_sections: ['image_generation', 'video_generation'],
      theme: 'dark',
      gallery_page_size: 20,
      startup_view: 'workspace',
    },
    output_dir: '/tmp/output',
    quantize_options: [4, 8],
    image_ratios: ['2:3', '16:9'],
    video_ratios: ['16:9'],
    image_size_options: { '2:3': ['m'], '16:9': ['l'] },
    video_size_options: { '16:9': ['m'] },
    scheduler_options: ['beta'],
    workflow_contract: {
      values: ['txt2img', 'img2img', 'txt2vid', 'img2vid'],
      legacy_aliases: {
        image: 'txt2img',
        i2i: 'img2img',
        video: 'txt2vid',
        i2v: 'img2vid',
      },
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
    },
    ...overrides,
  };
}

async function settle(): Promise<void> {
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

function expectPrimaryFocusTreatment(control: Element | null, label: string): void {
  expect(control, `Expected ${label} control to be rendered`).not.toBeNull();
  const className = control!.getAttribute('class') ?? '';
  expect(className, `${label} should use the 4px primary focus ring`).toContain('focus:ring-4');
  expect(className, `${label} should use the primary ring color`).toContain('focus:ring-primary-main');
  expect(className, `${label} should use the primary focus border`).toContain('focus:border-primary-main');
  expect(className, `${label} should not keep the stale 1px focus ring`).not.toContain('focus:ring-1');
}

function expectRoundedMdTreatment(control: Element | null, label: string): void {
  expect(control, `Expected ${label} control to be rendered`).not.toBeNull();
  const classTokens = (control!.getAttribute('class') ?? '').split(/\s+/).filter(Boolean);
  expect(classTokens, `${label} should use the 6px rounded-md radius`).toContain('rounded-md');
  expect(classTokens, `${label} should not keep the stale 4px rounded radius`).not.toContain('rounded');
}

function expectShellFocusTreatment(shell: Element | null, label: string): void {
  expect(shell, `Expected ${label} to be rendered`).not.toBeNull();
  expect(shell!.getAttribute('data-focused'), `${label} should expose runtime focus state`).toBe('false');
  const className = shell!.getAttribute('class') ?? '';
  expect(className, `${label} should not rely on focus-within utility indirection anymore`).not.toContain('focus-within:ring-4');
  expect(className, `${label} should not use the group-focus-within indirection`).not.toContain('group-focus-within:ring-4');
}

function expectTruthfulShellSurface(shell: Element | null, label: string): void {
  expect(shell, `Expected ${label} to be rendered`).not.toBeNull();
  const classTokens = (shell!.getAttribute('class') ?? '').split(/\s+/).filter(Boolean);
  expect(classTokens, `${label} should use the truthful control surface token`).toContain('bg-bg-surface');
  expect(classTokens, `${label} should use the subtle border token`).toContain('border-border-subtle');
}

function expectShellHoverTreatment(shell: Element | null, label: string): void {
  expect(shell, `Expected ${label} to be rendered`).not.toBeNull();
  expect(shell!.getAttribute('data-hovered'), `${label} should expose runtime hover state`).toBe('false');
  const classTokens = (shell!.getAttribute('class') ?? '').split(/\s+/).filter(Boolean);
  expect(classTokens, `${label} should not rely on hover pseudo-class indirection anymore`).not.toContain('hover:bg-bg-surface-hover');
  expect(classTokens, `${label} should not keep the stale direct hover border treatment`).not.toContain('hover:border-zinc-600');
  expect(classTokens, `${label} should not use the stale group-level hover indirection`).not.toContain('group-hover:bg-bg-surface-hover');
}

function expectShellHoverAndFocusRuntime(
  shell: HTMLElement | null,
  select: HTMLSelectElement | null,
  label: string,
): void {
  expect(shell, `Expected ${label} shell to be rendered`).not.toBeNull();
  expect(select, `Expected ${label} select to be rendered`).not.toBeNull();

  const initialTokens = (shell!.getAttribute('class') ?? '').split(/\s+/).filter(Boolean);
  expect(shell!.getAttribute('data-hovered'), `${label} should start idle`).toBe('false');
  expect(shell!.getAttribute('data-focused'), `${label} should start unfocused`).toBe('false');
  expect(initialTokens, `${label} should start on the truthful default surface`).toContain('bg-bg-surface');
  expect(initialTokens, `${label} should start with the subtle border`).toContain('border-border-subtle');

  select!.dispatchEvent(new MouseEvent('mouseenter'));
  flushSync();

  const hoveredTokens = (shell!.getAttribute('class') ?? '').split(/\s+/).filter(Boolean);
  expect(shell!.getAttribute('data-hovered'), `${label} should expose hovered runtime state`).toBe('true');
  expect(hoveredTokens, `${label} should use the hover surface token while hovered`).toContain('bg-bg-surface-hover');

  select!.dispatchEvent(new MouseEvent('mouseleave'));
  flushSync();

  expect(shell!.getAttribute('data-hovered'), `${label} should clear hovered runtime state`).toBe('false');

  select!.dispatchEvent(new FocusEvent('focus'));
  flushSync();

  const focusedTokens = (shell!.getAttribute('class') ?? '').split(/\s+/).filter(Boolean);
  expect(shell!.getAttribute('data-focused'), `${label} should expose focused runtime state`).toBe('true');
  expect(focusedTokens, `${label} should apply the 4px primary focus ring while focused`).toContain('ring-4');
  expect(focusedTokens, `${label} should apply the primary focus ring color while focused`).toContain('ring-primary-main');
  expect(focusedTokens, `${label} should apply the primary focus border while focused`).toContain('border-primary-main');

  select!.dispatchEvent(new FocusEvent('blur'));
  flushSync();

  const blurredTokens = (shell!.getAttribute('class') ?? '').split(/\s+/).filter(Boolean);
  expect(shell!.getAttribute('data-focused'), `${label} should clear focused runtime state`).toBe('false');
  expect(blurredTokens, `${label} should return to the truthful default surface after blur`).toContain('bg-bg-surface');
  expect(blurredTokens, `${label} should restore the subtle border after blur`).toContain('border-border-subtle');
}

function expectShellTextAndSpacingTreatment(shell: Element | null, label: string): void {
  expect(shell, `Expected ${label} to be rendered`).not.toBeNull();
  const classTokens = (shell!.getAttribute('class') ?? '').split(/\s+/).filter(Boolean);
  expect(classTokens, `${label} should render primary text color on the shell itself`).toContain('text-text-primary');
  expect(classTokens, `${label} should use the default control text size`).toContain('text-sm');
  expect(classTokens, `${label} should use the spec horizontal spacing token`).toContain('px-3');
  expect(classTokens, `${label} should use the spec vertical spacing token`).toContain('py-2');
}

describe('WorkspacePage', () => {
  let target: HTMLDivElement;
  let app: Record<string, unknown> | null = null;

  beforeEach(() => {
    localStorage.clear();
    draft.reset();
    historyStore.seedHistory([]);
    jobStore.clearJob();
    workspaceApiMocks.getWorkspaceContext.mockReset();
    workspaceApiMocks.submitGenerate.mockReset();
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
    promptFileApiMocks.openPathPicker.mockResolvedValue({ status: 'cancelled', path: null, message: null });
    Object.defineProperty(URL, 'createObjectURL', {
      configurable: true,
      value: vi.fn(() => 'blob:test-image'),
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
    workspaceApiMocks.getWorkspaceContext.mockResolvedValue(context);
    app = flushSync(() => mount(WorkspacePage, { target }));
    await settle();
  }

  it('renders a loading-safe shell before workspace authority resolves', async () => {
    draft.update('model', 'stale-model');
    draft.update('prompt', 'stale prompt');
    draft.update('steps', 99);
    workspaceApiMocks.getWorkspaceContext.mockImplementation(
      () => new Promise<WorkspaceContext>(() => undefined)
    );

    app = flushSync(() => mount(WorkspacePage, { target }));
    await settle();

    expect(target.querySelector('#ws-prompt')).toBeNull();
    expect((target.querySelector('#ws-submit') as HTMLButtonElement | null)?.disabled).toBe(true);
    expect((target.querySelector('#ws-model') as HTMLSelectElement | null)?.disabled).toBe(true);
    expect(target.textContent).toContain('Loading Workspace Controls');
    expect(target.textContent).not.toContain('stale-model');
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
      draft.update('videoUpscaleFactor', 4);
      await settle();

      expect(target.querySelector('#ws-video-upscale')).not.toBeNull();
      expect(target.querySelector('input[name="video_upscale_factor"]')).not.toBeNull();

      const form = target.querySelector('form');
      expect(form).not.toBeNull();
      const serializedFormData = new FormData(form!);
      expect(serializedFormData.get('workflow')).toBe(workflow);
      expect(serializedFormData.get('upscale')).toBe('4');
      expect(serializedFormData.get('video_upscale_factor')).toBe('4');
    }
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

  it('renders the required 4px primary focus treatment on workspace form inputs', async () => {
    const context = makeContext();
    draft.hydrateFromContext(context, null);
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

    expectPrimaryFocusTreatment(target.querySelector('#ws-prompt'), 'Prompt textarea');
    expectPrimaryFocusTreatment(target.querySelector('#ws-negative-prompt'), 'Negative prompt textarea');
    expectPrimaryFocusTreatment(target.querySelector('#ws-ratio'), 'Aspect ratio select');
    expectPrimaryFocusTreatment(target.querySelector('#ws-size'), 'Size select');
    expectPrimaryFocusTreatment(target.querySelector('#ws-runs'), 'Batch size input');
    expectPrimaryFocusTreatment(target.querySelector('#ws-seed'), 'Seed input');

    const customDimensionsButton = Array.from(target.querySelectorAll('button')).find(
      (button) => button.textContent?.trim() === 'Custom W/H'
    ) as HTMLButtonElement | undefined;
    expect(customDimensionsButton).not.toBeUndefined();
    customDimensionsButton!.click();
    await settle();

    expectPrimaryFocusTreatment(target.querySelector('#ws-width'), 'Width input');
    expectPrimaryFocusTreatment(target.querySelector('#ws-height'), 'Height input');

    draft.update('workflow', 'img2img');
    await settle();

    expectPrimaryFocusTreatment(target.querySelector('#ws-image-path'), 'Reference image path input');
  });

  it('uses the 6px rounded-md radius on workspace toolbar controls', async () => {
    draft.update('workflow', 'txt2img');
    draft.update('model', 'zit');

    const context = makeContext();
    await mountWorkspace(context);

    // The model shell is always rendered; test it immediately.
    const modelShell = target.querySelector('[data-testid="model-shell"]');
    expectRoundedMdTreatment(modelShell, 'Model selector shell');

    // The quantize block is gated on {#if supportsQuantize} which requires the
    // async context to load. An extra settle() lets the onMount promise chain
    // (getWorkspaceContext → context = ctx → reactive re-render) complete.
    await settle();
    expect(target.querySelector('select[name="quantize"]'), 'Quantize select must be rendered for zit model (supports_quantize: true)').not.toBeNull();
    const quantizeShell = target.querySelector('[data-testid="quantize-shell"]');
    const addLoraButton = Array.from(target.querySelectorAll('button')).find(
      (button) => button.textContent?.trim() === 'Add LoRA'
    ) as HTMLButtonElement | undefined;

    expectRoundedMdTreatment(quantizeShell, 'Quantize selector shell');
    expectRoundedMdTreatment(addLoraButton ?? null, 'Add LoRA button');
  });

  it('keeps both toolbar selector shells on the same runtime focus path', async () => {
    draft.update('workflow', 'txt2img');
    draft.update('model', 'zit');

    const context = makeContext();
    await mountWorkspace(context);

    // Model shell is always rendered.
    const modelShell = target.querySelector('[data-testid="model-shell"]') as HTMLElement | null;
    const modelSelect = target.querySelector('#ws-model') as HTMLSelectElement | null;
    expectShellFocusTreatment(modelShell, 'Model selector shell');
    expectShellHoverAndFocusRuntime(modelShell, modelSelect, 'Model selector shell');

    // Wait for the quantize block (requires async context to load and supportsQuantize=true).
    // An extra settle() lets the onMount promise chain complete before asserting.
    await settle();
    expect(target.querySelector('select[name="quantize"]'), 'Quantize select must be rendered for zit model (supports_quantize: true)').not.toBeNull();
    const quantizeShell = target.querySelector('[data-testid="quantize-shell"]') as HTMLElement | null;
    const quantizeSelect = target.querySelector('select[name="quantize"]') as HTMLSelectElement | null;
    expectShellFocusTreatment(quantizeShell, 'Quantize selector shell');
    expectShellHoverAndFocusRuntime(quantizeShell, quantizeSelect, 'Quantize selector shell');
  });

  it('uses the truthful surface and subtle border tokens on toolbar selector shells', async () => {
    draft.update('workflow', 'txt2img');
    draft.update('model', 'zit');

    const context = makeContext();
    await mountWorkspace(context);

    const modelShell = target.querySelector('[data-testid="model-shell"]');
    expectTruthfulShellSurface(modelShell, 'Model selector shell');

    await settle();
    expect(target.querySelector('select[name="quantize"]')).not.toBeNull();
    const quantizeShell = target.querySelector('[data-testid="quantize-shell"]');
    expectTruthfulShellSurface(quantizeShell, 'Quantize selector shell');
  });

  it('keeps both toolbar selector shells on the same runtime hover path', async () => {
    draft.update('workflow', 'txt2img');
    draft.update('model', 'zit');

    const context = makeContext();
    await mountWorkspace(context);

    const modelShell = target.querySelector('[data-testid="model-shell"]');
    expectShellHoverTreatment(modelShell, 'Model selector shell');

    await settle();
    expect(target.querySelector('select[name="quantize"]')).not.toBeNull();
    const quantizeShell = target.querySelector('[data-testid="quantize-shell"]');
    expectShellHoverTreatment(quantizeShell, 'Quantize selector shell');
  });

  it('renders text color and spacing tokens directly on toolbar selector shells', async () => {
    draft.update('workflow', 'txt2img');
    draft.update('model', 'zit');

    const context = makeContext();
    await mountWorkspace(context);

    const modelShell = target.querySelector('[data-testid="model-shell"]');
    expectShellTextAndSpacingTreatment(modelShell, 'Model selector shell');

    await settle();
    expect(target.querySelector('select[name="quantize"]')).not.toBeNull();
    const quantizeShell = target.querySelector('[data-testid="quantize-shell"]');
    expectShellTextAndSpacingTreatment(quantizeShell, 'Quantize selector shell');
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
    const loadButton = Array.from(target.querySelectorAll('button')).find((button) => button.textContent?.trim() === 'Load');
    expect(loadButton).not.toBeUndefined();
    loadButton!.dispatchEvent(new MouseEvent('click', { bubbles: true }));
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

  it('keeps the previous prompt-file selection when a manual reload fails', async () => {
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
    const loadButton = Array.from(target.querySelectorAll('button')).find((button) => button.textContent?.trim() === 'Load');
    loadButton!.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    await settle();

    const optionSelect = target.querySelector('#ws-prompt-option') as HTMLSelectElement | null;
    optionSelect!.value = 'portrait:0';
    optionSelect!.dispatchEvent(new Event('change', { bubbles: true }));
    await settle();

    promptFileApiMocks.inspectPromptFile.mockRejectedValueOnce(new Error('POST /api/prompt-files/inspect → 422: missing file'));
    pathInput!.value = '/missing/prompts.yaml';
    pathInput!.dispatchEvent(new Event('input', { bubbles: true }));
    loadButton!.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    await settle();

    const hiddenPath = target.querySelector('input[name="prompts_file"]') as HTMLInputElement | null;
    expect(pathInput!.value).toBe('/server/prompts.yaml');
    expect(hiddenPath?.value).toBe('/server/prompts.yaml');
    expect((target.querySelector('#ws-prompt-option') as HTMLSelectElement | null)?.value).toBe('portrait:0');
    expect((target.querySelector('#ws-submit') as HTMLButtonElement | null)?.disabled).toBe(false);
    expect(target.textContent).toContain('missing file');
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
    const loadButton = Array.from(target.querySelectorAll('button')).find((button) => button.textContent?.trim() === 'Load');
    loadButton!.dispatchEvent(new MouseEvent('click', { bubbles: true }));
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
});