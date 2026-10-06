import { describe, it, expect, beforeEach } from 'vitest';
import { draft, offeredSizes, settingDefaultsFor } from './draft.svelte';
import type { WorkspaceContext, ImageModelDefaults, VideoModelDefaults } from '$lib/types';

// ── Helpers ──────────────────────────────────────────────────────────────────

function makeImageDefaults(overrides: Partial<ImageModelDefaults> = {}): ImageModelDefaults {
  return {
    ratio: '2:3',
    size: 'm',
    steps: 10,
    guidance: 3.5,
    width: 832,
    height: 1216,
    scheduler: null,
    supports_negative_prompt: false,
    supports_quantize: true,
    quantize: null,
    image_strength: 0.5,
    postprocess: { sharpen: 0.8, contrast: false, saturation: false },
    upscale: { enabled: false, factor: null, denoise: null, steps: null, guidance: null, sharpen: true, save_pre: false },
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
    width: 848,
    height: 480,
    frame_count: 97,
    audio: true,
    low_memory: true,
    supports_i2v: false,
    supports_quantize: false,
    quantize: null,
    max_steps: 8,
    fps: 24,
    upscale: { enabled: false, factor: 2, steps: null },
    ...overrides,
  };
}

function makeContext(overrides: Partial<WorkspaceContext> = {}): WorkspaceContext {
  const imgDefaults = makeImageDefaults();
  const vidDefaults = makeVideoDefaults();
  return {
    image_models: [{ id: 'flux-dev', label: 'FLUX Dev', type: 'image' }],
    video_models: [{ id: 'ltx-v-0.9', label: 'LTX Video', type: 'video' }],
    loras: [],
    history_assets: [],
    active_job: null,
    defaults: imgDefaults,
    video_defaults: vidDefaults,
    image_model_defaults: { 'flux-dev': imgDefaults },
    video_model_defaults: { 'ltx-v-0.9': vidDefaults },
    current_image_model: 'flux-dev',
    current_video_model: 'ltx-v-0.9',
    config: { gallery_page_size: 20 },
    output_dir: '/tmp/output',
    quantize_options: [4, 8],
    image_ratios: ['1:1', '2:3', '16:9'],
    video_ratios: ['16:9', '9:16'],
    image_size_options: { '1:1': ['s', 'm', 'l'], '2:3': ['s', 'm', 'l'], '16:9': ['s', 'm', 'l'] },
    video_size_options: { '16:9': ['s', 'm'], '9:16': ['s', 'm'] },
    image_size_dimensions: {},
    scheduler_options: ['euler', 'dpm'],
    workflow_contract: {
      values: ['txt2img', 'img2img', 'txt2vid', 'img2vid'],
      definitions: {
        txt2img: {
          mode: 'image',
          model_kind: 'image',
          visible_controls: ['workflow', 'model', 'prompt_inline', 'ratio', 'size', 'custom_dimensions', 'runs', 'steps', 'guidance', 'seed'],
          supports_reference_image: false,
          requires_reference_image: false,
          clear_fields: ['image_path', 'image_strength', 'frames', 'audio', 'low_memory'],
        },
        img2img: {
          mode: 'image',
          model_kind: 'image',
          visible_controls: ['workflow', 'model', 'prompt_inline', 'negative_prompt', 'reference_image', 'reference_image_path', 'reference_image_clear', 'ratio', 'size', 'custom_dimensions', 'runs', 'steps', 'guidance', 'image_strength', 'seed'],
          supports_reference_image: true,
          requires_reference_image: true,
          clear_fields: ['frames', 'audio', 'low_memory'],
        },
        txt2vid: {
          mode: 'video',
          model_kind: 'video',
          visible_controls: ['workflow', 'model', 'prompt_inline', 'ratio', 'size', 'custom_dimensions', 'runs', 'frame_count', 'steps', 'seed', 'audio', 'low_memory'],
          supports_reference_image: false,
          requires_reference_image: false,
          clear_fields: ['negative_prompt', 'guidance', 'image_path', 'image_strength', 'quantize'],
        },
        img2vid: {
          mode: 'video',
          model_kind: 'video',
          visible_controls: ['workflow', 'model', 'prompt_inline', 'reference_image', 'reference_image_path', 'reference_image_clear', 'ratio', 'size', 'custom_dimensions', 'runs', 'frame_count', 'steps', 'seed', 'audio', 'low_memory'],
          supports_reference_image: true,
          requires_reference_image: true,
          clear_fields: ['negative_prompt', 'guidance', 'quantize'],
        },
      },
      field_precedence: { defaults: [], dimensions: '' },
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

// ── Tests ─────────────────────────────────────────────────────────────────────

describe('draft store', () => {
  beforeEach(() => {
    localStorage.clear();
    draft.reset();
  });

  it('loads default state when localStorage is empty', () => {
    expect(draft.state.workflow).toBe('txt2img');
    expect(draft.state.prompt).toBe('');
  });

  it('updates a field and saves to localStorage', () => {
    draft.update('prompt', 'test prompt');
    expect(draft.state.prompt).toBe('test prompt');
    expect(localStorage.getItem('ziv-workspace-draft-v1')).toContain('test prompt');
  });

  it('applies URL prefill correctly', () => {
    draft.loadFromUrl({ workflow: 'img2img', prompt: 'from url', model: 'flux-dev' }, makeContext());
    expect(draft.state.workflow).toBe('img2img');
    expect(draft.state.prompt).toBe('from url');
  });

  it('applies a reused negative prompt and scheduler when the workflow shows them', () => {
    const ctx = makeContext();
    ctx.workflow_contract.definitions.txt2img.visible_controls.push('negative_prompt', 'scheduler');
    draft.loadFromUrl({ workflow: 'txt2img', negative_prompt: 'blurry', scheduler: 'beta' }, ctx);
    expect(draft.state.negativePrompt).toBe('blurry');
    expect(draft.state.scheduler).toBe('beta');
  });

  it('leaves a reused scheduler out when the workflow hides it', () => {
    draft.loadFromUrl({ workflow: 'txt2img', scheduler: 'beta' }, makeContext());
    expect(draft.state.scheduler).not.toBe('beta');
  });

  it('ignores invalid workflow in URL prefill', () => {
    draft.loadFromUrl({ workflow: 'invalid-workflow' }, makeContext());
    expect(draft.state.workflow).toBe('txt2img'); // unchanged
  });

  describe('loadFromUrl – workflow handling', () => {
    it('ignores unsupported workflow aliases', () => {
      draft.loadFromUrl({ workflow: 'image' }, makeContext());
      expect(draft.state.workflow).toBe('txt2img');
    });

    it('ignores non-canonical workflow spellings', () => {
      draft.loadFromUrl({ workflow: 'txt-2-img' }, makeContext());
      expect(draft.state.workflow).toBe('txt2img');
    });

    it('applies ratio, size, frames, seed, and image_path URL params', () => {
      draft.loadFromUrl({ ratio: '16:9', size: 'l', frames: '121', seed: '9876', image_path: '/tmp/ref.png' });
      expect(draft.state.ratio).toBe('16:9');
      expect(draft.state.size).toBe('l');
      expect(draft.state.frameCount).toBe(121);
      expect(draft.state.seed).toBe(9876);
      expect(draft.state.referenceImagePath).toBe('/tmp/ref.png');
    });
  });

  describe('hydrateFromContext', () => {
    it('hydrates the backend default prompt source into draft state', () => {
      const ctx = makeContext({ default_prompt_source: 'file' });
      draft.update('workflow', 'txt2img');
      draft.update('promptSource', 'inline');

      draft.hydrateFromContext(ctx, null);

      expect(draft.state.promptSource).toBe('file');
    });

    it('sets image model and defaults for txt2img workflow', () => {
      const ctx = makeContext();
      draft.update('workflow', 'txt2img');
      draft.hydrateFromContext(ctx, null);

      expect(draft.state.model).toBe('flux-dev');
      expect(draft.state.ratio).toBe('2:3');
      expect(draft.state.size).toBe('m');
      expect(draft.state.steps).toBe(10);
      expect(draft.state.guidance).toBe(3.5);
      expect(draft.state.width).toBe(832);
      expect(draft.state.height).toBe(1216);
      expect(draft.state.referenceImageStrength).toBe(0.5);
    });

    it('sets video model and defaults for txt2vid workflow', () => {
      const ctx = makeContext();
      draft.update('workflow', 'txt2vid');
      draft.hydrateFromContext(ctx, null);

      expect(draft.state.model).toBe('ltx-v-0.9');
      expect(draft.state.ratio).toBe('16:9');
      expect(draft.state.width).toBe(848);
      expect(draft.state.height).toBe(480);
      expect(draft.state.frameCount).toBe(97);
      expect(draft.state.audio).toBe(true);
      expect(draft.state.lowMemory).toBe(true);
    });

    it('hydrates image width, height, and reference image strength from backend defaults', () => {
      const ctx = makeContext({
        image_model_defaults: {
          'flux-dev': makeImageDefaults({ width: 1024, height: 576, image_strength: 0.0, guidance: 5.5 }),
        },
      });

      draft.update('workflow', 'img2img');
      draft.update('width', 64);
      draft.update('height', 64);
      draft.update('referenceImageStrength', 0.7);

      draft.hydrateFromContext(ctx, null);

      expect(draft.state.width).toBe(1024);
      expect(draft.state.height).toBe(576);
      expect(draft.state.referenceImageStrength).toBe(0.0);
      expect(draft.state.guidance).toBe(5.5);
    });

    it('hydrates video width and height from backend defaults', () => {
      const ctx = makeContext({
        video_model_defaults: {
          'ltx-v-0.9': makeVideoDefaults({ width: 960, height: 544, frame_count: 81 }),
        },
      });

      draft.update('workflow', 'txt2vid');
      draft.update('width', 64);
      draft.update('height', 64);

      draft.hydrateFromContext(ctx, null);

      expect(draft.state.width).toBe(960);
      expect(draft.state.height).toBe(544);
      expect(draft.state.frameCount).toBe(81);
    });

    it('uses preferredModel when it is valid for the current workflow', () => {
      const ctx = makeContext({
        image_models: [
          { id: 'flux-dev', label: 'FLUX Dev', type: 'image' },
          { id: 'flux-schnell', label: 'FLUX Schnell', type: 'image' },
        ],
        image_model_defaults: {
          'flux-dev': makeImageDefaults({ steps: 10 }),
          'flux-schnell': makeImageDefaults({ steps: 4 }),
        },
      });
      draft.update('workflow', 'txt2img');
      draft.hydrateFromContext(ctx, 'flux-schnell');

      expect(draft.state.model).toBe('flux-schnell');
      expect(draft.state.steps).toBe(4);
    });

    it('falls back to context default when preferredModel is invalid for the workflow', () => {
      const ctx = makeContext();
      draft.update('workflow', 'txt2vid');
      // 'flux-dev' is an image model, not valid for video workflow
      draft.hydrateFromContext(ctx, 'flux-dev');

      expect(draft.state.model).toBe('ltx-v-0.9');
    });

    it('replaces a stale stored image model with the backend current image model defaults', () => {
      const ctx = makeContext({
        current_image_model: 'flux-dev',
        image_model_defaults: {
          'flux-dev': makeImageDefaults({ ratio: '16:9', size: 'l', steps: 18, guidance: 4.2 }),
        },
      });
      draft.update('workflow', 'txt2img');
      draft.update('model', 'stale-model');
      draft.update('ratio', '1:1');
      draft.update('size', 's');
      draft.update('steps', 2);
      draft.update('guidance', 0.1);

      draft.hydrateFromContext(ctx, null);

      expect(draft.state.model).toBe('flux-dev');
      expect(draft.state.ratio).toBe('16:9');
      expect(draft.state.size).toBe('l');
      expect(draft.state.steps).toBe(18);
      expect(draft.state.guidance).toBe(4.2);
    });

    it('corrects a stale image model when workflow is txt2vid', () => {
      const ctx = makeContext();
      // Simulate stale draft with image model set, then workflow changes to video
      draft.update('model', 'flux-dev');
      draft.update('workflow', 'txt2vid');
      draft.hydrateFromContext(ctx, null);

      expect(draft.state.model).toBe('ltx-v-0.9');
    });

    it('clears a kept scheduler on a model that uses its own sampler', () => {
      const kreaDefaults = makeImageDefaults({ steps: 8, guidance: 1, supports_scheduler: false });
      const ctx = makeContext({
        image_models: [{ id: 'krea2', label: 'Krea 2', type: 'image' }],
        current_image_model: 'krea2',
        defaults: kreaDefaults,
        image_model_defaults: { krea2: kreaDefaults },
      });

      draft.update('workflow', 'txt2img');
      draft.hydrateFromContext(ctx, 'krea2');
      draft.update('scheduler', 'beta');

      draft.hydrateFromContext(ctx, 'krea2', { keepSettings: true });

      expect(draft.state.scheduler).toBeNull();
    });

    it('clears structured-json and first-sigma draft fields when switching to a non-supporting image model', () => {
      const ideogramDefaults = makeIdeogramDefaults();
      const permissiveDefaults = makeImageDefaults({ ratio: '16:9', size: 'l', width: 1664, height: 928 });
      const ctx = makeContext({
        image_models: [
          { id: 'ideo', label: 'Ideogram 4', type: 'image' },
          { id: 'flux-dev', label: 'FLUX Dev', type: 'image' },
        ],
        current_image_model: 'ideo',
        defaults: ideogramDefaults,
        image_model_defaults: {
          ideo: ideogramDefaults,
          'flux-dev': permissiveDefaults,
        },
      });

      draft.update('workflow', 'txt2img');
      draft.hydrateFromContext(ctx, 'ideo');
      draft.update('jsonPromptEnabled', true);
      draft.update('jsonPrompt', '{"high_level_description":"caption"}');
      draft.update('firstSigma', 1.004);

      draft.hydrateFromContext(ctx, 'flux-dev');

      expect(draft.state.jsonPromptEnabled).toBe(false);
      expect(draft.state.jsonPrompt).toBe('');
      expect(draft.state.firstSigma).toBeNull();
    });

    it('retains structured-json and first-sigma draft fields when hydrating on an ideogram-capable model', () => {
      const ideogramDefaults = makeIdeogramDefaults();
      const ctx = makeContext({
        image_models: [{ id: 'ideo', label: 'Ideogram 4', type: 'image' }],
        current_image_model: 'ideo',
        defaults: ideogramDefaults,
        image_model_defaults: {
          ideo: ideogramDefaults,
        },
      });

      draft.update('workflow', 'txt2img');
      draft.hydrateFromContext(ctx, 'ideo');
      draft.update('jsonPromptEnabled', true);
      draft.update('jsonPrompt', '{"high_level_description":"caption"}');
      draft.update('firstSigma', 1.006);

      draft.hydrateFromContext(ctx, 'ideo');

      expect(draft.state.jsonPromptEnabled).toBe(true);
      expect(draft.state.jsonPrompt).toBe('{"high_level_description":"caption"}');
      expect(draft.state.firstSigma).toBe(1.006);
    });

    it('clears structured-json and first-sigma draft fields when hydrating a video workflow', () => {
      const ideogramDefaults = makeIdeogramDefaults();
      const ctx = makeContext({
        image_models: [{ id: 'ideo', label: 'Ideogram 4', type: 'image' }],
        current_image_model: 'ideo',
        defaults: ideogramDefaults,
        image_model_defaults: {
          ideo: ideogramDefaults,
        },
      });

      draft.update('workflow', 'txt2img');
      draft.hydrateFromContext(ctx, 'ideo');
      draft.update('jsonPromptEnabled', true);
      draft.update('jsonPrompt', '{"high_level_description":"caption"}');
      draft.update('firstSigma', 1.004);
      draft.update('workflow', 'txt2vid');

      draft.hydrateFromContext(ctx, null);

      expect(draft.state.jsonPromptEnabled).toBe(false);
      expect(draft.state.jsonPrompt).toBe('');
      expect(draft.state.firstSigma).toBeNull();
    });
  });

  describe('resetSelections', () => {
    it('restores model defaults while keeping model, loras, quantize, and prompt', () => {
      const ctx = makeContext();
      draft.update('workflow', 'txt2img');
      draft.hydrateFromContext(ctx, null);
      draft.update('prompt', 'a cat');
      draft.update('promptSource', 'file');
      draft.update('promptFilePath', '/tmp/prompts.yaml');
      draft.update('loraString', 'style:0.8');
      draft.update('quantize', 8);
      draft.update('steps', 42);
      draft.update('seed', 1234);
      draft.update('runs', 5);
      draft.update('ratio', '1:1');
      draft.update('upscaleEnabled', true);

      draft.resetSelections(ctx);

      expect(draft.state.model).toBe('flux-dev');
      expect(draft.state.prompt).toBe('a cat');
      expect(draft.state.promptSource).toBe('file');
      expect(draft.state.promptFilePath).toBe('/tmp/prompts.yaml');
      expect(draft.state.loraString).toBe('style:0.8');
      expect(draft.state.quantize).toBe(8);
      expect(draft.state.steps).toBe(10);
      expect(draft.state.seed).toBeNull();
      expect(draft.state.runs).toBe(1);
      expect(draft.state.ratio).toBe('2:3');
      expect(draft.state.upscaleEnabled).toBe(false);
    });
  });

  describe('onWorkflowChange', () => {
    it('switches to video model and defaults when changing to txt2vid', () => {
      const ctx = makeContext();
      draft.update('workflow', 'txt2img');
      draft.hydrateFromContext(ctx, null);
      // Simulate workflow switch
      draft.onWorkflowChange('txt2vid', ctx);

      expect(draft.state.workflow).toBe('txt2vid');
      expect(draft.state.model).toBe('ltx-v-0.9');
      expect(draft.state.ratio).toBe('16:9');
    });

    it('switches back to image model when changing from txt2vid to txt2img', () => {
      const ctx = makeContext();
      draft.update('workflow', 'txt2vid');
      draft.hydrateFromContext(ctx, null);
      draft.onWorkflowChange('txt2img', ctx);

      expect(draft.state.workflow).toBe('txt2img');
      expect(draft.state.model).toBe('flux-dev');
      expect(draft.state.ratio).toBe('2:3');
    });
  });
});

describe('draft store – saved drafts from the previous version', () => {
  beforeEach(() => {
    localStorage.clear();
    draft.reset();
  });

  function loadSavedV2(saved: Record<string, unknown>): void {
    localStorage.setItem('ziv-workspace-draft-v1', JSON.stringify({ version: 2, prompt: 'kept prompt', ...saved }));
    draft.loadDraft();
  }

  it('turns the old fixed sharpen default into auto and keeps the rest', () => {
    loadSavedV2({ postprocessSharpenAmount: 0.8 });
    expect(draft.state.postprocessSharpenAmount).toBeNull();
    expect(draft.state.prompt).toBe('kept prompt');
  });

  it('keeps a sharpen amount the user chose', () => {
    loadSavedV2({ postprocessSharpenAmount: 0.6 });
    expect(draft.state.postprocessSharpenAmount).toBe(0.6);
  });

  it('lowers a saved amount above the new limit to the limit', () => {
    loadSavedV2({ postprocessSharpenAmount: 1.8 });
    expect(draft.state.postprocessSharpenAmount).toBe(1.5);
  });
});

describe('draft store – prompt enhancer fields', () => {
  beforeEach(() => {
    localStorage.clear();
    draft.reset();
  });

  it('persists enhancer fields', () => {
    draft.update('enhancedPrompt', 'A fox in snow.');
    draft.update('enhancedFrom', { prompt: 'a fox', mode: 'image' });
    draft.update('enhanceAuto', true);
    draft.update('enhanceSettings', { style: 'photo', mood: 'keep', details: [], length: 'longer', motion: [] });
    draft.loadDraft();
    expect(draft.state.enhancedPrompt).toBe('A fox in snow.');
    expect(draft.state.enhancedFrom).toEqual({ prompt: 'a fox', mode: 'image' });
    expect(draft.state.enhanceAuto).toBe(true);
    expect(draft.state.enhanceSettings?.style).toBe('photo');
  });

  it('clears the Enhanced box when a prompt is reused from the URL', () => {
    draft.update('enhancedPrompt', 'Old enhanced text');
    draft.update('enhancedFrom', { prompt: 'old', mode: 'image' });
    draft.loadFromUrl({ prompt: 'reused prompt' }, makeContext());
    expect(draft.state.prompt).toBe('reused prompt');
    expect(draft.state.enhancedPrompt).toBe('');
    expect(draft.state.enhancedFrom).toBeNull();
  });

  it('keeps the Enhanced box when the URL has no prompt', () => {
    draft.update('enhancedPrompt', 'Kept');
    draft.loadFromUrl({ steps: '12' }, makeContext());
    expect(draft.state.enhancedPrompt).toBe('Kept');
  });

  it('resetSelections keeps the Enhanced prompt but resets enhancement options', () => {
    const ctx = makeContext();
    draft.hydrateFromContext(ctx, null);
    draft.update('enhancedPrompt', 'A fox in snow.');
    draft.update('enhancedFrom', { prompt: 'a fox', mode: 'image' });
    draft.update('enhanceAuto', true);
    draft.update('enhanceSettings', { style: 'photo', mood: 'keep', details: [], length: 'longer', motion: [] });
    draft.resetSelections(ctx);
    expect(draft.state.enhancedPrompt).toBe('A fox in snow.');
    expect(draft.state.enhancedFrom).toEqual({ prompt: 'a fox', mode: 'image' });
    expect(draft.state.enhanceAuto).toBe(false);
    expect(draft.state.enhanceSettings).toBeNull();
  });
});

describe('settingDefaultsFor', () => {
  beforeEach(() => {
    localStorage.clear();
    draft.reset();
  });

  it('returns the model defaults that resetSelections would apply, without touching the draft', () => {
    const ctx = makeContext();
    draft.hydrateFromContext(ctx);
    draft.patch({ steps: 42, seed: 7, guidance: 9 });

    const defaults = settingDefaultsFor(ctx, draft.state);

    expect(defaults).toMatchObject({ steps: 10, guidance: 3.5, seed: null, runs: 1, ratio: '2:3', size: 'm' });
    expect(draft.state.steps).toBe(42);
    draft.resetSelections(ctx);
    expect(draft.state).toMatchObject({ steps: defaults.steps, guidance: defaults.guidance, seed: defaults.seed });
  });

  it('uses the video defaults for video workflows', () => {
    const ctx = makeContext();
    draft.update('workflow', 'txt2vid');
    draft.hydrateFromContext(ctx);

    expect(settingDefaultsFor(ctx, draft.state)).toMatchObject({ steps: 8, frameCount: 97, audio: true, lowMemory: true });
  });
});

describe('draft settings persistence', () => {
  beforeEach(() => {
    localStorage.clear();
    draft.reset();
  });

  it('keeps saved settings when the workspace reloads for the same model', () => {
    const ctx = makeContext();
    draft.hydrateFromContext(ctx);
    draft.patch({ steps: 2, guidance: 1.2, runs: 40, ratio: '16:9', size: 'l', promptSource: 'file' });

    draft.loadDraft();
    draft.hydrateFromContext(ctx, null, { keepSettings: true });

    expect(draft.state).toMatchObject({ steps: 2, guidance: 1.2, runs: 40, ratio: '16:9', size: 'l', promptSource: 'file' });
  });

  it('applies the new model defaults when the model changes', () => {
    const ctx = makeContext({
      image_models: [{ id: 'flux-dev', label: 'FLUX Dev', type: 'image' }, { id: 'other', label: 'Other', type: 'image' }],
      image_model_defaults: { 'flux-dev': makeImageDefaults(), other: makeImageDefaults({ steps: 30 }) },
    });
    draft.hydrateFromContext(ctx);
    draft.update('steps', 2);

    draft.hydrateFromContext(ctx, 'other', { keepSettings: true });

    expect(draft.state.steps).toBe(30);
  });

  it('falls back to the default size when a saved preset no longer exists', () => {
    const ctx = makeContext();
    draft.hydrateFromContext(ctx);
    draft.patch({ steps: 2, ratio: '21:9', size: 'xl' });

    draft.hydrateFromContext(ctx, null, { keepSettings: true });

    expect(draft.state).toMatchObject({ steps: 2, ratio: '2:3', size: 'm' });
  });

  it('keeps settings when switching between workflows of the same mode', () => {
    const ctx = makeContext();
    draft.hydrateFromContext(ctx);
    draft.patch({ steps: 3, guidance: 2 });

    draft.onWorkflowChange('img2img', ctx);

    expect(draft.state).toMatchObject({ workflow: 'img2img', steps: 3, guidance: 2 });
  });

  it('switches reused assets back to their preset unless their size was custom', () => {
    const ctx = makeContext();
    draft.hydrateFromContext(ctx);
    draft.patch({ dimensionMode: 'custom', width: 1000, height: 1000 });

    draft.loadFromUrl({ ratio: '1:1', size: 'm', width: '1024', height: '1024' }, ctx);
    expect(draft.state.dimensionMode).toBe('ratio');

    draft.loadFromUrl({ ratio: '1:1', size: 'custom', width: '640', height: '480' }, ctx);
    expect(draft.state.dimensionMode).toBe('custom');
  });

  it('keeps a custom size across reloads', () => {
    const ctx = makeContext();
    draft.hydrateFromContext(ctx);
    draft.loadFromUrl({ ratio: '1:1', size: 'custom', width: '640', height: '480' }, ctx);

    draft.hydrateFromContext(ctx, null, { keepSettings: true });

    expect(draft.state).toMatchObject({ dimensionMode: 'custom', width: 640, height: 480 });
  });
});

describe('offeredSizes', () => {
  it('leaves out image presets beyond the model dimension limit', () => {
    const constrained = makeImageDefaults({ dimension_max: 1024 });
    const ctx = makeContext({
      defaults: constrained,
      image_model_defaults: { 'flux-dev': constrained },
      image_size_options: { '16:9': ['m', 'l'] },
      image_size_dimensions: { '16:9': { m: [1024, 576], l: [1344, 768] } },
    });
    expect(offeredSizes(ctx, 'txt2img', 'flux-dev', '16:9')).toEqual(['m']);
  });

  it('offers every video preset for the ratio', () => {
    expect(offeredSizes(makeContext(), 'txt2vid', 'ltx-v-0.9', '16:9')).toEqual(['s', 'm']);
  });
});
