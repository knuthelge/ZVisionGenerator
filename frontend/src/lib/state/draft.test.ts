import { describe, it, expect, beforeEach } from 'vitest';
import { legacySource, sceneSource, sourceInfo } from './extend.fixtures';
import { anyEnhancedUsed } from './promptEnhance';
import { draft, offeredSizes, settingDefaultsFor, videoTimingFor } from './draft.svelte';
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
          requires_source_video: false,
          clear_fields: ['image_path', 'image_strength', 'frames', 'audio', 'low_memory'],
        },
        img2img: {
          mode: 'image',
          model_kind: 'image',
          visible_controls: ['workflow', 'model', 'prompt_inline', 'negative_prompt', 'reference_image', 'reference_image_path', 'reference_image_clear', 'ratio', 'size', 'custom_dimensions', 'runs', 'steps', 'guidance', 'image_strength', 'seed'],
          supports_reference_image: true,
          requires_reference_image: true,
          requires_source_video: false,
          clear_fields: ['frames', 'audio', 'low_memory'],
        },
        txt2vid: {
          mode: 'video',
          model_kind: 'video',
          visible_controls: ['workflow', 'model', 'prompt_inline', 'prompt_scene', 'ratio', 'size', 'custom_dimensions', 'runs', 'frame_count', 'steps', 'seed', 'audio', 'low_memory'],
          supports_reference_image: false,
          requires_reference_image: false,
          requires_source_video: false,
          clear_fields: ['negative_prompt', 'guidance', 'image_path', 'image_strength', 'quantize'],
        },
        img2vid: {
          mode: 'video',
          model_kind: 'video',
          visible_controls: ['workflow', 'model', 'prompt_inline', 'prompt_scene', 'reference_image', 'reference_image_path', 'reference_image_clear', 'ratio', 'size', 'custom_dimensions', 'runs', 'frame_count', 'steps', 'seed', 'audio', 'low_memory'],
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
      field_precedence: { defaults: [], dimensions: '' },
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

  it('reuses a JSON caption into the JSON box, laid out, with the text boxes cleared', () => {
    draft.update('prompt', 'older text');
    draft.loadFromUrl({ workflow: 'txt2img', json_prompt: '{"high_level_description":"A {red} fox."}' }, makeContext());
    expect(draft.state.jsonPromptEnabled).toBe(true);
    expect(draft.state.jsonPrompt).toBe('{\n  "high_level_description": "A {red} fox."\n}');
    expect(draft.state.prompt).toBe('');
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

    it('takes the frame rate from the video model defaults', () => {
      const ctx = makeContext({
        video_model_defaults: { 'ltx-v-0.9': makeVideoDefaults({ fps: 30 }) },
      });

      draft.update('workflow', 'txt2vid');
      draft.hydrateFromContext(ctx, null);

      expect(draft.state.fps).toBe(30);
    });

    it('aligns a reused frame count to one the model accepts', () => {
      const ctx = makeContext({
        video_model_defaults: { 'ltx-v-0.9': makeVideoDefaults({ frame_alignment: 8 }) },
      });

      draft.update('workflow', 'txt2vid');
      draft.hydrateFromContext(ctx, null);
      draft.loadFromUrl({ frames: '50' }, ctx);

      expect(draft.state.frameCount).toBe(49);
    });

    it('aligns a reused frame count with the default video model when the named one has no defaults', () => {
      const aligned = makeVideoDefaults({ frame_alignment: 8 });
      const ctx = makeContext({ video_defaults: aligned, video_model_defaults: { 'ltx-v-0.9': aligned } });

      draft.update('workflow', 'txt2vid');
      draft.hydrateFromContext(ctx, null);
      draft.loadFromUrl({ model: 'stale-ltx', frames: '50' }, ctx);

      expect(draft.state.frameCount).toBe(49);
    });

    it('aligns a kept frame count that predates snapping', () => {
      const ctx = makeContext({
        video_model_defaults: { 'ltx-v-0.9': makeVideoDefaults({ frame_alignment: 8 }) },
      });

      draft.update('workflow', 'txt2vid');
      draft.hydrateFromContext(ctx, null);
      draft.update('frameCount', 100);
      draft.hydrateFromContext(ctx, null, { keepSettings: true });

      expect(draft.state.frameCount).toBe(97);
    });

    it('aligns an off-grid configured default so Reset matches the hydrated count', () => {
      const ctx = makeContext({
        video_model_defaults: { 'ltx-v-0.9': makeVideoDefaults({ frame_count: 50, frame_alignment: 8 }) },
      });

      draft.update('workflow', 'txt2vid');
      draft.hydrateFromContext(ctx, null);

      expect(draft.state.frameCount).toBe(49);
      expect(settingDefaultsFor(ctx, { workflow: 'txt2vid', model: 'ltx-v-0.9' }).frameCount).toBe(49);
    });

    it('takes fps and alignment together from the default video model for a model without defaults', () => {
      const ctx = makeContext({ video_defaults: makeVideoDefaults({ fps: 30, frame_alignment: 8 }) });

      expect(videoTimingFor(ctx, 'unknown-video')).toEqual({ fps: 30, frameAlignment: 8 });
      expect(videoTimingFor(ctx, 'ltx-v-0.9')).toEqual({ fps: 24, frameAlignment: 1 });
      // A source's frame rate replaces the model's, and the alignment stays the model's.
      expect(videoTimingFor(ctx, 'ltx-v-0.9', 30)).toEqual({ fps: 30, frameAlignment: 1 });
      expect(videoTimingFor(ctx, 'ltx-v-0.9', null)).toEqual({ fps: 24, frameAlignment: 1 });
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

  it('seeds the video options from options saved before images and video kept their own', () => {
    const saved = { style: 'photo', mood: 'keep', theme: 'keep', camera: 'keep', pace: 'keep', details: [], length: 'longer', motion: [], sound: [], note: '' };
    loadSavedV2({ enhanceSettings: saved });
    expect(draft.state.enhanceSettings).toEqual(saved);
    expect(draft.state.enhanceSettingsVideo).toEqual(saved);
  });

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
    draft.update('enhanceSettings', { style: 'photo', mood: 'keep', theme: 'keep', camera: 'keep', pace: 'keep', details: [], length: 'longer', motion: [], sound: [], note: '' });
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

  describe('reusing a video scene and prompt', () => {
    beforeEach(() => draft.update('workflow', 'txt2vid'));

    it('fills the scene and the prompt', () => {
      draft.loadFromUrl({ scene: 'A castle at dusk.', prompt: 'A knight rides in.' }, makeContext());
      expect(draft.state).toMatchObject({ scene: 'A castle at dusk.', prompt: 'A knight rides in.', enhancedScene: '', enhancedPrompt: '', enhancedFrom: null });
    });

    it('fills a scene-only asset and empties the prompt box', () => {
      draft.update('prompt', 'left over');
      draft.loadFromUrl({ scene: 'A castle at dusk.' }, makeContext());
      expect(draft.state).toMatchObject({ scene: 'A castle at dusk.', prompt: '' });
    });

    it('fills both tabs and keeps the enhanced pair current for the reused inputs', () => {
      draft.loadFromUrl({ scene: 'A castle.', prompt: 'a knight', enhanced_scene: 'A castle at dusk.', enhanced_prompt: 'A knight rides in.' }, makeContext());
      expect(draft.state).toMatchObject({ scene: 'A castle.', prompt: 'a knight', enhancedScene: 'A castle at dusk.', enhancedPrompt: 'A knight rides in.' });
      expect(draft.state.enhancedFrom).toEqual({ scene: 'A castle.', prompt: 'a knight', mode: 'video' });
    });

    it('fills an Enhanced box only from the enhanced parameter present, never from the plain text', () => {
      draft.loadFromUrl({ scene: 'A castle.', prompt: 'a knight', enhanced_prompt: 'A knight rides in.' }, makeContext());
      expect(draft.state).toMatchObject({ scene: 'A castle.', prompt: 'a knight', enhancedScene: '', enhancedPrompt: 'A knight rides in.' });
      expect(draft.state.enhancedFrom).toEqual({ scene: 'A castle.', prompt: 'a knight', mode: 'video' });
      expect(anyEnhancedUsed(draft.state)).toBe(true);
      draft.loadFromUrl({ scene: 'A castle.', prompt: 'a knight', enhanced_scene: 'A castle at dusk.' }, makeContext());
      expect(draft.state).toMatchObject({ enhancedScene: 'A castle at dusk.', enhancedPrompt: '' });
    });

    it('clears an earlier Enhanced pair when the reused text has none', () => {
      draft.patch({ enhancedScene: 'Old.', enhancedPrompt: 'Old.', enhancedFrom: { scene: 'x', prompt: 'y', mode: 'video' } });
      draft.loadFromUrl({ scene: 'A castle.', prompt: 'a knight' }, makeContext());
      expect(draft.state).toMatchObject({ enhancedScene: '', enhancedPrompt: '', enhancedFrom: null });
    });

    it('leaves the scene alone when an image is reused', () => {
      draft.patch({ workflow: 'txt2img', scene: 'kept' });
      draft.loadFromUrl({ prompt: 'a fox' }, makeContext());
      expect(draft.state).toMatchObject({ scene: 'kept', prompt: 'a fox' });
    });

    it('does not leave an Enhanced scene behind when an image is reused', () => {
      draft.patch({ enhancedScene: 'A castle at dusk.', enhancedPrompt: 'Old.', enhancedFrom: { scene: 'A castle.', prompt: 'x', mode: 'video' } });
      draft.patch({ workflow: 'txt2img' });
      draft.loadFromUrl({ prompt: 'a fox' }, makeContext());
      draft.update('workflow', 'txt2vid');
      expect(draft.state).toMatchObject({ prompt: 'a fox', enhancedScene: '', enhancedPrompt: '', enhancedFrom: null });
      expect(anyEnhancedUsed(draft.state)).toBe(false);
    });

    it('survives a reset of the selections', () => {
      const ctx = makeContext();
      draft.hydrateFromContext(ctx, null);
      draft.patch({ scene: 'A castle.', enhancedScene: 'A castle at dusk.' });
      draft.resetSelections(ctx);
      expect(draft.state).toMatchObject({ scene: 'A castle.', enhancedScene: 'A castle at dusk.' });
    });
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
    draft.update('enhanceSettings', { style: 'photo', mood: 'keep', theme: 'keep', camera: 'keep', pace: 'keep', details: [], length: 'longer', motion: [], sound: [], note: '' });
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

  it('reuses an asset at its own size when its preset has since changed size', () => {
    const ctx = makeContext({ image_size_dimensions: { '1:1': { m: [1024, 1024] } } });
    draft.hydrateFromContext(ctx);

    draft.loadFromUrl({ ratio: '1:1', size: 'm', width: '1024', height: '1024' }, ctx);
    expect(draft.state).toMatchObject({ dimensionMode: 'ratio', size: 'm' });

    draft.loadFromUrl({ ratio: '1:1', size: 'm', width: '1088', height: '1088' }, ctx);
    expect(draft.state).toMatchObject({ dimensionMode: 'custom', size: 'custom', width: 1088, height: 1088 });
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

describe('extend', () => {
  const extendContext = (overrides: Partial<WorkspaceContext> = {}): WorkspaceContext => {
    const ctx = makeContext({
      video_models: [{ id: 'ltx-v-0.9', label: 'LTX Video', type: 'video' }, { id: 'ltx-2', label: 'LTX 2', type: 'video' }],
      video_model_defaults: { 'ltx-v-0.9': makeVideoDefaults(), 'ltx-2': makeVideoDefaults({ steps: 12, frame_alignment: 8 }) },
      ...overrides,
    });
    ctx.workflow_contract.values = ['txt2img', 'img2img', 'txt2vid', 'img2vid', 'extend'];
    return ctx;
  };
  const SOURCE = '/clips/walk.mp4';

  beforeEach(() => {
    localStorage.clear();
    draft.reset();
    draft.update('workflow', 'extend');
  });

  describe('URL and Reuse prefill', () => {
    it('selects the extend workflow with its source and new frames', () => {
      draft.loadFromUrl({ workflow: 'extend', source_video: SOURCE, extend_frames: '48', scene: 'A street.', model: 'ltx-2', lora: 'style:0.7', steps: '6', seed: '5' }, extendContext());
      expect(draft.state).toMatchObject({
        workflow: 'extend', extendSourcePath: SOURCE, extendFrames: 48, scene: 'A street.', prompt: '', model: 'ltx-2', loraString: 'style:0.7', steps: 6, seed: 5,
        missingSourceVideo: false,
      });
    });

    it('does not align the new frames to 8k+1, only to a multiple of 8', () => {
      draft.loadFromUrl({ workflow: 'extend', source_video: SOURCE, extend_frames: '48' }, extendContext());
      expect(draft.state.extendFrames).toBe(48);
      draft.loadFromUrl({ workflow: 'extend', source_video: SOURCE, extend_frames: '50' }, extendContext());
      expect(draft.state.extendFrames).toBe(48);
      draft.loadFromUrl({ workflow: 'extend', source_video: SOURCE, extend_frames: '3' }, extendContext());
      expect(draft.state.extendFrames).toBe(8);
    });

    it('ignores size, clip length and reference fields, which an extension does not have', () => {
      draft.loadFromUrl({ workflow: 'extend', source_video: SOURCE, frames: '121', width: '640', height: '480', ratio: '9:16', size: 's', image_path: '/a.png', guidance: '4' }, extendContext());
      expect(draft.state.frameCount).not.toBe(121);
      expect(draft.state.width).not.toBe(640);
      expect(draft.state.ratio).not.toBe('9:16');
      expect(draft.state.referenceImagePath).toBeNull();
      expect(draft.state.guidance).not.toBe(4);
    });

    it('flags a source that moved and keeps its recorded path', () => {
      draft.loadFromUrl({ workflow: 'extend', source_video: SOURCE, extend_frames: '48', missing_source_video: 'true', scene: 'A street.' }, extendContext());
      expect(draft.state).toMatchObject({ extendSourcePath: SOURCE, missingSourceVideo: true, scene: 'A street.' });
    });

    it('flags a source with no recorded path', () => {
      draft.loadFromUrl({ workflow: 'extend', missing_source_video: 'true' }, extendContext());
      expect(draft.state).toMatchObject({ extendSourcePath: null, missingSourceVideo: true });
    });

    it('replaces the text of the boxes with the reused pair, blank where the URL has none', () => {
      draft.patch({ scene: 'old scene', prompt: 'old prompt', enhancedPrompt: 'old enhanced', enhancedScene: 'old enhanced scene' });
      draft.loadFromUrl({ workflow: 'extend', source_video: SOURCE, scene: 'A street.' }, extendContext());
      expect(draft.state).toMatchObject({ scene: 'A street.', prompt: '', enhancedPrompt: '', enhancedScene: '', enhancedFrom: null });
    });

    it('fills the Enhanced pair of a reused extension, current for its source and length', () => {
      draft.loadFromUrl({
        workflow: 'extend', source_video: SOURCE, extend_frames: '48', scene: 'A street.', prompt: 'She waves.', enhanced_scene: 'A wet street.', enhanced_prompt: 'She waves slowly.',
      }, extendContext());
      expect(draft.state).toMatchObject({ enhancedScene: 'A wet street.', enhancedPrompt: 'She waves slowly.' });
      expect(draft.state.enhancedFrom).toEqual({ prompt: 'She waves.', scene: 'A street.', mode: 'video', sourceVideoPath: SOURCE, extendFrames: 48 });
    });

    it('leaves the extend fields alone for other workflows', () => {
      draft.loadFromUrl({ workflow: 'txt2vid', source_video: SOURCE, extend_frames: '48', prompt: 'p' }, extendContext());
      expect(draft.state).toMatchObject({ workflow: 'txt2vid', extendSourcePath: null });
    });

    it('ignores an extension URL where the contract does not offer extend', () => {
      draft.update('workflow', 'txt2vid');
      draft.loadFromUrl({ workflow: 'extend', source_video: SOURCE }, makeContext());
      expect(draft.state.workflow).toBe('txt2vid');
      expect(draft.state.extendSourcePath).toBeNull();
    });
  });

  describe('applyExtendSource', () => {
    beforeEach(() => {
      draft.setExtendSourcePath(SOURCE);
    });

    it('fills Scene from a recorded scene and leaves Prompt empty', () => {
      draft.applyExtendSource(sceneSource());
      expect(draft.state).toMatchObject({ scene: 'A woman on a rainy street.', prompt: '' });
      expect(draft.extendSource?.path).toBe(SOURCE);
    });

    it('leaves both empty when no prompt is recorded', () => {
      draft.patch({ scene: 'something', prompt: 'else' });
      draft.applyExtendSource(sourceInfo(), null, { replaceText: true });
      expect(draft.state).toMatchObject({ scene: '', prompt: '' });
    });

    it('takes the source frame rate and the default length, and clears the "moved" notice', () => {
      draft.patch({ missingSourceVideo: true, fps: 24 });
      draft.applyExtendSource(sourceInfo({ fps: 30000 / 1001, extend_frames: 56 }));
      expect(draft.state).toMatchObject({ fps: 30000 / 1001, extendFrames: 56, missingSourceVideo: false });
    });

    it('keeps a length already chosen', () => {
      draft.patch({ extendFrames: 96 });
      draft.applyExtendSource(sourceInfo({ extend_frames: 48 }));
      expect(draft.state.extendFrames).toBe(96);
    });

    it('takes the model and LoRAs of the source and loads the model defaults', () => {
      const ctx = extendContext();
      draft.hydrateFromContext(ctx);
      draft.applyExtendSource(sourceInfo({ model: 'ltx-2', lora: 'style:0.7' }), ctx);
      expect(draft.state).toMatchObject({ model: 'ltx-2', loraString: 'style:0.7', steps: 12 });
    });

    it('clears the LoRAs when the source recorded settings without any', () => {
      draft.patch({ model: 'ltx-v-0.9', loraString: 'mine:1' });
      draft.applyExtendSource(sourceInfo({ model: 'ltx-v-0.9', lora: '' }));
      expect(draft.state.loraString).toBe('');
    });

    it('keeps the model and LoRAs when the source offers none', () => {
      draft.patch({ model: 'ltx-v-0.9', loraString: 'mine:1' });
      draft.applyExtendSource(sourceInfo());
      expect(draft.state).toMatchObject({ model: 'ltx-v-0.9', loraString: 'mine:1' });
    });

    it('clears the Enhanced pair when the boxes are filled', () => {
      draft.patch({ enhancedPrompt: 'x', enhancedScene: 'y', enhancedFrom: { prompt: '', mode: 'video' } });
      draft.applyExtendSource(sceneSource());
      expect(draft.state).toMatchObject({ enhancedPrompt: '', enhancedScene: '', enhancedFrom: null });
    });

    it('keeps a box the user edited when a different source is chosen', () => {
      draft.applyExtendSource(legacySource());
      draft.patch({ prompt: 'She turns and leaves.' });
      draft.setExtendSourcePath('/clips/other.mp4');
      draft.applyExtendSource(sceneSource({ path: '/clips/other.mp4' }));
      expect(draft.state).toMatchObject({ scene: 'A woman on a rainy street.', prompt: 'She turns and leaves.' });
    });

  });

  describe('source path', () => {
    it('forgets the previous source facts and the moved notice when another is chosen', () => {
      draft.setExtendSourcePath(SOURCE);
      draft.applyExtendSource(sourceInfo());
      draft.patch({ missingSourceVideo: true });
      draft.setExtendSourcePath('/clips/other.mp4');
      expect(draft.extendSource).toBeNull();
      expect(draft.state).toMatchObject({ extendSourcePath: '/clips/other.mp4', missingSourceVideo: false });
    });

    it('keeps the facts when the same path is set again', () => {
      draft.setExtendSourcePath(SOURCE);
      draft.applyExtendSource(sourceInfo());
      draft.setExtendSourcePath(SOURCE);
      expect(draft.extendSource).not.toBeNull();
    });

  });

  describe('hydrateFromContext', () => {
    it('keeps the new frames as a multiple of 8, not 8k+1', () => {
      const ctx = extendContext({ video_defaults: makeVideoDefaults({ frame_alignment: 8 }), video_model_defaults: { 'ltx-v-0.9': makeVideoDefaults({ frame_alignment: 8 }) } });
      draft.patch({ extendFrames: 48 });
      draft.hydrateFromContext(ctx, null, { keepSettings: true });
      expect(draft.state.extendFrames).toBe(48);
    });

    it('runs at the source frame rate instead of the model frame rate, and other workflows at the model frame rate', () => {
      const ctx = extendContext();
      draft.setExtendSourcePath(SOURCE);
      draft.applyExtendSource(sourceInfo({ fps: 30 }));
      draft.hydrateFromContext(ctx, null, { keepSettings: true });
      expect(draft.state.fps).toBe(30);
      draft.onWorkflowChange('txt2vid', ctx);
      expect(draft.state.fps).toBe(24);
    });

  });

  describe('resetSelections', () => {
    it('keeps the source and text and returns the length to two seconds at the source frame rate', () => {
      const ctx = extendContext();
      draft.setExtendSourcePath(SOURCE);
      draft.applyExtendSource(sourceInfo({ fps: 30 }));
      draft.patch({ extendFrames: 96, scene: 'Kept.' });
      draft.resetSelections(ctx);
      expect(draft.state).toMatchObject({ extendSourcePath: SOURCE, extendFrames: 64, scene: 'Kept.', fps: 30 });
    });
  });
});

describe('LoRAs remembered per model family', () => {
  const loras = [{ name: 'zi-style', path: '/loras/zi-style.safetensors' }, { name: 'flux-style', path: '/loras/flux-style.safetensors' }];
  const familyContext = (): WorkspaceContext => {
    const ctx = makeContext({
      image_models: [
        { id: 'zit', label: 'Z-Image Turbo', type: 'image' },
        { id: 'zib', label: 'Z-Image Base', type: 'image' },
        { id: 'flux-dev', label: 'FLUX Dev', type: 'image' },
        { id: 'mystery', label: 'Mystery', type: 'image' },
      ],
      image_model_defaults: {
        zit: makeImageDefaults({ family: 'zimage' }),
        zib: makeImageDefaults({ family: 'zimage' }),
        'flux-dev': makeImageDefaults({ family: 'flux1' }),
        mystery: makeImageDefaults({ family: null }),
      },
      video_model_defaults: { 'ltx-v-0.9': makeVideoDefaults({ family: 'ltx' }) },
      current_image_model: 'zit',
      loras,
    });
    ctx.workflow_contract.definitions.txt2img.visible_controls.push('loras');
    ctx.workflow_contract.definitions.txt2vid.visible_controls.push('loras');
    return ctx;
  };

  beforeEach(() => {
    localStorage.clear();
    draft.reset();
  });

  it('restores the LoRAs last chosen for a family when switching back to it', () => {
    const ctx = familyContext();
    draft.hydrateFromContext(ctx, 'zit');
    draft.update('loraString', 'zi-style:0.8');
    draft.hydrateFromContext(ctx, 'flux-dev');
    expect(draft.state.loraString).toBe('');
    draft.update('loraString', 'flux-style:1');
    draft.hydrateFromContext(ctx, 'zit');
    expect(draft.state.loraString).toBe('zi-style:0.8');
    draft.hydrateFromContext(ctx, 'flux-dev');
    expect(draft.state.loraString).toBe('flux-style:1');
  });

  it('keeps the LoRAs when switching between models of the same family', () => {
    const ctx = familyContext();
    draft.hydrateFromContext(ctx, 'zit');
    draft.update('loraString', 'zi-style:0.8');
    draft.hydrateFromContext(ctx, 'zib');
    expect(draft.state.loraString).toBe('zi-style:0.8');
  });

  it('treats a model of unknown family as its own family', () => {
    const ctx = familyContext();
    draft.hydrateFromContext(ctx, 'zit');
    draft.update('loraString', 'zi-style:0.8');
    draft.hydrateFromContext(ctx, 'mystery');
    expect(draft.state.loraString).toBe('');
  });

  it('remembers image LoRAs across a switch to a video workflow and back', () => {
    const ctx = familyContext();
    draft.hydrateFromContext(ctx, 'zit');
    draft.update('loraString', 'zi-style:0.8');
    draft.onWorkflowChange('txt2vid', ctx);
    expect(draft.state.loraString).toBe('');
    draft.onWorkflowChange('txt2img', ctx);
    expect(draft.state).toMatchObject({ model: 'zit', loraString: 'zi-style:0.8' });
  });

  it('drops remembered LoRAs that are no longer installed', () => {
    const ctx = familyContext();
    draft.hydrateFromContext(ctx, 'zit');
    draft.update('loraString', 'zi-style:0.8,gone:1');
    draft.hydrateFromContext(ctx, 'flux-dev');
    draft.hydrateFromContext(ctx, 'zit');
    expect(draft.state.loraString).toBe('zi-style:0.8');
  });

  it('keeps the remembered LoRAs through Reset', () => {
    const ctx = familyContext();
    draft.hydrateFromContext(ctx, 'zit');
    draft.update('loraString', 'zi-style:0.8');
    draft.hydrateFromContext(ctx, 'flux-dev');
    draft.resetSelections(ctx);
    draft.hydrateFromContext(ctx, 'zit');
    expect(draft.state.loraString).toBe('zi-style:0.8');
  });

  describe('Reuse', () => {
    const reuse = (ctx: WorkspaceContext, params: Record<string, string>): void => {
      draft.hydrateFromContext(ctx, params.model ?? null);
      draft.loadFromUrl(params, ctx);
    };

    it('clears the chosen LoRAs when the source had none', () => {
      const ctx = familyContext();
      draft.hydrateFromContext(ctx, 'zit');
      draft.update('loraString', 'zi-style:0.8');
      reuse(ctx, { workflow: 'txt2img', model: 'zib', lora: '' });
      expect(draft.state.loraString).toBe('');
    });

    it('takes the source LoRAs over the remembered ones and remembers them for its family', () => {
      const ctx = familyContext();
      draft.hydrateFromContext(ctx, 'zit');
      draft.update('loraString', 'zi-style:0.8');
      draft.hydrateFromContext(ctx, 'flux-dev');
      reuse(ctx, { workflow: 'txt2img', model: 'zit', lora: '/loras/zi-style.safetensors:0.3' });
      expect(draft.state.loraString).toBe('/loras/zi-style.safetensors:0.3');
      draft.hydrateFromContext(ctx, 'flux-dev');
      draft.hydrateFromContext(ctx, 'zit');
      expect(draft.state.loraString).toBe('zi-style:0.3');
    });

    it('remembers the current LoRAs when a URL changes the model without re-hydrating', () => {
      const ctx = familyContext();
      draft.hydrateFromContext(ctx, 'zit');
      draft.update('loraString', 'zi-style:0.8');
      draft.loadFromUrl({ model: 'flux-dev', lora: 'flux-style:1' }, ctx);
      expect(draft.state.loraString).toBe('flux-style:1');
      draft.hydrateFromContext(ctx, 'zit');
      expect(draft.state.loraString).toBe('zi-style:0.8');
    });

    it('leaves the LoRAs alone when the URL names none, as a reference image does', () => {
      const ctx = familyContext();
      draft.hydrateFromContext(ctx, 'zit');
      draft.update('loraString', 'zi-style:0.8');
      draft.loadFromUrl({ workflow: 'txt2img', prompt: 'a cat' }, ctx);
      expect(draft.state.loraString).toBe('zi-style:0.8');
    });
  });
});
