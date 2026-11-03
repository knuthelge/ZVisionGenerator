import { describe, it, expect, beforeEach } from 'vitest';
import { legacySource, sceneSource, sourceInfo } from './extend.fixtures';
import { anyEnhancedUsed, isEnhancedStale, rememberedParts } from './promptEnhance';
import { presetDiffers } from './presets';
import { downloadStore } from './downloads.svelte';
import { changedSettings, draft, extendStepFor, fitDimensions, offeredSizes, settingDefaultsFor, snapDimension, stepOptionsFor, videoTimingFor } from './draft.svelte';
import { heldUploads } from './heldUploads.svelte';
import { voiceConsent } from './voiceConsent.svelte';
import { memoryUploadStore } from './uploadStore';
import { navTab, videoKindOf } from './workspaceTab';
import type { DraftState, WorkspaceContext, ImageModelDefaults, SettingsPreset, VideoModelDefaults, Workflow } from '$lib/types';

// ── Helpers ──────────────────────────────────────────────────────────────────

/** Open the tab a workflow belongs to the way the top bar and the Workspace do: pick it, then re-hydrate for it. */
function openTab(workflow: Workflow, ctx: WorkspaceContext): void {
  const kind = videoKindOf(workflow);
  if (kind === null) draft.setTab(navTab(workflow));
  else {
    draft.setTab('video');
    draft.setVideoKind(kind, ctx);
  }
  draft.onTabChange(ctx);
}

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
    max_reference_images: 0,
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
    supports_i2v: true,
    supports_quantize: false,
    quantize: null,
    max_steps: 8,
    fps: 24,
    frame_rule: { alignment: 1, offset: 1, round: 'nearest' },
    supports_extend: true,
    supports_audio_track_upscale: true,
    extend_frame_step: 8,
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
          visible_controls: ['workflow', 'model', 'prompt_inline', 'reference_image', 'reference_image_path', 'reference_image_clear', 'ratio', 'size', 'custom_dimensions', 'runs', 'steps', 'guidance', 'seed'],
          requires_reference_image: false,
          requires_source_video: false,
          clear_fields: ['frames', 'audio', 'low_memory'],
        },
        img2img: {
          mode: 'image',
          model_kind: 'image',
          visible_controls: ['workflow', 'model', 'prompt_inline', 'negative_prompt', 'reference_image', 'reference_image_path', 'reference_image_clear', 'ratio', 'size', 'custom_dimensions', 'runs', 'steps', 'guidance', 'image_strength', 'seed'],
          requires_reference_image: true,
          requires_source_video: false,
          clear_fields: ['frames', 'audio', 'low_memory'],
        },
        txt2vid: {
          mode: 'video',
          model_kind: 'video',
          visible_controls: ['workflow', 'model', 'prompt_inline', 'prompt_scene', 'reference_image', 'reference_image_path', 'reference_image_clear', 'end_image', 'end_image_path', 'end_image_clear', 'ratio', 'size', 'custom_dimensions', 'runs', 'frame_count', 'steps', 'seed', 'audio', 'low_memory'],
          requires_reference_image: false,
          requires_source_video: false,
          clear_fields: ['negative_prompt', 'guidance', 'quantize'],
        },
        img2vid: {
          mode: 'video',
          model_kind: 'video',
          visible_controls: ['workflow', 'model', 'prompt_inline', 'prompt_scene', 'reference_image', 'reference_image_path', 'reference_image_clear', 'end_image', 'end_image_path', 'end_image_clear', 'ratio', 'size', 'custom_dimensions', 'runs', 'frame_count', 'steps', 'seed', 'audio', 'low_memory'],
          requires_reference_image: true,
          requires_source_video: false,
          clear_fields: ['negative_prompt', 'guidance', 'quantize'],
        },
        extend: {
          mode: 'video',
          model_kind: 'video',
          visible_controls: ['workflow', 'model', 'loras', 'prompt_source', 'prompt_scene', 'prompt_inline', 'source_video', 'source_video_path', 'source_video_clear', 'extend_length', 'runs', 'steps', 'seed', 'audio', 'low_memory'],
          requires_reference_image: false,
          requires_source_video: true,
          clear_fields: ['negative_prompt', 'guidance', 'quantize', 'ratio', 'size', 'width', 'height', 'frames', 'video_upscale_enabled', 'video_upscale_factor'],
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
    draft.loadFromUrl({ workflow: 'img2img', image_path: '/in/a.png', prompt: 'from url', model: 'flux-dev' }, makeContext());
    expect(draft.state.workflow).toBe('img2img');
    expect(draft.state.prompt).toBe('from url');
  });

  it('loadFromUrl derives the workflow from the images: an image workflow without a start image runs from text', () => {
    draft.loadFromUrl({ workflow: 'img2img', prompt: 'from url' }, makeContext());
    expect(draft.state.workflow).toBe('txt2img');
    draft.loadFromUrl({ workflow: 'img2vid' }, makeContext());
    expect(draft.state.workflow).toBe('txt2vid');
    draft.loadFromUrl({ workflow: 'txt2vid', image_path: '/in/first.png' }, makeContext());
    expect(draft.state.workflow).toBe('img2vid');
    draft.loadFromUrl({ workflow: 'txt2img', image_path: '/in/start.png' }, makeContext());
    expect(draft.state.workflow).toBe('img2img');
    // A full prefill without images empties the slots, so the workflow follows.
    draft.loadFromUrl({ workflow: 'img2img', prompt: 'p' }, makeContext(), { replaceSlots: true });
    expect(draft.state.workflow).toBe('txt2img');
  });

  it('loadFromUrl without a context sets the image but derives nothing', () => {
    draft.loadFromUrl({ image_path: '/in/a.png' });
    expect(draft.state).toMatchObject({ workflow: 'txt2img', referenceImagePath: '/in/a.png' });
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

  describe('loadFromUrl – reused quantization', () => {
    function quantizeContext(): WorkspaceContext {
      const ctx = makeContext();
      ctx.workflow_contract.definitions.txt2img.visible_controls.push('quantize');
      return ctx;
    }

    it('applies a level the model offers', () => {
      draft.loadFromUrl({ workflow: 'txt2img', model: 'flux-dev', quantize: '8' }, quantizeContext());
      expect(draft.state.quantize).toBe(8);
    });

    it('clears the level when the URL names it blank', () => {
      draft.update('quantize', 4);
      draft.loadFromUrl({ workflow: 'txt2img', model: 'flux-dev', quantize: '' }, quantizeContext());
      expect(draft.state.quantize).toBeNull();
    });

    it('drops a level the context does not offer', () => {
      draft.update('quantize', 4);
      draft.loadFromUrl({ workflow: 'txt2img', model: 'flux-dev', quantize: '6' }, quantizeContext());
      expect(draft.state.quantize).toBeNull();
    });

    it('drops the level for a model without quantization support', () => {
      const ctx = quantizeContext();
      ctx.image_model_defaults['flux-dev'] = { ...ctx.image_model_defaults['flux-dev'], supports_quantize: false };
      draft.loadFromUrl({ workflow: 'txt2img', model: 'flux-dev', quantize: '8' }, ctx);
      expect(draft.state.quantize).toBeNull();
    });

    it('keeps the current level when the URL does not name one', () => {
      draft.update('quantize', 4);
      draft.loadFromUrl({ workflow: 'txt2img', model: 'flux-dev' }, quantizeContext());
      expect(draft.state.quantize).toBe(4);
    });
  });

  describe('loadFromUrl – reused finishing settings', () => {
    /** The params a Reuse of an image asset sends (the server's `reuse_params`). */
    /** A context whose image workflows show every finishing control, and whose video ones clear Transparent. */
    const finishingContext = (): WorkspaceContext => {
      const ctx = makeContext();
      for (const id of ['txt2img', 'img2img'] as const) {
        ctx.workflow_contract.definitions[id].visible_controls.push(
          'postprocess_sharpen', 'postprocess_contrast', 'postprocess_saturation', 'postprocess_transparent',
          'image_upscale_enabled', 'image_upscale_factor', 'image_upscale_denoise', 'image_upscale_steps', 'image_upscale_guidance', 'image_upscale_sharpen',
        );
      }
      ctx.workflow_contract.definitions.img2img.visible_controls.push('image_strength');
      for (const id of ['txt2vid', 'img2vid'] as const) ctx.workflow_contract.definitions[id].clear_fields.push('transparent');
      return ctx;
    };
    const REUSE = {
      workflow: 'img2img', image_path: '/in/a.png', transparent: 'true', sharpen_enabled: 'true', sharpen_amount: '0.6', contrast_enabled: 'true', contrast_amount: '1.1',
      saturation_enabled: 'true', saturation_amount: '1.3', upscale: '4', upscale_denoise: '0.3', upscale_steps: '12', upscale_guidance: '1.5', upscale_sharpen: 'false', image_strength: '0.7',
    };

    it('sets sharpen, contrast, saturation, upscale, strength and Transparent', () => {
      draft.loadFromUrl(REUSE, finishingContext());

      expect(draft.state).toMatchObject({
        workflow: 'img2img', postprocessTransparent: true, postprocessSharpenEnabled: true, postprocessSharpenAmount: 0.6, postprocessContrastEnabled: true, postprocessContrastAmount: 1.1,
        postprocessSaturationEnabled: true, postprocessSaturationAmount: 1.3, upscaleEnabled: true, upscaleFactor: 4, upscaleDenoise: 0.3, upscaleSteps: 12, upscaleGuidance: 1.5,
        upscaleSharpen: false, referenceImageStrength: 0.7,
      });
    });

    it('clears an "on" Transparent when the asset was opaque', () => {
      draft.update('postprocessTransparent', true);
      draft.loadFromUrl({ workflow: 'txt2img', transparent: 'false', sharpen_enabled: 'true', contrast_enabled: 'false', saturation_enabled: 'false' }, finishingContext());

      expect(draft.state.postprocessTransparent).toBe(false);
    });

    it('turns Upscale off when the asset was made without one', () => {
      draft.update('upscaleEnabled', true);
      draft.loadFromUrl({ workflow: 'txt2img', transparent: 'false', sharpen_enabled: 'true', contrast_enabled: 'false', saturation_enabled: 'false' }, finishingContext());

      expect(draft.state.upscaleEnabled).toBe(false);
    });

    it('sets only Transparent for a viewer-upscaled asset, leaving post-processing and Upscale as they were', () => {
      draft.patch({ postprocessSharpenEnabled: false, postprocessContrastEnabled: true, postprocessContrastAmount: 1.4, upscaleEnabled: true, upscaleFactor: 4, referenceImageStrength: 0.3 });
      draft.loadFromUrl({ workflow: 'txt2img', transparent: 'true' }, finishingContext());

      expect(draft.state).toMatchObject({
        postprocessTransparent: true, postprocessSharpenEnabled: false, postprocessContrastEnabled: true, postprocessContrastAmount: 1.4, upscaleEnabled: true, upscaleFactor: 4, referenceImageStrength: 0.3,
      });
    });

    it('leaves Transparent alone when the URL does not name it', () => {
      draft.update('postprocessTransparent', true);
      draft.loadFromUrl({ workflow: 'txt2img', prompt: 'from a link' }, finishingContext());

      expect(draft.state.postprocessTransparent).toBe(true);
    });

    it('skips the settings of controls the workflow does not show', () => {
      draft.loadFromUrl({ ...REUSE, workflow: 'txt2img' }, finishingContext());
      expect(draft.state.referenceImageStrength).not.toBe(0.7);

      const ctx = finishingContext();
      ctx.workflow_contract.definitions.txt2img.visible_controls = ctx.workflow_contract.definitions.txt2img.visible_controls.filter((id) => id !== 'postprocess_transparent');
      draft.update('postprocessTransparent', false);
      draft.loadFromUrl({ ...REUSE, workflow: 'txt2img' }, ctx);
      expect(draft.state.postprocessTransparent).toBe(false);
    });

    it('does not set finishing settings for a video workflow', () => {
      draft.loadFromUrl({ workflow: 'txt2vid', transparent: 'true', sharpen_enabled: 'false' }, finishingContext());

      expect(draft.state.postprocessTransparent).toBe(false);
      expect(draft.state.postprocessSharpenEnabled).toBe(true);
    });

    it('clears Transparent when a video workflow is opened', () => {
      draft.update('postprocessTransparent', true);
      draft.update('workflow', 'txt2vid');
      draft.hydrateFromContext(finishingContext());

      expect(draft.state.postprocessTransparent).toBe(false);
    });
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
        video_model_defaults: { 'ltx-v-0.9': makeVideoDefaults({ frame_rule: { alignment: 8, offset: 1, round: 'nearest' } }) },
      });

      draft.update('workflow', 'txt2vid');
      draft.hydrateFromContext(ctx, null);
      draft.loadFromUrl({ frames: '50' }, ctx);

      expect(draft.state.frameCount).toBe(49);
    });

    it('aligns a reused frame count with the default video model when the named one has no defaults', () => {
      const aligned = makeVideoDefaults({ frame_rule: { alignment: 8, offset: 1, round: 'nearest' } });
      const ctx = makeContext({ video_defaults: aligned, video_model_defaults: { 'ltx-v-0.9': aligned } });

      draft.update('workflow', 'txt2vid');
      draft.hydrateFromContext(ctx, null);
      draft.loadFromUrl({ model: 'stale-ltx', frames: '50' }, ctx);

      expect(draft.state.frameCount).toBe(49);
    });

    it('aligns a kept frame count that predates snapping', () => {
      const ctx = makeContext({
        video_model_defaults: { 'ltx-v-0.9': makeVideoDefaults({ frame_rule: { alignment: 8, offset: 1, round: 'nearest' } }) },
      });

      draft.update('workflow', 'txt2vid');
      draft.hydrateFromContext(ctx, null);
      draft.update('frameCount', 100);
      draft.hydrateFromContext(ctx, null, { keepSettings: true });

      expect(draft.state.frameCount).toBe(97);
    });

    it('aligns an off-grid configured default so Reset matches the hydrated count', () => {
      const ctx = makeContext({
        video_model_defaults: { 'ltx-v-0.9': makeVideoDefaults({ frame_count: 50, frame_rule: { alignment: 8, offset: 1, round: 'nearest' } }) },
      });

      draft.update('workflow', 'txt2vid');
      draft.hydrateFromContext(ctx, null);

      expect(draft.state.frameCount).toBe(49);
      expect(settingDefaultsFor(ctx, { workflow: 'txt2vid', model: 'ltx-v-0.9' }).frameCount).toBe(49);
    });

    it('takes fps and alignment together from the default video model for a model without defaults', () => {
      const ctx = makeContext({ video_defaults: makeVideoDefaults({ fps: 30, frame_rule: { alignment: 8, offset: 1, round: 'nearest' } }) });

      expect(videoTimingFor(ctx, 'unknown-video')).toEqual({ fps: 30, frameRule: { alignment: 8, offset: 1, round: 'nearest' } });
      expect(videoTimingFor(ctx, 'ltx-v-0.9')).toEqual({ fps: 24, frameRule: { alignment: 1, offset: 1, round: 'nearest' } });
      // A source's frame rate replaces the model's, and the alignment stays the model's.
      expect(videoTimingFor(ctx, 'ltx-v-0.9', 30)).toEqual({ fps: 30, frameRule: { alignment: 1, offset: 1, round: 'nearest' } });
      expect(videoTimingFor(ctx, 'ltx-v-0.9', null)).toEqual({ fps: 24, frameRule: { alignment: 1, offset: 1, round: 'nearest' } });
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
      const mossDefaults = makeImageDefaults({ steps: 4, guidance: 1, supports_scheduler: false });
      const ctx = makeContext({
        image_models: [{ id: 'moss', label: 'MOSS-Image Turbo', type: 'image' }],
        current_image_model: 'moss',
        defaults: mossDefaults,
        image_model_defaults: { moss: mossDefaults },
      });

      draft.update('workflow', 'txt2img');
      draft.hydrateFromContext(ctx, 'moss');
      draft.update('scheduler', 'beta');

      draft.hydrateFromContext(ctx, 'moss', { keepSettings: true });

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

  describe('setTab and onTabChange', () => {
    it('switches to video model and defaults when changing to the Video tab', () => {
      const ctx = makeContext();
      draft.update('workflow', 'txt2img');
      draft.hydrateFromContext(ctx, null);
      // Simulate workflow switch
      openTab('txt2vid', ctx);

      expect(draft.state.workflow).toBe('txt2vid');
      expect(draft.state.model).toBe('ltx-v-0.9');
      expect(draft.state.ratio).toBe('16:9');
    });

    it('switches back to image model when changing from the Video tab to the Image tab', () => {
      const ctx = makeContext();
      draft.update('workflow', 'txt2vid');
      draft.hydrateFromContext(ctx, null);
      openTab('txt2img', ctx);

      expect(draft.state.workflow).toBe('txt2img');
      expect(draft.state.model).toBe('flux-dev');
      expect(draft.state.ratio).toBe('2:3');
    });
  });
});

describe('draft store – saved drafts', () => {
  beforeEach(() => {
    localStorage.clear();
    draft.reset();
  });

  function loadSaved(saved: Record<string, unknown>): void {
    localStorage.setItem('ziv-workspace-draft-v1', JSON.stringify({ prompt: 'kept prompt', ...saved }));
    draft.loadDraft();
  }

  it('starts from the defaults when the saved draft is of another schema version', () => {
    loadSaved({ version: 2, postprocessSharpenAmount: 0.6 });
    expect(draft.state).toMatchObject({ prompt: '', postprocessSharpenAmount: null });
    loadSaved({ version: 3 });
    expect(draft.state.prompt).toBe('kept prompt');
  });
});

describe('draft store – prompt enhancer fields', () => {
  beforeEach(() => {
    localStorage.clear();
    draft.reset();
  });

  it('persists enhancer fields', () => {
    draft.update('enhancedPrompt', 'A fox in snow.');
    draft.update('enhancedFrom', { prompt: 'a fox', scene: '', mode: 'image' });
    draft.update('enhanceAuto', true);
    draft.update('enhanceSettings', { style: 'photo', mood: 'keep', theme: 'keep', shot: 'keep', angle: 'keep', palette: 'keep', colors: [], camera: 'keep', pace: 'keep', details: [], length: 'longer', motion: [], sound: [], note: '' });
    draft.loadDraft();
    expect(draft.state.enhancedPrompt).toBe('A fox in snow.');
    expect(draft.state.enhancedFrom).toEqual({ prompt: 'a fox', scene: '', mode: 'image' });
    expect(draft.state.enhanceAuto).toBe(true);
    expect(draft.state.enhanceSettings?.style).toBe('photo');
  });

  it('clears the Enhanced box when a prompt is reused from the URL', () => {
    draft.update('enhancedPrompt', 'Old enhanced text');
    draft.update('enhancedFrom', { prompt: 'old', scene: '', mode: 'image' });
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

    it('restores how the reused rewrite was made: the options, the enhancer and the record kept with it', () => {
      const enhance = JSON.stringify({ settings: { style: 'cinematic', camera: 'orbit', parts: 'action' }, model: 'org/smart', tier: 'smart', format: 'text' });
      draft.update('enhanceSettings', { style: 'photo', mood: 'keep', theme: 'keep', shot: 'keep', angle: 'keep', palette: 'keep', colors: [], camera: 'keep', pace: 'keep', details: [], length: 'same', motion: [], sound: [], note: '' });
      draft.loadFromUrl({ workflow: 'txt2vid', scene: 'A castle.', prompt: 'a knight', enhanced_prompt: 'A knight rides in.', enhance }, makeContext());
      expect(draft.state.enhanceSettingsVideo).toMatchObject({ style: 'cinematic', camera: 'orbit', partsByWorkflow: { txt2vid: 'action' } });
      expect(draft.state.enhanceSettings?.style).toBe('photo');
      expect(draft.state.enhanceModel).toBe('smart');
      expect(draft.state.enhancedFrom?.record).toEqual({ settings: { style: 'cinematic', camera: 'orbit', parts: 'action' }, model: 'org/smart', format: 'text' });
    });

    it('ignores an enhance record without Enhanced text, and a malformed one', () => {
      draft.loadFromUrl({ workflow: 'txt2vid', scene: 'A castle.', prompt: 'a knight', enhance: '{"settings":{"style":"cinematic"},"model":"org/fast","tier":"fast"}' }, makeContext());
      expect(draft.state.enhanceSettingsVideo).toBeNull();
      draft.loadFromUrl({ workflow: 'txt2vid', scene: 'A castle.', prompt: 'a knight', enhanced_prompt: 'A knight rides in.', enhance: '{broken' }, makeContext());
      expect(draft.state.enhanceSettingsVideo).toBeNull();
      expect(draft.state.enhancedFrom?.record).toBeUndefined();
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

  describe('Use as with the Image tab prompt', () => {
    const image = { prompt: 'my image prompt', enhancedPrompt: 'My enhanced image prompt.', enhancedFrom: { prompt: 'my image prompt', scene: '', mode: 'image' as const } };

    it('first frame from Image sets Scene, leaves Action empty and keeps the Image prompt', () => {
      const ctx = makeContext();
      draft.hydrateFromContext(ctx, null);
      draft.patch(image);
      draft.loadFromUrl({ workflow: 'img2vid', image_path: '/in/a.png', use_as: 'start', scene: 'a red fox' }, ctx);
      expect(draft.state).toMatchObject({ workflow: 'img2vid', scene: 'a red fox', prompt: '', enhancedScene: '', enhancedPrompt: '', enhancedFrom: null, firstFramePath: '/in/a.png' });
      openTab('txt2img', ctx);
      expect(draft.state).toMatchObject({ ...image, scene: '' });
      expect(isEnhancedStale(draft.state)).toBe(false);
    });

    it('first frame replaces an earlier scene, Action and Enhanced text of Video', () => {
      const ctx = makeContext();
      draft.hydrateFromContext(ctx, null);
      openTab('txt2vid', ctx);
      draft.patch({ scene: 'old scene', prompt: 'old action', enhancedScene: 'Old enhanced scene.' });
      draft.loadFromUrl({ workflow: 'img2vid', image_path: '/in/a.png', use_as: 'start', scene: 'a red fox' }, ctx);
      expect(draft.state).toMatchObject({ scene: 'a red fox', prompt: '', enhancedScene: '' });
    });

    it('first frame of an asset without a prompt changes no text', () => {
      const ctx = makeContext();
      draft.hydrateFromContext(ctx, null);
      draft.patch(image);
      openTab('txt2vid', ctx);
      draft.patch({ scene: 'kept scene', prompt: 'kept action' });
      openTab('txt2img', ctx);
      draft.loadFromUrl({ workflow: 'img2vid', image_path: '/in/a.png', use_as: 'start' }, ctx);
      expect(draft.state).toMatchObject({ scene: 'kept scene', prompt: 'kept action' });
      openTab('txt2img', ctx);
      expect(draft.state).toMatchObject(image);
    });

    it('last frame changes no text on either tab', () => {
      const ctx = makeContext();
      draft.hydrateFromContext(ctx, null);
      draft.patch(image);
      openTab('txt2vid', ctx);
      draft.patch({ scene: 'a scene', prompt: 'an action' });
      openTab('txt2img', ctx);
      draft.loadFromUrl({ workflow: 'img2vid', end_image_path: '/in/b.png', use_as: 'last' }, ctx);
      expect(draft.state).toMatchObject({ endImagePath: '/in/b.png', scene: 'a scene', prompt: 'an action' });
      openTab('txt2img', ctx);
      expect(draft.state).toMatchObject(image);
    });

    it('Start image from Video keeps the scene and Action of Video', () => {
      const ctx = makeContext();
      draft.hydrateFromContext(ctx, null);
      draft.patch(image);
      openTab('txt2vid', ctx);
      draft.patch({ scene: 'a castle', prompt: 'a knight rides in', enhancedScene: 'A castle at dusk.' });
      draft.loadFromUrl({ workflow: 'img2img', image_path: '/in/a.png', use_as: 'start', prompt: 'a red fox' }, ctx);
      expect(draft.state).toMatchObject({ workflow: 'img2img', prompt: 'a red fox', enhancedPrompt: '', enhancedFrom: null });
      openTab('txt2vid', ctx);
      expect(draft.state).toMatchObject({ scene: 'a castle', prompt: 'a knight rides in', enhancedScene: 'A castle at dusk.' });
    });

    it('a Use as that carries a prompt of its own for Video still replaces both boxes', () => {
      const ctx = makeContext();
      draft.hydrateFromContext(ctx, null);
      draft.patch(image);
      draft.loadFromUrl({ workflow: 'img2vid', image_path: '/in/a.png', use_as: 'start', scene: 'A castle.', prompt: 'a knight' }, ctx);
      expect(draft.state).toMatchObject({ scene: 'A castle.', prompt: 'a knight', enhancedPrompt: '' });
      openTab('txt2img', ctx);
      expect(draft.state).toMatchObject(image);
    });

    it('Start image in the Image tab still replaces the prompt and clears the Enhanced pair', () => {
      draft.patch(image);
      draft.loadFromUrl({ workflow: 'img2img', image_path: '/in/a.png', use_as: 'start', prompt: 'a red fox' }, makeContext());
      expect(draft.state).toMatchObject({ workflow: 'img2img', prompt: 'a red fox', enhancedPrompt: '', enhancedFrom: null });
    });

    it('a Reuse of a video sets the scene, the prompt and a matching Enhanced pair', () => {
      draft.patch(image);
      draft.loadFromUrl({ workflow: 'txt2vid', scene: 'A castle.', prompt: 'a knight', enhanced_prompt: 'A knight rides in.' }, makeContext());
      expect(draft.state).toMatchObject({ scene: 'A castle.', prompt: 'a knight', enhancedScene: '', enhancedPrompt: 'A knight rides in.', enhancedFrom: { scene: 'A castle.', prompt: 'a knight', mode: 'video' } });
    });

    it('a Reuse of a video with an image start keeps replacing the prompt', () => {
      draft.patch(image);
      draft.loadFromUrl({ workflow: 'img2vid', image_path: '/in/a.png', scene: 'A castle.', prompt: 'a knight' }, makeContext(), { replaceSlots: true });
      expect(draft.state).toMatchObject({ workflow: 'img2vid', scene: 'A castle.', prompt: 'a knight', enhancedPrompt: '', enhancedFrom: null });
    });
  });

  describe('each tab keeps its own text', () => {
    /** A context whose image model takes a negative prompt, so the capability gates leave it alone. */
    const textContext = (): WorkspaceContext => {
      const defaults = makeImageDefaults({ supports_negative_prompt: true });
      return makeContext({ defaults, image_model_defaults: { 'flux-dev': defaults } });
    };
    const imageText = { prompt: 'a fox in snow', negativePrompt: 'blurry', enhancedPrompt: 'A red fox in deep snow.', enhancedFrom: { prompt: 'a fox in snow', scene: '', mode: 'image' as const } };
    const videoText = { scene: 'A castle at dusk.', prompt: 'a knight rides in', enhancedScene: 'A grey castle at dusk.', enhancedPrompt: 'A knight rides in slowly.', enhancedFrom: { prompt: 'a knight rides in', scene: 'A castle at dusk.', mode: 'video' as const } };

    it('each tab keeps its own prompt across tab switches and a reload', () => {
      const ctx = textContext();
      draft.hydrateFromContext(ctx, null);
      draft.patch(imageText);
      openTab('txt2vid', ctx);
      expect(draft.state).toMatchObject({ prompt: '', scene: '', negativePrompt: '', enhancedPrompt: '', enhancedScene: '', enhancedFrom: null, enhancedHistory: [] });
      draft.patch({ ...videoText, enhancedHistory: [{ prompt: 'A knight rides in slowly.', scene: 'A grey castle at dusk.', from: videoText.enhancedFrom }] });

      openTab('txt2img', ctx);
      expect(draft.state).toMatchObject({ ...imageText, scene: '', enhancedHistory: [] });
      expect(isEnhancedStale(draft.state)).toBe(false);

      draft.loadDraft();
      expect(draft.state).toMatchObject(imageText);
      openTab('txt2vid', ctx);
      expect(draft.state).toMatchObject({ ...videoText, negativePrompt: '' });
      expect(draft.state.enhancedHistory).toHaveLength(1);
      expect(isEnhancedStale(draft.state)).toBe(false);
    });

    it('keeps the JSON caption and its toggle with Image', () => {
      draft.patch({ jsonPromptEnabled: true, jsonPrompt: '{"subject":"fox"}' });
      draft.setTab('video');
      expect(draft.state).toMatchObject({ jsonPromptEnabled: false, jsonPrompt: '' });
      draft.setTab('image');
      expect(draft.state).toMatchObject({ jsonPromptEnabled: true, jsonPrompt: '{"subject":"fox"}' });
    });

    it('keeps the prompt source and prompt file shared between the tabs', () => {
      draft.patch({ promptSource: 'file', promptFilePath: '/prompts/set.yaml', promptFileOptionIds: ['one'] });
      draft.setTab('video');
      expect(draft.state).toMatchObject({ promptSource: 'file', promptFilePath: '/prompts/set.yaml', promptFileOptionIds: ['one'] });
    });

    it('an old draft without saved text opens the other tab empty', () => {
      localStorage.setItem('ziv-workspace-draft-v1', JSON.stringify({ version: 3, prompt: 'old prompt', scene: 'old scene' }));
      draft.loadDraft();
      expect(draft.state.savedText).toEqual({});
      draft.setTab('video');
      expect(draft.state).toMatchObject({ prompt: '', scene: '' });
      draft.setTab('image');
      expect(draft.state).toMatchObject({ prompt: 'old prompt', scene: 'old scene' });
    });

    it('Extend shares the text of Video', () => {
      const ctx = textContext();
      ctx.workflow_contract.values.push('extend');
      draft.hydrateFromContext(ctx, null);
      draft.patch(imageText);
      openTab('txt2vid', ctx);
      draft.patch(videoText);
      draft.setVideoKind('extend', ctx);
      expect(draft.state).toMatchObject({ workflow: 'extend', scene: videoText.scene, prompt: videoText.prompt, enhancedScene: videoText.enhancedScene });
      draft.patch({ prompt: 'what happens next' });
      draft.setVideoKind('new', ctx);
      expect(draft.state).toMatchObject({ scene: videoText.scene, prompt: 'what happens next' });
      draft.setVideoKind('extend', ctx);
      draft.setTab('image');
      expect(draft.state).toMatchObject(imageText);
      expect(draft.state.savedText.video).toMatchObject({ scene: videoText.scene, prompt: 'what happens next' });
    });

    it('Reuse fills only the opened tab\'s text', () => {
      const ctx = textContext();
      draft.hydrateFromContext(ctx, null);
      draft.patch(imageText);
      openTab('txt2vid', ctx);
      draft.patch({ scene: 'my scene', prompt: 'my action' });
      // A video reused while Image is open lands on Video and leaves Image alone.
      openTab('txt2img', ctx);
      draft.loadFromUrl({ workflow: 'txt2vid', scene: 'A reused scene.', prompt: 'a reused action' }, ctx, { replaceSlots: true });
      expect(draft.state).toMatchObject({ workflow: 'txt2vid', scene: 'A reused scene.', prompt: 'a reused action' });
      openTab('txt2img', ctx);
      expect(draft.state).toMatchObject(imageText);
      // An image reused while Video is open lands on Image and leaves Video alone.
      openTab('txt2vid', ctx);
      draft.loadFromUrl({ workflow: 'txt2img', prompt: 'a reused image' }, ctx, { replaceSlots: true });
      expect(draft.state).toMatchObject({ workflow: 'txt2img', prompt: 'a reused image' });
      openTab('txt2vid', ctx);
      expect(draft.state).toMatchObject({ scene: 'A reused scene.', prompt: 'a reused action' });
    });

    it('survives a reset of the selections on both tabs', () => {
      const ctx = textContext();
      draft.hydrateFromContext(ctx, null);
      draft.patch(imageText);
      openTab('txt2vid', ctx);
      draft.patch({ scene: 'a scene', prompt: 'an action' });
      draft.resetSelections(ctx);
      expect(draft.state).toMatchObject({ scene: 'a scene', prompt: 'an action' });
      openTab('txt2img', ctx);
      expect(draft.state).toMatchObject(imageText);
    });

    it('the enhancer\'s state stays with its tab', () => {
      const ctx = textContext();
      draft.hydrateFromContext(ctx, null);
      const imageSettings = { style: 'photo', mood: 'keep', theme: 'keep', shot: 'keep', angle: 'keep', palette: 'keep', colors: [], camera: 'keep', pace: 'keep', details: [], length: 'longer', motion: [], sound: [], note: '' };
      draft.patch({ ...imageText, enhanceSettings: imageSettings });
      openTab('txt2vid', ctx);
      expect(draft.state).toMatchObject({ enhancedPrompt: '', enhancedFrom: null, enhanceSettings: imageSettings, enhanceSettingsVideo: null });
      draft.patch({ ...videoText, enhanceSettingsVideo: { ...imageSettings, style: 'dreamy' } });
      openTab('txt2img', ctx);
      expect(draft.state).toMatchObject({ enhancedPrompt: imageText.enhancedPrompt, enhancedScene: '', enhancedFrom: imageText.enhancedFrom, enhanceSettings: imageSettings });
      expect(draft.state.enhanceSettingsVideo?.style).toBe('dreamy');
      expect(isEnhancedStale(draft.state)).toBe(false);
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
    draft.update('enhancedFrom', { prompt: 'a fox', scene: '', mode: 'image' });
    draft.update('enhanceAuto', true);
    draft.update('enhanceSettings', { style: 'photo', mood: 'keep', theme: 'keep', shot: 'keep', angle: 'keep', palette: 'keep', colors: [], camera: 'keep', pace: 'keep', details: [], length: 'longer', motion: [], sound: [], note: '' });
    draft.resetSelections(ctx);
    expect(draft.state.enhancedPrompt).toBe('A fox in snow.');
    expect(draft.state.enhancedFrom).toEqual({ prompt: 'a fox', scene: '', mode: 'image' });
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

  it('keeps settings when a tab opens again for the same model', () => {
    const ctx = makeContext();
    draft.hydrateFromContext(ctx);
    draft.patch({ steps: 3, guidance: 2 });

    openTab('txt2img', ctx);

    expect(draft.state).toMatchObject({ workflow: 'txt2img', steps: 3, guidance: 2 });
  });

  it('setStartImage switches the workflow within its tab and keeps every setting', () => {
    const ctx = makeContext();
    draft.hydrateFromContext(ctx);
    draft.patch({ steps: 3, guidance: 2, referenceImageStrength: 0.9, ratio: '1:1', size: 's' });

    draft.setStartImage('image', { path: '/in/a.png' }, ctx);
    expect(draft.state).toMatchObject({ workflow: 'img2img', referenceImagePath: '/in/a.png', referenceImageUpload: null, model: 'flux-dev', steps: 3, guidance: 2, referenceImageStrength: 0.9, ratio: '1:1', size: 's' });

    draft.setStartImage('image', null, ctx);
    expect(draft.state).toMatchObject({ workflow: 'txt2img', referenceImagePath: null, steps: 3, guidance: 2, referenceImageStrength: 0.9, ratio: '1:1', size: 's' });
  });

  it('setStartImage stores an upload as its file name alone, and holds the file', () => {
    const ctx = makeContext();
    draft.hydrateFromContext(ctx);
    const file = new File(['x'], 'start.png', { type: 'image/png' });

    draft.setStartImage('image', { file }, ctx);

    expect(draft.state).toMatchObject({ workflow: 'img2img', referenceImagePath: null, referenceImageUpload: 'start.png' });
    expect(heldUploads.start('image')).toBe(file);
    draft.setStartImage('image', { path: '/in/b.png' }, ctx);
    expect(draft.state).toMatchObject({ referenceImagePath: '/in/b.png', referenceImageUpload: null });
    expect(heldUploads.start('image')).toBeNull();
  });

  it('setStartImage drops the missing flag of the path it replaces, unless another slot still holds it', () => {
    const ctx = makeContext();
    draft.hydrateFromContext(ctx);
    draft.patch({ referenceImagePath: '/in/gone.png', referenceImagePaths: ['/in/shared.png'], missingReferencePaths: ['/in/gone.png', '/in/shared.png'] });

    draft.setStartImage('image', { path: '/in/new.png' }, ctx);
    expect(draft.state.missingReferencePaths).toEqual(['/in/shared.png']);

    draft.setStartImage('image', { path: '/in/shared.png' }, ctx);
    draft.setStartImage('image', null, ctx);
    expect(draft.state.missingReferencePaths).toEqual(['/in/shared.png']);
  });

  it('setLastFrame stores a path or an upload, never both, and keeps the workflow', () => {
    const ctx = makeContext();
    draft.update('workflow', 'txt2vid');
    draft.hydrateFromContext(ctx);
    const file = new File(['x'], 'end.png', { type: 'image/png' });

    draft.setLastFrame({ path: '/in/end.png' });
    expect(draft.state).toMatchObject({ workflow: 'txt2vid', endImagePath: '/in/end.png', endImageUpload: null });
    draft.setLastFrame({ file });
    expect(draft.state).toMatchObject({ endImagePath: null, endImageUpload: 'end.png' });
    expect(heldUploads.lastFrame).toBe(file);
    draft.setLastFrame(null);
    expect(draft.state).toMatchObject({ endImagePath: null, endImageUpload: null });
    expect(heldUploads.lastFrame).toBeNull();
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
      video_model_defaults: { 'ltx-v-0.9': makeVideoDefaults(), 'ltx-2': makeVideoDefaults({ steps: 12, frame_rule: { alignment: 8, offset: 1, round: 'nearest' } }) },
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
      draft.patch({ enhancedPrompt: 'x', enhancedScene: 'y', enhancedFrom: { prompt: '', scene: '', mode: 'video' } });
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
      const ctx = extendContext({ video_defaults: makeVideoDefaults({ frame_rule: { alignment: 8, offset: 1, round: 'nearest' } }), video_model_defaults: { 'ltx-v-0.9': makeVideoDefaults({ frame_rule: { alignment: 8, offset: 1, round: 'nearest' } }) } });
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
      openTab('txt2vid', ctx);
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

  it('remembers LoRAs per Klein size, so 4B and 9B swap them', () => {
    const ctx = familyContext();
    ctx.image_models.push({ id: 'klein-4b', label: 'Klein 4B', type: 'image' }, { id: 'klein-9b', label: 'Klein 9B', type: 'image' });
    ctx.image_model_defaults = {
      ...ctx.image_model_defaults,
      'klein-4b': makeImageDefaults({ family: 'flux2_klein', size_variant: '4b' }),
      'klein-9b': makeImageDefaults({ family: 'flux2_klein', size_variant: '9b' }),
    };
    draft.hydrateFromContext(ctx, 'klein-4b');
    draft.update('loraString', 'zi-style:0.5');
    draft.hydrateFromContext(ctx, 'klein-9b');
    expect(draft.state.loraString).toBe('');
    draft.update('loraString', 'flux-style:1');
    draft.hydrateFromContext(ctx, 'klein-4b');
    expect(draft.state.loraString).toBe('zi-style:0.5');
    draft.hydrateFromContext(ctx, 'klein-9b');
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
    openTab('txt2vid', ctx);
    expect(draft.state.loraString).toBe('');
    openTab('txt2img', ctx);
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

describe('changedSettings', () => {
  const baseline = settingDefaultsFor(makeContext(), { workflow: 'txt2img', model: 'flux-dev' });

  it('returns only the settings that differ from the baseline', () => {
    expect(changedSettings({ ...baseline, steps: 4, runs: 3 }, baseline)).toEqual({ steps: 4, runs: 3 });
  });

  it('returns nothing when every setting is at the baseline', () => {
    expect(changedSettings({ ...baseline }, baseline)).toEqual({});
  });

  it('keeps the size settings together when any one changed', () => {
    const current = { ...baseline, width: 1000, dimensionMode: 'custom' as const };
    const changed = changedSettings(current, baseline);
    expect(changed).toMatchObject({ ratio: baseline.ratio, size: baseline.size, dimensionMode: 'custom', width: 1000, height: baseline.height });
  });
});

describe('snapDimension', () => {
  const limits = { min: 256, max: 2048, step: 16 };

  it('rounds to the nearest multiple of the step', () => {
    expect(snapDimension(1000, limits)).toBe(1008);
    expect(snapDimension(1001, limits)).toBe(1008);
    expect(snapDimension(1007, limits)).toBe(1008);
  });

  it('clamps to the smallest and largest multiples within the limits', () => {
    expect(snapDimension(100, limits)).toBe(256);
    expect(snapDimension(5000, limits)).toBe(2048);
    expect(snapDimension(10, { min: 20, max: 100, step: 16 })).toBe(32);
    expect(snapDimension(500, { min: 20, max: 100, step: 16 })).toBe(96);
  });

  it('has no upper bound without a maximum', () => {
    expect(snapDimension(4000, { min: 16, max: null, step: 16 })).toBe(4000);
  });
});

describe('fitDimensions', () => {
  const limits = { min: 256, max: 2048, step: 16 };

  it('scales a size above the maximum down keeping its aspect ratio', () => {
    expect(fitDimensions(3072, 2048, limits)).toEqual({ width: 2048, height: 1360 });
  });

  it('scales a size below the minimum up keeping its aspect ratio', () => {
    expect(fitDimensions(128, 200, limits)).toEqual({ width: 256, height: 400 });
  });

  it('only snaps a size already within the limits', () => {
    expect(fitDimensions(1000, 1000, limits)).toEqual({ width: 1008, height: 1008 });
  });

  it('snaps without scaling when the model has no maximum', () => {
    expect(fitDimensions(4000, 3000, { min: 16, max: null, step: 16 })).toEqual({ width: 4000, height: 3008 });
  });
});

describe('hydrateFromContext – keepChangedSettings', () => {
  const imageContext = (overrides: Partial<WorkspaceContext> = {}): WorkspaceContext => makeContext({
    image_models: [{ id: 'flux-dev', label: 'FLUX Dev', type: 'image' }, { id: 'other', label: 'Other', type: 'image' }],
    image_model_defaults: {
      'flux-dev': makeImageDefaults({ steps: 10, guidance: 3.5, family: 'flux1' }),
      other: makeImageDefaults({ steps: 30, guidance: 5, family: 'other' }),
    },
    ...overrides,
  });
  const switchTo = (ctx: WorkspaceContext, model: string): void => draft.hydrateFromContext(ctx, model, { keepChangedSettings: true });

  beforeEach(() => {
    localStorage.clear();
    draft.reset();
  });

  it('keeps a changed setting and gives unchanged ones the new model defaults', () => {
    const ctx = imageContext();
    draft.hydrateFromContext(ctx, 'flux-dev');
    draft.update('steps', 4);

    switchTo(ctx, 'other');

    expect(draft.state).toMatchObject({ model: 'other', steps: 4, guidance: 5 });
  });

  it('treats a setting edited back to the old default as unchanged', () => {
    const ctx = imageContext();
    draft.hydrateFromContext(ctx, 'flux-dev');
    draft.update('steps', 4);
    draft.update('steps', 10);

    switchTo(ctx, 'other');

    expect(draft.state.steps).toBe(30);
  });

  it('keeps the size together when only one part changed', () => {
    const ctx = imageContext({ image_size_options: { '1:1': ['s', 'm', 'l'], '2:3': ['s', 'm', 'l'], '16:9': ['s', 'm', 'l'] } });
    draft.hydrateFromContext(ctx, 'flux-dev');
    draft.patch({ ratio: '16:9' });

    switchTo(ctx, 'other');

    expect(draft.state).toMatchObject({ ratio: '16:9', size: 'm' });
  });

  it('keeps everything for the same model, like keepSettings', () => {
    const ctx = imageContext();
    draft.hydrateFromContext(ctx, 'flux-dev');
    draft.patch({ steps: 4, runs: 7 });

    draft.hydrateFromContext(ctx, 'flux-dev', { keepChangedSettings: true });

    expect(draft.state).toMatchObject({ steps: 4, runs: 7 });
  });

  it('leaves the seed and run count alone when the model changes', () => {
    const ctx = imageContext();
    draft.hydrateFromContext(ctx, 'flux-dev');
    draft.patch({ seed: 1234, runs: 5 });

    switchTo(ctx, 'other');

    expect(draft.state).toMatchObject({ model: 'other', seed: 1234, runs: 5 });
  });

  it('keeps nothing when the old model has no known defaults', () => {
    const ctx = imageContext();
    draft.hydrateFromContext(ctx, 'flux-dev');
    draft.patch({ model: 'removed-model', steps: 4 });

    switchTo(ctx, 'other');

    expect(draft.state.steps).toBe(30);
  });

  it('keeps the prompt source when the new context still offers it', () => {
    const ctx = imageContext();
    draft.hydrateFromContext(ctx, 'flux-dev');
    draft.update('promptSource', 'file');

    switchTo(ctx, 'other');

    expect(draft.state.promptSource).toBe('file');
  });

  it('does not change the settings, model or LoRAs when only the quantization changes', () => {
    const ctx = imageContext();
    draft.hydrateFromContext(ctx, 'flux-dev');
    draft.update('loraString', 'style:1');
    const before = { ...draft.state };

    draft.update('quantize', 8);

    expect(draft.state).toEqual({ ...before, quantize: 8 });
  });

  it('still resets to the new model defaults without the option', () => {
    const ctx = imageContext();
    draft.hydrateFromContext(ctx, 'flux-dev');
    draft.update('steps', 4);

    draft.hydrateFromContext(ctx, 'other');

    expect(draft.state.steps).toBe(30);
  });

  describe('size', () => {
    const sizedContext = (): WorkspaceContext => imageContext({
      image_model_defaults: {
        'flux-dev': makeImageDefaults(),
        other: makeImageDefaults({ dimension_max: 1024 }),
      },
      image_size_options: { '2:3': ['s', 'm', 'l', 'xl'] },
      image_size_dimensions: { '2:3': { s: [512, 768], m: [832, 1216], l: [1024, 1536], xl: [1280, 1920] } },
    });

    it('picks the largest offered preset when the chosen one is not offered', () => {
      const ctx = imageContext({
        image_model_defaults: { 'flux-dev': makeImageDefaults(), other: makeImageDefaults({ dimension_max: 1100 }) },
        image_size_options: { '2:3': ['s', 'm', 'l', 'xl'] },
        image_size_dimensions: { '2:3': { s: [512, 768], m: [832, 1000], l: [1024, 1100], xl: [1280, 1920] } },
      });
      draft.hydrateFromContext(ctx, 'flux-dev');
      draft.patch({ size: 'xl' });

      switchTo(ctx, 'other');

      expect(draft.state).toMatchObject({ ratio: '2:3', size: 'l' });
    });

    it('picks the last listed preset when no pixel sizes are known', () => {
      const ctx = imageContext({ image_size_options: { '2:3': ['s', 'm', 'l'] } });
      ctx.image_model_defaults!.other = makeImageDefaults({ dimension_max: 1024 });
      draft.hydrateFromContext(ctx, 'flux-dev');
      draft.patch({ size: 'xl' });

      switchTo(ctx, 'other');

      expect(draft.state.size).toBe('l');
    });

    it('takes the new model size when no preset of the chosen ratio is offered', () => {
      const ctx = imageContext({
        image_model_defaults: { 'flux-dev': makeImageDefaults(), other: makeImageDefaults({ ratio: '1:1', size: 's', width: 512, height: 512, dimension_max: 100 }) },
        image_size_options: { '1:1': ['s'], '2:3': ['s', 'm'] },
        image_size_dimensions: { '2:3': { s: [512, 768], m: [832, 1216] } },
      });
      draft.hydrateFromContext(ctx, 'flux-dev');
      draft.patch({ size: 'm' });

      switchTo(ctx, 'other');

      expect(draft.state).toMatchObject({ ratio: '1:1', size: 's', dimensionMode: 'ratio', width: 512, height: 512 });
    });

    it('scales a custom size into the new model limits keeping its aspect ratio', () => {
      const ctx = imageContext({
        image_model_defaults: { 'flux-dev': makeImageDefaults(), other: makeIdeogramDefaults() },
      });
      draft.hydrateFromContext(ctx, 'flux-dev');
      draft.patch({ dimensionMode: 'custom', width: 3072, height: 2048 });

      switchTo(ctx, 'other');

      expect(draft.state).toMatchObject({ dimensionMode: 'custom', width: 2048, height: 1360 });
    });

    it('keeps an offered preset', () => {
      const ctx = sizedContext();
      draft.hydrateFromContext(ctx, 'flux-dev');
      draft.patch({ size: 's' });

      switchTo(ctx, 'other');

      expect(draft.state.size).toBe('s');
    });
  });

  describe('capability gates', () => {
    it('clears a negative prompt the new model cannot use', () => {
      const ctx = imageContext({
        image_model_defaults: {
          'flux-dev': makeImageDefaults({ supports_negative_prompt: true }),
          other: makeImageDefaults({ supports_negative_prompt: false }),
        },
      });
      draft.hydrateFromContext(ctx, 'flux-dev');
      draft.update('negativePrompt', 'blurry');

      switchTo(ctx, 'other');

      expect(draft.state.negativePrompt).toBe('');
    });

    it('gives the new model its default scheduler when the old one had none', () => {
      const ctx = imageContext({
        image_model_defaults: {
          'flux-dev': makeImageDefaults({ supports_scheduler: false, scheduler: null }),
          other: makeImageDefaults({ supports_scheduler: true, scheduler: 'euler' }),
        },
      });
      draft.hydrateFromContext(ctx, 'flux-dev');
      expect(draft.state.scheduler).toBeNull();

      switchTo(ctx, 'other');

      expect(draft.state.scheduler).toBe('euler');
    });

    it('keeps a changed scheduler the new model supports', () => {
      const ctx = imageContext({
        image_model_defaults: {
          'flux-dev': makeImageDefaults({ scheduler: 'euler' }),
          other: makeImageDefaults({ scheduler: 'dpm' }),
        },
      });
      draft.hydrateFromContext(ctx, 'flux-dev');
      draft.update('scheduler', 'beta');

      switchTo(ctx, 'other');

      expect(draft.state.scheduler).toBe('beta');
    });
  });

  describe('LoRAs', () => {
    it('follow the model family while settings are kept', () => {
      const loras = [{ name: 'a', path: '/loras/a.safetensors' }, { name: 'b', path: '/loras/b.safetensors' }];
      const ctx = imageContext({ loras });
      draft.hydrateFromContext(ctx, 'flux-dev');
      draft.patch({ loraString: 'a:1', steps: 4 });

      switchTo(ctx, 'other');
      expect(draft.state).toMatchObject({ loraString: '', steps: 4 });

      switchTo(ctx, 'flux-dev');
      expect(draft.state.loraString).toBe('a:1');
    });
  });

  describe('video', () => {
    const videoContext = (): WorkspaceContext => makeContext({
      video_models: [{ id: 'ltx-v-0.9', label: 'LTX Video', type: 'video' }, { id: 'ltx-2', label: 'LTX 2', type: 'video' }],
      video_model_defaults: {
        'ltx-v-0.9': makeVideoDefaults({ steps: 30, max_steps: null }),
        'ltx-2': makeVideoDefaults({ steps: 8, max_steps: 8, frame_rule: { alignment: 8, offset: 1, round: 'nearest' } }),
      },
      current_video_model: 'ltx-v-0.9',
    });

    beforeEach(() => draft.update('workflow', 'txt2vid'));

    it('caps the steps at the new model step limit', () => {
      const ctx = videoContext();
      draft.hydrateFromContext(ctx, 'ltx-v-0.9');
      draft.update('steps', 20);

      switchTo(ctx, 'ltx-2');

      expect(draft.state.steps).toBe(8);
    });

    it('leaves steps below the limit alone', () => {
      const ctx = videoContext();
      draft.hydrateFromContext(ctx, 'ltx-v-0.9');
      draft.update('steps', 5);

      switchTo(ctx, 'ltx-2');

      expect(draft.state.steps).toBe(5);
    });

    it('aligns a kept frame count to the new model grid', () => {
      const ctx = videoContext();
      draft.hydrateFromContext(ctx, 'ltx-v-0.9');
      draft.update('frameCount', 50);

      switchTo(ctx, 'ltx-2');

      expect(draft.state.frameCount).toBe(49);
    });

    describe('a family with its own rules', () => {
      const H3_RULE = { alignment: 17, offset: 5, round: 'up' as const };
      const familyContext = (h3: Partial<VideoModelDefaults> = {}): WorkspaceContext => {
        const ctx = makeContext({
          video_models: [{ id: 'ltx-2', label: 'LTX 2', type: 'video' }, { id: 'h3', label: 'MiniMax H3', type: 'video' }],
          video_model_defaults: {
            'ltx-2': makeVideoDefaults({ steps: 8, max_steps: 8, frame_rule: { alignment: 8, offset: 1, round: 'nearest' } }),
            h3: makeVideoDefaults({ steps: 8, max_steps: 8, frame_count: 124, frame_rule: H3_RULE, step_options: [4, 8], extend_frame_step: 17, hidden_controls: ['low_memory'], ...h3 }),
          },
          current_video_model: 'ltx-2',
        });
        ctx.workflow_contract.values = ['txt2img', 'img2img', 'txt2vid', 'img2vid', 'extend'];
        return ctx;
      };

      it('reads the frame rule and the extend step of the model', () => {
        const ctx = familyContext();

        expect(videoTimingFor(ctx, 'h3')).toEqual({ fps: 24, frameRule: H3_RULE });
        expect(extendStepFor(ctx, 'h3')).toBe(17);
        expect(extendStepFor(ctx, 'ltx-2')).toBe(8);
        expect(stepOptionsFor(ctx, 'h3')).toEqual([4, 8]);
        expect(stepOptionsFor(ctx, 'ltx-2')).toBeNull();
      });

      it('moves a kept frame count to the next count of the new rule', () => {
        const ctx = familyContext();
        draft.hydrateFromContext(ctx, 'ltx-2');
        draft.update('frameCount', 121);

        switchTo(ctx, 'h3');

        expect(draft.state.frameCount).toBe(124);
      });

      it('moves the steps to an offered count (the higher on a tie)', () => {
        const ctx = familyContext();
        draft.hydrateFromContext(ctx, 'ltx-2');
        draft.update('steps', 6);

        switchTo(ctx, 'h3');

        expect(draft.state.steps).toBe(8);
      });

      it('clears the low-memory and upscale settings the model lacks', () => {
        const ctx = familyContext({ hidden_controls: ['low_memory', 'video_upscale_enabled', 'video_upscale_factor', 'video_upscale_review'] });
        draft.hydrateFromContext(ctx, 'ltx-2');
        draft.patch({ lowMemory: false, videoUpscaleEnabled: true, videoUpscaleReview: true });

        switchTo(ctx, 'h3');

        expect(draft.state.lowMemory).toBe(true);
        expect(draft.state.videoUpscaleEnabled).toBe(false);
        expect(draft.state.videoUpscaleReview).toBe(false);
      });

      it('leaves those settings alone for a model that has the controls', () => {
        const ctx = familyContext();
        draft.hydrateFromContext(ctx, 'ltx-2');
        draft.patch({ lowMemory: false, videoUpscaleEnabled: true });

        draft.hydrateFromContext(ctx, 'ltx-2', { keepSettings: true });

        expect(draft.state.lowMemory).toBe(false);
        expect(draft.state.videoUpscaleEnabled).toBe(true);
      });

      it('aligns a reused frame count to the rule of the reused model', () => {
        const ctx = familyContext();
        draft.hydrateFromContext(ctx, 'h3');

        draft.loadFromUrl({ model: 'h3', frames: '125' }, ctx);

        expect(draft.state.frameCount).toBe(141);
      });

      describe('the last frame', () => {
        const END_CONTROLS = ['end_image', 'end_image_clear', 'end_image_path'];
        // LTX hides the last-frame controls, H3 keeps them; no model change or tab change drops the image.
        const lastFrameContext = (): WorkspaceContext => {
          const ctx = familyContext();
          ctx.video_model_defaults['ltx-2'] = { ...ctx.video_model_defaults['ltx-2'] as VideoModelDefaults, hidden_controls: END_CONTROLS };
          return ctx;
        };

        beforeEach(() => draft.update('workflow', 'txt2vid'));

        it('starts empty', () => {
          expect([draft.state.endImagePath, draft.state.endImageUpload]).toEqual([null, null]);
        });

        it('keeps a chosen last frame on a model that has the slot', () => {
          const ctx = lastFrameContext();
          draft.hydrateFromContext(ctx, 'h3');
          draft.update('endImagePath', '/in/end.png');

          draft.hydrateFromContext(ctx, 'h3', { keepSettings: true });

          expect(draft.state.endImagePath).toBe('/in/end.png');
        });

        it('keeps the last frame when another tab opens', () => {
          const ctx = lastFrameContext();
          draft.hydrateFromContext(ctx, 'h3');
          draft.setLastFrame({ path: '/in/end.png' });

          openTab('txt2img', ctx);

          expect(draft.state.endImagePath).toBe('/in/end.png');
        });

        it('reuses a recorded last frame with the model that made it', () => {
          const ctx = lastFrameContext();
          draft.hydrateFromContext(ctx, 'h3');

          draft.loadFromUrl({ workflow: 'img2vid', model: 'h3', image_path: '/in/start.png', end_image_path: '/in/end.png' }, ctx);

          expect([draft.state.firstFramePath, draft.state.endImagePath]).toEqual(['/in/start.png', '/in/end.png']);
        });

        it('ignores a last frame in a reuse URL for the Image tab, which has no such slot', () => {
          const ctx = lastFrameContext();
          draft.hydrateFromContext(ctx, 'h3');
          draft.update('workflow', 'txt2img');

          draft.loadFromUrl({ workflow: 'txt2img', end_image_path: '/in/end.png' }, ctx);

          expect(draft.state.endImagePath).toBeNull();
        });
      });
    });
  });
});

describe('applyPreset', () => {
  const LORAS = [{ name: 'style-a', path: '/loras/style-a.safetensors' }, { name: 'detail-b', path: '/loras/detail-b.safetensors' }];

  /** A context whose workflows show every control a preset sets, with two image and two video models. */
  const presetContext = (overrides: Partial<WorkspaceContext> = {}): WorkspaceContext => {
    const json = makeImageDefaults({ family: 'zimage', supports_negative_prompt: true, supports_json_prompt: true, supports_first_sigma: true, steps: 9, guidance: 0, max_reference_images: 4 });
    const plain = makeImageDefaults({ family: 'flux1', supports_negative_prompt: false, supports_json_prompt: false, supports_first_sigma: false, steps: 28, guidance: 3.5, max_reference_images: 4 });
    const ctx = makeContext({
      image_models: [{ id: 'json-model', label: 'JSON', type: 'image' }, { id: 'plain-model', label: 'Plain', type: 'image' }, { id: 'twin-model', label: 'Twin', type: 'image' }],
      image_model_defaults: { 'json-model': json, 'plain-model': plain, 'twin-model': { ...json } },
      current_image_model: 'json-model',
      video_models: [{ id: 'ltx-v-0.9', label: 'LTX Video', type: 'video' }, { id: 'ltx-2', label: 'LTX 2', type: 'video' }],
      video_model_defaults: { 'ltx-v-0.9': makeVideoDefaults({ family: 'ltx' }), 'ltx-2': makeVideoDefaults({ family: 'ltx', steps: 6, max_steps: 8, frame_rule: { alignment: 8, offset: 1, round: 'nearest' } }) },
      loras: LORAS,
      ...overrides,
    });
    ctx.workflow_contract.values = ['txt2img', 'img2img', 'txt2vid', 'img2vid', 'extend'];
    for (const definition of Object.values(ctx.workflow_contract.definitions)) definition.visible_controls.push('loras', 'quantize', 'scheduler');
    return ctx;
  };

  const IMAGE_PRESET: SettingsPreset = {
    slug: 'look', name: 'Look', mode: 'image', model: 'twin-model',
    settings: {
      quantize: 8, lora: 'style-a:0.8', ratio: '16:9', size: 'l', width: null, height: null, runs: 4, steps: 17, guidance: 2.5, image_strength: 0.7, scheduler: 'dpm', first_sigma: 0.4,
      sharpen_enabled: true, sharpen_amount: 0.6, contrast_enabled: true, contrast_amount: 1.3, saturation_enabled: false, saturation_amount: null, transparent: true,
      upscale: 4, upscale_denoise: 0.3, upscale_steps: 12, upscale_guidance: 1.5, upscale_sharpen: false, review: true,
    },
  };

  /** The draft keys an image preset sets, with the values IMAGE_PRESET gives them. */
  const IMAGE_PRESET_DRAFT: Partial<DraftState> = {
    model: 'twin-model', quantize: 8, loraString: 'style-a:0.8', ratio: '16:9', size: 'l', dimensionMode: 'ratio', runs: 4, steps: 17, guidance: 2.5, referenceImageStrength: 0.7, scheduler: 'dpm', firstSigma: 0.4,
    postprocessSharpenEnabled: true, postprocessSharpenAmount: 0.6, postprocessContrastEnabled: true, postprocessContrastAmount: 1.3,
    postprocessSaturationEnabled: false, postprocessSaturationAmount: 1, postprocessTransparent: true, upscaleEnabled: true, upscaleFactor: 4, upscaleDenoise: 0.3, upscaleSteps: 12, upscaleGuidance: 1.5, upscaleSharpen: false, upscaleReview: true,
  };

  /** A distinctive value for each draft key that is not a preset key. */
  const MARKED: Partial<DraftState> = {
    workflow: 'img2img', promptSource: 'file', prompt: 'a prompt', scene: 'a scene', jsonPromptEnabled: true, jsonPrompt: '{"a":1}', negativePrompt: 'blurry',
    promptFilePath: '/prompts/set.yaml', promptFileOptionIds: ['one', 'two'], promptFileLoaded: { path: '/prompts/set.yaml', count: 2 },
    seed: 42, referenceImagePath: '/refs/ref.png', referenceImageUpload: 'start.png', firstFramePath: '/refs/first.png', firstFrameUpload: 'first.png',
    referenceImagePaths: ['/refs/a.png', '/refs/b.png'], missingReferencePaths: ['/refs/b.png'], endImagePath: '/refs/end.png', endImageUpload: 'end.png', audioTrackPath: '/audio/song.mp3', audioTrackUpload: 'memo.m4a', audioTrackStart: 90,
    extendAudioTrackPath: '/audio/other.mp3', extendAudioTrackUpload: 'extend.m4a', extendAudioTrackStart: 12, frameCount: 33, fps: 30, audio: true, lowMemory: true,
    historyCollapsed: true, sidebarCollapsed: true, lastGeneratedAt: '2026-01-01T00:00:00Z',
    enhancedPrompt: '{"caption":"x"}', enhancedScene: 'enhanced scene', enhancedFrom: { prompt: 'a prompt', scene: 'a scene', mode: 'image' },
    savedText: { video: { prompt: 'an action', scene: 'a video scene', negativePrompt: '', jsonPromptEnabled: false, jsonPrompt: '', enhancedPrompt: '', enhancedScene: '', enhancedFrom: null, enhancedHistory: [] } },
    enhancedHistory: [{ prompt: 'plain rewrite', scene: '', from: { prompt: 'a prompt', scene: 'a scene', mode: 'image' } }, { prompt: '{"caption":"x"}', scene: '', from: { prompt: 'a prompt', scene: 'a scene', mode: 'image' } }],
    enhanceSettings: { style: 'cinematic' } as DraftState['enhanceSettings'], enhanceSettingsVideo: { style: 'dreamy' } as DraftState['enhanceSettingsVideo'],
    enhanceAuto: true, enhanceModel: 'smart', enhanceFormat: 'json',
    extendSourcePath: '/clips/source.mp4', extendFrames: 16, missingSourceVideo: true, extendPrefill: { path: '/clips/source.mp4', scene: 's', prompt: 'p' },
    videoUpscaleEnabled: true, videoUpscaleFactor: 2, videoUpscaleReview: true,
    videoKind: 'extend', audioKind: 'speech', audioModels: { music: 'yue2', speech: 'higgs3', sound: 'sa3' }, audioStyle: 'a style', lyrics: '[verse]\nla', speechText: 'some text', soundPrompt: 'rain',
    audioSeconds: 30, voiceMode: 'recording', voicePath: '/voices/me.wav', voiceUpload: 'me.wav', voiceText: 'hello',
  };

  beforeEach(() => {
    localStorage.clear();
    draft.reset();
  });

  it('changes only the preset keys, and the capability gates of the model it brings', () => {
    const ctx = presetContext();
    draft.hydrateFromContext(ctx, 'json-model');
    draft.patch({ ...MARKED, loraString: 'detail-b:1', quantize: 4, lorasByFamily: { flux1: 'detail-b:2' } });
    const before = { ...draft.state };

    draft.applyPreset(ctx, IMAGE_PRESET);

    const after = draft.state;
    expect(after).toMatchObject(IMAGE_PRESET_DRAFT);
    const presetOwned = new Set([...Object.keys(IMAGE_PRESET_DRAFT), 'lorasByFamily', 'lastImageModel', 'lastVideoModel', 'appliedPreset', 'width', 'height', 'upscaleFactor']);
    // Every key is either marked above or a preset key, so a new draft key must be classified here.
    const unclassified = Object.keys(after).filter((key) => !(key in MARKED) && !presetOwned.has(key) && key !== 'version');
    expect(unclassified).toEqual([]);
    for (const key of Object.keys(MARKED) as (keyof DraftState)[]) expect(after[key], `${key} must be left alone`).toEqual(before[key]);
    expect(after.version).toBe(before.version);
    expect(after.appliedPreset).toBe('look');
  });

  describe('the workflow follows the start image and the preset\'s model', () => {
    /** presetContext plus "no-i2i", an image model without image-to-image. */
    const withoutImg2img = (): WorkspaceContext => {
      const ctx = presetContext();
      ctx.image_models = [...ctx.image_models, { id: 'no-i2i', label: 'No img2img', type: 'image' }];
      ctx.image_model_defaults = { ...ctx.image_model_defaults, 'no-i2i': makeImageDefaults({ family: 'flux1', supports_img2img: false }) };
      return ctx;
    };

    it('a preset that brings a model without image-to-image runs from text, keeping the start image', () => {
      const ctx = withoutImg2img();
      draft.hydrateFromContext(ctx, 'json-model');
      draft.setStartImage('image', { path: '/in/start.png' }, ctx);
      expect(draft.state.workflow).toBe('img2img');

      draft.applyPreset(ctx, { ...IMAGE_PRESET, model: 'no-i2i' });

      expect(draft.state).toMatchObject({ model: 'no-i2i', workflow: 'txt2img', referenceImagePath: '/in/start.png' });
    });

    it('a preset that brings a model with image-to-image starts from the kept start image', () => {
      const ctx = withoutImg2img();
      draft.hydrateFromContext(ctx, 'no-i2i');
      draft.setStartImage('image', { path: '/in/start.png' }, ctx);
      expect(draft.state.workflow).toBe('txt2img');

      draft.applyPreset(ctx, IMAGE_PRESET);

      expect(draft.state).toMatchObject({ model: 'twin-model', workflow: 'img2img', referenceImagePath: '/in/start.png' });
    });
  });

  it('clears what a model that cannot use it, but only that, when the preset switches from a JSON model to a plain one', () => {
    const ctx = presetContext();
    draft.hydrateFromContext(ctx, 'json-model');
    draft.patch({ ...MARKED });
    const before = { ...draft.state };

    draft.applyPreset(ctx, { ...IMAGE_PRESET, model: 'plain-model' });

    const after = draft.state;
    expect(after.model).toBe('plain-model');
    // The gates of a manual model switch: the JSON caption and box, the negative prompt, the first sigma.
    expect(after).toMatchObject({ enhancedPrompt: '', enhancedFrom: null, jsonPromptEnabled: false, jsonPrompt: '', negativePrompt: '', firstSigma: null });
    // The rewrite history keeps its plain rewrites and loses the JSON captions.
    expect(after.enhancedHistory.map((entry) => entry.prompt)).toEqual(['plain rewrite']);
    // Everything else stays: prompt, scene, seed, prompt source, enhance options, workflow, reference, extend source.
    for (const key of ['prompt', 'scene', 'seed', 'promptSource', 'promptFilePath', 'promptFileOptionIds', 'enhancedScene', 'enhanceSettings', 'enhanceSettingsVideo', 'enhanceAuto',
      'enhanceModel', 'enhanceFormat', 'workflow', 'referenceImagePath', 'referenceImagePaths', 'extendSourcePath', 'extendFrames', 'sidebarCollapsed'] as (keyof DraftState)[]) {
      expect(after[key], `${key} must be left alone`).toEqual(before[key]);
    }
  });

  it('never replaces the prompt, seed, workflow or reference image', () => {
    const ctx = presetContext();
    draft.hydrateFromContext(ctx, 'json-model');
    draft.patch({ prompt: 'keep me', seed: 7, referenceImagePath: '/refs/r.png' });
    draft.update('workflow', 'img2img');

    draft.applyPreset(ctx, IMAGE_PRESET);

    expect(draft.state).toMatchObject({ prompt: 'keep me', seed: 7, referenceImagePath: '/refs/r.png', workflow: 'img2img' });
  });

  it('clears the LoRAs and quantization when the preset has none, and the LoRAs remembered for the family stay out', () => {
    const ctx = presetContext();
    draft.hydrateFromContext(ctx, 'json-model');
    draft.patch({ loraString: 'style-a:1', quantize: 8, lorasByFamily: { flux1: 'detail-b:0.5' } });

    draft.applyPreset(ctx, { ...IMAGE_PRESET, model: 'plain-model', settings: { ...IMAGE_PRESET.settings, lora: '', quantize: null } });

    expect(draft.state).toMatchObject({ model: 'plain-model', loraString: '', quantize: null });
    // The family left behind is remembered, as on a manual switch.
    expect(draft.state.lorasByFamily).toMatchObject({ zimage: 'style-a:1' });
  });

  it('leaves LoRAs that are not installed out, and keeps the current model when the preset names one that is not offered', () => {
    const ctx = presetContext();
    draft.hydrateFromContext(ctx, 'plain-model');

    draft.applyPreset(ctx, { ...IMAGE_PRESET, model: 'removed-model', settings: { ...IMAGE_PRESET.settings, lora: 'style-a:0.8,gone-one:1,gone-two:0.5' } });

    expect(draft.state).toMatchObject({ model: 'plain-model', loraString: 'style-a:0.8', steps: 17, ratio: '16:9' });
  });

  it('keeps a LoRA whose file is still downloading across a family switch and back, and drops one that is merely missing', () => {
    const ctx = presetContext();
    draft.hydrateFromContext(ctx, 'json-model');
    downloadStore.adopt({ id: 'd1', kind: 'lora', name: 'coming', label: 'LoRA coming', state: 'running', message: null });
    draft.patch({ loraString: 'style-a:1,coming:0.7,never-was:1' });

    draft.hydrateFromContext(ctx, 'plain-model');
    expect(draft.state.loraString).toBe('');
    draft.hydrateFromContext(ctx, 'json-model');

    expect(draft.state.loraString).toBe('style-a:1,coming:0.7');
    downloadStore.reset();
  });

  it('removes a deleted LoRA from the chips and from what every model family remembers', () => {
    const ctx = presetContext();
    draft.hydrateFromContext(ctx, 'json-model');
    draft.patch({ loraString: 'style-a:1,detail-b:0.5', lorasByFamily: { flux1: 'detail-b:2,style-a:1', ltx: 'detail-b:1' } });

    draft.removeLora('detail-b');

    expect(draft.state.loraString).toBe('style-a:1');
    expect(draft.state.lorasByFamily).toEqual({ flux1: 'style-a:1', ltx: '' });
  });

  it('applies a custom size as a custom size', () => {
    const ctx = presetContext();
    draft.hydrateFromContext(ctx, 'json-model');

    draft.applyPreset(ctx, { ...IMAGE_PRESET, settings: { ...IMAGE_PRESET.settings, size: 'custom', ratio: '16:9', width: 640, height: 960 } });

    expect(draft.state).toMatchObject({ dimensionMode: 'custom', width: 640, height: 960 });
  });

  it('sets every video key and leaves the image keys alone', () => {
    const ctx = presetContext();
    draft.update('workflow', 'txt2vid');
    draft.hydrateFromContext(ctx, 'ltx-v-0.9');
    draft.patch({ prompt: 'a clip', scene: 'a street', seed: 5, guidance: 9, quantize: 4, scheduler: 'dpm', upscaleEnabled: true, upscaleFactor: 4, postprocessSharpenAmount: 0.9, referenceImagePath: '/refs/r.png' });
    const before = { ...draft.state };
    const preset: SettingsPreset = {
      slug: 'clip', name: 'Clip', mode: 'video', model: 'ltx-2',
      settings: { lora: 'detail-b:0.4', ratio: '9:16', size: 's', width: null, height: null, runs: 2, steps: 20, frames: 50, audio: false, low_memory: false, video_upscale_factor: 2, review: true },
    };

    draft.applyPreset(ctx, preset);

    const after = draft.state;
    expect(after).toMatchObject({
      model: 'ltx-2', loraString: 'detail-b:0.4', ratio: '9:16', size: 's', dimensionMode: 'ratio', audio: false, lowMemory: false,
      videoUpscaleEnabled: true, videoUpscaleFactor: 2, videoUpscaleReview: true, runs: 2, appliedPreset: 'clip',
    });
    // The new model's frame grid and step limit apply, as on a manual switch.
    expect(after.frameCount).toBe(49);
    expect(after.steps).toBe(8);
    for (const key of ['prompt', 'scene', 'seed', 'guidance', 'scheduler', 'upscaleEnabled', 'upscaleFactor', 'postprocessSharpenAmount', 'referenceImagePath', 'workflow'] as (keyof DraftState)[]) {
      expect(after[key], `${key} must be left alone`).toEqual(before[key]);
    }
    expect(after.quantize).toBeNull();
  });

  it('in an extension sets every preset key too, and leaves the source, prompt and extend length alone', () => {
    const ctx = presetContext();
    draft.update('workflow', 'extend');
    draft.hydrateFromContext(ctx, 'ltx-2');
    draft.patch({ prompt: 'p', scene: 's', seed: 7, extendSourcePath: '/clips/source.mp4', extendFrames: 24 });
    const before = { ...draft.state };
    const preset: SettingsPreset = {
      slug: 'clip', name: 'Clip', mode: 'video', model: 'ltx-v-0.9',
      settings: { lora: 'style-a:0.4', ratio: '9:16', size: 's', width: null, height: null, runs: 1, steps: 5, frames: 97, audio: false, low_memory: false, video_upscale_factor: 2, review: true },
    };

    draft.applyPreset(ctx, preset);

    expect(draft.state).toMatchObject({
      model: 'ltx-v-0.9', loraString: 'style-a:0.4', ratio: '9:16', size: 's', steps: 5, audio: false, lowMemory: false,
      videoUpscaleEnabled: true, videoUpscaleFactor: 2, videoUpscaleReview: true, appliedPreset: 'clip',
    });
    for (const key of ['prompt', 'scene', 'seed', 'extendSourcePath', 'extendFrames', 'workflow'] as (keyof DraftState)[]) {
      expect(draft.state[key], `${key} must be left alone`).toEqual(before[key]);
    }
  });

  it('fits a custom size and the steps to the model it ends on, as after a manual switch', () => {
    const ctx = presetContext({ image_model_defaults: { 'json-model': makeImageDefaults({ supports_json_prompt: true, dimension_max: 1024 }), 'plain-model': makeImageDefaults({ dimension_max: 1024 }), 'twin-model': makeImageDefaults({ dimension_max: 1024 }) } });
    draft.hydrateFromContext(ctx, 'plain-model');

    // The preset's model is not installed, so the current model stays and its limits apply.
    draft.applyPreset(ctx, { ...IMAGE_PRESET, model: 'removed-model', settings: { ...IMAGE_PRESET.settings, size: 'custom', width: 2048, height: 1024 } });

    expect(draft.state.model).toBe('plain-model');
    expect(draft.state.dimensionMode).toBe('custom');
    expect(Math.max(draft.state.width, draft.state.height)).toBeLessThanOrEqual(1024);
    expect(draft.state.width / draft.state.height).toBeCloseTo(2, 1);
  });

  it('remembers the applied slug, persisted with the draft', () => {
    const ctx = presetContext();
    draft.hydrateFromContext(ctx, 'json-model');
    draft.applyPreset(ctx, IMAGE_PRESET);

    expect(draft.state.appliedPreset).toBe('look');
    draft.loadDraft();
    expect(draft.state.appliedPreset).toBe('look');
  });

  it('moves and clears the applied slug', () => {
    draft.setAppliedPreset('look');
    draft.setAppliedPreset('new-look');
    expect(draft.state.appliedPreset).toBe('new-look');
    draft.setAppliedPreset(null);
    expect(draft.state.appliedPreset).toBeNull();
  });

  it('forgets the preset when a new extend source is chosen, not when the same one is set again', () => {
    const ctx = presetContext();
    draft.update('workflow', 'extend');
    draft.hydrateFromContext(ctx, 'ltx-2');
    draft.setAppliedPreset('clip');
    draft.setExtendSourcePath('/clips/b.mp4');
    expect(draft.state.appliedPreset).toBeNull();

    draft.setAppliedPreset('clip');
    draft.setExtendSourcePath('/clips/b.mp4');
    expect(draft.state.appliedPreset).toBe('clip');
  });

  it('writes a saved auto upscale denoise, steps and guidance over the model default', () => {
    const base = makeImageDefaults({ family: 'zimage', upscale: { enabled: false, factor: 2, denoise: 0.4, steps: 10, guidance: 1, sharpen: true, save_pre: false } });
    const ctx = presetContext({ image_model_defaults: { 'json-model': base } });
    draft.hydrateFromContext(ctx, 'json-model');
    expect(draft.state.upscaleDenoise).toBe(0.4);

    const preset: SettingsPreset = {
      ...IMAGE_PRESET, model: 'json-model',
      settings: { ...IMAGE_PRESET.settings, first_sigma: null, upscale: 2, upscale_denoise: null, upscale_steps: null, upscale_guidance: null },
    };
    draft.applyPreset(ctx, preset);

    expect(draft.state).toMatchObject({ upscaleEnabled: true, upscaleDenoise: null, upscaleSteps: null, upscaleGuidance: null });
    expect(presetDiffers(preset, draft.state, ctx)).toBe(false);
  });

  it('keeps the preset when the same extend source is chosen again, when params carry no source settings, and when the source is inspected', () => {
    const ctx = presetContext();
    draft.update('workflow', 'extend');
    draft.hydrateFromContext(ctx, 'ltx-2');
    draft.setExtendSourcePath('/clips/a.mp4');
    draft.setAppliedPreset('clip');

    draft.setExtendSourcePath('/clips/a.mp4');
    draft.loadFromUrl({ prompt: 'p' }, ctx);
    draft.applyExtendSource(sourceInfo({ model: 'ltx-2', lora: 'style-a:1' }), ctx, { replaceText: true });

    expect(draft.state.appliedPreset).toBe('clip');
  });

  it('forgets the preset on a switch between image and video, not when an image changes the workflow within a tab, in the order the top nav makes it', () => {
    const ctx = presetContext();
    draft.hydrateFromContext(ctx, 'json-model');
    draft.setAppliedPreset('look');

    draft.setStartImage('image', { path: '/in/a.png' }, ctx);
    expect(draft.state).toMatchObject({ workflow: 'img2img', appliedPreset: 'look' });
    // The top nav picks the tab first; the workspace's tab-change handler runs after.
    draft.setTab('video');
    expect(draft.state.appliedPreset).toBeNull();
    draft.onTabChange(ctx);
    expect(draft.state.appliedPreset).toBeNull();
    // New and Extend are one mode.
    draft.setAppliedPreset('clip');
    draft.setVideoKind('extend', ctx);
    expect(draft.state.appliedPreset).toBe('clip');
  });

  it('keeps the preset when workflow and mode stay put, and when loadFromUrl names anything', () => {
    const ctx = presetContext();
    draft.hydrateFromContext(ctx, 'json-model');
    draft.setAppliedPreset('look');
    draft.loadFromUrl({ workflow: 'txt2vid', model: 'plain-model', steps: '9', source_video: '/clips/a.mp4' }, ctx);
    expect(draft.state.appliedPreset).toBe('look');
  });

  it('clears the applied preset on Reset', () => {
    const ctx = presetContext();
    draft.hydrateFromContext(ctx, 'json-model');
    draft.setAppliedPreset('look');
    draft.resetSelections(ctx);
    expect(draft.state.appliedPreset).toBeNull();
  });
});

describe('model remembered per mode', () => {
  /** Klein takes 4 references and krea none; the server's default image model is krea, its default video model is ltx-a. */
  const modeContext = (overrides: Partial<WorkspaceContext> = {}): WorkspaceContext => {
    const ctx = makeContext({
      image_models: [{ id: 'krea', label: 'Krea', type: 'image' }, { id: 'klein4b', label: 'Klein 4B', type: 'image' }],
      video_models: [{ id: 'ltx-a', label: 'LTX A', type: 'video' }, { id: 'ltx-b', label: 'LTX B', type: 'video' }],
      image_model_defaults: { krea: makeImageDefaults(), klein4b: makeImageDefaults({ max_reference_images: 4 }) },
      video_model_defaults: { 'ltx-a': makeVideoDefaults(), 'ltx-b': makeVideoDefaults() },
      current_image_model: 'krea',
      current_video_model: 'ltx-a',
      ...overrides,
    });
    ctx.workflow_contract.definitions.txt2img.visible_controls.push('references', 'references_clear');
    return ctx;
  };

  beforeEach(() => {
    localStorage.clear();
    draft.reset();
  });

  it('returns to the image model last used after a video workflow', () => {
    const ctx = modeContext();
    draft.hydrateFromContext(ctx, 'klein4b');

    openTab('img2vid', ctx);
    expect(draft.state.model).toBe('ltx-a');
    openTab('txt2img', ctx);

    expect(draft.state.model).toBe('klein4b');
  });

  it('returns to the video model last used after an image workflow', () => {
    const ctx = modeContext();
    openTab('img2vid', ctx);
    draft.update('model', 'ltx-b');

    openTab('txt2img', ctx);
    expect(draft.state.model).toBe('krea');
    openTab('txt2vid', ctx);

    expect(draft.state.model).toBe('ltx-b');
  });

  it('remembers the model picked by hand through a model switch', () => {
    const ctx = modeContext();
    draft.hydrateFromContext(ctx, 'krea');
    draft.hydrateFromContext(ctx, 'klein4b', { keepChangedSettings: true });

    openTab('txt2vid', ctx);
    openTab('txt2img', ctx);

    expect(draft.state.model).toBe('klein4b');
  });

  it('keeps both remembered models through Reset', () => {
    const ctx = modeContext();
    draft.hydrateFromContext(ctx, 'klein4b');
    openTab('txt2vid', ctx);
    draft.update('model', 'ltx-b');

    draft.resetSelections(ctx);
    expect(draft.state.model).toBe('ltx-b');
    openTab('txt2img', ctx);
    expect(draft.state.model).toBe('klein4b');
    openTab('txt2vid', ctx);

    expect(draft.state.model).toBe('ltx-b');
  });

  it('survives a reload of the saved draft', () => {
    const ctx = modeContext();
    draft.hydrateFromContext(ctx, 'klein4b');
    openTab('txt2vid', ctx);

    draft.loadDraft();
    openTab('txt2img', ctx);

    expect(draft.state.model).toBe('klein4b');
  });

  it('lets a model named by the URL win over the remembered one', () => {
    const ctx = modeContext();
    draft.hydrateFromContext(ctx, 'klein4b');
    openTab('txt2vid', ctx);

    // The page's prefill: workflow first, then the model the URL names, then the rest of the params.
    const params = { workflow: 'txt2img', model: 'krea' };
    draft.loadFromUrl({ workflow: params.workflow }, ctx);
    draft.hydrateFromContext(ctx, params.model);
    draft.loadFromUrl(params, ctx);

    expect(draft.state.model).toBe('krea');
    expect(draft.state.lastImageModel).toBe('krea');
  });

  it('falls back to the default when the remembered model is no longer offered', () => {
    draft.hydrateFromContext(modeContext(), 'klein4b');
    openTab('txt2vid', modeContext());
    const later = modeContext({
      image_models: [{ id: 'krea', label: 'Krea', type: 'image' }],
      image_model_defaults: { krea: makeImageDefaults() },
    });

    openTab('txt2img', later);

    expect(draft.state.model).toBe('krea');
  });

});

describe('references', () => {
  /** Klein takes 4, "pair" takes 2, "flux-dev" takes none; only the image workflows list the controls. */
  const referencesContext = (): WorkspaceContext => {
    const ctx = makeContext({
      image_models: [{ id: 'klein', label: 'Klein', type: 'image' }, { id: 'pair', label: 'Pair', type: 'image' }, { id: 'flux-dev', label: 'FLUX Dev', type: 'image' }],
      image_model_defaults: {
        klein: makeImageDefaults({ max_reference_images: 4 }),
        pair: makeImageDefaults({ max_reference_images: 2 }),
        'flux-dev': makeImageDefaults(),
      },
      current_image_model: 'klein',
    });
    const { definitions } = ctx.workflow_contract;
    for (const workflow of ['txt2img', 'img2img'] as const) definitions[workflow].visible_controls.push('references', 'references_clear');
    return ctx;
  };
  const PATHS = ['/in/a.png', '/in/b.png', '/in/c.png'];

  beforeEach(() => {
    localStorage.clear();
    draft.reset();
  });

  it('keeps the whole list through a model that takes none or fewer, and back', () => {
    const ctx = referencesContext();
    draft.hydrateFromContext(ctx, 'klein');
    draft.update('referenceImagePaths', PATHS);

    draft.hydrateFromContext(ctx, 'flux-dev');
    expect(draft.state.referenceImagePaths).toEqual(PATHS);
    draft.hydrateFromContext(ctx, 'pair');
    expect(draft.state.referenceImagePaths).toEqual(PATHS);
    draft.hydrateFromContext(ctx, 'klein');
    expect(draft.state.referenceImagePaths).toEqual(PATHS);
  });

  it('keeps them while a video workflow is open, and brings them back to the image workflow', () => {
    const ctx = referencesContext();
    draft.hydrateFromContext(ctx, 'klein');
    draft.patch({ referenceImagePaths: PATHS, missingReferencePaths: ['/in/c.png'] });

    openTab('txt2vid', ctx);
    expect(draft.state.referenceImagePaths).toEqual(PATHS);

    openTab('txt2img', ctx);
    expect(draft.state.referenceImagePaths).toEqual(PATHS);
    expect(draft.state.missingReferencePaths).toEqual(['/in/c.png']);
  });

  describe('Reuse', () => {
    it('brings back the references in order and drops the flags of the rows it replaces', () => {
      const ctx = referencesContext();
      draft.hydrateFromContext(ctx, 'klein');
      draft.patch({ referenceImagePaths: ['/in/old.png'], missingReferencePaths: ['/in/old.png'] });

      draft.loadFromUrl({ workflow: 'txt2img', model: 'klein', reference_path: PATHS }, ctx);

      expect(draft.state.referenceImagePaths).toEqual(PATHS);
      expect(draft.state.missingReferencePaths).toEqual([]);
    });

    it('takes a single reference as a list of one', () => {
      const ctx = referencesContext();
      draft.hydrateFromContext(ctx, 'klein');

      draft.loadFromUrl({ reference_path: '/in/a.png' }, ctx);

      expect(draft.state.referenceImagePaths).toEqual(['/in/a.png']);
    });

    it('clears the references already chosen when the URL names a blank list', () => {
      const ctx = referencesContext();
      draft.hydrateFromContext(ctx, 'klein');
      draft.update('referenceImagePaths', PATHS);

      draft.loadFromUrl({ reference_path: '' }, ctx);

      expect(draft.state.referenceImagePaths).toEqual([]);
    });

    it('leaves the references alone when the URL does not name them', () => {
      const ctx = referencesContext();
      draft.hydrateFromContext(ctx, 'klein');
      draft.update('referenceImagePaths', PATHS);

      draft.loadFromUrl({ steps: '8' }, ctx);

      expect(draft.state.referenceImagePaths).toEqual(PATHS);
    });

    it('ignores references for a workflow that does not show the section', () => {
      const ctx = referencesContext();
      draft.hydrateFromContext(ctx, 'klein');

      draft.loadFromUrl({ workflow: 'txt2vid', reference_path: PATHS }, ctx);

      expect(draft.state.referenceImagePaths).toEqual([]);
    });
  });
});

describe('image slots', () => {
  /** Klein takes a start image and 4 references; "ideogram" has no image-to-image; "no-end" is a video model without a last frame. */
  const slotContext = (): WorkspaceContext => {
    const ctx = makeContext({
      image_models: [{ id: 'klein', label: 'Klein', type: 'image' }, { id: 'ideogram', label: 'Ideogram', type: 'image' }],
      image_model_defaults: { klein: makeImageDefaults({ max_reference_images: 4 }), ideogram: makeIdeogramDefaults() },
      current_image_model: 'klein',
      video_models: [{ id: 'ltx-v-0.9', label: 'LTX Video', type: 'video' }, { id: 'no-end', label: 'No end', type: 'video' }],
      video_model_defaults: { 'ltx-v-0.9': makeVideoDefaults(), 'no-end': makeVideoDefaults({ hidden_controls: ['end_image', 'end_image_path', 'end_image_clear'] }) },
    });
    ctx.workflow_contract.values = ['txt2img', 'img2img', 'txt2vid', 'img2vid', 'extend'];
    for (const workflow of ['txt2img', 'img2img'] as const) ctx.workflow_contract.definitions[workflow].visible_controls.push('references', 'references_clear');
    return ctx;
  };
  const file = (name: string): File => new File(['x'], name, { type: 'image/png' });

  beforeEach(() => {
    localStorage.clear();
    draft.reset();
  });

  it('replacing a held file with a path forgets the file in the store', async () => {
    const ctx = slotContext();
    const store = memoryUploadStore();
    heldUploads.useStore(store);
    await heldUploads.restore();
    draft.hydrateFromContext(ctx, 'klein');
    draft.setStartImage('image', { file: file('start.png') }, ctx);
    await heldUploads.persisted();
    expect(await store.get('image')).not.toBeNull();

    draft.setStartImage('image', { path: '/in/start.png' }, ctx);
    await heldUploads.persisted();

    expect(heldUploads.start('image')).toBeNull();
    expect(await store.get('image')).toBeNull();
  });

  it('replacing a held first frame or last frame with a path forgets each file in the store', async () => {
    const ctx = slotContext();
    const store = memoryUploadStore();
    heldUploads.useStore(store);
    await heldUploads.restore();
    draft.setTab('video');
    draft.onTabChange(ctx);
    draft.setStartImage('video', { file: file('first.png') }, ctx);
    draft.setLastFrame({ file: file('last.png') });
    await heldUploads.persisted();
    expect(await store.get('video')).toMatchObject({ start: { name: 'first.png' }, end: { name: 'last.png' } });

    draft.setLastFrame({ path: '/in/last.png' });
    await heldUploads.persisted();
    expect(await store.get('video')).toMatchObject({ start: { name: 'first.png' }, end: null });

    draft.setStartImage('video', { path: '/in/first.png' }, ctx);
    await heldUploads.persisted();

    expect(heldUploads.start('video')).toBeNull();
    expect(heldUploads.lastFrame).toBeNull();
    expect(await store.get('video')).toBeNull();
  });

  it('Video with a last frame -> Image -> Video keeps both frames', () => {
    const ctx = slotContext();
    draft.setTab('video');
    draft.onTabChange(ctx);
    draft.setStartImage('video', { path: '/in/first.png' }, ctx);
    draft.setLastFrame({ path: '/in/last.png' });
    expect(draft.state.workflow).toBe('img2vid');

    openTab('txt2img', ctx);
    expect(draft.state).toMatchObject({ workflow: 'txt2img', firstFramePath: '/in/first.png', endImagePath: '/in/last.png' });
    openTab('txt2vid', ctx);

    expect(draft.state).toMatchObject({ workflow: 'img2vid', firstFramePath: '/in/first.png', endImagePath: '/in/last.png' });
  });

  it('Image -> Extend -> Image keeps the start image', () => {
    const ctx = slotContext();
    draft.hydrateFromContext(ctx);
    draft.setStartImage('image', { path: '/in/start.png' }, ctx);
    draft.patch({ referenceImagePaths: ['/in/ref.png'] });

    openTab('extend', ctx);
    expect(draft.state).toMatchObject({ workflow: 'extend', referenceImagePath: '/in/start.png' });
    openTab('txt2img', ctx);

    expect(draft.state).toMatchObject({ workflow: 'img2img', referenceImagePath: '/in/start.png', referenceImagePaths: ['/in/ref.png'] });
  });

  it('a model change keeps every image: the start image and references, the first and the last frame', () => {
    const ctx = slotContext();
    draft.hydrateFromContext(ctx, 'klein');
    draft.setStartImage('image', { path: '/in/start.png' }, ctx);
    draft.patch({ referenceImagePaths: ['/in/ref.png'] });

    draft.hydrateFromContext(ctx, 'ideogram', { keepChangedSettings: true });

    expect(draft.state).toMatchObject({ model: 'ideogram', referenceImagePath: '/in/start.png', referenceImagePaths: ['/in/ref.png'] });

    openTab('txt2vid', ctx);
    draft.setStartImage('video', { path: '/in/first.png' }, ctx);
    draft.setLastFrame({ path: '/in/end.png' });

    draft.hydrateFromContext(ctx, 'no-end', { keepChangedSettings: true });
    expect(draft.state).toMatchObject({ model: 'no-end', workflow: 'img2vid', firstFramePath: '/in/first.png', endImagePath: '/in/end.png' });

    draft.hydrateFromContext(ctx, 'ltx-v-0.9', { keepChangedSettings: true });
    expect(draft.state).toMatchObject({ model: 'ltx-v-0.9', firstFramePath: '/in/first.png', endImagePath: '/in/end.png', referenceImagePath: '/in/start.png' });
  });

  it('a start image on a model without img2img stays and the workflow is txt2img', () => {
    const ctx = slotContext();
    draft.hydrateFromContext(ctx, 'klein');
    draft.setStartImage('image', { path: '/in/start.png' }, ctx);
    expect(draft.state.workflow).toBe('img2img');

    draft.hydrateFromContext(ctx, 'ideogram', { keepChangedSettings: true });
    expect(draft.state).toMatchObject({ workflow: 'txt2img', referenceImagePath: '/in/start.png' });

    draft.hydrateFromContext(ctx, 'klein', { keepChangedSettings: true });
    expect(draft.state.workflow).toBe('img2img');

    // Set directly on a model without img2img, it stays too.
    draft.setStartImage('image', null, ctx);
    draft.hydrateFromContext(ctx, 'ideogram', { keepChangedSettings: true });
    draft.setStartImage('image', { path: '/in/other.png' }, ctx);
    expect(draft.state).toMatchObject({ workflow: 'txt2img', referenceImagePath: '/in/other.png' });
  });

  it('a start image set for another tab does not move this tab off its workflow', () => {
    const ctx = slotContext();
    draft.hydrateFromContext(ctx);
    openTab('extend', ctx);

    draft.setStartImage('image', { path: '/in/start.png' }, ctx);
    draft.setStartImage('video', { path: '/in/first.png' }, ctx);

    expect(draft.state.workflow).toBe('extend');
  });

  it('keeps Strength while no start image is set, and resets it with the model like Steps', () => {
    const ctx = slotContext();
    draft.hydrateFromContext(ctx, 'klein');
    draft.update('referenceImageStrength', 0.9);

    draft.setStartImage('image', { path: '/in/start.png' }, ctx);
    draft.setStartImage('image', null, ctx);
    openTab('txt2img', ctx);
    expect(draft.state.referenceImageStrength).toBe(0.9);

    draft.hydrateFromContext(ctx, 'ideogram');
    expect(draft.state.referenceImageStrength).toBe(0.5);
  });

  it('adding a first frame switches rememberedParts to action', () => {
    const ctx = slotContext();
    draft.setTab('video');
    draft.onTabChange(ctx);
    expect(rememberedParts(draft.state)).toBe('both');

    draft.setStartImage('video', { path: '/in/first.png' }, ctx);
    expect(rememberedParts(draft.state)).toBe('action');

    draft.setStartImage('video', null, ctx);
    expect(rememberedParts(draft.state)).toBe('both');
  });

  describe('resetSelections and images', () => {
    /** Both tabs hold every image: Image a start image, a reference and Strength 0.9; Video a first and a last frame. Ends on Image. */
    function holdImagesOnBothTabs(ctx: WorkspaceContext): void {
      draft.hydrateFromContext(ctx, 'klein');
      openTab('txt2vid', ctx);
      draft.setStartImage('video', { file: file('first.png') }, ctx);
      draft.setLastFrame({ file: file('last.png') });
      openTab('txt2img', ctx);
      draft.setStartImage('image', { file: file('start.png') }, ctx);
      draft.patch({ referenceImagePaths: ['/in/ref.png'], missingReferencePaths: ['/in/ref.png'], referenceImageStrength: 0.9, prompt: 'kept' });
      heldUploads.setReferences([file('ref.png')]);
    }

    const imageSlots = () => ({ referenceImagePath: null, referenceImageUpload: null, referenceImagePaths: [] as string[] });
    const videoSlots = () => ({ firstFramePath: null, firstFrameUpload: null, endImagePath: null, endImageUpload: null });

    it('on Image clears the start image, references and Strength, and keeps the first and last frame', () => {
      const ctx = slotContext();
      holdImagesOnBothTabs(ctx);

      draft.resetSelections(ctx);

      expect(draft.state).toMatchObject({ workflow: 'txt2img', prompt: 'kept', model: 'klein', referenceImageStrength: 0.5, missingReferencePaths: [], ...imageSlots() });
      expect(draft.state).toMatchObject({ firstFrameUpload: 'first.png', endImageUpload: 'last.png' });
      expect([heldUploads.startImage, heldUploads.references]).toEqual([null, []]);
      expect([heldUploads.firstFrame?.name, heldUploads.lastFrame?.name]).toEqual(['first.png', 'last.png']);

      openTab('txt2vid', ctx);
      expect(draft.state.workflow).toBe('img2vid');
    });

    it('on Video clears the first and last frame, and keeps the start image, references and Strength', () => {
      const ctx = slotContext();
      holdImagesOnBothTabs(ctx);
      openTab('txt2vid', ctx);
      expect(draft.state.workflow).toBe('img2vid');

      draft.resetSelections(ctx);

      expect(draft.state).toMatchObject({ workflow: 'txt2vid', referenceImageStrength: 0.9, missingReferencePaths: ['/in/ref.png'], ...videoSlots() });
      expect(draft.state).toMatchObject({ referenceImageUpload: 'start.png', referenceImagePaths: ['/in/ref.png'] });
      expect([heldUploads.firstFrame, heldUploads.lastFrame]).toEqual([null, null]);
      expect([heldUploads.startImage?.name, heldUploads.references.map((held) => held?.name)]).toEqual(['start.png', ['ref.png']]);

      openTab('txt2img', ctx);
      expect(draft.state.workflow).toBe('img2img');
    });

    it('on Extend clears no images', () => {
      const ctx = slotContext();
      holdImagesOnBothTabs(ctx);
      openTab('extend', ctx);
      expect(draft.state.workflow).toBe('extend');

      draft.resetSelections(ctx);

      expect(draft.state).toMatchObject({
        workflow: 'extend', referenceImageUpload: 'start.png', referenceImagePaths: ['/in/ref.png'], referenceImageStrength: 0.9,
        missingReferencePaths: ['/in/ref.png'], firstFrameUpload: 'first.png', endImageUpload: 'last.png',
      });
      expect([heldUploads.startImage?.name, heldUploads.firstFrame?.name, heldUploads.lastFrame?.name]).toEqual(['start.png', 'first.png', 'last.png']);
      expect(heldUploads.references).toHaveLength(1);
    });

    it('on Extend keeps frames and a start image chosen by path', () => {
      const ctx = slotContext();
      draft.hydrateFromContext(ctx, 'klein');
      openTab('txt2vid', ctx);
      draft.setStartImage('video', { path: '/in/first.png' }, ctx);
      draft.setLastFrame({ path: '/in/last.png' });
      openTab('txt2img', ctx);
      draft.setStartImage('image', { path: '/in/start.png' }, ctx);
      openTab('extend', ctx);

      draft.resetSelections(ctx);

      expect(draft.state).toMatchObject({
        workflow: 'extend', firstFramePath: '/in/first.png', endImagePath: '/in/last.png', referenceImagePath: '/in/start.png',
      });
    });
  });

  it('setStartImage and setLastFrame leave the changed settings alone, and only a tab change re-hydrates them', () => {
    const ctx = slotContext();
    draft.hydrateFromContext(ctx, 'klein');
    draft.update('steps', 23);

    draft.setStartImage('image', { path: '/in/start.png' }, ctx);
    draft.setStartImage('video', { file: file('first.png') }, ctx);
    draft.setLastFrame({ path: '/in/last.png' });
    draft.setStartImage('image', null, ctx);
    expect(draft.state.steps).toBe(23);

    draft.setTab('video');
    expect(draft.state.steps).toBe(23);
    draft.onTabChange(ctx);
    expect(draft.state.steps).not.toBe(23);
  });

  it('reset clears the held uploads too', () => {
    draft.setStartImage('image', { file: file('start.png') }, null);
    draft.reset();
    expect(heldUploads.startImage).toBeNull();
  });

  describe('prefill', () => {
    it('a full prefill sets every slot of the tab it opens, empty when absent', () => {
      const ctx = slotContext();
      draft.hydrateFromContext(ctx, 'klein');
      draft.setStartImage('image', { file: file('old.png') }, ctx);
      draft.patch({ referenceImagePaths: ['/in/old-ref.png'], missingReferencePaths: ['/in/old-ref.png'] });

      draft.loadFromUrl({ workflow: 'txt2img', prompt: 'a text-made asset' }, ctx, { replaceSlots: true });
      expect(draft.state).toMatchObject({ workflow: 'txt2img', referenceImagePath: null, referenceImageUpload: null, referenceImagePaths: [], missingReferencePaths: [] });

      draft.loadFromUrl({ workflow: 'img2img', image_path: '/in/start.png', reference_path: ['/in/a.png'] }, ctx, { replaceSlots: true });
      expect(draft.state).toMatchObject({ workflow: 'img2img', referenceImagePath: '/in/start.png', referenceImagePaths: ['/in/a.png'] });
    });

    it('a full prefill of a tab leaves the other tab\'s images alone', () => {
      const ctx = slotContext();
      draft.hydrateFromContext(ctx, 'klein');
      draft.setStartImage('video', { path: '/in/first.png' }, ctx);
      draft.setLastFrame({ path: '/in/last.png' });
      draft.patch({ missingReferencePaths: ['/in/first.png'] });

      draft.loadFromUrl({ workflow: 'txt2img', prompt: 'p' }, ctx, { replaceSlots: true });

      expect(draft.state).toMatchObject({ firstFramePath: '/in/first.png', endImagePath: '/in/last.png', missingReferencePaths: ['/in/first.png'] });
    });

    it('a full prefill in Video sets the first and the last frame and keeps the references', () => {
      const ctx = slotContext();
      draft.hydrateFromContext(ctx, 'klein');
      draft.patch({ referenceImagePath: '/in/start.png', referenceImagePaths: ['/in/ref.png'] });
      draft.setStartImage('video', { path: '/in/old-first.png' }, ctx);
      draft.setLastFrame({ path: '/in/old-last.png' });

      draft.loadFromUrl({ workflow: 'img2vid', image_path: '/in/first.png' }, ctx, { replaceSlots: true });

      expect(draft.state).toMatchObject({
        workflow: 'img2vid', firstFramePath: '/in/first.png', firstFrameUpload: null, endImagePath: null, endImageUpload: null,
        referenceImagePath: '/in/start.png', referenceImagePaths: ['/in/ref.png'],
      });
    });

    it('image_path routes to firstFramePath in Video and to referenceImagePath in Image', () => {
      const ctx = slotContext();
      draft.hydrateFromContext(ctx, 'klein');

      draft.loadFromUrl({ workflow: 'img2vid', image_path: '/in/a.png' }, ctx);
      expect(draft.state).toMatchObject({ firstFramePath: '/in/a.png', referenceImagePath: null });

      draft.loadFromUrl({ workflow: 'img2img', image_path: '/in/b.png' }, ctx);
      expect(draft.state).toMatchObject({ referenceImagePath: '/in/b.png', firstFramePath: '/in/a.png' });
    });

    it('a path drops the upload marker of its slot', () => {
      const ctx = slotContext();
      draft.hydrateFromContext(ctx, 'klein');
      draft.setStartImage('image', { file: file('start.png') }, ctx);

      draft.loadFromUrl({ workflow: 'img2img', image_path: '/in/b.png', use_as: 'start' }, ctx);

      expect(draft.state).toMatchObject({ referenceImagePath: '/in/b.png', referenceImageUpload: null });
    });

    it('a use_as prefill sets only its slot', () => {
      const ctx = slotContext();
      draft.hydrateFromContext(ctx, 'klein');
      draft.setStartImage('video', { path: '/in/first.png' }, ctx);
      draft.setLastFrame({ path: '/in/last.png' });
      draft.patch({ referenceImagePath: '/in/start.png', referenceImagePaths: ['/in/ref.png'] });

      draft.loadFromUrl({ workflow: 'img2vid', end_image_path: '/in/new-last.png', use_as: 'last' }, ctx, { replaceSlots: true });
      expect(draft.state).toMatchObject({ firstFramePath: '/in/first.png', endImagePath: '/in/new-last.png', referenceImagePath: '/in/start.png', referenceImagePaths: ['/in/ref.png'] });

      draft.loadFromUrl({ workflow: 'img2vid', image_path: '/in/new-first.png', use_as: 'start' }, ctx, { replaceSlots: true });
      expect(draft.state).toMatchObject({ firstFramePath: '/in/new-first.png', endImagePath: '/in/new-last.png' });

      draft.loadFromUrl({ workflow: 'img2img', image_path: '/in/new-start.png', use_as: 'start' }, ctx, { replaceSlots: true });
      expect(draft.state).toMatchObject({ referenceImagePath: '/in/new-start.png', referenceImagePaths: ['/in/ref.png'], firstFramePath: '/in/new-first.png' });
    });
  });

  describe('setTab', () => {
    it('picks the workflow its images call for, without a context', () => {
      draft.setStartImage('video', { path: '/in/first.png' }, null);
      draft.setTab('video');
      expect(draft.state.workflow).toBe('img2vid');
      draft.setVideoKind('extend', null);
      expect(draft.state.workflow).toBe('extend');
      draft.setTab('image');
      expect(draft.state.workflow).toBe('txt2img');
    });

    it('opens Video on the kind it was left on, and keeps each kind\'s own inputs while the kind changes', () => {
      draft.setStartImage('video', { path: '/in/first.png' }, null);
      draft.setTab('video');
      draft.patch({ frameCount: 49 });
      draft.setVideoKind('extend', null);
      draft.setExtendSourcePath('/clips/source.mp4');
      draft.patch({ extendFrames: 48 });
      draft.setAudioTrack('extend', { path: '/audio/extend.wav' });
      draft.setVideoKind('new', null);
      expect(draft.state).toMatchObject({ workflow: 'img2vid', videoKind: 'new', frameCount: 49, firstFramePath: '/in/first.png', extendSourcePath: '/clips/source.mp4', extendFrames: 48, extendAudioTrackPath: '/audio/extend.wav' });

      draft.setVideoKind('extend', null);
      draft.setTab('audio');
      draft.setTab('video');
      expect(draft.state).toMatchObject({ workflow: 'extend', videoKind: 'extend' });
      draft.setTab('image');
      draft.setTab('video');
      expect(draft.state.workflow).toBe('extend');

      draft.setVideoKind('new', null);
      draft.setTab('image');
      draft.setTab('video');
      expect(draft.state.workflow).toBe('img2vid');
    });

    it('opens Video on the kind a prefill named, once the draft is left and entered again', () => {
      draft.setTab('video');
      draft.setVideoKind('new', null);
      draft.setTab('image');
      draft.setTab('video');
      const ctx = makeContext();
      ctx.workflow_contract.values.push('extend');
      draft.loadFromUrl({ workflow: 'extend', source_video: '/clips/source.mp4' }, ctx);
      expect(draft.state.workflow).toBe('extend');
      draft.setTab('image');
      draft.setTab('video');
      expect(draft.state.workflow).toBe('extend');
    });

    it('gives the Extend kind back to New on a machine that cannot extend', () => {
      draft.setVideoKind('extend', null);
      const ctx = makeContext();
      ctx.workflow_contract.values = ctx.workflow_contract.values.filter((value) => value !== 'extend');
      draft.hydrateFromContext(ctx, null, { keepSettings: true });
      expect(draft.state).toMatchObject({ workflow: 'txt2vid', videoKind: 'new' });
    });
  });
});

describe('audio track', () => {
  /** A video model that follows a track, and one that does not (its audio controls are hidden). */
  const trackContext = (): WorkspaceContext => {
    const ctx = makeContext({
      image_models: [{ id: 'klein', label: 'Klein', type: 'image' }],
      image_model_defaults: { klein: makeImageDefaults() },
      current_image_model: 'klein',
      video_models: [{ id: 'ltx-2', label: 'LTX 2', type: 'video' }, { id: 'plain', label: 'Plain', type: 'video' }],
      video_model_defaults: { 'ltx-2': makeVideoDefaults(), plain: makeVideoDefaults({ hidden_controls: ['audio_track', 'audio_track_path', 'audio_track_start'] }) },
    });
    ctx.workflow_contract.values = ['txt2img', 'img2img', 'txt2vid', 'img2vid', 'extend'];
    for (const workflow of ['txt2vid', 'img2vid', 'extend'] as const) ctx.workflow_contract.definitions[workflow].visible_controls.push('audio_track', 'audio_track_path', 'audio_track_start');
    return ctx;
  };
  const memo = (name: string): File => new File(['x'], name, { type: 'audio/mp4' });

  beforeEach(() => {
    localStorage.clear();
    draft.reset();
    heldUploads.useStore(memoryUploadStore());
  });

  describe('per tab', () => {
    it('holds a path or an upload marker, never both, and starts at 0', () => {
      draft.setAudioTrack('video', { path: '/audio/song.mp3' });
      draft.setAudioTrackStart('video', 90);
      expect(draft.state).toMatchObject({ audioTrackPath: '/audio/song.mp3', audioTrackUpload: null, audioTrackStart: 90 });

      draft.setAudioTrack('video', { file: memo('memo.m4a') });
      expect(draft.state).toMatchObject({ audioTrackPath: null, audioTrackUpload: 'memo.m4a', audioTrackStart: 0 });
      expect(heldUploads.audioTrack?.name).toBe('memo.m4a');
    });

    it('keeps Video and Extend apart', () => {
      draft.setAudioTrack('video', { file: memo('video.m4a') });
      draft.setAudioTrack('extend', { path: '/audio/extend.mp3' });
      draft.setAudioTrackStart('extend', 12);

      expect(draft.state).toMatchObject({ audioTrackUpload: 'video.m4a', audioTrackStart: 0, extendAudioTrackPath: '/audio/extend.mp3', extendAudioTrackStart: 12 });
      expect([heldUploads.audioTrack?.name, heldUploads.extendAudioTrack]).toEqual(['video.m4a', null]);
    });

    it('clearing puts Start at back to 0 and forgets the held file', () => {
      draft.setAudioTrack('video', { file: memo('memo.m4a') });
      draft.setAudioTrackStart('video', 30);

      draft.setAudioTrack('video', null);

      expect(draft.state).toMatchObject({ audioTrackPath: null, audioTrackUpload: null, audioTrackStart: 0 });
      expect(heldUploads.audioTrack).toBeNull();
    });

    it('keeps the track when the model cannot follow it, and the model that can brings it back', () => {
      const ctx = trackContext();
      draft.hydrateFromContext(ctx, 'ltx-2');
      draft.update('workflow', 'txt2vid');
      draft.setAudioTrack('video', { path: '/audio/song.mp3' });

      draft.hydrateFromContext(ctx, 'plain', { keepSettings: true });
      expect(draft.state).toMatchObject({ model: 'plain', audioTrackPath: '/audio/song.mp3' });
      draft.hydrateFromContext(ctx, 'ltx-2', { keepSettings: true });
      expect(draft.state.audioTrackPath).toBe('/audio/song.mp3');
    });

    it('is left alone by a preset', () => {
      const ctx = trackContext();
      draft.hydrateFromContext(ctx, 'ltx-2');
      draft.update('workflow', 'txt2vid');
      draft.setAudioTrack('video', { path: '/audio/song.mp3' });
      draft.setAudioTrackStart('video', 90);

      draft.applyPreset(ctx, { slug: 'video-look', name: 'Video look', mode: 'video', model: 'ltx-2', settings: { steps: 6 } } as SettingsPreset);

      expect(draft.state).toMatchObject({ audioTrackPath: '/audio/song.mp3', audioTrackStart: 90 });
    });
  });

  describe('Reset', () => {
    it('on Video clears the Video track and its held file, and keeps the Extend track', () => {
      const ctx = trackContext();
      draft.hydrateFromContext(ctx, 'ltx-2');
      openTab('txt2vid', ctx);
      draft.setAudioTrack('video', { file: memo('video.m4a') });
      draft.setAudioTrackStart('video', 5);
      draft.setAudioTrack('extend', { path: '/audio/extend.mp3' });
      draft.setAudioTrackStart('extend', 12);

      draft.resetSelections(ctx);

      expect(draft.state).toMatchObject({ audioTrackPath: null, audioTrackUpload: null, audioTrackStart: 0, extendAudioTrackPath: '/audio/extend.mp3', extendAudioTrackStart: 12 });
      expect(heldUploads.audioTrack).toBeNull();
    });

    it('on Extend clears the Extend track and keeps the Video track', () => {
      const ctx = trackContext();
      draft.hydrateFromContext(ctx, 'ltx-2');
      openTab('txt2vid', ctx);
      draft.setAudioTrack('video', { file: memo('video.m4a') });
      draft.setAudioTrack('extend', { file: memo('extend.m4a') });
      openTab('extend', ctx);

      draft.resetSelections(ctx);

      expect(draft.state).toMatchObject({ extendAudioTrackPath: null, extendAudioTrackUpload: null, extendAudioTrackStart: 0, audioTrackUpload: 'video.m4a' });
      expect([heldUploads.audioTrack?.name, heldUploads.extendAudioTrack]).toEqual(['video.m4a', null]);
    });

  });

  describe('Reuse and Load settings', () => {
    it('put the track and Start at in Video for a text or image job', () => {
      const ctx = trackContext();
      draft.hydrateFromContext(ctx, 'ltx-2');

      draft.loadFromUrl({ workflow: 'txt2vid', prompt: 'a singer', audio_track_path: '/audio/song.mp3', audio_track_start: '90' }, ctx, { replaceSlots: true });

      expect(draft.state).toMatchObject({ audioTrackPath: '/audio/song.mp3', audioTrackUpload: null, audioTrackStart: 90, extendAudioTrackPath: null });
    });

    it('put the track and Start at in Extend for an extension', () => {
      const ctx = trackContext();
      draft.hydrateFromContext(ctx, 'ltx-2');
      draft.setAudioTrack('video', { path: '/audio/video.mp3' });

      draft.loadFromUrl({ workflow: 'extend', source_video: '/clips/a.mp4', scene: 's', audio_track_path: '/audio/song.mp3', audio_track_start: '6.65' }, ctx, { replaceSlots: true });

      expect(draft.state).toMatchObject({ extendAudioTrackPath: '/audio/song.mp3', extendAudioTrackStart: 6.65, audioTrackPath: '/audio/video.mp3' });
    });

    it('clear the opened tab\'s track for a job without one, and leave the other tab\'s', () => {
      const ctx = trackContext();
      draft.hydrateFromContext(ctx, 'ltx-2');
      draft.setAudioTrack('video', { file: memo('video.m4a') });
      draft.setAudioTrackStart('video', 5);
      draft.setAudioTrack('extend', { path: '/audio/extend.mp3' });

      draft.loadFromUrl({ workflow: 'txt2vid', prompt: 'no track' }, ctx, { replaceSlots: true });

      expect(draft.state).toMatchObject({ audioTrackPath: null, audioTrackUpload: null, audioTrackStart: 0, extendAudioTrackPath: '/audio/extend.mp3' });
    });

    it('leave the track alone for a Use as', () => {
      const ctx = trackContext();
      draft.hydrateFromContext(ctx, 'ltx-2');
      draft.setAudioTrack('video', { path: '/audio/song.mp3' });

      draft.loadFromUrl({ workflow: 'img2vid', image_path: '/in/first.png', use_as: 'start' }, ctx, { replaceSlots: true });

      expect(draft.state.audioTrackPath).toBe('/audio/song.mp3');
    });

  });

  describe('Extend prefill', () => {
    const SOURCE = '/clips/song-clip.mp4';

    beforeEach(() => {
      draft.update('workflow', 'extend');
      draft.setExtendSourcePath(SOURCE);
    });

    it('fills the track from a source that followed one, starting where its sound ended', () => {
      draft.applyExtendSource(sceneSource({ audio_track_path: '/audio/song.mp3', audio_track_end: 6.65 }));

      expect(draft.state).toMatchObject({ extendAudioTrackPath: '/audio/song.mp3', extendAudioTrackUpload: null, extendAudioTrackStart: 6.65 });
    });

    it('brings the source\'s track back on Reset, from where its sound ended', () => {
      draft.applyExtendSource(sceneSource({ audio_track_path: '/audio/song.mp3', audio_track_end: 6.65 }));
      draft.setAudioTrackStart('extend', 2);

      draft.resetSelections(trackContext());

      expect(draft.state).toMatchObject({ extendSourcePath: SOURCE, extendAudioTrackPath: '/audio/song.mp3', extendAudioTrackUpload: null, extendAudioTrackStart: 6.65 });
    });

    it('clears a track filled in from another source when the new source has none', () => {
      draft.applyExtendSource(sceneSource({ audio_track_path: '/audio/song.mp3', audio_track_end: 6.65 }));
      draft.setExtendSourcePath('/clips/plain.mp4');

      draft.applyExtendSource(sceneSource({ path: '/clips/plain.mp4' }));

      expect(draft.state).toMatchObject({ extendAudioTrackPath: null, extendAudioTrackUpload: null, extendAudioTrackStart: 0 });
    });

    it('keeps a track the user uploaded when the source changes', () => {
      draft.setAudioTrack('extend', { file: memo('memo.m4a') });

      draft.applyExtendSource(sceneSource({ audio_track_path: '/audio/song.mp3', audio_track_end: 3 }));

      expect(draft.state).toMatchObject({ extendAudioTrackPath: null, extendAudioTrackUpload: 'memo.m4a' });
      expect(heldUploads.extendAudioTrack?.name).toBe('memo.m4a');
    });

    it('keeps a track the user chose by path, and one whose Start at they changed, when the source changes', () => {
      draft.applyExtendSource(sceneSource({ audio_track_path: '/audio/song.mp3', audio_track_end: 6.65 }));
      draft.setAudioTrackStart('extend', 10);
      draft.setExtendSourcePath('/clips/other.mp4');

      draft.applyExtendSource(sceneSource({ path: '/clips/other.mp4', audio_track_path: '/audio/other.mp3', audio_track_end: 2 }));

      expect(draft.state).toMatchObject({ extendAudioTrackPath: '/audio/song.mp3', extendAudioTrackStart: 10 });
    });

    it('replaces a track an earlier source prefilled, and drops it for a source without one', () => {
      draft.applyExtendSource(sceneSource({ audio_track_path: '/audio/song.mp3', audio_track_end: 6.65 }));
      draft.setExtendSourcePath('/clips/other.mp4');
      draft.applyExtendSource(sceneSource({ path: '/clips/other.mp4', audio_track_path: '/audio/other.mp3', audio_track_end: 2 }));
      expect(draft.state).toMatchObject({ extendAudioTrackPath: '/audio/other.mp3', extendAudioTrackStart: 2 });

      draft.setExtendSourcePath('/clips/plain.mp4');
      draft.applyExtendSource(sceneSource({ path: '/clips/plain.mp4' }), null, { replaceText: true });

      expect(draft.state).toMatchObject({ extendAudioTrackPath: null, extendAudioTrackUpload: null, extendAudioTrackStart: 0 });
    });

    it('does not touch the track on a later answer for the same source', () => {
      draft.applyExtendSource(sceneSource({ audio_track_path: '/audio/song.mp3', audio_track_end: 3 }));
      draft.setAudioTrack('extend', { path: '/audio/mine.mp3' });

      draft.applyExtendSource(sceneSource({ audio_track_path: '/audio/song.mp3', audio_track_end: 3 }));

      expect(draft.state.extendAudioTrackPath).toBe('/audio/mine.mp3');
    });
  });
});
