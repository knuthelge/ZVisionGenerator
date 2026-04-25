import { describe, it, expect, beforeEach } from 'vitest';
import { draft } from './draft.svelte';
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
    ...overrides,
  };
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
    defaults: imgDefaults,
    video_defaults: vidDefaults,
    image_model_defaults: { 'flux-dev': imgDefaults },
    video_model_defaults: { 'ltx-v-0.9': vidDefaults },
    current_image_model: 'flux-dev',
    current_video_model: 'ltx-v-0.9',
    config: { visible_sections: [], theme: 'dark', gallery_page_size: 20 },
    output_dir: '/tmp/output',
    quantize_options: [4, 8],
    image_ratios: ['1:1', '2:3', '16:9'],
    video_ratios: ['16:9', '9:16'],
    image_size_options: { '1:1': ['s', 'm', 'l'], '2:3': ['s', 'm', 'l'], '16:9': ['s', 'm', 'l'] },
    video_size_options: { '16:9': ['s', 'm'], '9:16': ['s', 'm'] },
    scheduler_options: ['euler', 'dpm'],
    workflow_contract: {
      values: ['txt2img', 'img2img', 'txt2vid', 'img2vid'],
      legacy_aliases: {},
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
    draft.loadFromUrl({ workflow: 'img2img', prompt: 'from url', model: 'flux-dev' });
    expect(draft.state.workflow).toBe('img2img');
    expect(draft.state.prompt).toBe('from url');
  });

  it('ignores invalid workflow in URL prefill', () => {
    draft.loadFromUrl({ workflow: 'invalid-workflow' });
    expect(draft.state.workflow).toBe('txt2img'); // unchanged
  });

  describe('loadFromUrl – legacy alias resolution', () => {
    it('resolves legacy "image" alias to txt2img', () => {
      draft.loadFromUrl({ workflow: 'image' });
      expect(draft.state.workflow).toBe('txt2img');
    });

    it('resolves legacy "video" alias to txt2vid', () => {
      draft.loadFromUrl({ workflow: 'video' });
      expect(draft.state.workflow).toBe('txt2vid');
    });

    it('resolves legacy "i2i" alias to img2img', () => {
      draft.loadFromUrl({ workflow: 'i2i' });
      expect(draft.state.workflow).toBe('img2img');
    });

    it('resolves legacy "i2v" alias to img2vid', () => {
      draft.loadFromUrl({ workflow: 'i2v' });
      expect(draft.state.workflow).toBe('img2vid');
    });

    it('applies ratio, size, frames, and image_path URL params', () => {
      draft.loadFromUrl({ ratio: '16:9', size: 'l', frames: '121', image_path: '/tmp/ref.png' });
      expect(draft.state.ratio).toBe('16:9');
      expect(draft.state.size).toBe('l');
      expect(draft.state.frameCount).toBe(121);
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
