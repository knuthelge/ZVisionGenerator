import { describe, expect, it } from 'vitest';

import type { DraftState } from '$lib/types';

import { clipMemoryRequest, extensionMemoryRequest, imageMemoryRequest, type ClipMemoryControls, type ImageMemoryControls } from './memoryRequest';

function draftState(overrides: Partial<DraftState> = {}): DraftState {
  return {
    workflow: 'txt2vid', model: 'ltx-8', dimensionMode: 'ratio', ratio: '16:9', size: 'm', width: 704, height: 448, frameCount: 121, lowMemory: true,
    videoUpscaleEnabled: false, videoUpscaleFactor: 2, videoUpscaleReview: false, loraString: '', ...overrides,
  } as DraftState;
}

const CONTROLS: ClipMemoryControls = { frameRule: { alignment: 8, offset: 1, round: 'nearest' }, lowMemory: true, upscale: true };

describe('clipMemoryRequest', () => {
  it('tells an Image to Video clip from a Text to Video clip and sends the steps only for a model that offers them', () => {
    expect(clipMemoryRequest(draftState({ workflow: 'img2vid', steps: 4 }), { ...CONTROLS, steps: true })).toMatchObject({ image: true, steps: 4 });
    const text = clipMemoryRequest(draftState({ steps: 4 }), CONTROLS);
    expect(text).not.toHaveProperty('image');
    expect(text).not.toHaveProperty('steps');
  });

  it('asks for what the job form submits: the model, a preset size, the frames, low memory and upscale', () => {
    expect(clipMemoryRequest(draftState(), CONTROLS)).toEqual({ media: 'video', kind: 'clip', model: 'ltx-8', ratio: '16:9', size: 'm', frames: 121, low_memory: true, upscale: null });
  });

  it('asks for the width and height of a custom size, and not for its ratio and size', () => {
    const request = clipMemoryRequest(draftState({ dimensionMode: 'custom', width: 640, height: 384 }), CONTROLS);

    expect(request).toEqual({ media: 'video', kind: 'clip', model: 'ltx-8', width: 640, height: 384, frames: 121, low_memory: true, upscale: null });
    expect(request).not.toHaveProperty('ratio');
  });

  it('has nothing to ask for a custom size that is not complete', () => {
    expect(clipMemoryRequest(draftState({ dimensionMode: 'custom', width: 640, height: 0 }), CONTROLS)).toBeNull();
    expect(clipMemoryRequest(draftState({ dimensionMode: 'custom', width: 0, height: 384 }), CONTROLS)).toBeNull();
  });

  it('sends the frame count the job would use', () => {
    expect(clipMemoryRequest(draftState({ frameCount: 100 }), CONTROLS)?.frames).toBe(97);
    expect(clipMemoryRequest(draftState({ frameCount: 100 }), { ...CONTROLS, frameRule: { alignment: 1, offset: 1, round: 'nearest' } })?.frames).toBe(100);
    expect(clipMemoryRequest(draftState({ frameCount: 125 }), { ...CONTROLS, frameRule: { alignment: 17, offset: 5, round: 'up' } })?.frames).toBe(141);
  });

  it('sends the upscale factor only while the toggle is on and offered', () => {
    expect(clipMemoryRequest(draftState({ videoUpscaleEnabled: true }), CONTROLS)?.upscale).toBe(2);
    expect(clipMemoryRequest(draftState({ videoUpscaleEnabled: true }), { ...CONTROLS, upscale: false })?.upscale).toBeNull();
  });

  it('leaves Low memory on when the model does not offer the toggle', () => {
    expect(clipMemoryRequest(draftState({ lowMemory: false }), CONTROLS)?.low_memory).toBe(false);
    expect(clipMemoryRequest(draftState({ lowMemory: false }), { ...CONTROLS, lowMemory: false })?.low_memory).toBe(true);
  });

  it('sends the selected LoRAs as the job form submits them, and none when there are none', () => {
    expect(clipMemoryRequest(draftState({ loraString: 'detail:0.8,style:1' }), CONTROLS)?.lora).toBe('detail:0.8,style:1');
    expect(clipMemoryRequest(draftState({ loraString: '  ' }), CONTROLS)).not.toHaveProperty('lora');
    expect(clipMemoryRequest(draftState({ loraString: '' }), CONTROLS)).not.toHaveProperty('lora');
  });

  it('sends how many references the job holds, and nothing when there are none', () => {
    expect(clipMemoryRequest(draftState(), { ...CONTROLS, references: 2 })?.references).toBe(2);
    expect(clipMemoryRequest(draftState(), { ...CONTROLS, references: 0 })).not.toHaveProperty('references');
    expect(clipMemoryRequest(draftState(), CONTROLS)).not.toHaveProperty('references');
  });

  it('sends Review only while an upscale is on and the model offers the toggle', () => {
    const reviewing = { videoUpscaleEnabled: true, videoUpscaleReview: true };
    const controls = { ...CONTROLS, review: true };
    expect(clipMemoryRequest(draftState(reviewing), controls)?.review).toBe(true);
    expect(clipMemoryRequest(draftState({ ...reviewing, videoUpscaleReview: false }), controls)).not.toHaveProperty('review');
    expect(clipMemoryRequest(draftState({ ...reviewing, videoUpscaleEnabled: false }), controls)).not.toHaveProperty('review');
    expect(clipMemoryRequest(draftState(reviewing), CONTROLS)).not.toHaveProperty('review');
    expect(clipMemoryRequest(draftState(reviewing), { ...controls, upscale: false })).not.toHaveProperty('review');
  });

  it('checks Image to Video like Text to Video', () => {
    expect(clipMemoryRequest(draftState({ workflow: 'img2vid' }), CONTROLS)).not.toBeNull();
  });

  it.each(['extend', 'txt2img', 'img2img'] as const)('does not check %s as a clip', (workflow) => {
    expect(clipMemoryRequest(draftState({ workflow }), CONTROLS)).toBeNull();
  });

  it('has nothing to ask without a model or a length', () => {
    expect(clipMemoryRequest(draftState({ model: '' }), CONTROLS)).toBeNull();
    expect(clipMemoryRequest(draftState({ frameCount: 0 }), CONTROLS)).toBeNull();
  });
});

describe('imageMemoryRequest', () => {
  const NO_EDIT: ImageMemoryControls = { quantize: true, referenceSizes: [], referencePaths: [], upscale: false, edit: null };
  const EDIT = { mode: 'paint', width: 1024, height: 768, painted: false, extension: null } as const;

  function imageState(overrides: Partial<DraftState> = {}): DraftState {
    return { workflow: 'txt2img', model: 'z-image', dimensionMode: 'ratio', ratio: '2:3', size: 'm', width: 832, height: 1216, quantize: null, loraString: '', ...overrides } as DraftState;
  }

  it('asks for a preset size with the ratio and the size names', () => {
    expect(imageMemoryRequest(imageState(), NO_EDIT)).toEqual({ media: 'image', kind: 'image', model: 'z-image', ratio: '2:3', size: 'm' });
  });

  it('sends the upscale factor while Upscale is on and offered, and nothing otherwise', () => {
    expect(imageMemoryRequest(imageState({ upscaleEnabled: true, upscaleFactor: 4 }), { ...NO_EDIT, upscale: true })).toMatchObject({ upscale: 4 });
    expect(imageMemoryRequest(imageState({ upscaleEnabled: false, upscaleFactor: 4 }), { ...NO_EDIT, upscale: true })).not.toHaveProperty('upscale');
    expect(imageMemoryRequest(imageState({ upscaleEnabled: true, upscaleFactor: 4 }), NO_EDIT)).not.toHaveProperty('upscale');
  });

  it('sends image when the job starts from a start image (Image to Image), and not for Text to Image', () => {
    expect(imageMemoryRequest(imageState({ workflow: 'img2img' }), NO_EDIT)).toEqual({ media: 'image', kind: 'image', model: 'z-image', ratio: '2:3', size: 'm', image: true });
    expect(imageMemoryRequest(imageState({ workflow: 'txt2img' }), NO_EDIT)).not.toHaveProperty('image');
  });

  it('sends neither for an edit, whose mask and extension say what it starts from', () => {
    const request = imageMemoryRequest(imageState({ workflow: 'edit', upscaleEnabled: true }), { ...NO_EDIT, upscale: true, edit: { ...EDIT } as ImageMemoryControls['edit'] });

    expect(request).not.toHaveProperty('upscale');
    expect(request).not.toHaveProperty('image');
  });

  it('asks for a custom size in pixels, and nothing while it is incomplete', () => {
    expect(imageMemoryRequest(imageState({ dimensionMode: 'custom' }), NO_EDIT)).toEqual({ media: 'image', kind: 'image', model: 'z-image', width: 832, height: 1216 });
    expect(imageMemoryRequest(imageState({ dimensionMode: 'custom', width: 0 }), NO_EDIT)).toBeNull();
    expect(imageMemoryRequest(imageState({ dimensionMode: 'custom', height: 0 }), NO_EDIT)).toBeNull();
  });

  it('sends the measured size of each uploaded reference and the path of each file on the server, and only when there are some', () => {
    const request = imageMemoryRequest(imageState(), { ...NO_EDIT, referenceSizes: [{ width: 512, height: 512 }, { width: 2000, height: 1500 }], referencePaths: ['/in/a.png'] });

    expect(request).toMatchObject({ reference_sizes: [[512, 512], [2000, 1500]], reference_paths: ['/in/a.png'] });
    expect(request).not.toHaveProperty('references');
    expect(imageMemoryRequest(imageState(), NO_EDIT)).not.toHaveProperty('reference_sizes');
    expect(imageMemoryRequest(imageState(), NO_EDIT)).not.toHaveProperty('reference_paths');
  });

  it('sends the quantization only where the model offers it and one is chosen', () => {
    expect(imageMemoryRequest(imageState({ quantize: 8 }), NO_EDIT)).toMatchObject({ quantize: 8 });
    expect(imageMemoryRequest(imageState({ quantize: 8 }), { ...NO_EDIT, quantize: false })).not.toHaveProperty('quantize');
    expect(imageMemoryRequest(imageState({ quantize: null }), NO_EDIT)).not.toHaveProperty('quantize');
  });

  it('has nothing to check without a model or for another workflow', () => {
    expect(imageMemoryRequest(imageState({ model: '' }), NO_EDIT)).toBeNull();
    for (const workflow of ['txt2vid', 'img2vid', 'extend', 'speech'] as const) expect(imageMemoryRequest(imageState({ workflow }), NO_EDIT)).toBeNull();
  });

  describe('an edit', () => {
    const edit = (overrides: Partial<typeof EDIT> | Record<string, unknown> = {}) => imageMemoryRequest(imageState({ workflow: 'edit' }), { ...NO_EDIT, edit: { ...EDIT, ...overrides } as ImageMemoryControls['edit'] });

    it('sends the picture size of a whole-picture edit and no size or references of its own', () => {
      expect(edit()).toEqual({ media: 'image', kind: 'edit', model: 'z-image', source_width: 1024, source_height: 768 });
    });

    it('sends masked for a painted mask in Paint', () => {
      expect(edit({ painted: true })).toEqual({ media: 'image', kind: 'edit', model: 'z-image', source_width: 1024, source_height: 768, masked: true });
    });

    it('sends the extension as side=amount in Extend', () => {
      expect(edit({ mode: 'extend', extension: { side: 'right', amount: 256 } })).toMatchObject({ extend: 'right=256' });
      expect(edit({ mode: 'extend', extension: null })).not.toHaveProperty('extend');
    });

    it('sends no extension in Paint with a side left over from Extend', () => {
      expect(edit({ mode: 'paint', extension: { side: 'right', amount: 256 } })).not.toHaveProperty('extend');
    });

    it('sends no mask in Extend with a mask left over from Paint', () => {
      expect(edit({ mode: 'extend', painted: true, extension: { side: 'top', amount: 128 } })).not.toHaveProperty('masked');
    });

    it('has nothing to check while the picture size is unknown or without the open edit', () => {
      expect(edit({ width: null })).toBeNull();
      expect(edit({ height: null })).toBeNull();
      expect(imageMemoryRequest(imageState({ workflow: 'edit' }), NO_EDIT)).toBeNull();
    });
  });
});

describe('imageMemoryRequest LoRAs', () => {
  const NO_EDIT: ImageMemoryControls = { quantize: true, referenceSizes: [], referencePaths: [], upscale: false, edit: null };

  it('sends the selected LoRAs of an image or an edit as the job form submits them, and none when there are none', () => {
    const state = { workflow: 'txt2img', model: 'z-image', dimensionMode: 'ratio', ratio: '2:3', size: 'm', width: 832, height: 1216, quantize: null } as DraftState;

    expect(imageMemoryRequest({ ...state, loraString: 'detail:0.8' }, NO_EDIT)?.lora).toBe('detail:0.8');
    expect(imageMemoryRequest({ ...state, loraString: ' ' }, NO_EDIT)).not.toHaveProperty('lora');
    const edit = { mode: 'paint', width: 1024, height: 768, painted: false, extension: null } as const;
    expect(imageMemoryRequest({ ...state, workflow: 'edit', loraString: 'detail:0.8' }, { ...NO_EDIT, edit }, )?.lora).toBe('detail:0.8');
  });
});

describe('extensionMemoryRequest', () => {
  const extend = (overrides: Partial<DraftState> = {}) => draftState({ workflow: 'extend', extendSourcePath: '/clips/walk.mp4', extendFrames: 48, steps: 4, ...overrides });
  const CONTROLS_EXTEND = { lowMemory: true, steps: false, references: 0 };

  it('asks for the source, the length, the model and Low memory', () => {
    expect(extensionMemoryRequest(extend(), CONTROLS_EXTEND)).toEqual({ media: 'video', kind: 'extension', model: 'ltx-8', source: '/clips/walk.mp4', low_memory: true, extend_frames: 48 });
  });

  it('leaves the length out until one is set, so the server\'s default applies', () => {
    expect(extensionMemoryRequest(extend({ extendFrames: null }), CONTROLS_EXTEND)).not.toHaveProperty('extend_frames');
  });

  it('sends the steps, LoRAs and references only where the job does', () => {
    expect(extensionMemoryRequest(extend({ loraString: 'detail:0.8' }), { ...CONTROLS_EXTEND, steps: true, references: 2 })).toMatchObject({ steps: 4, lora: 'detail:0.8', references: 2 });
    const plain = extensionMemoryRequest(extend({ loraString: '' }), CONTROLS_EXTEND);
    expect(plain).not.toHaveProperty('steps');
    expect(plain).not.toHaveProperty('lora');
    expect(plain).not.toHaveProperty('references');
  });

  it('keeps Low memory on when the model does not offer the switch', () => {
    expect(extensionMemoryRequest(extend({ lowMemory: false }), { ...CONTROLS_EXTEND, lowMemory: false })?.low_memory).toBe(true);
    expect(extensionMemoryRequest(extend({ lowMemory: false }), CONTROLS_EXTEND)?.low_memory).toBe(false);
  });

  it('has nothing to ask without a source or a model, or for another workflow', () => {
    expect(extensionMemoryRequest(extend({ extendSourcePath: null }), CONTROLS_EXTEND)).toBeNull();
    expect(extensionMemoryRequest(extend({ model: '' }), CONTROLS_EXTEND)).toBeNull();
    expect(extensionMemoryRequest(draftState({ extendSourcePath: '/clips/walk.mp4' }), CONTROLS_EXTEND)).toBeNull();
  });
});
