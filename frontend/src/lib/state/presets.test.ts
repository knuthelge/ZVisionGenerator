import { beforeEach, describe, expect, it } from 'vitest';

import type { DraftState, PresetSettings, SettingsPreset, WorkspaceContext } from '$lib/types';
import { draft } from './draft.svelte';
import {
  appliedPreset,
  presetDiffers,
  presetDraftKeys,
  presetPrefill,
  presetSettingsFromDraft,
  PRESET_KEYS,
} from './presets';

// Only what the helpers read: the model lists and the installed LoRAs.
const ctx = {
  image_models: [{ id: 'zit', label: 'zit', type: 'image' }, { id: 'flux-lite', label: 'flux-lite', type: 'image' }],
  video_models: [{ id: 'ltx-8', label: 'ltx-8', type: 'video' }, { id: 'ltx-2', label: 'ltx-2', type: 'video' }],
  loras: [
    { name: 'style-a', path: '/loras/style-a.safetensors' },
    { name: 'detail-b', path: '/loras/detail-b.safetensors' },
  ],
} as unknown as WorkspaceContext;

function imageState(overrides: Partial<DraftState> = {}): DraftState {
  draft.reset();
  return {
    ...draft.state,
    workflow: 'txt2img', model: 'zit', ratio: '2:3', size: 'm', dimensionMode: 'ratio', width: 832, height: 1216, steps: 9, guidance: 0,
    quantize: 8, loraString: 'style-a:0.8', scheduler: null, firstSigma: null,
    postprocessSharpenEnabled: true, postprocessSharpenAmount: null, postprocessContrastEnabled: false, postprocessContrastAmount: 1,
    postprocessSaturationEnabled: false, postprocessSaturationAmount: 1,
    upscaleEnabled: true, upscaleFactor: 2, upscaleDenoise: null, upscaleSteps: null, upscaleGuidance: null, upscaleSharpen: true, upscaleReview: false,
    ...overrides,
  };
}

function videoState(overrides: Partial<DraftState> = {}): DraftState {
  draft.reset();
  return {
    ...draft.state,
    workflow: 'txt2vid', model: 'ltx-8', ratio: '16:9', size: 'm', dimensionMode: 'ratio', width: 704, height: 448, steps: 8, frameCount: 97,
    audio: true, lowMemory: true, videoUpscaleEnabled: false, videoUpscaleFactor: 2, videoUpscaleReview: false, loraString: '',
    ...overrides,
  };
}

function presetOf(state: DraftState, overrides: Partial<SettingsPreset> = {}, settings: PresetSettings = {}): SettingsPreset {
  const snapshot = presetSettingsFromDraft(state, ctx);
  return {
    slug: 'soft-portrait', name: 'Soft portrait', mode: state.workflow === 'txt2vid' || state.workflow === 'extend' ? 'video' : 'image',
    model: snapshot.model, ...overrides, settings: { ...snapshot.settings, ...settings },
  };
}

beforeEach(() => {
  localStorage.clear();
});

describe('presetSettingsFromDraft', () => {
  it('holds exactly the keys of the mode, with the model on top', () => {
    const image = presetSettingsFromDraft(imageState(), ctx);
    expect(Object.keys(image.settings)).toEqual([...PRESET_KEYS.image]);
    expect(image.model).toBe('zit');
    expect(Object.keys(presetSettingsFromDraft(videoState(), ctx).settings)).toEqual([...PRESET_KEYS.video]);
  });

  it('reads post-processing amounts only while the effect is on, and upscale and review only while upscale is on', () => {
    const on = presetSettingsFromDraft(imageState({ postprocessSharpenAmount: 0.6, postprocessContrastEnabled: true, postprocessContrastAmount: 1.3, upscaleReview: true }), ctx).settings;
    expect(on).toMatchObject({ sharpen_enabled: true, sharpen_amount: 0.6, contrast_enabled: true, contrast_amount: 1.3, saturation_enabled: false, saturation_amount: null, upscale: 2, review: true });
    const off = presetSettingsFromDraft(imageState({ postprocessSharpenEnabled: false, postprocessSharpenAmount: 0.6, upscaleEnabled: false, upscaleReview: true }), ctx).settings;
    expect(off).toMatchObject({ sharpen_enabled: false, sharpen_amount: null, upscale: null, review: null });
  });

  it('stores null for the options and review of an upscale that is off, image and video', () => {
    const image = presetSettingsFromDraft(imageState({ upscaleEnabled: false, upscaleDenoise: 0.3, upscaleSteps: 12, upscaleGuidance: 1.5, upscaleSharpen: false, upscaleReview: true }), ctx).settings;
    expect(image).toMatchObject({ upscale: null, upscale_denoise: null, upscale_steps: null, upscale_guidance: null, upscale_sharpen: null, review: null });
    const video = presetSettingsFromDraft(videoState({ videoUpscaleEnabled: false, videoUpscaleReview: true }), ctx).settings;
    expect(video).toMatchObject({ video_upscale_factor: null, review: null });
  });

  it('writes a custom size as custom with its width and height, and a preset size with neither', () => {
    expect(presetSettingsFromDraft(imageState({ dimensionMode: 'custom', width: 640, height: 960 }), ctx).settings).toMatchObject({ size: 'custom', width: 640, height: 960 });
    expect(presetSettingsFromDraft(imageState(), ctx).settings).toMatchObject({ ratio: '2:3', size: 'm', width: null, height: null });
  });

  it('writes the LoRAs by installed name, weights kept', () => {
    expect(presetSettingsFromDraft(imageState({ loraString: '/loras/style-a.safetensors:0.5, detail-b' }), ctx).settings.lora).toBe('style-a:0.5,detail-b:1');
  });

  it('reads video frames, toggles and the upscale factor', () => {
    expect(presetSettingsFromDraft(videoState({ audio: false, videoUpscaleEnabled: true, videoUpscaleFactor: 2, videoUpscaleReview: true }), ctx).settings).toEqual({
      lora: '', ratio: '16:9', size: 'm', width: null, height: null, runs: 1, steps: 8, frames: 97, audio: false, low_memory: true, video_upscale_factor: 2, review: true,
    });
  });
});

describe('presetDiffers', () => {
  it('is false for the state a preset was read from, and true after a setting moves', () => {
    const state = imageState();
    const preset = presetOf(state);
    expect(presetDiffers(preset, state, ctx)).toBe(false);
    expect(presetDiffers(preset, { ...state, steps: 12 }, ctx)).toBe(true);
    expect(presetDiffers(preset, { ...state, runs: 4 }, ctx)).toBe(true);
    expect(presetDiffers(preset, { ...state, referenceImageStrength: 0.5 }, ctx)).toBe(true);
    expect(presetDiffers(preset, { ...state, model: 'flux-lite' }, ctx)).toBe(true);
    expect(presetDiffers(preset, { ...state, postprocessSharpenAmount: 0.5 }, ctx)).toBe(true);
    expect(presetDiffers(preset, { ...state, upscaleEnabled: false }, ctx)).toBe(true);
  });

  it('is false right after applying a preset saved with upscale off, whatever its hidden upscale options hold', () => {
    const preset = presetOf(imageState({ upscaleEnabled: false, upscaleDenoise: 0.3, upscaleReview: true }));
    const { patch } = presetPrefill(preset, imageState(), ctx);
    expect(presetDiffers(preset, { ...imageState(), ...patch }, ctx)).toBe(false);
    expect(presetDiffers(preset, { ...imageState(), ...patch, upscaleDenoise: 0.5 }, ctx)).toBe(false);
    expect(presetDiffers(preset, { ...imageState(), ...patch, upscaleEnabled: true }, ctx)).toBe(true);
  });

  it('ignores the model of a preset that names none', () => {
    const state = imageState();
    expect(presetDiffers(presetOf(state, { model: null }), { ...state, model: 'flux-lite' }, ctx)).toBe(false);
  });

  it('reads LoRA lists as lists: a missing weight is the weight 1', () => {
    const state = imageState({ loraString: 'style-a:1' });
    const preset = presetOf(state, {}, { lora: 'style-a' });
    expect(presetDiffers(preset, state, ctx)).toBe(false);
    expect(presetDiffers(preset, { ...state, loraString: 'style-a:0.5' }, ctx)).toBe(true);
    expect(presetDiffers(preset, { ...state, loraString: '' }, ctx)).toBe(true);
  });

  it('compares the size keys as one unit', () => {
    const state = imageState();
    const preset = presetOf(state);
    // The pixel size of a preset size is the model's business; width and height only count for a custom size.
    expect(presetDiffers(preset, { ...state, width: 1, height: 2 }, ctx)).toBe(false);
    expect(presetDiffers(preset, { ...state, size: 'l' }, ctx)).toBe(true);
    const custom = imageState({ dimensionMode: 'custom', width: 640, height: 960 });
    const customPreset = presetOf(custom);
    expect(presetDiffers(customPreset, custom, ctx)).toBe(false);
    expect(presetDiffers(customPreset, { ...custom, height: 944 }, ctx)).toBe(true);
    expect(presetDiffers(customPreset, { ...custom, dimensionMode: 'ratio' }, ctx)).toBe(true);
  });

  it('compares every key in an extension too', () => {
    const extend = videoState({ workflow: 'extend' });
    const preset = presetOf(extend);
    expect(presetDiffers(preset, extend, ctx)).toBe(false);
    expect(presetDiffers(preset, { ...extend, model: 'ltx-2' }, ctx)).toBe(true);
    expect(presetDiffers(preset, { ...extend, frameCount: 49 }, ctx)).toBe(true);
    expect(presetDiffers(preset, { ...extend, loraString: 'style-a:1' }, ctx)).toBe(true);
  });

  it('is true for a preset of the other mode', () => {
    expect(presetDiffers(presetOf(videoState()), imageState(), ctx)).toBe(true);
  });
});

describe('appliedPreset', () => {
  it('finds the applied preset when it is of the current mode', () => {
    const image = presetOf(imageState());
    const video = { ...presetOf(videoState()), slug: 'clip', name: 'Clip' };
    expect(appliedPreset([image, video], { workflow: 'txt2img', appliedPreset: 'soft-portrait' })).toBe(image);
    expect(appliedPreset([image, video], { workflow: 'txt2vid', appliedPreset: 'clip' })).toBe(video);
    expect(appliedPreset([image, video], { workflow: 'txt2vid', appliedPreset: 'soft-portrait' })).toBeNull();
    expect(appliedPreset([video], { workflow: 'txt2img', appliedPreset: 'soft-portrait' })).toBeNull();
    expect(appliedPreset([image], { workflow: 'txt2img', appliedPreset: null })).toBeNull();
  });

  it('matches by slug and mode when both modes hold the same slug', () => {
    const image = presetOf(imageState());
    const video = { ...presetOf(videoState()), slug: image.slug };
    expect(appliedPreset([image, video], { workflow: 'txt2img', appliedPreset: image.slug })).toBe(image);
    expect(appliedPreset([image, video], { workflow: 'extend', appliedPreset: image.slug })).toBe(video);
  });
});

describe('presetPrefill', () => {
  it('gives the model, the reuse params and the patch for every key', () => {
    const preset = presetOf(imageState(), { model: 'flux-lite' }, {
      quantize: 4, lora: 'style-a:0.8,detail-b:0.5', ratio: '16:9', size: 'l', runs: 3, steps: 17, guidance: 2.5, image_strength: 0.6, scheduler: 'dpm', first_sigma: 0.4,
      sharpen_enabled: true, sharpen_amount: 0.6, contrast_enabled: true, contrast_amount: 1.3, saturation_enabled: false,
      upscale: 4, upscale_denoise: 0.3, upscale_steps: 12, upscale_guidance: 1.5, upscale_sharpen: false, review: true,
    });
    const result = presetPrefill(preset, imageState(), ctx);

    expect(result.model).toBe('flux-lite');
    expect(result.params).toEqual({ lora: 'style-a:0.8,detail-b:0.5', ratio: '16:9', size: 'l', steps: '17', quantize: '4', guidance: '2.5' });
    expect(result.patch).toEqual({
      runs: 3, referenceImageStrength: 0.6, scheduler: 'dpm', firstSigma: 0.4,
      postprocessSharpenEnabled: true, postprocessSharpenAmount: 0.6, postprocessContrastEnabled: true, postprocessContrastAmount: 1.3, postprocessSaturationEnabled: false,
      upscaleEnabled: true, upscaleFactor: 4, upscaleReview: true, upscaleDenoise: 0.3, upscaleSteps: 12, upscaleGuidance: 1.5, upscaleSharpen: false,
    });
  });

  it('sends a blank LoRA list and quantize when the preset has none, so remembered ones do not come back', () => {
    const preset = presetOf(imageState({ loraString: '', quantize: null, upscaleEnabled: false }));
    const { params, patch } = presetPrefill(preset, imageState(), ctx);
    expect(params.lora).toBe('');
    expect(params.quantize).toBe('');
    expect(patch).toMatchObject({ upscaleEnabled: false, upscaleReview: false });
    expect(patch).not.toHaveProperty('upscaleFactor');
  });

  it('drops LoRAs that are not installed', () => {
    const preset = presetOf(imageState(), {}, { lora: 'style-a:0.8,gone-one:1,gone-two:1' });
    const result = presetPrefill(preset, imageState(), ctx);
    expect(result.params.lora).toBe('style-a:0.8');
  });

  it('keeps the current model when the preset names one that is not offered, or none', () => {
    const state = imageState({ model: 'flux-lite' });
    expect(presetPrefill(presetOf(imageState(), { model: 'removed-model' }), state, ctx).model).toBe('flux-lite');
    expect(presetPrefill(presetOf(imageState(), { model: null }), state, ctx).model).toBe('flux-lite');
  });

  it('turns a custom size into size custom with its width and height', () => {
    const preset = presetOf(imageState({ dimensionMode: 'custom', width: 640, height: 960 }));
    expect(presetPrefill(preset, imageState(), ctx).params).toMatchObject({ size: 'custom', width: '640', height: '960' });
  });

  it('maps the video keys, with the upscale factor and review as one choice', () => {
    const preset = presetOf(videoState(), { model: 'ltx-2' }, { frames: 49, audio: false, low_memory: false, video_upscale_factor: 2, review: true, steps: 6 });
    const result = presetPrefill(preset, videoState(), ctx);
    expect(result).toMatchObject({ model: 'ltx-2', params: { frames: '49', steps: '6', lora: '', ratio: '16:9', size: 'm' } });
    expect(result.patch).toEqual({ runs: 1, audio: false, lowMemory: false, videoUpscaleEnabled: true, videoUpscaleFactor: 2, videoUpscaleReview: true });
    expect(presetPrefill({ ...preset, settings: { ...preset.settings, video_upscale_factor: null } }, videoState(), ctx).patch).toMatchObject({ videoUpscaleEnabled: false, videoUpscaleReview: false });
  });

  it('applies every key in an extension, model and LoRAs included', () => {
    const extend = videoState({ workflow: 'extend', model: 'ltx-2' });
    const preset = presetOf(videoState(), { model: 'ltx-8' }, { lora: 'style-a:1', frames: 49, steps: 6, audio: false, video_upscale_factor: 2, review: true, ratio: '9:16', size: 's' });
    const result = presetPrefill(preset, extend, ctx);
    expect(result.model).toBe('ltx-8');
    expect(result.params).toEqual({ lora: 'style-a:1', ratio: '9:16', size: 's', steps: '6', frames: '49' });
    expect(result.patch).toEqual({ runs: 1, audio: false, lowMemory: true, videoUpscaleEnabled: true, videoUpscaleFactor: 2, videoUpscaleReview: true });
  });
});

describe('presetDraftKeys', () => {
  it('lists what an apply may write, the same in every workflow', () => {
    const image = presetDraftKeys(imageState());
    for (const owned of ['model', 'loraString', 'lorasByFamily', 'dimensionMode', 'steps', 'runs', 'referenceImageStrength', 'upscaleEnabled']) expect(image).toContain(owned);
    for (const kept of ['prompt', 'seed', 'workflow', 'referenceImagePath', 'videoUpscaleEnabled', 'frameCount']) expect(image).not.toContain(kept);
    const extend = presetDraftKeys(videoState({ workflow: 'extend' }));
    expect(extend).toEqual(presetDraftKeys(videoState()));
    for (const owned of ['model', 'loraString', 'ratio', 'size', 'frameCount', 'videoUpscaleEnabled', 'videoUpscaleReview', 'audio', 'runs']) expect(extend).toContain(owned);
    for (const kept of ['prompt', 'scene', 'extendSourcePath', 'extendFrames']) expect(extend).not.toContain(kept);
  });
});
