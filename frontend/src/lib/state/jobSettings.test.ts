import { describe, expect, it } from 'vitest';

import { jobSettingsPrefill } from './jobSettings';

describe('jobSettingsPrefill', () => {
  it('splits an inline image job into prefill params and the remaining settings', () => {
    const { params, patch } = jobSettingsPrefill({
      mode: 'image', workflow: 'img2img', model: 'zit', prompt_source: 'inline', prompt: 'a lake', negative_prompt: 'blur',
      ratio: '2:3', size: 'm', steps: '9', guidance: '3.5', seed: '', runs: '4', image_path: '/out/.web_uploads/ref.png', image_strength: '0.6',
      lora: 'style:0.7', quantize: '8', scheduler: '', first_sigma: 'null',
      sharpen_enabled: 'true', sharpen_amount: '0.8', contrast_enabled: 'false', saturation_enabled: 'true', saturation_amount: '1.2',
      upscale: '2', upscale_denoise: '', upscale_steps: '6', upscale_guidance: '', upscale_sharpen: 'false',
      enhance_auto: 'true', enhance_settings: '{"style":"photo","mood":"keep","details":["lighting"],"length":"longer"}',
    });

    expect(params).toEqual({ workflow: 'img2img', model: 'zit', steps: '9', guidance: '3.5', ratio: '2:3', size: 'm', lora: 'style:0.7', image_path: '/out/.web_uploads/ref.png' });
    expect(patch).toMatchObject({
      promptSource: 'inline', prompt: 'a lake', enhancedPrompt: '', negativePrompt: 'blur', seed: null, runs: 4, dimensionMode: 'ratio',
      referenceImageStrength: 0.6, quantize: 8, scheduler: null, firstSigma: null, jsonPromptEnabled: false,
      postprocessSharpenEnabled: true, postprocessSharpenAmount: 0.8, postprocessContrastEnabled: false, postprocessSaturationEnabled: true, postprocessSaturationAmount: 1.2,
      upscaleEnabled: true, upscaleFactor: 2, upscaleDenoise: null, upscaleSteps: 6, upscaleGuidance: null, upscaleSharpen: false,
      enhanceAuto: true, enhanceSettings: { style: 'photo', mood: 'keep', theme: 'keep', camera: 'keep', pace: 'keep', details: ['lighting'], length: 'longer', motion: [] },
    });
  });

  it('keeps a blank LoRA list so loading a job without LoRAs clears them', () => {
    const { params } = jobSettingsPrefill({ mode: 'image', workflow: 'txt2img', model: 'zit', lora: '', steps: '' });

    expect(params).toEqual({ workflow: 'txt2img', model: 'zit', lora: '' });
  });

  it('restores the enhancer and format a job ran with', () => {
    expect(jobSettingsPrefill({ mode: 'image', prompt: 'a', enhance_auto: 'true', enhance_model: 'smart', enhance_format: 'json' }).patch).toMatchObject({ enhanceModel: 'smart', enhanceFormat: 'json' });
    expect(jobSettingsPrefill({ mode: 'image', prompt: 'a' }).patch).toMatchObject({ enhanceModel: 'fast', enhanceFormat: 'text' });
  });

  it('restores auto sharpening when the job sent no sharpen amount', () => {
    const { patch } = jobSettingsPrefill({ mode: 'image', workflow: 'txt2img', prompt: 'a lake', sharpen_enabled: 'true' });

    expect(patch.postprocessSharpenEnabled).toBe(true);
    expect(patch).toHaveProperty('postprocessSharpenAmount', null);
  });

  it('leaves the sharpen amount alone when the job had sharpening off', () => {
    const { patch } = jobSettingsPrefill({ mode: 'image', workflow: 'txt2img', prompt: 'a lake', sharpen_enabled: 'false' });

    expect(patch.postprocessSharpenEnabled).toBe(false);
    expect(patch).not.toHaveProperty('postprocessSharpenAmount');
  });

  it('restores a prompt-file job with several selected prompts and a custom size', () => {
    const { params, patch } = jobSettingsPrefill({
      mode: 'image', workflow: 'txt2img', model: 'zit', prompt_source: 'file', prompts_file: '/p/prompts.yaml',
      prompt_option_id: ['a:0', 'a:1', 'b:0'], width: '1000', height: '600', seed: '42', contrast_enabled: 'false',
    });

    expect(params).toMatchObject({ width: '1000', height: '600', seed: '42' });
    expect(patch).toMatchObject({ promptSource: 'file', promptFilePath: '/p/prompts.yaml', promptFileOptionIds: ['a:0', 'a:1', 'b:0'], dimensionMode: 'custom', seed: 42, upscaleEnabled: false, enhanceAuto: false });
  });

  it('reads video toggles sent with a hidden false fallback', () => {
    const { patch } = jobSettingsPrefill({ mode: 'video', workflow: 'txt2vid', model: 'ltx-8', prompt: 'waves', frames: '49', audio: ['on', 'false'], low_memory: 'false', video_upscale_factor: '2', upscale: '2' });

    expect(patch).toMatchObject({ audio: true, lowMemory: false, videoUpscaleEnabled: true, videoUpscaleFactor: 2 });
    expect(patch).not.toHaveProperty('upscaleEnabled');
  });

  it('restores a video scene and prompt into the boxes', () => {
    const { patch } = jobSettingsPrefill({ mode: 'video', workflow: 'txt2vid', scene: 'A castle.', prompt: 'a knight' });

    expect(patch).toMatchObject({ scene: 'A castle.', prompt: 'a knight', enhancedScene: '', enhancedPrompt: '', enhancedFrom: null });
  });

  it('restores the user text and the Enhanced pair of a video that ran enhanced', () => {
    const { patch } = jobSettingsPrefill({
      mode: 'video', workflow: 'txt2vid', scene: 'A castle at dusk.', prompt: 'A knight rides in.', scene_input: 'A castle.', prompt_input: 'a knight',
    });

    expect(patch).toMatchObject({ scene: 'A castle.', prompt: 'a knight', enhancedScene: 'A castle at dusk.', enhancedPrompt: 'A knight rides in.' });
    expect(patch.enhancedFrom).toEqual({ scene: 'A castle.', prompt: 'a knight', mode: 'video' });
  });

  it('fills only the Enhanced box of a part that ran enhanced, not the one that ran as written', () => {
    const { patch } = jobSettingsPrefill({
      mode: 'video', workflow: 'extend', source_video: '/clips/walk.mp4', scene: 'A street.', prompt: 'She boards the tram.', scene_input: 'A street.', prompt_input: 'She waves.',
    });

    expect(patch).toMatchObject({ scene: 'A street.', prompt: 'She waves.', enhancedScene: '', enhancedPrompt: 'She boards the tram.' });
    expect(patch.enhancedFrom).toMatchObject({ scene: 'A street.', prompt: 'She waves.', mode: 'video', sourceVideoPath: '/clips/walk.mp4' });
  });

  it('leaves both Enhanced boxes empty when neither part differs from its recorded input', () => {
    const { patch } = jobSettingsPrefill({ mode: 'video', workflow: 'txt2vid', scene: 'A castle.', prompt: 'a knight', scene_input: 'A castle.', prompt_input: 'a knight' });

    expect(patch).toMatchObject({ enhancedScene: '', enhancedPrompt: '', enhancedFrom: null });
  });

  it('remembers the Rewrite choice of a queued job for its own workflow', () => {
    const { patch } = jobSettingsPrefill({
      mode: 'video', workflow: 'extend', source_video: '/clips/walk.mp4', prompt: 'She waves.',
      enhance_auto: 'true', enhance_settings: '{"style":"keep","mood":"keep","details":[],"length":"same","motion":[],"parts":"both"}',
    });

    expect(patch.enhanceSettings?.partsByWorkflow).toEqual({ extend: 'both' });
  });

  it('keeps a typed scene when an image job is loaded', () => {
    const { patch } = jobSettingsPrefill({ mode: 'image', workflow: 'txt2img', prompt: 'a fox' });

    expect(patch).not.toHaveProperty('scene');
  });

  it('restores a JSON caption', () => {
    const { patch } = jobSettingsPrefill({ mode: 'image', json_prompt: '{"high_level_description":"x"}' });
    expect(patch).toMatchObject({ jsonPromptEnabled: true, jsonPrompt: '{"high_level_description":"x"}' });
  });

  it('restores the review choice of an upscaled image job', () => {
    expect(jobSettingsPrefill({ mode: 'image', prompt: 'a lake', upscale: '2', review: 'true' }).patch).toMatchObject({ upscaleEnabled: true, upscaleReview: true });
    expect(jobSettingsPrefill({ mode: 'image', prompt: 'a lake', upscale: '2' }).patch.upscaleReview).toBe(false);
    expect(jobSettingsPrefill({ mode: 'image', prompt: 'a lake', review: 'true' }).patch.upscaleReview).toBe(false);
  });

  it('restores the review choice of an upscaled video job', () => {
    const video = { mode: 'video', prompt: 'waves', video_upscale_factor: '2', upscale: '2' };
    expect(jobSettingsPrefill({ ...video, review: 'true' }).patch).toMatchObject({ videoUpscaleEnabled: true, videoUpscaleReview: true });
    expect(jobSettingsPrefill(video).patch.videoUpscaleReview).toBe(false);
    expect(jobSettingsPrefill({ mode: 'video', prompt: 'waves', review: 'true' }).patch.videoUpscaleReview).toBe(false);
    expect(jobSettingsPrefill({ ...video, review: 'true' }).patch.upscaleReview).toBeUndefined();
  });

  it('restores an extension from its source, length and text', () => {
    const { params, patch } = jobSettingsPrefill({
      mode: 'video', workflow: 'extend', model: 'ltx', prompt_source: 'inline', source_video: '/clips/walk.mp4', extend_frames: '48',
      scene: 'A rainy street.', prompt: 'She waves.', steps: '8', seed: '7', audio: 'true', low_memory: 'true',
    });

    expect(params).toEqual({ workflow: 'extend', model: 'ltx', steps: '8', seed: '7', source_video: '/clips/walk.mp4', extend_frames: '48' });
    expect(patch).toMatchObject({ scene: 'A rainy street.', prompt: 'She waves.', seed: 7, audio: true, lowMemory: true, videoUpscaleEnabled: false });
    // The text is the job's own, so inspecting its source must not replace it.
    expect(patch.extendPrefill).toEqual({ path: '/clips/walk.mp4', scene: '', prompt: '' });
  });

  it('keeps the blank scene of an extension that sent none', () => {
    const { patch } = jobSettingsPrefill({ mode: 'video', workflow: 'extend', source_video: '/clips/walk.mp4', scene: '', prompt: '' });

    expect(patch).toMatchObject({ scene: '', prompt: '' });
  });
});
