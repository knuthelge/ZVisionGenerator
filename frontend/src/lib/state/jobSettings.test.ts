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
      enhanceAuto: true, enhanceSettings: { style: 'photo', mood: 'keep', details: ['lighting'], length: 'longer', motion: [] },
    });
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

  it('restores a JSON caption', () => {
    const { patch } = jobSettingsPrefill({ mode: 'image', json_prompt: '{"high_level_description":"x"}' });
    expect(patch).toMatchObject({ jsonPromptEnabled: true, jsonPrompt: '{"high_level_description":"x"}' });
  });
});
