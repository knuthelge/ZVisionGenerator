import { describe, expect, it } from 'vitest';

import settingsCases from '../../../../tests/fixtures/audio_settings_cases.json';
import { hasJobSettings, jobSettingsPrefill, outputSettingsPatch } from './jobSettings';
import type { JobSettings } from '$lib/types';

describe('jobSettingsPrefill', () => {
  it('loads an edit job back into Edit: the image, the extension and the paint to do again, never the mask', () => {
    const settings = {
      mode: 'image', workflow: 'edit', edit_source_path: '/out/a.png', edit_mask_file: 'mask.png', edit_painted: 'true', edit_extend_side: 'top', edit_extend_amount: '256',
      prompt: 'a red door', model: 'klein', steps: '4', seed: '9', runs: '2', lora: 'detail:0.8', quantize: '8',
    };
    const { params, patch } = jobSettingsPrefill(settings);

    expect(params).toMatchObject({
      workflow: 'edit', edit_source_path: '/out/a.png', edit_repaint: 'true', edit_extend_side: 'top', edit_extend_amount: '256', model: 'klein', steps: '4', seed: '9', lora: 'detail:0.8',
    });
    expect(Object.keys(params)).not.toContain('edit_mask_file');
    expect(patch).toMatchObject({ prompt: 'a red door', runs: 2, quantize: 8, seed: 9 });
  });

  it('asks for no repaint from an edit that painted nothing, and names no image for an uploaded one', () => {
    const { params } = jobSettingsPrefill({ mode: 'image', workflow: 'edit', edit_source_file: 'photo.png', prompt: 'x', model: 'klein' });
    expect(params).not.toHaveProperty('edit_repaint');
    expect(params).not.toHaveProperty('edit_source_path');
  });

  it('loads an audio job into the Audio tab instead of reading it as an image job', () => {
    expect(jobSettingsPrefill({ mode: 'audio', workflow: 'music', prompt: 'a calm piano', lyrics: '[verse]\nla', model: 'yue2', steps: '9' })).toEqual({
      params: { workflow: 'music', prompt: 'a calm piano', lyrics: '[verse]\nla', model: 'yue2' },
      patch: {},
    });
    expect(jobSettingsPrefill({ mode: 'image', prompt: 'a calm piano' }).patch).toMatchObject({ prompt: 'a calm piano' });
  });

  it.each(settingsCases.cases.map((item) => [item.name, item] as const))('loads the settings of an audio job as a Reuse of the same take does: %s', (_name, item) => {
    const { voice_consent: _consent, ...form } = item.form as unknown as JobSettings;

    expect(jobSettingsPrefill(form).params).toEqual(item.retry);
  });

  it('reads a job as Enhanced when it sent an enhance record, not when its own text is non-empty', () => {
    const base = { mode: 'audio', workflow: 'sound', model: 'sa3', prompt: 'Heavy rain on a tin roof.' };
    const record = '{"settings":{},"model":"fast","tier":"fast","format":"text","task":"sound"}';

    expect(jobSettingsPrefill({ ...base, prompt_input: 'rain', enhance_record: record }).params).toMatchObject({ prompt: 'rain', enhanced_prompt: 'Heavy rain on a tin roof.', enhance: record });
    expect(jobSettingsPrefill({ ...base, prompt_input: '', enhance_record: record }).params).toMatchObject({ prompt: '', enhanced_prompt: 'Heavy rain on a tin roof.' });
    for (const enhance_record of [undefined, '', '{}', 'null', 'not json']) {
      const { params } = jobSettingsPrefill({ ...base, prompt_input: 'rain', ...(enhance_record === undefined ? {} : { enhance_record }) });
      expect(params.prompt).toBe('Heavy rain on a tin roof.');
      expect(params).not.toHaveProperty('enhanced_prompt');
      expect(params).not.toHaveProperty('enhance');
    }
  });

  it('loads the built-in voice of a speech job, and a recording when it followed one', () => {
    expect(jobSettingsPrefill({ mode: 'audio', workflow: 'speech', model: 'qwen3_tts', prompt: 'Hello', speaker: 'vivian', runs: '1' }).params).toEqual({ workflow: 'speech', model: 'qwen3_tts', prompt: 'Hello', speaker: 'vivian', runs: '1' });
    expect(jobSettingsPrefill({ mode: 'audio', workflow: 'speech', model: 'qwen3_tts', prompt: 'Hello', voice_path: '/v/me.wav', voice_text: 'Hi', voice_consent: 'on' }).params).toEqual({
      workflow: 'speech', model: 'qwen3_tts', prompt: 'Hello', voice_path: '/v/me.wav', voice_text: 'Hi',
    });
  });

  it('loads the saved voice of a speech job as the slug alone', () => {
    expect(jobSettingsPrefill({ mode: 'audio', workflow: 'speech', model: 'qwen3_tts', prompt: 'Hello', saved_voice: 'narrator', runs: '1' }).params).toEqual({
      workflow: 'speech', model: 'qwen3_tts', prompt: 'Hello', saved_voice: 'narrator', runs: '1',
    });
  });

  it('splits an inline image job into prefill params and the remaining settings', () => {
    const { params, patch } = jobSettingsPrefill({
      mode: 'image', workflow: 'img2img', model: 'zit', prompt_source: 'inline', prompt: 'a lake', negative_prompt: 'blur',
      ratio: '2:3', size: 'm', steps: '9', guidance: '3.5', seed: '', runs: '4', image_path: '/out/.web_uploads/ref.png', image_strength: '0.6',
      lora: 'style:0.7', quantize: '8', scheduler: '', first_sigma: 'null',
      sharpen_enabled: 'true', sharpen_amount: '0.8', contrast_enabled: 'false', saturation_enabled: 'true', saturation_amount: '1.2',
      upscale: '2', upscale_denoise: '', upscale_steps: '6', upscale_guidance: '', upscale_sharpen: 'false',
      enhance_auto: 'true', enhance_settings: '{"style":"photo","mood":"keep","details":["lighting"],"length":"longer"}',
    });

    expect(params).toEqual({ workflow: 'img2img', model: 'zit', steps: '9', guidance: '3.5', ratio: '2:3', size: 'm', lora: 'style:0.7', image_path: '/out/.web_uploads/ref.png', reference_path: [] });
    expect(patch).toMatchObject({
      promptSource: 'inline', prompt: 'a lake', enhancedPrompt: '', negativePrompt: 'blur', seed: null, runs: 4, dimensionMode: 'ratio',
      referenceImageStrength: 0.6, quantize: 8, scheduler: null, firstSigma: null, jsonPromptEnabled: false,
      postprocessSharpenEnabled: true, postprocessSharpenAmount: 0.8, postprocessContrastEnabled: false, postprocessSaturationEnabled: true, postprocessSaturationAmount: 1.2,
      upscaleEnabled: true, upscaleFactor: 2, upscaleDenoise: null, upscaleSteps: 6, upscaleGuidance: null, upscaleSharpen: false,
      enhanceAuto: true, enhanceSettings: { style: 'photo', mood: 'keep', theme: 'keep', camera: 'keep', pace: 'keep', details: ['lighting'], length: 'longer', motion: [] },
    });
  });

  it('carries the references an image job ran with, in order, and none for a job without them', () => {
    const { params } = jobSettingsPrefill({ mode: 'image', workflow: 'txt2img', model: 'klein', prompt: 'a', reference_path: ['/in/a.png', '/out/.web_uploads/b.png'] });
    expect(params.reference_path).toEqual(['/in/a.png', '/out/.web_uploads/b.png']);

    expect(jobSettingsPrefill({ mode: 'image', workflow: 'txt2img', model: 'klein', prompt: 'a', reference_path: '/in/only.png' }).params.reference_path).toEqual(['/in/only.png']);
    expect(jobSettingsPrefill({ mode: 'video', workflow: 'txt2vid', model: 'ltx', prompt: 'a' }).params).not.toHaveProperty('reference_path');
  });

  it('carries a submitted last frame as a prefill param, like the first frame', () => {
    const { params } = jobSettingsPrefill({ mode: 'video', workflow: 'img2vid', model: 'h3', prompt: 'a', image_path: '/in/start.png', end_image_path: '/out/.web_uploads/end.png' });

    expect(params).toMatchObject({ workflow: 'img2vid', image_path: '/in/start.png', end_image_path: '/out/.web_uploads/end.png' });
  });

  it('carries a submitted audio track and its Start at as prefill params, for a clip and for an extension', () => {
    const clip = jobSettingsPrefill({ mode: 'video', workflow: 'img2vid', model: 'ltx', prompt: 'a', image_path: '/in/start.png', audio_track_path: '/out/.web_uploads/memo.m4a', audio_track_start: '90' });
    expect(clip.params).toMatchObject({ audio_track_path: '/out/.web_uploads/memo.m4a', audio_track_start: '90' });

    const extension = jobSettingsPrefill({ mode: 'video', workflow: 'extend', model: 'ltx', source_video: '/clips/walk.mp4', audio_track_path: '/in/song.mp3', audio_track_start: '6.65' });
    expect(extension.params).toMatchObject({ workflow: 'extend', audio_track_path: '/in/song.mp3', audio_track_start: '6.65' });
  });

  it('names no track for a job made without one, which clears the tab\'s track', () => {
    const { params } = jobSettingsPrefill({ mode: 'video', workflow: 'txt2vid', model: 'ltx', prompt: 'a', audio: 'true' });

    expect(params).not.toHaveProperty('audio_track_path');
    expect(params).not.toHaveProperty('audio_track_start');
  });

  it('leaves the Audio switch alone for a job that followed a track, which sent none', () => {
    const track = jobSettingsPrefill({ mode: 'video', workflow: 'txt2vid', model: 'ltx', prompt: 'a', audio_track_path: '/in/song.mp3' });
    const plain = jobSettingsPrefill({ mode: 'video', workflow: 'txt2vid', model: 'ltx', prompt: 'a', audio: 'false' });

    expect(track.patch).not.toHaveProperty('audio');
    expect(plain.patch.audio).toBe(false);
  });

  it('keeps a blank LoRA list so loading a job without LoRAs clears them', () => {
    const { params } = jobSettingsPrefill({ mode: 'image', workflow: 'txt2img', model: 'zit', lora: '', steps: '' });

    expect(params).toEqual({ workflow: 'txt2img', model: 'zit', lora: '', reference_path: [] });
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

  it('turns Transparent on for a job that sent it, and off for one that did not', () => {
    expect(jobSettingsPrefill({ mode: 'image', workflow: 'txt2img', prompt: 'a', transparent: 'true' }).patch.postprocessTransparent).toBe(true);
    expect(jobSettingsPrefill({ mode: 'image', workflow: 'txt2img', prompt: 'a', transparent: 'false' }).patch.postprocessTransparent).toBe(false);
    expect(jobSettingsPrefill({ mode: 'image', workflow: 'txt2img', prompt: 'a' }).patch.postprocessTransparent).toBe(false);
    expect(jobSettingsPrefill({ mode: 'video', workflow: 'txt2vid', prompt: 'a' }).patch).not.toHaveProperty('postprocessTransparent');
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

  it('restores the record of how the Enhanced text was made', () => {
    const record = { settings: { style: 'cinematic' }, model: 'org/fast@r1', format: 'text' };
    const { patch } = jobSettingsPrefill({
      mode: 'video', workflow: 'txt2vid', scene: 'A castle at dusk.', prompt: 'A knight rides in.', scene_input: 'A castle.', prompt_input: 'a knight', enhance_record: JSON.stringify(record),
    });

    expect(patch.enhancedFrom?.record).toEqual(record);
  });

  it('restores an enhanced image job as the own prompt plus its Enhanced text', () => {
    const { patch } = jobSettingsPrefill({ mode: 'image', workflow: 'txt2img', prompt: 'A red fox curls up in deep snow.', prompt_input: 'a fox in snow' });

    expect(patch).toMatchObject({ prompt: 'a fox in snow', enhancedPrompt: 'A red fox curls up in deep snow.', enhancedFrom: { prompt: 'a fox in snow', mode: 'image' } });
  });

  it('restores an Enhanced JSON caption into the Enhanced box, not the { } box', () => {
    const { patch } = jobSettingsPrefill({ mode: 'image', workflow: 'txt2img', json_prompt: '{"a":1}', prompt_input: 'a fox' });

    expect(patch).toMatchObject({ prompt: 'a fox', enhancedPrompt: '{\n  "a": 1\n}', jsonPromptEnabled: false });
    expect(patch.jsonPrompt).toBeUndefined();
  });

  it('keeps a caption typed in the { } box as the JSON prompt', () => {
    const { patch } = jobSettingsPrefill({ mode: 'image', workflow: 'txt2img', json_prompt: '{"a":1}' });

    expect(patch).toMatchObject({ jsonPromptEnabled: true, jsonPrompt: '{"a":1}', enhancedPrompt: '', enhancedFrom: null });
  });

  it('remembers the Rewrite choice of a queued job for its own workflow', () => {
    const { patch } = jobSettingsPrefill({
      mode: 'video', workflow: 'extend', source_video: '/clips/walk.mp4', prompt: 'She waves.',
      enhance_auto: 'true', enhance_settings: '{"style":"keep","mood":"keep","details":[],"length":"same","motion":[],"parts":"both"}',
    });

    expect(patch.enhanceSettingsVideo?.partsByWorkflow).toEqual({ extend: 'both' });
    expect(patch.enhanceSettings).toBeUndefined();
  });

  it('keeps the Rewrite choices the draft remembers for other video workflows', () => {
    const { patch } = jobSettingsPrefill(
      { mode: 'video', workflow: 'img2vid', prompt: 'She waves.', enhance_auto: 'true', enhance_settings: '{"parts":"both"}' },
      { txt2vid: 'action', img2vid: 'action' },
    );

    expect(patch.enhanceSettingsVideo?.partsByWorkflow).toEqual({ txt2vid: 'action', img2vid: 'both' });
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

  it('restores the point an extension continued from', () => {
    const { params } = jobSettingsPrefill({ mode: 'video', workflow: 'extend', source_video: '/clips/walk.mp4', extend_frames: '48', extend_from: '3.208333', scene: 'A street.' });
    expect(params).toMatchObject({ source_video: '/clips/walk.mp4', extend_from: '3.208333' });
    expect(jobSettingsPrefill({ mode: 'video', workflow: 'extend', source_video: '/clips/walk.mp4', scene: 'A street.' }).params).not.toHaveProperty('extend_from');
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

describe('hasJobSettings', () => {
  it('is false for no settings and true once any field was submitted', () => {
    expect(hasJobSettings({})).toBe(false);
    expect(hasJobSettings(undefined)).toBe(false);
    expect(hasJobSettings(null)).toBe(false);
    expect(hasJobSettings({ model: 'zit' })).toBe(true);
  });
});

describe('outputSettingsPatch', () => {
  const FINISHING = {
    transparent: 'true', sharpen_enabled: 'true', sharpen_amount: '0.6', contrast_enabled: 'true', contrast_amount: '1.1', saturation_enabled: 'false',
    upscale: '2', upscale_denoise: '0.3', upscale_steps: '12', upscale_guidance: '1.5', upscale_sharpen: 'false', image_strength: '0.7',
  };

  it('maps every finishing setting of an image to its draft key', () => {
    expect(outputSettingsPatch(FINISHING, false)).toEqual({
      postprocessTransparent: true, postprocessSharpenEnabled: true, postprocessSharpenAmount: 0.6, postprocessContrastEnabled: true, postprocessContrastAmount: 1.1,
      postprocessSaturationEnabled: false, upscaleEnabled: true, upscaleFactor: 2, upscaleDenoise: 0.3, upscaleSteps: 12, upscaleGuidance: 1.5, upscaleSharpen: false,
      referenceImageStrength: 0.7,
    });
  });

  it('reads a missing upscale as off, and never touches the Review choice', () => {
    const patch = outputSettingsPatch({ sharpen_enabled: 'true' }, false);
    expect(patch).toMatchObject({ upscaleEnabled: false, postprocessTransparent: false });
    expect(patch).not.toHaveProperty('upscaleReview');
  });

  it('maps only the post-processing of a video', () => {
    expect(outputSettingsPatch({ ...FINISHING }, true)).toEqual({
      postprocessSharpenEnabled: true, postprocessSharpenAmount: 0.6, postprocessContrastEnabled: true, postprocessContrastAmount: 1.1, postprocessSaturationEnabled: false,
    });
  });
});
