import { describe, it, expect } from 'vitest';
import {
  canSubmitText,
  effectiveEnhanceSettings,
  entryEnhanceSpec,
  enhanceContext,
  enhancedSourceFor,
  anyEnhancedUsed,
  actionOnlyBlocked,
  defaultParts,
  effectiveFormat,
  effectiveTier,
  isJsonCaption,
  prettyCaption,
  submittedCaption,
  enhancedPartUsed,
  hasAudio,
  isEnhancedStale,
  rememberedParts,
  runParts,
  settingsPayload,
  submittedPair,
  submittedPrompt,
  withRememberedParts,
  workflowMode,
  DEFAULT_ENHANCE_SETTINGS,
  enhanceSettingsKey,
  isDefaultEnhanceSettings,
} from './promptEnhance';
import { createNdjsonParser } from '$lib/api/promptEnhance';
import type { DraftState, EnhanceFrame, EnhanceSettings, PromptEnhancerContract } from '$lib/types';

function state(overrides: Partial<DraftState> = {}): DraftState {
  return {
    workflow: 'txt2img',
    promptSource: 'inline',
    prompt: 'a fox',
    scene: '',
    jsonPromptEnabled: false,
    enhanceAuto: false,
    enhancedPrompt: '',
    enhancedScene: '',
    enhancedFrom: null,
    enhanceSettings: null,
    ...overrides,
  } as DraftState;
}

const SETTINGS: EnhanceSettings = { style: 'keep', mood: 'keep', theme: 'keep', camera: 'keep', pace: 'keep', details: [], length: 'same', motion: [], sound: [], note: '' };

describe('Fast/Smart and Text/JSON', () => {
  const contract = (smart: PromptEnhancerContract['smart']): PromptEnhancerContract =>
    ({ matrix: { axes: [], defaults: SETTINGS }, model: 'fast/m', revision: null, downloaded: true, download_size_label: null, default_max_words: 300, smart, error: null }) as PromptEnhancerContract;

  it('runs Smart only when chosen and available', () => {
    const available = contract({ download_size_label: '19 GB', unavailable: null });
    expect(effectiveTier(state({ enhanceModel: 'smart' }), available)).toBe('smart');
    expect(effectiveTier(state({ enhanceModel: 'fast' }), available)).toBe('fast');
    expect(effectiveTier(state({ enhanceModel: 'smart' }), contract({ download_size_label: null, unavailable: 'Needs about 9 GB of memory.' }))).toBe('fast');
    expect(effectiveTier(state({ enhanceModel: 'smart' }), contract(null))).toBe('fast');
  });

  it('writes JSON only for image models that take JSON captions', () => {
    expect(effectiveFormat(state({ enhanceFormat: 'json' }), true)).toBe('json');
    expect(effectiveFormat(state({ enhanceFormat: 'json' }), false)).toBe('text');
    expect(effectiveFormat(state({ enhanceFormat: 'json', workflow: 'txt2vid' }), true)).toBe('text');
  });

  it('detects and lays out captions', () => {
    expect(isJsonCaption('{"high_level_description":"x"}')).toBe(true);
    expect(isJsonCaption('{a|b} fox')).toBe(false);
    expect(prettyCaption('{"a":1}')).toBe('{\n  "a": 1\n}');
    expect(prettyCaption('a fox')).toBe('a fox');
  });

  it('submits an Enhanced caption as the JSON prompt', () => {
    expect(submittedCaption(state({ enhancedPrompt: '{\n  "a": 1\n}' }))).toBe('{\n  "a": 1\n}');
    expect(submittedCaption(state({ enhancedPrompt: 'A fox in snow.' }))).toBeNull();
    expect(submittedCaption(state({ enhancedPrompt: '{"a":1}', enhanceAuto: true }))).toBeNull();
    expect(submittedCaption(state({ enhancedPrompt: '{"a":1}', workflow: 'txt2vid' }))).toBeNull();
  });

});

describe('submission rule', () => {
  it.each([
    [{}, false, 'a fox'],
    [{ enhancedPrompt: 'A fox in snow.' }, true, 'A fox in snow.'],
    [{ enhancedPrompt: '   ' }, false, 'a fox'],
    [{ enhancedPrompt: 'A fox in snow.', promptSource: 'file' as const }, false, 'a fox'],
    [{ enhancedPrompt: 'A fox in snow.', jsonPromptEnabled: true }, false, 'a fox'],
    [{ enhancedPrompt: 'A fox in snow.', enhanceAuto: true }, false, 'a fox'],
  ])('%o → override %s', (overrides, active, prompt) => {
    expect(anyEnhancedUsed(state(overrides))).toBe(active);
    expect(submittedPrompt(state(overrides))).toBe(prompt);
  });
});

describe('isEnhancedStale', () => {
  const made = { prompt: 'a fox', scene: '', mode: 'image' as const };

  it.each([
    ['hand-typed text', { enhancedPrompt: 'typed', enhancedFrom: null }, false],
    ['prompt and mode match', { enhancedPrompt: 'x', enhancedFrom: made }, false],
    ['the prompt changes', { prompt: 'a cat', enhancedPrompt: 'x', enhancedFrom: made }, true],
    ['the workflow mode changes', { workflow: 'txt2vid' as const, enhancedPrompt: 'x', enhancedFrom: made }, true],
    ['the box is empty', { prompt: 'a cat', enhancedPrompt: '', enhancedFrom: made }, false],
  ])('is stale only when the text it was made from changed: %s', (_name, overrides, stale) => {
    expect(isEnhancedStale(state(overrides))).toBe(stale);
  });
});

describe('sound, direction and context', () => {
  it('sends sound only for video with audio', () => {
    const settings = { ...SETTINGS, motion: ['action'], sound: ['music'], note: 'at dusk' };
    expect(settingsPayload(settings, 'video', true)).toMatchObject({ motion: ['action'], sound: ['music'], note: 'at dusk' });
    expect(settingsPayload(settings, 'video', false)).not.toHaveProperty('sound');
    expect(settingsPayload(settings, 'image', true)).not.toHaveProperty('sound');
    expect(settingsPayload(settings, 'image').note).toBe('at dusk');
  });

  it('fills sound and note when the saved settings lack them', () => {
    const saved = { style: 'anime', mood: 'keep', theme: 'keep', camera: 'keep', pace: 'keep', details: [], length: 'same', motion: [] } as unknown as EnhanceSettings;
    const result = effectiveEnhanceSettings(state({ enhanceSettings: saved }), undefined);
    expect(result).toMatchObject({ style: 'anime', sound: [], note: '' });
  });

  it('describes the video job', () => {
    const video = state({ workflow: 'img2vid', ratio: '9:16', dimensionMode: 'ratio', width: 704, height: 448, frameCount: 97, fps: 24, audio: true });
    expect(enhanceContext(video)).toEqual({ ratio: '9:16', width: null, height: null, frames: 97, fps: 24, audio: true, reference_image: true, end_image: false });
    expect(hasAudio(video)).toBe(true);
  });

  it('tells the enhancer about a last frame for a video job only', () => {
    const video = state({ workflow: 'img2vid', frameCount: 97, fps: 24 });
    expect(enhanceContext(video, true, null, true).end_image).toBe(true);
    expect(enhanceContext(video).end_image).toBe(false);
    expect(enhanceContext(state({ workflow: 'img2img' }), true, null, true).end_image).toBe(false);
  });

  it('has no audio where the video backend writes none', () => {
    const video = state({ workflow: 'txt2vid', frameCount: 97, fps: 24, audio: true });
    expect(hasAudio(video, false)).toBe(false);
    expect(enhanceContext(video, false).audio).toBe(false);
  });

  it('describes the image job without video facts', () => {
    const image = state({ workflow: 'txt2img', ratio: '2:3', dimensionMode: 'custom', width: 1024, height: 576, frameCount: 97, fps: 24, audio: true });
    expect(enhanceContext(image)).toEqual({ ratio: '2:3', width: 1024, height: 576, frames: null, fps: null, audio: false, reference_image: false, end_image: false });
  });
});

describe('entryEnhanceSpec', () => {
  const cameraOnly = { enhance: 'camera=orbit', enhance_image: null, enhance_caption: 'style=keep' };

  it('applies a prompt-file entry as the run does', () => {
    expect(entryEnhanceSpec(cameraOnly, 'video', false)).toBe('camera=orbit');
    expect(entryEnhanceSpec(cameraOnly, 'image', false)).toBeNull();
    expect(entryEnhanceSpec(cameraOnly, 'image', true)).toBe('style=keep');
    expect(entryEnhanceSpec({ enhance: null, enhance_image: null, enhance_caption: null }, 'image', true)).toBeNull();
  });
});

describe('settings helpers', () => {
  it('strips motion for images', () => {
    expect(settingsPayload({ ...SETTINGS, motion: ['secondary'], camera: 'orbit' }, 'image')).toEqual({ style: 'keep', mood: 'keep', theme: 'keep', details: [], length: 'same' });
    expect(settingsPayload({ ...SETTINGS, camera: 'orbit', pace: 'fast' }, 'video')).toMatchObject({ camera: 'orbit', pace: 'fast' });
    expect(settingsPayload({ ...SETTINGS, motion: ['secondary'] }, 'video').motion).toEqual(['secondary']);
  });

  it('falls back to contract defaults without sharing arrays', () => {
    const contract = { matrix: { axes: [], defaults: { style: 'keep', mood: 'keep', theme: 'keep', camera: 'keep', pace: 'keep', details: ['lighting'], length: 'same', motion: ['action'], sound: [], note: '' } } } as unknown as PromptEnhancerContract;
    const settings = effectiveEnhanceSettings(state(), contract);
    expect(settings.details).toEqual(['lighting']);
    expect(settings.details).not.toBe(contract.matrix.defaults.details);
    expect(effectiveEnhanceSettings(state({ enhanceSettings: SETTINGS }), contract)).toEqual(SETTINGS);
  });

  it('fills axes missing from saved settings with contract defaults', () => {
    const contract = { matrix: { axes: [], defaults: { style: 'keep', mood: 'keep', theme: 'keep', camera: 'keep', pace: 'keep', details: ['lighting'], length: 'same', motion: ['action'], sound: [], note: '' } } } as unknown as PromptEnhancerContract;
    const saved = { style: 'photo', details: [], length: 'longer', motion: [], sound: [], note: '' } as unknown as EnhanceSettings;
    expect(effectiveEnhanceSettings(state({ enhanceSettings: saved }), contract)).toEqual({ ...saved, mood: 'keep', theme: 'keep', camera: 'keep', pace: 'keep' });
  });

  it('drops saved values the server no longer offers', () => {
    const option = (slug: string) => ({ slug, label: slug, description: '', group: null });
    const axes = [
      { key: 'style', label: 'Style', multi: false, video_only: false, audio_only: false, options: [option('keep'), option('photo')], default: ['keep'] },
      { key: 'motion', label: 'Describe', multi: true, video_only: true, audio_only: false, options: [option('action'), option('secondary')], default: [] },
    ];
    const contract = { matrix: { axes, defaults: { style: 'keep', mood: 'keep', theme: 'keep', camera: 'keep', pace: 'keep', details: [], length: 'same', motion: [], sound: [], note: '' } } } as unknown as PromptEnhancerContract;
    const saved = { ...SETTINGS, style: 'gone', motion: ['action', 'pacing'] };
    const settings = effectiveEnhanceSettings(state({ workflow: 'txt2vid', enhanceSettingsVideo: saved }), contract);
    expect(settings.style).toBe('keep');
    expect(settings.motion).toEqual(['action']);
  });

  it('maps workflows to modes', () => {
    expect(workflowMode('img2img')).toBe('image');
    expect(workflowMode('img2vid')).toBe('video');
    expect(workflowMode('extend')).toBe('video');
  });

});

describe('createNdjsonParser', () => {
  it('handles frames split across chunks and a trailing line without newline', () => {
    const frames: EnhanceFrame[] = [];
    const parser = createNdjsonParser((frame) => frames.push(frame));
    parser.push('{"type":"status","pha');
    parser.push('se":"loading"}\n{"type":"text","text":"A fox"}\n\n');
    parser.push('{"type":"done","prompt":"A fox.","clamped":false}');
    expect(frames).toHaveLength(2);
    parser.flush();
    expect(frames.map((frame) => frame.type)).toEqual(['status', 'text', 'done']);
  });

  it('ignores malformed lines', () => {
    const frames: EnhanceFrame[] = [];
    const parser = createNdjsonParser((frame) => frames.push(frame));
    parser.push('not json\n{"type":"error","detail":"x"}\n');
    expect(frames).toEqual([{ type: 'error', detail: 'x' }]);
  });
});

describe('scene and prompt pair', () => {
  const video = (overrides: Partial<DraftState> = {}) => state({ workflow: 'txt2vid', ...overrides });

  it.each([
    ['txt2vid', 'A castle.', '', true],
    ['txt2vid', '', 'A knight rides in.', true],
    ['txt2vid', '  ', '  ', false],
    ['img2vid', 'A castle.', '', true],
    ['txt2img', 'A castle.', '', false],
    ['txt2img', '', 'a fox', true],
  ] as const)('%s with scene %j and prompt %j can run text: %s', (workflow, scene, prompt, expected) => {
    expect(canSubmitText(state({ workflow, scene, prompt }))).toBe(expected);
  });

  it('uses the Enhanced text of each part whose Enhanced box has text', () => {
    expect(enhancedPartUsed(video({ enhancedScene: 'A castle at dusk.' }), 'scene')).toBe(true);
    expect(enhancedPartUsed(video({ enhancedScene: 'A castle at dusk.' }), 'action')).toBe(false);
    expect(enhancedPartUsed(video({ enhancedPrompt: 'A knight rides in.' }), 'action')).toBe(true);
    expect(enhancedPartUsed(video({ enhancedPrompt: 'A knight rides in.' }), 'scene')).toBe(false);
    expect(anyEnhancedUsed(video({ enhancedScene: ' ', enhancedPrompt: ' ' }))).toBe(false);
    // An image never carries a scene, so a stray Enhanced scene does not switch its text.
    expect(anyEnhancedUsed(state({ workflow: 'txt2img', enhancedScene: 'A castle.' }))).toBe(false);
  });

  it.each([
    ['neither part enhanced', {}, { scene: 'A castle.', prompt: 'a knight' }],
    ['both parts enhanced', { enhancedScene: 'A castle at dusk.', enhancedPrompt: 'A knight rides in.' }, { scene: 'A castle at dusk.', prompt: 'A knight rides in.' }],
    ['only the action enhanced', { enhancedPrompt: 'A knight rides in.' }, { scene: 'A castle.', prompt: 'A knight rides in.' }],
    ['only the scene enhanced', { enhancedScene: 'A castle at dusk.' }, { scene: 'A castle at dusk.', prompt: 'a knight' }],
    ['a blank Enhanced scene', { enhancedScene: '  ', enhancedPrompt: 'A knight rides in.' }, { scene: 'A castle.', prompt: 'A knight rides in.' }],
    ['file source', { enhancedScene: 'x', enhancedPrompt: 'y', promptSource: 'file' as const }, { scene: 'A castle.', prompt: 'a knight' }],
    ['auto-enhance', { enhancedScene: 'x', enhancedPrompt: 'y', enhanceAuto: true }, { scene: 'A castle.', prompt: 'a knight' }],
    ['a JSON caption', { enhancedScene: 'x', enhancedPrompt: 'y', jsonPromptEnabled: true }, { scene: 'A castle.', prompt: 'a knight' }],
  ])('submits each part enhanced or as written: %s', (_name, overrides, expected) => {
    expect(submittedPair(video({ scene: 'A castle.', prompt: 'a knight', ...overrides }))).toEqual(expected);
  });

  it('never submits a scene for an image', () => {
    expect(submittedPair(state({ workflow: 'txt2img', scene: 'ignored', prompt: 'a fox' }))).toEqual({ scene: '', prompt: 'a fox' });
  });

  describe('Rewrite', () => {
    it('defaults to Action only for an extension and Image to Video, and Scene and action for Text to Video', () => {
      expect(defaultParts('extend')).toBe('action');
      expect(defaultParts('img2vid')).toBe('action');
      expect(defaultParts('txt2vid')).toBe('both');
      expect(rememberedParts(state({ workflow: 'extend' }))).toBe('action');
      expect(rememberedParts(state({ workflow: 'img2vid' }))).toBe('action');
      expect(rememberedParts(state({ workflow: 'txt2vid' }))).toBe('both');
    });

    it('rewrites both parts of an Image to Video run while its Scene is blank', () => {
      expect(runParts(state({ workflow: 'img2vid', scene: '', prompt: 'She waves.' }))).toBe('both');
      expect(runParts(state({ workflow: 'img2vid', scene: 'A fox on a hill.', prompt: 'She waves.' }))).toBe('action');
    });

    it('remembers the choice for each video workflow apart', () => {
      const settings = withRememberedParts({ ...SETTINGS }, 'extend', 'both');
      expect(rememberedParts(state({ workflow: 'extend', enhanceSettingsVideo: settings }))).toBe('both');
      expect(rememberedParts(state({ workflow: 'txt2vid', enhanceSettingsVideo: settings }))).toBe('both');
      const action = withRememberedParts(settings, 'txt2vid', 'action');
      expect(rememberedParts(state({ workflow: 'txt2vid', enhanceSettingsVideo: action }))).toBe('action');
      expect(rememberedParts(state({ workflow: 'extend', enhanceSettingsVideo: action }))).toBe('both');
      expect(rememberedParts(state({ workflow: 'img2vid', enhanceSettingsVideo: action }))).toBe('action');
      expect(rememberedParts(state({ workflow: 'img2vid', enhanceSettingsVideo: withRememberedParts(action, 'img2vid', 'both') }))).toBe('both');
    });

    it('falls back to both while the Scene is blank, without changing the remembered choice', () => {
      const remembered = state({ workflow: 'extend', scene: '  ', prompt: 'She waves.' });
      expect(actionOnlyBlocked(remembered)).toBe(true);
      expect(runParts(remembered)).toBe('both');
      expect(rememberedParts(remembered)).toBe('action');
      expect(runParts({ ...remembered, scene: 'A street.' })).toBe('action');
      expect(runParts(video({ scene: 'A castle.' }))).toBe('both');
    });

    it('does not fall back for prompt-file entries, which bring their own scenes', () => {
      const file = state({ workflow: 'extend', scene: '', promptSource: 'file' });
      expect(actionOnlyBlocked(file)).toBe(false);
      expect(runParts(file)).toBe('action');
    });

    it('is sent with a video payload and left out for images and for the browser-only memory', () => {
      const settings = { ...SETTINGS, parts: 'action' as const };
      expect(settingsPayload(settings, 'video')).toEqual({ style: 'keep', mood: 'keep', theme: 'keep', camera: 'keep', pace: 'keep', details: [], length: 'same', parts: 'action', motion: [] });
      expect(settingsPayload(SETTINGS, 'video')).not.toHaveProperty('parts');
    });
  });

  describe('staleness', () => {
    const enhanced = { enhancedScene: 'A castle at dusk.', enhancedPrompt: 'A knight rides in.' };
    const made = { scene: 'A castle.', prompt: 'a knight', mode: 'video' as const };

    it.each([
      ['scene, prompt and workflow match what was enhanced', { scene: 'A castle.', prompt: 'a knight', ...enhanced, enhancedFrom: made }, false],
      ['the scene changes', { scene: 'A harbour.', prompt: 'a knight', ...enhanced, enhancedFrom: made }, true],
      ['the prompt changes', { scene: 'A castle.', prompt: 'a queen', ...enhanced, enhancedFrom: made }, true],
      ['the workflow mode changes', { workflow: 'txt2img' as const, prompt: 'a knight', enhancedPrompt: 'x', enhancedFrom: made }, true],
      ['only an Enhanced scene exists and the scene changes', { scene: 'A harbour.', enhancedScene: 'A castle at dusk.', enhancedFrom: { ...made, prompt: 'a fox' } }, true],
    ])('is stale only when the text it was made from changed: %s', (_name, overrides, stale) => {
      expect(isEnhancedStale(video(overrides))).toBe(stale);
    });
  });
});

describe('extend', () => {
  const extending = (overrides: Partial<DraftState> = {}) => state({
    workflow: 'extend', prompt: '', scene: '', audio: true, fps: 24, frameCount: 97, ratio: '16:9', dimensionMode: 'ratio', width: 848, height: 480,
    extendSourcePath: '/clips/walk.mp4', extendFrames: 48, model: 'ltx-2.3', ...overrides,
  });

  it('describes the job by its source and new frames, with no frame shape or reference image', () => {
    expect(enhanceContext(extending())).toEqual({
      ratio: null, width: null, height: null, frames: null, fps: null, audio: true, reference_image: false,
      source_video: '/clips/walk.mp4', extend_frames: 48, model: 'ltx-2.3',
    });
  });

  it('names the selected video model so the server applies its extend step', () => {
    expect(enhanceContext(extending({ model: 'h3-turbo', extendFrames: 51 }))).toMatchObject({ model: 'h3-turbo', extend_frames: 51 });
    expect(enhanceContext(extending({ model: '' })).model).toBeNull();
    expect(enhanceContext(state({ workflow: 'txt2vid', model: 'h3-turbo' }))).not.toHaveProperty('model');
  });

  it('has no audio where the backend writes none or the source is silent', () => {
    expect(enhanceContext(extending(), false).audio).toBe(false);
    expect(enhanceContext(extending(), true, false).audio).toBe(false);
    expect(enhanceContext(extending(), true, true).audio).toBe(true);
    expect(hasAudio(extending({ audio: false }))).toBe(false);
  });

  it('adds no source facts to the context of other workflows', () => {
    expect(enhanceContext(state({ workflow: 'txt2vid' }))).not.toHaveProperty('source_video');
  });

  it('allows an empty scene and prompt, which continue naturally', () => {
    expect(canSubmitText(extending())).toBe(true);
    expect(canSubmitText(state({ workflow: 'txt2vid', prompt: '', scene: '' }))).toBe(false);
    expect(canSubmitText(state({ workflow: 'txt2img', prompt: ' ' }))).toBe(false);
  });

  describe('staleness', () => {
    const enhanced = () => extending({ prompt: 'She waves.', enhancedPrompt: 'She waves slowly.', enhancedFrom: enhancedSourceFor(extending({ prompt: 'She waves.' })) });

    it('records the source and length the Enhanced text was made for', () => {
      expect(enhancedSourceFor(extending({ prompt: 'She waves.', scene: 'A street.' }))).toEqual({
        prompt: 'She waves.', scene: 'A street.', mode: 'video', sourceVideoPath: '/clips/walk.mp4', extendFrames: 48,
      });
      expect(enhancedSourceFor(state({ workflow: 'txt2vid', prompt: 'p' }))).toEqual({ prompt: 'p', scene: '', mode: 'video' });
      expect(enhancedSourceFor(state({ workflow: 'txt2img', prompt: 'p', scene: 'ignored' }))).toEqual({ prompt: 'p', scene: '', mode: 'image' });
    });

    it.each([
      ['the source and length are the same', {}, false],
      ['the source changes', { extendSourcePath: '/clips/other.mp4' }, true],
      ['the length changes', { extendFrames: 56 }, true],
      ['the workflow is no longer an extension', { workflow: 'txt2vid' as const }, true],
    ])('is out of date only when %s', (_name, overrides, stale) => {
      expect(isEnhancedStale({ ...enhanced(), ...overrides })).toBe(stale);
    });

    it('is out of date when text made for another video workflow is used to extend', () => {
      const made = state({ workflow: 'txt2vid', prompt: 'She waves.', enhancedPrompt: 'x', enhancedFrom: enhancedSourceFor(state({ workflow: 'txt2vid', prompt: 'She waves.' })) });
      expect(isEnhancedStale(made)).toBe(false);
      expect(isEnhancedStale({ ...made, workflow: 'extend', extendSourcePath: '/clips/walk.mp4', extendFrames: 48 })).toBe(true);
    });

    it('is not out of date for a video record that carries no extension source', () => {
      const old = state({ workflow: 'txt2vid', prompt: 'p', enhancedPrompt: 'x', enhancedFrom: { prompt: 'p', mode: 'video', scene: '' } });
      expect(isEnhancedStale(old)).toBe(false);
    });
  });
});

describe('options per mode', () => {
  it('keeps image and video options apart', () => {
    const image = { ...SETTINGS, style: 'photo' };
    const video = { ...SETTINGS, style: 'cinematic', camera: 'orbit' };
    expect(effectiveEnhanceSettings(state({ workflow: 'txt2img', enhanceSettings: image, enhanceSettingsVideo: video }), undefined).style).toBe('photo');
    expect(effectiveEnhanceSettings(state({ workflow: 'img2vid', enhanceSettings: image, enhanceSettingsVideo: video }), undefined)).toMatchObject({ style: 'cinematic', camera: 'orbit' });
    expect(enhanceSettingsKey('img2img')).toBe('enhanceSettings');
    expect(enhanceSettingsKey('extend')).toBe('enhanceSettingsVideo');
  });

  it('tells default options from picked ones, ignoring the remembered Rewrite choices', () => {
    expect(isDefaultEnhanceSettings({ ...DEFAULT_ENHANCE_SETTINGS, partsByWorkflow: { txt2vid: 'action' } })).toBe(true);
    expect(isDefaultEnhanceSettings({ ...DEFAULT_ENHANCE_SETTINGS, note: '  ' })).toBe(true);
    expect(isDefaultEnhanceSettings({ ...DEFAULT_ENHANCE_SETTINGS, theme: 'zen' })).toBe(false);
    expect(isDefaultEnhanceSettings({ ...DEFAULT_ENHANCE_SETTINGS, details: ['lighting'] })).toBe(false);
    expect(isDefaultEnhanceSettings({ ...DEFAULT_ENHANCE_SETTINGS, note: 'at dusk' })).toBe(false);
  });
});
