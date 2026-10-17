import { describe, it, expect } from 'vitest';
import {
  canSubmitText,
  clampedNote,
  effectiveEnhanceSettings,
  enhanceContext,
  enhancedSourceFor,
  enhancePhaseMessage,
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
  isNoOpSettings,
  rememberedParts,
  runParts,
  settingsPayload,
  autoSplitsPrompt,
  splitsPrompt,
  submittedPair,
  submittedPrompt,
  withRememberedParts,
  workflowMode,
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

const SETTINGS: EnhanceSettings = { style: 'keep', mood: 'keep', details: [], length: 'same', motion: [], sound: [], note: '' };

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

  it('a JSON caption is never a no-op', () => {
    expect(isNoOpSettings(SETTINGS, 'image')).toBe(true);
    expect(isNoOpSettings(SETTINGS, 'image', false, false, true)).toBe(false);
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

  it('names a caption run in the status line', () => {
    expect(enhancePhaseMessage('generating', null, true)).toBe('Writing a JSON caption…');
    expect(enhancePhaseMessage('generating', null)).toBe('Enhancing…');
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
  const made = { prompt: 'a fox', mode: 'image' as const };

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
  it('counts sound only for video with audio, and a direction everywhere', () => {
    expect(isNoOpSettings({ ...SETTINGS, sound: ['ambient'] }, 'video', true)).toBe(false);
    expect(isNoOpSettings({ ...SETTINGS, sound: ['ambient'] }, 'video', false)).toBe(true);
    expect(isNoOpSettings({ ...SETTINGS, sound: ['ambient'] }, 'image', true)).toBe(true);
    expect(isNoOpSettings({ ...SETTINGS, note: 'at night' }, 'image')).toBe(false);
    expect(isNoOpSettings({ ...SETTINGS, note: '   ' }, 'image')).toBe(true);
  });

  it('sends sound only for video with audio', () => {
    const settings = { ...SETTINGS, motion: ['action'], sound: ['music'], note: 'at dusk' };
    expect(settingsPayload(settings, 'video', true)).toMatchObject({ motion: ['action'], sound: ['music'], note: 'at dusk' });
    expect(settingsPayload(settings, 'video', false)).not.toHaveProperty('sound');
    expect(settingsPayload(settings, 'image', true)).not.toHaveProperty('sound');
    expect(settingsPayload(settings, 'image').note).toBe('at dusk');
  });

  it('fills sound and note for settings saved before they existed', () => {
    const saved = { style: 'anime', mood: 'keep', details: [], length: 'same', motion: [] } as unknown as EnhanceSettings;
    const result = effectiveEnhanceSettings(state({ enhanceSettings: saved }), undefined);
    expect(result).toMatchObject({ style: 'anime', sound: [], note: '' });
  });

  it('describes the video job', () => {
    const video = state({ workflow: 'img2vid', ratio: '9:16', dimensionMode: 'ratio', width: 704, height: 448, frameCount: 97, fps: 24, audio: true });
    expect(enhanceContext(video)).toEqual({ ratio: '9:16', width: null, height: null, frames: 97, fps: 24, audio: true, reference_image: true });
    expect(hasAudio(video)).toBe(true);
  });

  it('has no audio where the video backend writes none', () => {
    const video = state({ workflow: 'txt2vid', frameCount: 97, fps: 24, audio: true });
    expect(hasAudio(video, false)).toBe(false);
    expect(enhanceContext(video, false).audio).toBe(false);
  });

  it('describes the image job without video facts', () => {
    const image = state({ workflow: 'txt2img', ratio: '2:3', dimensionMode: 'custom', width: 1024, height: 576, frameCount: 97, fps: 24, audio: true });
    expect(enhanceContext(image)).toEqual({ ratio: '2:3', width: 1024, height: 576, frames: null, fps: null, audio: false, reference_image: false });
  });
});

describe('settings helpers', () => {
  it('detects the no-op combination per mode', () => {
    expect(isNoOpSettings(SETTINGS, 'image')).toBe(true);
    expect(isNoOpSettings({ ...SETTINGS, motion: ['action'] }, 'image')).toBe(true);
    expect(isNoOpSettings({ ...SETTINGS, motion: ['action'] }, 'video')).toBe(false);
    expect(isNoOpSettings({ ...SETTINGS, length: 'longer' }, 'image')).toBe(false);
    expect(isNoOpSettings({ ...SETTINGS, mood: 'eerie' }, 'image')).toBe(false);
  });

  it('treats an action-only video rewrite as a change, like the server', () => {
    expect(isNoOpSettings({ ...SETTINGS, parts: 'action' }, 'video')).toBe(false);
    expect(isNoOpSettings({ ...SETTINGS, parts: 'both' }, 'video')).toBe(true);
    expect(isNoOpSettings({ ...SETTINGS, parts: 'action' }, 'image')).toBe(true);
  });

  it('strips motion for images', () => {
    expect(settingsPayload({ ...SETTINGS, motion: ['pacing'] }, 'image')).toEqual({ style: 'keep', mood: 'keep', details: [], length: 'same' });
    expect(settingsPayload({ ...SETTINGS, motion: ['pacing'] }, 'video').motion).toEqual(['pacing']);
  });

  it('falls back to contract defaults without sharing arrays', () => {
    const contract = { matrix: { axes: [], defaults: { style: 'keep', mood: 'keep', details: ['lighting'], length: 'same', motion: ['action'], sound: [], note: '' } } } as unknown as PromptEnhancerContract;
    const settings = effectiveEnhanceSettings(state(), contract);
    expect(settings.details).toEqual(['lighting']);
    expect(settings.details).not.toBe(contract.matrix.defaults.details);
    expect(effectiveEnhanceSettings(state({ enhanceSettings: SETTINGS }), contract)).toEqual(SETTINGS);
  });

  it('fills axes missing from saved settings with contract defaults', () => {
    const contract = { matrix: { axes: [], defaults: { style: 'keep', mood: 'keep', details: ['lighting'], length: 'same', motion: ['action'], sound: [], note: '' } } } as unknown as PromptEnhancerContract;
    const saved = { style: 'photo', details: [], length: 'longer', motion: [], sound: [], note: '' } as unknown as EnhanceSettings;
    expect(effectiveEnhanceSettings(state({ enhanceSettings: saved }), contract)).toEqual({ ...saved, mood: 'keep' });
  });

  it('maps workflows to modes', () => {
    expect(workflowMode('img2img')).toBe('image');
    expect(workflowMode('img2vid')).toBe('video');
    expect(workflowMode('extend')).toBe('video');
  });

  it('formats phase and clamp messages', () => {
    expect(enhancePhaseMessage('downloading', '2.4 GB')).toContain('≈2.4 GB');
    expect(enhancePhaseMessage('loading', null)).toBe('Loading enhancer…');
    expect(enhancePhaseMessage('generating_cpu', null)).toContain('on the CPU');
    expect(clampedNote('shorter')).toContain('Too short');
    expect(clampedNote('extra')).toContain('maximum length');
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

  it('splits only a video prompt that has no scene', () => {
    expect(splitsPrompt(video({ scene: '', prompt: 'a knight' }))).toBe(true);
    expect(splitsPrompt(video({ scene: 'A castle.', prompt: 'a knight' }))).toBe(false);
    expect(splitsPrompt(video({ scene: '', prompt: '' }))).toBe(false);
    expect(splitsPrompt(state({ workflow: 'txt2img', scene: '', prompt: 'a fox' }))).toBe(false);
  });

  it('auto-enhance splits a video with a blank default scene whatever the prompt box holds, and any extension', () => {
    expect(autoSplitsPrompt(video({ scene: '', prompt: '' }))).toBe(true);
    expect(autoSplitsPrompt(video({ scene: '  ', prompt: 'a knight' }))).toBe(true);
    expect(autoSplitsPrompt(video({ scene: 'A castle.', prompt: '' }))).toBe(false);
    expect(autoSplitsPrompt(state({ workflow: 'txt2img', scene: '', prompt: 'a fox' }))).toBe(false);
    expect(autoSplitsPrompt(state({ workflow: 'extend', scene: 'A street.', prompt: 'She waves.' }))).toBe(true);
    expect(autoSplitsPrompt(state({ workflow: 'extend', scene: '', prompt: '' }))).toBe(true);
  });

  it('is never a no-op for a split video prompt, and unchanged for images', () => {
    expect(isNoOpSettings(SETTINGS, 'video', false, true)).toBe(false);
    expect(isNoOpSettings(SETTINGS, 'video', false, false)).toBe(true);
    expect(isNoOpSettings(SETTINGS, 'image', false, true)).toBe(true);
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
      expect(rememberedParts(state({ workflow: 'extend', enhanceSettings: settings }))).toBe('both');
      expect(rememberedParts(state({ workflow: 'txt2vid', enhanceSettings: settings }))).toBe('both');
      const action = withRememberedParts(settings, 'txt2vid', 'action');
      expect(rememberedParts(state({ workflow: 'txt2vid', enhanceSettings: action }))).toBe('action');
      expect(rememberedParts(state({ workflow: 'extend', enhanceSettings: action }))).toBe('both');
      expect(rememberedParts(state({ workflow: 'img2vid', enhanceSettings: action }))).toBe('action');
      expect(rememberedParts(state({ workflow: 'img2vid', enhanceSettings: withRememberedParts(action, 'img2vid', 'both') }))).toBe('both');
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
      expect(settingsPayload(settings, 'video')).toEqual({ style: 'keep', mood: 'keep', details: [], length: 'same', parts: 'action', motion: [] });
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
      ['the source was saved before scenes existed', { scene: '', prompt: 'a fox', enhancedPrompt: 'x', enhancedFrom: { prompt: 'a fox', mode: 'video' as const } }, false],
    ])('is stale only when the text it was made from changed: %s', (_name, overrides, stale) => {
      expect(isEnhancedStale(video(overrides))).toBe(stale);
    });
  });
});

describe('extend', () => {
  const extending = (overrides: Partial<DraftState> = {}) => state({
    workflow: 'extend', prompt: '', scene: '', audio: true, fps: 24, frameCount: 97, ratio: '16:9', dimensionMode: 'ratio', width: 848, height: 480,
    extendSourcePath: '/clips/walk.mp4', extendFrames: 48, ...overrides,
  });

  it('describes the job by its source and new frames, with no frame shape or reference image', () => {
    expect(enhanceContext(extending())).toEqual({
      ratio: null, width: null, height: null, frames: null, fps: null, audio: true, reference_image: false,
      source_video: '/clips/walk.mp4', extend_frames: 48,
    });
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

  it('is never a no-op, since an extension always writes a new action', () => {
    expect(splitsPrompt(extending())).toBe(true);
    expect(splitsPrompt(extending({ scene: 'A street.', prompt: 'She waves.' }))).toBe(true);
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

    it('is not out of date for Enhanced text from before extensions existed', () => {
      const old = state({ workflow: 'txt2vid', prompt: 'p', enhancedPrompt: 'x', enhancedFrom: { prompt: 'p', mode: 'video', scene: '' } });
      expect(isEnhancedStale(old)).toBe(false);
    });
  });
});
