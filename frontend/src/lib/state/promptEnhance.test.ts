import { describe, it, expect } from 'vitest';
import {
  canSubmitText,
  clampedNote,
  effectiveEnhanceSettings,
  enhanceContext,
  enhancePhaseMessage,
  enhancedOverrideActive,
  hasAudio,
  isEnhancedStale,
  isNoOpSettings,
  settingsPayload,
  splitsPrompt,
  submittedPair,
  submittedPrompt,
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

describe('submission rule', () => {
  it.each([
    [{}, false, 'a fox'],
    [{ enhancedPrompt: 'A fox in snow.' }, true, 'A fox in snow.'],
    [{ enhancedPrompt: '   ' }, false, 'a fox'],
    [{ enhancedPrompt: 'A fox in snow.', promptSource: 'file' as const }, false, 'a fox'],
    [{ enhancedPrompt: 'A fox in snow.', jsonPromptEnabled: true }, false, 'a fox'],
    [{ enhancedPrompt: 'A fox in snow.', enhanceAuto: true }, false, 'a fox'],
  ])('%o → override %s', (overrides, active, prompt) => {
    expect(enhancedOverrideActive(state(overrides))).toBe(active);
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

  it('is never a no-op for a split video prompt, and unchanged for images', () => {
    expect(isNoOpSettings(SETTINGS, 'video', false, true)).toBe(false);
    expect(isNoOpSettings(SETTINGS, 'video', false, false)).toBe(true);
    expect(isNoOpSettings(SETTINGS, 'image', false, true)).toBe(true);
  });

  it('uses the Enhanced pair when either Enhanced box has text', () => {
    expect(enhancedOverrideActive(video({ enhancedScene: 'A castle at dusk.' }))).toBe(true);
    expect(enhancedOverrideActive(video({ enhancedPrompt: 'A knight rides in.' }))).toBe(true);
    expect(enhancedOverrideActive(video({ enhancedScene: ' ', enhancedPrompt: ' ' }))).toBe(false);
    // An image never carries a scene, so a stray Enhanced scene does not switch its text.
    expect(enhancedOverrideActive(state({ workflow: 'txt2img', enhancedScene: 'A castle.' }))).toBe(false);
  });

  it('submits the Enhanced pair, with no scene when the enhanced scene is blank', () => {
    const base = { scene: 'A castle.', prompt: 'a knight' };
    expect(submittedPair(video(base))).toEqual({ scene: 'A castle.', prompt: 'a knight' });
    expect(submittedPair(video({ ...base, enhancedScene: 'A castle at dusk.', enhancedPrompt: 'A knight rides in.' }))).toEqual({ scene: 'A castle at dusk.', prompt: 'A knight rides in.' });
    expect(submittedPair(video({ ...base, enhancedPrompt: 'A knight rides in.' }))).toEqual({ scene: '', prompt: 'A knight rides in.' });
    expect(submittedPair(state({ workflow: 'txt2img', scene: 'ignored', prompt: 'a fox' }))).toEqual({ scene: '', prompt: 'a fox' });
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
