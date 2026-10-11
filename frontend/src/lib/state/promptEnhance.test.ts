import { describe, it, expect } from 'vitest';
import {
  clampedNote,
  effectiveEnhanceSettings,
  enhanceContext,
  enhancePhaseMessage,
  enhancedOverrideActive,
  hasAudio,
  isEnhancedStale,
  isNoOpSettings,
  settingsPayload,
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
    jsonPromptEnabled: false,
    enhanceAuto: false,
    enhancedPrompt: '',
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
  it('is false for hand-typed text', () => {
    expect(isEnhancedStale(state({ enhancedPrompt: 'typed', enhancedFrom: null }))).toBe(false);
  });

  it('is false when prompt and mode match', () => {
    expect(isEnhancedStale(state({ enhancedPrompt: 'x', enhancedFrom: { prompt: 'a fox', mode: 'image' } }))).toBe(false);
  });

  it('is true after the prompt changes', () => {
    expect(isEnhancedStale(state({ prompt: 'a cat', enhancedPrompt: 'x', enhancedFrom: { prompt: 'a fox', mode: 'image' } }))).toBe(true);
  });

  it('is true after the workflow mode changes', () => {
    expect(isEnhancedStale(state({ workflow: 'txt2vid', enhancedPrompt: 'x', enhancedFrom: { prompt: 'a fox', mode: 'image' } }))).toBe(true);
  });

  it('is false when the box is empty', () => {
    expect(isEnhancedStale(state({ prompt: 'a cat', enhancedPrompt: '', enhancedFrom: { prompt: 'a fox', mode: 'image' } }))).toBe(false);
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

  it('treats a non-keep mood as a change', () => {
    expect(isNoOpSettings({ ...SETTINGS, mood: 'eerie' }, 'image')).toBe(false);
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
