import { describe, expect, it } from 'vitest';

import type { DraftState, ExtendSourceInfo } from '$lib/types';
import sourceFixture from '../../../../tests/fixtures/extend_source_response.json';
import {
  CHOOSE_SOURCE_NOTE, REJECTED_SOURCE_NOTE, extendEnhanceHint, extendGuard, extendMemoryNote, extendPrefillFor, extendSourceFacts, extendSourceUsable,
  finalizeSceneFields, isOriginalLegacyPrompt,
} from './extend';
import { legacySource, sceneSource, sourceInfo } from './extend.fixtures';

describe('extendPrefillFor', () => {
  it('copies a recorded scene and leaves the prompt for what happens next', () => {
    expect(extendPrefillFor(sceneSource())).toEqual({ scene: 'A woman on a rainy street.', prompt: '' });
  });

  it('puts the single prompt of a legacy source into the prompt and leaves the scene empty', () => {
    expect(extendPrefillFor(legacySource())).toEqual({ scene: '', prompt: 'A woman stands in the rain and says "Hello".' });
  });

  it('leaves both empty when nothing is recorded', () => {
    expect(extendPrefillFor(sourceInfo())).toEqual({ scene: '', prompt: '' });
  });
});

describe('isOriginalLegacyPrompt', () => {
  it('is true right after a legacy prefill', () => {
    expect(isOriginalLegacyPrompt({ prompt: extendPrefillFor(legacySource()).prompt }, legacySource())).toBe(true);
  });

  it('ignores whitespace differences, like the server', () => {
    expect(isOriginalLegacyPrompt({ prompt: '  A woman stands\n in the rain   and says "Hello".  ' }, legacySource())).toBe(true);
  });

  it('is false once the text differs, even slightly', () => {
    expect(isOriginalLegacyPrompt({ prompt: 'A woman stands in the rain and says "Hello"!' }, legacySource())).toBe(false);
    expect(isOriginalLegacyPrompt({ prompt: '' }, legacySource())).toBe(false);
  });

  it('is false for sources that are not legacy and without a source', () => {
    expect(isOriginalLegacyPrompt({ prompt: 'She looks up.' }, sceneSource())).toBe(false);
    expect(isOriginalLegacyPrompt({ prompt: '' }, sourceInfo())).toBe(false);
    expect(isOriginalLegacyPrompt({ prompt: 'x' }, null)).toBe(false);
  });
});

describe('extendSourceUsable', () => {
  it('needs a readable source without problems', () => {
    expect(extendSourceUsable(sourceInfo())).toBe(true);
    expect(extendSourceUsable(sourceInfo({ problems: [{ code: 'too_short', message: 'x' }] }))).toBe(false);
    expect(extendSourceUsable(sourceInfo({ fps: null }))).toBe(false);
    expect(extendSourceUsable(null)).toBe(false);
  });
});

describe('finalizeSceneFields', () => {
  const draftState = (overrides: Partial<DraftState> = {}) => ({
    workflow: 'extend', promptSource: 'inline', prompt: '', scene: '', jsonPromptEnabled: false, enhanceAuto: false, enhancedPrompt: '', enhancedScene: '',
    extendSourcePath: '/clips/walk.mp4', extendFrames: 48, ...overrides,
  }) as DraftState;

  const formWith = (fields: Record<string, string>) => {
    const form = new FormData();
    for (const [key, value] of Object.entries(fields)) form.append(key, value);
    return form;
  };

  it('always sends scene for an extension, even blank, with the source and the new frames', () => {
    const form = formWith({ mode: 'video', workflow: 'extend', prompt: '', scene: '' });
    finalizeSceneFields(form, draftState());
    expect(form.get('scene')).toBe('');
    expect(form.get('source_video')).toBe('/clips/walk.mp4');
    expect(form.get('extend_frames')).toBe('48');
  });

  it('adds the scene field when the form has none', () => {
    const form = formWith({ mode: 'video', workflow: 'extend' });
    finalizeSceneFields(form, draftState({ scene: 'A street.' }));
    expect(form.get('scene')).toBe('A street.');
  });

  it('sends the Enhanced scene when the Enhanced pair applies and the form has no scene field', () => {
    const form = formWith({ mode: 'video', workflow: 'extend' });
    finalizeSceneFields(form, draftState({ scene: 'A street.', enhancedPrompt: 'She waves.', enhancedScene: 'A wet street.' }));
    expect(form.get('scene')).toBe('A wet street.');
  });

  it('sends no size, frame, reference or upscale fields', () => {
    const form = formWith({ ratio: '16:9', size: 'm', width: '848', height: '480', frames: '97', image_path: '/a.png', video_upscale_enabled: 'true', video_upscale_factor: '2', scene: 'x', steps: '8' });
    finalizeSceneFields(form, draftState());
    for (const key of ['ratio', 'size', 'width', 'height', 'frames', 'image_path', 'video_upscale_enabled', 'video_upscale_factor']) expect(form.has(key)).toBe(false);
    expect(form.get('steps')).toBe('8');
    expect(form.get('scene')).toBe('x');
  });

  it('leaves out the new frames and the source until they are chosen', () => {
    const form = formWith({ scene: '' });
    finalizeSceneFields(form, draftState({ extendSourcePath: null, extendFrames: null }));
    expect(form.has('source_video')).toBe(false);
    expect(form.has('extend_frames')).toBe(false);
  });

  it('drops a blank scene for other workflows and keeps a written one', () => {
    const blank = formWith({ scene: '  ' });
    finalizeSceneFields(blank, draftState({ workflow: 'txt2vid' }));
    expect(blank.has('scene')).toBe(false);

    const written = formWith({ scene: 'A castle.' });
    finalizeSceneFields(written, draftState({ workflow: 'txt2vid' }));
    expect(written.get('scene')).toBe('A castle.');
    expect(written.has('source_video')).toBe(false);
  });
});

describe('extendGuard', () => {
  const extending = { workflow: 'extend' as const, extendSourcePath: '/clips/walk.mp4' };

  it('never blocks a workflow that does not extend', () => {
    expect(extendGuard({ workflow: 'txt2vid', extendSourcePath: null }, null, 'idle')).toEqual({ kind: 'ok', blocked: false, note: null });
  });

  it('asks for a video when none is chosen or the chosen path has no video', () => {
    expect(extendGuard({ workflow: 'extend', extendSourcePath: null }, null, 'idle')).toEqual({ kind: 'none', blocked: true, note: CHOOSE_SOURCE_NOTE });
    expect(extendGuard(extending, null, 'not_found')).toEqual({ kind: 'none', blocked: true, note: CHOOSE_SOURCE_NOTE });
  });

  it('blocks without a note while the source has not been described yet', () => {
    expect(extendGuard(extending, null, 'checking')).toEqual({ kind: 'checking', blocked: true, note: null });
    expect(extendGuard(extending, null, 'idle')).toEqual({ kind: 'checking', blocked: true, note: null });
  });

  it('points to the Source video section for a source that was read and cannot be extended', () => {
    const rejected = sourceInfo({ problems: [{ code: 'hdr', message: 'It is HDR or wide gamut.' }] });
    expect(extendGuard(extending, rejected, 'ready')).toEqual({ kind: 'rejected', blocked: true, note: REJECTED_SOURCE_NOTE });
    expect(extendGuard(extending, null, 'failed')).toEqual({ kind: 'rejected', blocked: true, note: REJECTED_SOURCE_NOTE });
    expect(extendGuard(extending, sourceInfo({ fps: null }), 'ready').kind).toBe('rejected');
  });

  it('lets a usable source through, whether or not it is being checked again', () => {
    expect(extendGuard(extending, sceneSource(), 'ready')).toEqual({ kind: 'ok', blocked: false, note: null });
    expect(extendGuard(extending, sceneSource(), 'checking').blocked).toBe(false);
  });

  it('words the Enhance hint for each way an extension can be stuck', () => {
    expect(extendEnhanceHint(extendGuard(extending, null, 'not_found'))).toBe('Choose a video to extend first.');
    expect(extendEnhanceHint(extendGuard(extending, sourceInfo({ problems: [{ code: 'hdr', message: 'x' }] }), 'ready'))).toBe('Choose a video that can be extended first.');
    expect(extendEnhanceHint(extendGuard(extending, null, 'checking'))).toBeNull();
    expect(extendEnhanceHint(extendGuard(extending, sceneSource(), 'ready'))).toBeNull();
  });
});

describe('extendSourceFacts', () => {
  const texts = (info: Parameters<typeof extendSourceFacts>[0]) => extendSourceFacts(info).map((fact) => fact.text);

  it('lists the displayed size, frame rate, length and frames, then the audio', () => {
    expect(texts(sceneSource())).toEqual(['512 × 288', '24 fps', '4.04 s', '97 frames', 'Audio']);
  });

  it('keeps a fractional frame rate to two places and drops a trailing zero', () => {
    expect(texts(sceneSource({ fps: 29.97003 }))).toContain('29.97 fps');
    expect(texts(sceneSource({ fps: 30 }))).toContain('30 fps');
  });

  it('marks rotation with a tooltip, and a silent clip without audio', () => {
    const facts = extendSourceFacts(sceneSource({ rotation: 90, has_audio: false }));
    expect(facts.map((fact) => fact.text)).toEqual(['512 × 288', '24 fps', '4.04 s', '97 frames', 'Rotated', 'No audio']);
    expect(facts.find((fact) => fact.key === 'rotated')?.title).toBe('Shown upright. The original is re-encoded once when joined.');
  });

  it('says when nothing was recorded, and leaves out facts the server could not read', () => {
    expect(texts(sourceInfo())).toContain('No prompt recorded');
    expect(texts(legacySource())).not.toContain('No prompt recorded');
    expect(texts(sourceInfo({ width: null, height: null, fps: null, duration_seconds: null, frame_count: null, has_audio: null, rotation: null, prompt: 'x' }))).toEqual([]);
  });
});

describe('extendMemoryNote', () => {
  it('words a tight and a too-large fit', () => {
    expect(extendMemoryNote({ status: 'tight', required_gb: 21, budget_gb: 24 })).toEqual({ status: 'tight', text: 'Tight: needs ~21 GB of 24 GB. The Mac may slow down while generating.' });
    expect(extendMemoryNote({ status: 'too_large', required_gb: 40, budget_gb: 24 })?.text).toBe('Too large: needs ~40 GB of 24 GB. Expect heavy swapping or an out-of-memory error.');
  });

  it('has nothing to say when it fits or is unknown', () => {
    expect(extendMemoryNote({ status: 'fits', required_gb: 10, budget_gb: 24 })).toBeNull();
    expect(extendMemoryNote(null)).toBeNull();
  });
});

// tests/fixtures/extend_source_response.json is also checked by pytest against the inspect route's answer.
describe('the inspect route response', () => {
  const info: ExtendSourceInfo = sourceFixture;

  it('has exactly the fields of ExtendSourceInfo', () => {
    expect(Object.keys(sourceFixture).sort()).toEqual(Object.keys(sourceInfo()).sort());
  });

  it('is read by the helpers that consume it', () => {
    expect(extendSourceUsable(info)).toBe(true);
    expect(extendPrefillFor(info)).toEqual({ scene: 'A woman in a red raincoat on a rainy street at night.', prompt: '' });
    expect(extendSourceFacts(info).map((fact) => fact.text)).toEqual(['512 × 256', '24 fps', '4.04 s', '97 frames', 'Audio']);
    expect(extendMemoryNote(info.memory_fit)).toBeNull();
    expect(isOriginalLegacyPrompt({ prompt: info.prompt ?? '' }, info)).toBe(false);
  });
});
