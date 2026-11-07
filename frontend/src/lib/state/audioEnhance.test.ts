import { describe, expect, it } from 'vitest';

import { AUDIO_MATRIX, AUDIO_SETTINGS } from '../../test-utils/audio';
import { audioEnhanceContext, audioSettingsPayload, axesOf, effectiveAudioSettings, enhanceBlock, enhanceTaskOf, isDefaultAudioSettings, musicSings, parseAudioRecord, taskSings } from './audioEnhance';
import type { AudioEnhanceSettings, DraftState, PromptEnhancerContract } from '$lib/types';

const NONE = { sound: null, style: null };
const settings = (overrides: Partial<AudioEnhanceSettings> = {}): AudioEnhanceSettings => ({ ...AUDIO_MATRIX.defaults, ...overrides });
const contract = (overrides: Partial<PromptEnhancerContract> = {}) => ({ audio_matrix: AUDIO_MATRIX, model: 'owner/llm', error: null, ...overrides }) as PromptEnhancerContract;

describe('axesOf and enhanceTaskOf', () => {
  it("offers Sound's four axes to the sound task and Music's six to the style task", () => {
    expect(axesOf(AUDIO_MATRIX, 'sound').map((axis) => axis.key)).toEqual(['space', 'distance', 'over_time', 'movement']);
    expect(axesOf(AUDIO_MATRIX, 'style').map((axis) => axis.key)).toEqual(['genre', 'music_mood', 'instruments', 'without', 'singer', 'production']);
  });

  it('writes a Sound prompt and a Music style, and nothing for Speech, which is read as written', () => {
    expect([enhanceTaskOf('sound'), enhanceTaskOf('music'), enhanceTaskOf('speech')]).toEqual(['sound', 'style', null]);
  });
});

describe('effectiveAudioSettings', () => {
  it('starts from the defaults, then the choices made, and leaves out an option the server no longer offers', () => {
    expect(effectiveAudioSettings({ enhanceSettingsAudio: NONE }, 'sound', AUDIO_MATRIX)).toEqual(AUDIO_MATRIX.defaults);
    expect(effectiveAudioSettings({ enhanceSettingsAudio: { ...NONE, sound: { space: 'forest', note: 'dry' } } }, 'sound', AUDIO_MATRIX)).toMatchObject({ space: 'forest', distance: 'keep', note: 'dry' });
    expect(effectiveAudioSettings({ enhanceSettingsAudio: { ...NONE, sound: { space: 'a-gone-option' } } }, 'sound', AUDIO_MATRIX).space).toBe('keep');
  });
});

describe('audioSettingsPayload', () => {
  it('sends only what differs from the defaults on the task\'s own axes, and the direction when it is not blank', () => {
    expect(audioSettingsPayload(settings(), 'sound', AUDIO_MATRIX)).toEqual({});
    expect(audioSettingsPayload(settings({ space: 'forest', over_time: 'builds', note: '  ' }), 'sound', AUDIO_MATRIX)).toEqual({ space: 'forest', over_time: 'builds' });
    expect(audioSettingsPayload(settings({ note: 'keep it far off' }), 'sound', AUDIO_MATRIX)).toEqual({ note: 'keep it far off' });
  });

  it('leaves out an option of the other kind', () => {
    expect(audioSettingsPayload(settings({ genre: 'folk', distance: 'far' }), 'sound', AUDIO_MATRIX)).toEqual({ distance: 'far' });
  });

  it('tells whether every option is at its default', () => {
    expect(isDefaultAudioSettings(settings(), 'sound', AUDIO_MATRIX)).toBe(true);
    expect(isDefaultAudioSettings(settings({ movement: 'passing' }), 'sound', AUDIO_MATRIX)).toBe(false);
    expect(isDefaultAudioSettings(settings({ note: 'x' }), 'sound', AUDIO_MATRIX)).toBe(false);
  });
});

describe('audioEnhanceContext', () => {
  const NO_LYRICS = { shown: { showLyrics: false, showTempo: false, showKey: false }, music: null };
  const SONG = { shown: { showLyrics: true, showTempo: true, showKey: true }, music: { kind: 'song' as const } };
  const state = (overrides: Record<string, unknown> = {}) => ({ soundType: 'auto' as const, audioSeconds: 0, lyrics: '', audioTempo: null, audioKey: null, ...overrides });

  it('sends the Sound Type and the length set, or what Auto stands for at that type', () => {
    expect(audioEnhanceContext(state({ soundType: 'effect', audioSeconds: 8 }), 'sound', AUDIO_SETTINGS, NO_LYRICS)).toEqual({ sound_type: 'effect', seconds: 8 });
    expect(audioEnhanceContext(state({ soundType: 'one-shot' }), 'sound', AUDIO_SETTINGS, NO_LYRICS)).toEqual({ sound_type: 'one-shot', seconds: AUDIO_SETTINGS.auto_seconds['one-shot'] });
  });

  it('sends nothing of a sound to a style, and never the length', () => {
    expect(audioEnhanceContext(state({ soundType: 'effect', audioSeconds: 8 }), 'style', AUDIO_SETTINGS, NO_LYRICS)).toEqual({ instrumental: true });
  });

  it('tells a style for a song the start of its lyrics, its tempo and its key', () => {
    const lyrics = 'la '.repeat(400);
    const context = audioEnhanceContext(state({ lyrics, audioTempo: 96, audioKey: 'Am' }), 'style', AUDIO_SETTINGS, SONG);
    expect(context).toEqual({ instrumental: false, lyrics: lyrics.slice(0, 600), tempo: 96, key: 'Am' });
  });

  it('leaves out a tempo and key that are Auto or whose control the model does not show', () => {
    expect(audioEnhanceContext(state({ lyrics: 'la' }), 'style', AUDIO_SETTINGS, SONG)).toEqual({ instrumental: false, lyrics: 'la' });
    const hidden = { ...SONG, shown: { showLyrics: true, showTempo: false, showKey: false } };
    expect(audioEnhanceContext(state({ lyrics: 'la', audioTempo: 96, audioKey: 'Am' }), 'style', AUDIO_SETTINGS, hidden)).toEqual({ instrumental: false, lyrics: 'la' });
  });

  it('sends no lyrics for a model that sings none, even when the draft keeps some, and calls the take instrumental', () => {
    expect(audioEnhanceContext(state({ lyrics: 'la la' }), 'style', AUDIO_SETTINGS, NO_LYRICS)).toEqual({ instrumental: true });
  });

  it('calls a song with no words, or only section tags, instrumental', () => {
    const instrumental = { ...SONG, music: { kind: 'instrumental' as const } };
    expect(audioEnhanceContext(state({ lyrics: '[verse]' }), 'style', AUDIO_SETTINGS, instrumental)).toEqual({ instrumental: true });
  });
});

describe('Enhance on an empty Style', () => {
  it('is held back with the empty-text reason, whatever Describe it holds', () => {
    expect(enhanceBlock(contract(), 'style', { jobsActive: false, busy: false, text: '  ' })).toBe('Write the style first.');
    expect(enhanceBlock(contract(), 'style', { jobsActive: false, busy: false, text: 'folk' })).toBeNull();
  });
});

describe('musicSings and the Singer axis', () => {
  const SONG = { shown: { showLyrics: true, showTempo: true, showKey: true }, music: { kind: 'song' as const } };

  it('sings when the model takes lyrics and they hold words', () => {
    expect(musicSings({ lyrics: 'la' }, SONG)).toBe(true);
    expect(musicSings({ lyrics: '[verse]' }, { ...SONG, music: { kind: 'instrumental' } })).toBe(false);
    expect(musicSings({ lyrics: '' }, { ...SONG, music: null })).toBe(false);
    expect(musicSings({ lyrics: 'la' }, { ...SONG, music: null })).toBe(true);
    expect(musicSings({ lyrics: 'la' }, { ...SONG, shown: { ...SONG.shown, showLyrics: false } })).toBe(false);
  });

  it('has a voice for a sound, which has no Singer to hide', () => {
    expect(taskSings({ lyrics: '' }, 'sound', { shown: { ...SONG.shown, showLyrics: false }, music: null })).toBe(true);
    expect(taskSings({ lyrics: '' }, 'style', { shown: { ...SONG.shown, showLyrics: false }, music: null })).toBe(false);
  });

  it('leaves Singer out of the axes and the payload of a take that does not sing', () => {
    expect(axesOf(AUDIO_MATRIX, 'style', false).map((axis) => axis.key)).toEqual(['genre', 'music_mood', 'instruments', 'without', 'production']);
    const chosen = settings({ singer: 'duet', genre: 'folk', instruments: ['piano'], without: ['drums'] });
    expect(audioSettingsPayload(chosen, 'style', AUDIO_MATRIX)).toEqual({ genre: 'folk', instruments: ['piano'], without: ['drums'], singer: 'duet' });
    expect(audioSettingsPayload(chosen, 'style', AUDIO_MATRIX, false)).toEqual({ genre: 'folk', instruments: ['piano'], without: ['drums'] });
    expect(isDefaultAudioSettings(settings({ singer: 'duet' }), 'style', AUDIO_MATRIX, false)).toBe(true);
    expect(isDefaultAudioSettings(settings({ singer: 'duet' }), 'style', AUDIO_MATRIX)).toBe(false);
  });
});

describe('enhanceBlock', () => {
  const ready = { jobsActive: false, busy: false, text: 'rain' };

  it('is null when the enhancer is there, no take is being made and there is text', () => {
    expect(enhanceBlock(contract(), 'sound', ready)).toBeNull();
  });

  it.each([
    ['a take is running or queued', { ...ready, jobsActive: true }, 'Available when the take is done.'],
    ['a take is being submitted', { ...ready, busy: true }, 'Available once the take is submitted.'],
    ['the box is empty', { ...ready, text: '  ' }, 'Describe the sound first.'],
  ])('says why when %s', (_name, state, hint) => {
    expect(enhanceBlock(contract(), 'sound', state)).toBe(hint);
  });

  it('says so when no enhancer model is configured', () => {
    expect(enhanceBlock(contract({ model: null, error: 'No model.' }), 'sound', ready)).toBe('No model.');
  });
});

describe('parseAudioRecord', () => {
  it('reads a saved record, keeping its tier', () => {
    const json = '{"format":"text","model":"owner/llm","settings":{"space":"forest"},"task":"sound","tier":"fast"}';

    expect(parseAudioRecord(json)).toEqual({ settings: { space: 'forest' }, model: 'owner/llm', tier: 'fast', format: 'text', task: 'sound' });
  });

  it('reads the record of a script the helper wrote: what it did, and for a fit the time and the pace', () => {
    expect(parseAudioRecord('{"format":"text","model":"owner/llm","settings":{"actions":["delivery"]},"task":"script","tier":"smart"}')).toEqual({ settings: { actions: ['delivery'] }, model: 'owner/llm', tier: 'smart', format: 'text', task: 'script' });
    expect(parseAudioRecord('{"model":"owner/llm","settings":{"actions":["fit"],"seconds":20,"pace":"slow","extra":1},"task":"script"}')).toEqual({ settings: { actions: ['fit'], seconds: 20, pace: 'slow' }, model: 'owner/llm', format: 'text', task: 'script' });
  });

  it('reads the record of a fit and delivery in one run, in run order', () => {
    expect(parseAudioRecord('{"model":"owner/llm","settings":{"actions":["delivery","fit"],"seconds":20},"task":"script"}')).toEqual({ settings: { actions: ['fit', 'delivery'], seconds: 20 }, model: 'owner/llm', format: 'text', task: 'script' });
  });

  it.each(['{"model":"m","task":"script"}', '{"model":"m","task":"script","settings":{"actions":["fit"]}}', '{"model":"m","task":"script","settings":{"actions":["write"]}}', '{"model":"m","task":"script","settings":{"action":"delivery"}}'])('reads %j as no record', (value) => {
    expect(parseAudioRecord(value)).toBeNull();
  });

  it.each([undefined, '', 'not json', '{}', '{"model":"m"}', '{"model":"m","task":"lyrics"}', '{"model":"m","task":"fill"}'])('reads %j as no record', (value) => {
    expect(parseAudioRecord(value)).toBeNull();
  });
});
