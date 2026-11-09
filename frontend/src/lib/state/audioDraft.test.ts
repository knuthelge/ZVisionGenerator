import { describe, expect, it } from 'vitest';

import { AUDIO_MODELS, withAudio } from '../../test-utils/audio';
import { makeContext } from '../../test-utils/workspace';
import { AUDIO_FIELD_DEFAULTS, audioModelById, audioModelsOf, audioPrefill, availableKinds, clampSeconds, hydrateAudio, isAudioModelId, musicSecondsShown, replacesVoice, secondsRange, settleMusicSeconds, settledSpeaker, voicesOf } from './audioDraft';
import type { AudioModelOption, DraftState } from '$lib/types';

const ctx = withAudio(makeContext());

/** A draft on the Audio tab, as the page would hold it. */
function state(overrides: Partial<DraftState> = {}): DraftState {
  return { workflow: 'music', model: '', audioKind: 'music', audioModels: { music: '', speech: '', sound: '' }, audioSeconds: 10, ...overrides } as DraftState;
}

describe('availableKinds', () => {
  it('lists the kinds the contract lists, in the Kind control\'s order', () => {
    expect(availableKinds(ctx)).toEqual(['music', 'speech', 'sound']);
    expect(availableKinds(withAudio(makeContext(), AUDIO_MODELS.filter((model) => model.kind === 'sound')))).toEqual(['sound']);
    expect(availableKinds(makeContext())).toEqual([]);
  });
});

/** Stable Audio as a model of two kinds: sound, and music without lyrics whose length is chosen. */
const SA3_MUSIC: AudioModelOption = { ...AUDIO_MODELS[2], kind: 'music', hidden_controls: ['lyrics'], limits: { min_seconds: 1, max_seconds: 380, default_seconds: 30 } };
const twoKinds = withAudio(makeContext(), [...AUDIO_MODELS, SA3_MUSIC]);

describe('a model of several kinds', () => {
  it('is listed under each of its kinds, in config order', () => {
    expect(audioModelsOf(twoKinds, 'music').map((model) => model.id)).toEqual(['yue2', 'sa3']);
    expect(audioModelsOf(twoKinds, 'sound').map((model) => model.id)).toEqual(['sa3']);
    expect(audioModelsOf(twoKinds, 'speech').map((model) => model.id)).toEqual(['qwen3_tts']);
  });

  it('is looked up by id and kind: each kind is its own row', () => {
    expect(audioModelById(twoKinds, 'sa3', 'music')).toBe(SA3_MUSIC);
    expect(audioModelById(twoKinds, 'sa3', 'sound')).toBe(AUDIO_MODELS[2]);
    expect(audioModelById(twoKinds, 'sa3', 'speech')).toBeNull();
    expect(audioModelById(twoKinds, 'nope', 'music')).toBeNull();
    expect(audioModelById(null, 'sa3', 'music')).toBeNull();
  });

  it('is an audio model whatever the kind', () => {
    expect([isAudioModelId(twoKinds, 'sa3'), isAudioModelId(twoKinds, 'yue2'), isAudioModelId(twoKinds, 'zit'), isAudioModelId(null, 'sa3')]).toEqual([true, true, false, false]);
  });

  it('keeps its model when the draft moves between its kinds', () => {
    const music = hydrateAudio(twoKinds, state({ workflow: 'sound', audioKind: 'sound', model: 'sa3' }));
    const asMusic = hydrateAudio(twoKinds, state({ model: 'sa3' }));

    expect([music.model, asMusic.model]).toEqual(['sa3', 'sa3']);
  });

  it('settles a Reuse of a music take on the music row of the model, from whichever kind and model the draft was on', () => {
    const reused = audioPrefill({ model: 'sa3', prompt: 'calm piano', seed: '7' }, twoKinds, 'music');
    const fromYue = hydrateAudio(twoKinds, state({ model: 'yue2' }), reused.patch.model ?? null);
    const fromSound = hydrateAudio(twoKinds, state({ workflow: 'music', audioKind: 'music', model: 'sa3', audioModels: { music: '', speech: '', sound: 'sa3' } }), reused.patch.model ?? null);

    for (const settled of [fromYue, fromSound]) {
      expect(settled).toMatchObject({ workflow: 'music', audioKind: 'music', model: 'sa3' });
      expect(settled.audioModels.music).toBe('sa3');
      expect(audioModelById(twoKinds, settled.model, 'music')).toBe(SA3_MUSIC);
    }
  });

  it('takes a model in a Reuse for the kind it makes', () => {
    expect(audioPrefill({ model: 'sa3', prompt: 'piano' }, twoKinds, 'music').patch.model).toBe('sa3');
    expect(audioPrefill({ model: 'sa3', prompt: 'hello' }, twoKinds, 'speech').patch.model).toBeUndefined();
  });
});

describe('hydrateAudio', () => {

  it('prefers the asked model, then the draft\'s, then the one last chosen for the kind', () => {
    const two = withAudio(makeContext(), [...AUDIO_MODELS, { ...AUDIO_MODELS[0], id: 'yue3', label: 'YuE3' }]);

    expect(hydrateAudio(two, state({ model: 'yue2' }), 'yue3').model).toBe('yue3');
    expect(hydrateAudio(two, state({ model: 'yue3' })).model).toBe('yue3');
    expect(hydrateAudio(two, state({ model: 'zit', audioModels: { music: 'yue3', speech: '', sound: '' } })).model).toBe('yue3');
    expect(hydrateAudio(two, state({ model: 'zit' }), 'nope').model).toBe('yue2');
  });

  it('moves to a kind the machine runs when it runs none of the open one', () => {
    const soundOnly = withAudio(makeContext(), AUDIO_MODELS.filter((model) => model.kind === 'sound'));

    expect(hydrateAudio(soundOnly, state())).toMatchObject({ workflow: 'sound', audioKind: 'sound', model: 'sa3' });
  });

  it('leaves a sound\'s length on Auto (0) and keeps one that was chosen', () => {
    const sound = { workflow: 'sound', audioKind: 'sound' } as const;

    expect(hydrateAudio(ctx, state({ ...sound, audioSeconds: 0 })).audioSeconds).toBe(0);
    expect(hydrateAudio(ctx, state({ ...sound, audioSeconds: 10 })).audioSeconds).toBe(10);
    expect(hydrateAudio(ctx, state({ audioSeconds: 0 })).audioSeconds).toBe(0);
  });

  it('moves a sound\'s length into the model\'s range', () => {
    expect(hydrateAudio(ctx, state({ workflow: 'sound', audioKind: 'sound', audioSeconds: 900 })).audioSeconds).toBe(380);
    expect(hydrateAudio(ctx, state({ workflow: 'sound', audioKind: 'sound', audioSeconds: 0.2 })).audioSeconds).toBe(1);
    expect(clampSeconds(Number.NaN, AUDIO_MODELS[2])).toBe(12);
  });
});

describe('the built-in voice of the draft', () => {
  const speech = (overrides: Partial<DraftState> = {}) => state({ workflow: 'speech', audioKind: 'speech', model: 'qwen3_tts', ...overrides });

  it('keeps the model\'s default voice as no choice, and one the model has as itself', () => {
    expect(settledSpeaker('ryan', AUDIO_MODELS[1])).toBeNull();
    expect(settledSpeaker('vivian', AUDIO_MODELS[1])).toBe('vivian');
    expect(settledSpeaker(null, AUDIO_MODELS[1])).toBeNull();
  });

  it('drops a voice the model does not have', () => {
    expect(settledSpeaker('nobody', AUDIO_MODELS[1])).toBeNull();
    expect(settledSpeaker('vivian', AUDIO_MODELS[0])).toBeNull();
    expect(settledSpeaker('vivian', null)).toBeNull();
  });

  it('settles the voice when the draft settles on its model', () => {
    expect(hydrateAudio(ctx, speech({ speechSpeaker: 'ryan' })).speechSpeaker).toBeNull();
    expect(hydrateAudio(ctx, speech({ speechSpeaker: 'sohee' })).speechSpeaker).toBe('sohee');
    expect(hydrateAudio(ctx, speech({ speechSpeaker: 'nobody' })).speechSpeaker).toBeNull();
  });

  it('lists the voices of a speech model, and none for another kind', () => {
    expect(voicesOf(AUDIO_MODELS[1]).map((voice) => voice.id)).toEqual(['ryan', 'aiden', 'vivian', 'serena', 'uncle_fu', 'dylan', 'eric', 'ono_anna', 'sohee']);
    expect(voicesOf(AUDIO_MODELS[0])).toEqual([]);
    expect(voicesOf(null)).toEqual([]);
  });
});

describe('audioPrefill', () => {

  it('makes a Reuse of a song without lyrics an instrumental, and leaves a bare link\'s boxes alone', () => {
    expect(audioPrefill({ workflow: 'music', prompt: 'Dreamy' }, ctx, 'music').patch.lyrics).toBe('');
    expect(audioPrefill({ workflow: 'music' }, ctx, 'music').patch).toEqual({});
  });

  it('reads a sound\'s length, and a Reuse of speech without a recording as the default built-in voice', () => {
    expect(audioPrefill({ workflow: 'sound', prompt: 'rain', seconds: '45' }, ctx, 'sound').patch).toMatchObject({ soundPrompt: 'rain', audioSeconds: 45 });
    const speech = audioPrefill({ workflow: 'speech', prompt: 'Hello' }, ctx, 'speech');
    expect(speech.patch).toMatchObject({ speechText: 'Hello', voiceMode: 'builtin', speechSpeaker: null, voicePath: null, voiceUpload: null, voiceText: '' });
    expect(speech.voice).toEqual({ exempt: false });
  });

  it('reads the built-in voice a Reuse of speech names, and replaces the voice when only the speaker is named', () => {
    expect(audioPrefill({ workflow: 'speech', prompt: 'Hello', speaker: 'vivian' }, ctx, 'speech').patch).toMatchObject({ voiceMode: 'builtin', speechSpeaker: 'vivian', voicePath: null });
    expect(audioPrefill({ workflow: 'speech', speaker: 'vivian' }, ctx, 'speech').patch).toMatchObject({ voiceMode: 'builtin', speechSpeaker: 'vivian' });
    expect(replacesVoice({ speaker: 'vivian' }, 'speech')).toBe(true);
  });

  it('follows the saved voice a Reuse names, without a recording, a transcript, a speaker or a delivery, and needs no exemption', () => {
    const { patch, voice } = audioPrefill({ workflow: 'speech', prompt: 'Hello', saved_voice: 'narrator', voice_text: 'ignored', emotion: 'sad' }, ctx, 'speech');

    expect(patch).toMatchObject({ speechText: 'Hello', voiceMode: 'saved', savedVoice: 'narrator', voicePath: null, voiceUpload: null, voiceText: '' });
    expect(patch).not.toHaveProperty('speechSpeaker');
    expect(patch).not.toHaveProperty('speechDelivery');
    expect(voice).toEqual({ exempt: false });
    expect(replacesVoice({ saved_voice: 'narrator' }, 'speech')).toBe(true);
    expect(replacesVoice({ saved_voice: 'narrator' }, 'music')).toBe(false);
  });

  it('clears the saved voice when a Reuse names a built-in voice or a recording', () => {
    expect(audioPrefill({ workflow: 'speech', prompt: 'Hello', speaker: 'vivian' }, ctx, 'speech').patch).toMatchObject({ voiceMode: 'builtin', savedVoice: null });
    expect(audioPrefill({ workflow: 'speech', prompt: 'Hello', voice_path: '/voices/me.wav', voice_text: 'Hi' }, ctx, 'speech').patch).toMatchObject({ voiceMode: 'recording', savedVoice: null });
  });

  it('prefers the recording to a speaker when a link names both', () => {
    const { patch } = audioPrefill({ workflow: 'speech', prompt: 'Hello', speaker: 'vivian', voice_path: '/voices/me.wav', voice_text: 'Hi' }, ctx, 'speech');

    expect(patch).toMatchObject({ voiceMode: 'recording', voicePath: '/voices/me.wav' });
    expect(patch).not.toHaveProperty('speechSpeaker');
  });

  it('follows a recording and its words, and never exempts a Reuse from the tick', () => {
    const { patch, voice } = audioPrefill({ workflow: 'speech', prompt: 'Hello', voice_path: '/voices/me.wav', voice_text: 'Hi' }, ctx, 'speech');

    expect(patch).toMatchObject({ speechText: 'Hello', voiceMode: 'recording', voicePath: '/voices/me.wav', voiceText: 'Hi' });
    expect(voice).toEqual({ exempt: false });
  });

  it('takes a model only when the machine runs it for the kind', () => {
    expect(audioPrefill({ workflow: 'music', model: 'qwen3_tts', prompt: 'x' }, ctx, 'music').patch.model).toBeUndefined();
  });
});

describe('the settings of each kind in a prefill', () => {
  it('reads the tempo and key of a song, a blank one as Auto', () => {
    expect(audioPrefill({ workflow: 'music', prompt: 'x', tempo: '92', key: 'F#m' }, ctx, 'music').patch).toMatchObject({ audioTempo: 92, audioKey: 'F#m' });
    expect(audioPrefill({ workflow: 'music', prompt: 'x', tempo: '', key: '' }, ctx, 'music').patch).toMatchObject({ audioTempo: null, audioKey: null });
    expect(audioPrefill({ workflow: 'music', prompt: 'x', tempo: '92.4' }, ctx, 'music').patch.audioTempo).toBe(92);
  });

  it('names no tempo or key when the params have none, so a link leaves them alone', () => {
    const { patch } = audioPrefill({ workflow: 'music', prompt: 'x' }, ctx, 'music');

    expect(patch).not.toHaveProperty('audioTempo');
    expect(patch).not.toHaveProperty('audioKey');
  });

  it('reads the length by kind: a sound\'s into the sound length, a song\'s into the song length, a blank one as 0', () => {
    expect(audioPrefill({ workflow: 'sound', prompt: 'x', seconds: '45' }, ctx, 'sound').patch).toMatchObject({ audioSeconds: 45 });
    expect(audioPrefill({ workflow: 'sound', prompt: 'x', seconds: '' }, ctx, 'sound').patch).toMatchObject({ audioSeconds: 0 });
    expect(audioPrefill({ workflow: 'music', prompt: 'x', seconds: '45' }, twoKinds, 'music').patch).toMatchObject({ musicSeconds: 45 });
    expect(audioPrefill({ workflow: 'music', prompt: 'x', seconds: '45' }, twoKinds, 'music').patch).not.toHaveProperty('audioSeconds');
    expect(audioPrefill({ workflow: 'music', prompt: 'x', seconds: '' }, twoKinds, 'music').patch).toMatchObject({ musicSeconds: 0 });
    expect(audioPrefill({ workflow: 'speech', prompt: 'x', seconds: '45' }, ctx, 'speech').patch).not.toHaveProperty('audioSeconds');
  });

  it('reads the type of a sound, and skips one the vocabulary does not have', () => {
    expect(audioPrefill({ workflow: 'sound', prompt: 'x', sound_type: 'one-shot' }, ctx, 'sound').patch.soundType).toBe('one-shot');
    expect(audioPrefill({ workflow: 'sound', prompt: 'x', sound_type: 'song' }, ctx, 'sound').patch).not.toHaveProperty('soundType');
    expect(audioPrefill({ workflow: 'music', prompt: 'x', sound_type: 'effect' }, ctx, 'music').patch).not.toHaveProperty('soundType');
  });

  it('reads the delivery of a built-in voice, the values the params do not name at their defaults', () => {
    const named = { workflow: 'speech', prompt: 'x', emotion: 'sad', manner: 'whisper', pace: 'slow', pitch: 'low', expressiveness: 'high' };

    expect(audioPrefill(named, ctx, 'speech').patch.speechDelivery).toEqual({ emotion: 'sad', manner: 'whisper', pace: 'slow', pitch: 'low', expressiveness: 'high' });
    expect(audioPrefill({ workflow: 'speech', prompt: 'x', pace: 'fast' }, ctx, 'speech').patch.speechDelivery).toEqual({ ...AUDIO_FIELD_DEFAULTS.speechDelivery, pace: 'fast' });
    expect(audioPrefill({ workflow: 'speech', prompt: 'x' }, ctx, 'speech').patch).not.toHaveProperty('speechDelivery');
  });

  it('reads no delivery for a take that followed a recording, or for another kind', () => {
    expect(audioPrefill({ workflow: 'speech', prompt: 'x', voice_path: '/voices/me.wav', voice_text: 'Hi', emotion: 'sad' }, ctx, 'speech').patch).not.toHaveProperty('speechDelivery');
    expect(audioPrefill({ workflow: 'music', prompt: 'x', emotion: 'sad' }, ctx, 'music').patch).not.toHaveProperty('speechDelivery');
  });

  it('skips a setting whose control the kind does not list, unless ungated', () => {
    const bare = { ...ctx, workflow_contract: { ...ctx.workflow_contract, definitions: { ...ctx.workflow_contract.definitions, music: { ...ctx.workflow_contract.definitions.music!, visible_controls: ['music_style', 'lyrics'] } } } };

    expect(audioPrefill({ workflow: 'music', prompt: 'x', tempo: '92', key: 'C' }, bare, 'music').patch).not.toHaveProperty('audioTempo');
    expect(audioPrefill({ workflow: 'music', prompt: 'x', tempo: '92', key: 'C' }, bare, 'music', true).patch).toMatchObject({ audioTempo: 92, audioKey: 'C' });
  });
});

describe('the length of a song', () => {
  it('has a range on a music row that sets its length, and none on one that follows its lyrics', () => {
    expect(secondsRange(SA3_MUSIC)).toEqual({ min: 1, max: 380 });
    expect(secondsRange(AUDIO_MODELS[0])).toBeNull();
    expect(secondsRange(AUDIO_MODELS[1])).toBeNull();
  });

  it('moves into the range of the music row when the draft settles, and 0 stays the model\'s default', () => {
    const music = { workflow: 'music', audioKind: 'music', model: 'sa3' } as const;

    expect(hydrateAudio(twoKinds, state({ ...music, musicSeconds: 900 })).musicSeconds).toBe(380);
    expect(hydrateAudio(twoKinds, state({ ...music, musicSeconds: 0 })).musicSeconds).toBe(0);
    expect(hydrateAudio(twoKinds, state({ ...music, model: 'yue2', musicSeconds: 900 })).musicSeconds).toBe(900);
    expect(clampSeconds(500, SA3_MUSIC)).toBe(380);
  });

  it('shows the row\'s default length for 0, and none for a row that sets no length', () => {
    expect(musicSecondsShown(0, SA3_MUSIC)).toBe(30);
    expect(musicSecondsShown(45, SA3_MUSIC)).toBe(45);
    expect(musicSecondsShown(45, AUDIO_MODELS[0])).toBeNull();
    expect(musicSecondsShown(0, AUDIO_MODELS[2])).toBeNull();
    expect(musicSecondsShown(0, null)).toBeNull();
  });

  it('keeps a typed length in the range, and the default length as 0 so it is not a change', () => {
    expect(settleMusicSeconds(45, SA3_MUSIC)).toBe(45);
    expect(settleMusicSeconds(900, SA3_MUSIC)).toBe(380);
    expect(settleMusicSeconds(0.2, SA3_MUSIC)).toBe(1);
    expect(settleMusicSeconds(30, SA3_MUSIC)).toBe(0);
    expect(settleMusicSeconds(45, AUDIO_MODELS[0])).toBe(45);
  });

  it('is kept apart from a sound\'s length across the kinds', () => {
    const settled = hydrateAudio(twoKinds, state({ workflow: 'music', audioKind: 'music', model: 'sa3', audioSeconds: 0, musicSeconds: 45 }));

    expect(settled).toMatchObject({ audioSeconds: 0, musicSeconds: 45 });
  });
});

describe('replacesVoice', () => {
  it('is true for Speech params that name a recording or a take, false otherwise', () => {
    expect(replacesVoice({ prompt: 'x' }, 'speech')).toBe(true);
    expect(replacesVoice({ voice_path: '/v.wav' }, 'speech')).toBe(true);
    expect(replacesVoice({ use_as: 'voice', voice_path: '/v.wav' }, 'speech')).toBe(true);
    expect(replacesVoice({ use_as: 'voice' }, 'speech')).toBe(false);
    expect(replacesVoice({ workflow: 'speech' }, 'speech')).toBe(false);
    expect(replacesVoice({ prompt: 'x' }, 'music')).toBe(false);
  });
});

describe('audioPrefill of Enhanced text', () => {
  const record = JSON.stringify({ format: 'text', model: 'owner/llm', settings: { space: 'forest' }, task: 'sound', tier: 'smart' });
  const reuse = { workflow: 'sound', model: 'sa3', prompt: 'rain', enhanced_prompt: 'Heavy rain.', enhance: record };

  it('puts the rewrite in the Sound box with what it was made from, its options and its enhancer', () => {
    const { patch } = audioPrefill(reuse, ctx, 'sound');

    expect(patch.soundPrompt).toBe('rain');
    expect(patch.audioRewrites?.sound).toMatchObject({ text: 'Heavy rain.', from: { input: 'rain', model: 'sa3', task: 'sound', record: { model: 'owner/llm', tier: 'smart', settings: { space: 'forest' } } } });
    expect(patch.audioRewrites?.sound.history.map((entry) => entry.text)).toEqual(['Heavy rain.']);
    expect(patch.enhanceSettingsAudio).toEqual({ sound: { space: 'forest' }, style: null });
    expect(patch.enhanceModel).toBe('smart');
  });

  it('keeps the other boxes\' Enhanced text and options, and the box\'s earlier versions of the same text', () => {
    const earlier = { text: 'Soft rain.', from: { input: 'rain', model: 'sa3', task: 'sound' as const }, history: [{ text: 'Soft rain.', from: { input: 'rain', model: 'sa3', task: 'sound' as const } }] };
    const other = { text: 'Folk.', from: null, history: [] };
    const current = { audioRewrites: { ...AUDIO_FIELD_DEFAULTS.audioRewrites, sound: earlier, style: other }, enhanceSettingsAudio: { sound: null, style: { genre: 'folk' } } };

    const { patch } = audioPrefill(reuse, ctx, 'sound', false, current);

    expect(patch.audioRewrites?.style).toBe(other);
    expect(patch.audioRewrites?.sound.history.map((entry) => entry.text)).toEqual(['Soft rain.', 'Heavy rain.']);
    expect(patch.enhanceSettingsAudio?.style).toEqual({ genre: 'folk' });
  });

  it('puts the Written text of a Speech take in its box with the script record, and sets no Enhance options', () => {
    const record = JSON.stringify({ format: 'text', model: 'owner/llm', settings: { actions: ['fit', 'delivery'], seconds: 20 }, task: 'script', tier: 'fast' });
    const { patch } = audioPrefill({ workflow: 'speech', model: 'qwen3_tts', prompt: 'Hello.', enhanced_prompt: '[sad] Hello.', enhance: record }, ctx, 'speech');

    expect(patch.speechText).toBe('Hello.');
    expect(patch.audioRewrites?.text).toMatchObject({ text: '[sad] Hello.', from: { input: 'Hello.', model: 'qwen3_tts', task: 'script', record: { settings: { actions: ['fit', 'delivery'], seconds: 20 }, tier: 'fast' } } });
    expect(patch.enhanceSettingsAudio).toBeUndefined();
    expect(patch.enhanceModel).toBe('fast');
  });

  it('keeps a Written text whose record is broken as typed-in text of the script task', () => {
    const { patch } = audioPrefill({ workflow: 'speech', model: 'qwen3_tts', prompt: 'Hello.', enhanced_prompt: '[sad] Hello.', enhance: 'not json' }, ctx, 'speech');

    expect(patch.audioRewrites?.text).toMatchObject({ text: '[sad] Hello.', from: { input: 'Hello.', task: 'script' } });
    expect(patch.audioRewrites?.text.from).not.toHaveProperty('record');
  });

  it('empties the box\'s Enhanced text for a take that had none, and for a link that names no text it sets nothing', () => {
    const current = { audioRewrites: { ...AUDIO_FIELD_DEFAULTS.audioRewrites, sound: { text: 'older', from: null, history: [] } }, enhanceSettingsAudio: AUDIO_FIELD_DEFAULTS.enhanceSettingsAudio };

    expect(audioPrefill({ workflow: 'sound', model: 'sa3', prompt: 'a door slam' }, ctx, 'sound', false, current).patch.audioRewrites?.sound).toMatchObject({ text: '', from: null });
    expect(audioPrefill({ workflow: 'sound', model: 'sa3' }, ctx, 'sound', false, current).patch.audioRewrites).toBeUndefined();
  });

  it('keeps an Enhanced text whose record is broken, as typed-in text of the kind\'s own task, and sets no options', () => {
    const { patch } = audioPrefill({ ...reuse, enhance: 'not json' }, ctx, 'sound');

    expect(patch.audioRewrites?.sound).toMatchObject({ text: 'Heavy rain.', from: { input: 'rain', task: 'sound' } });
    expect(patch.audioRewrites?.sound.from).not.toHaveProperty('record');
    expect(patch.enhanceSettingsAudio).toBeUndefined();
  });
});
