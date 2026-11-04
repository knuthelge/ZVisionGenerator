import { describe, expect, it } from 'vitest';

import { AUDIO_MODELS, withAudio } from '../../test-utils/audio';
import { makeContext } from '../../test-utils/workspace';
import { audioPrefill, availableKinds, clampSeconds, hydrateAudio, replacesVoice } from './audioDraft';
import type { DraftState } from '$lib/types';

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

  it('gives a length nobody chose the default of the sound model, and keeps one that was chosen', () => {
    const slower = withAudio(makeContext(), AUDIO_MODELS.map((model) => (model.kind === 'sound' ? { ...model, limits: { ...model.limits, default_seconds: 25 } } : model)));
    const sound = { workflow: 'sound', audioKind: 'sound' } as const;

    expect(hydrateAudio(slower, state({ ...sound, audioSeconds: 0 })).audioSeconds).toBe(25);
    expect(hydrateAudio(slower, state({ ...sound, audioSeconds: 10 })).audioSeconds).toBe(10);
    expect(hydrateAudio(slower, state({ audioSeconds: 0 })).audioSeconds).toBe(0);
  });

  it('moves a sound\'s length into the model\'s range', () => {
    expect(hydrateAudio(ctx, state({ workflow: 'sound', audioKind: 'sound', audioSeconds: 900 })).audioSeconds).toBe(380);
    expect(hydrateAudio(ctx, state({ workflow: 'sound', audioKind: 'sound', audioSeconds: 0.2 })).audioSeconds).toBe(1);
    expect(clampSeconds(Number.NaN, AUDIO_MODELS[2])).toBe(10);
  });
});

describe('audioPrefill', () => {

  it('makes a Reuse of a song without lyrics an instrumental, and leaves a bare link\'s boxes alone', () => {
    expect(audioPrefill({ workflow: 'music', prompt: 'Dreamy' }, ctx, 'music').patch.lyrics).toBe('');
    expect(audioPrefill({ workflow: 'music' }, ctx, 'music').patch).toEqual({});
  });

  it('reads a sound\'s length, and a Reuse of speech without a recording as Any voice', () => {
    expect(audioPrefill({ workflow: 'sound', prompt: 'rain', seconds: '45' }, ctx, 'sound').patch).toMatchObject({ soundPrompt: 'rain', audioSeconds: 45 });
    const speech = audioPrefill({ workflow: 'speech', prompt: 'Hello' }, ctx, 'speech');
    expect(speech.patch).toMatchObject({ speechText: 'Hello', voiceMode: 'any', voicePath: null, voiceUpload: null, voiceText: '' });
    expect(speech.voice).toEqual({ exempt: false });
  });

  it('follows a recording and its words, and never exempts a Reuse from the tick', () => {
    const { patch, voice } = audioPrefill({ workflow: 'speech', prompt: 'Hello', voice_path: '/voices/me.wav', voice_text: 'Hi' }, ctx, 'speech');

    expect(patch).toMatchObject({ speechText: 'Hello', voiceMode: 'recording', voicePath: '/voices/me.wav', voiceText: 'Hi' });
    expect(voice).toEqual({ exempt: false });
  });

  it('takes a model only when the machine runs it for the kind', () => {
    expect(audioPrefill({ workflow: 'music', model: 'higgs3', prompt: 'x' }, ctx, 'music').patch.model).toBeUndefined();
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
