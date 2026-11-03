import { describe, expect, it } from 'vitest';

import { AUDIO_MODELS } from '../../test-utils/audio';
import type { DraftState } from '$lib/types';
import { audioView, type AudioViewInput } from './audioView';

const YUE2 = AUDIO_MODELS[0];
const HIGGS = AUDIO_MODELS[1];
const SA3 = AUDIO_MODELS[2];

function state(overrides: Partial<DraftState> = {}): DraftState {
  return {
    workflow: 'music', model: 'yue2', audioKind: 'music', audioStyle: '', lyrics: '', speechText: '', soundPrompt: '', audioSeconds: 10,
    voiceMode: 'any', voicePath: null, voiceUpload: null, voiceText: '', runs: 1, seed: null,
    ...overrides,
  } as DraftState;
}

function view(overrides: Partial<DraftState> = {}, input: Partial<AudioViewInput> = {}) {
  const full = state(overrides);
  const kind = full.workflow as 'music' | 'speech' | 'sound';
  const model = { music: YUE2, speech: HIGGS, sound: SA3 }[kind];
  return audioView({
    state: full,
    caps: { isAudio: true, audioKind: kind, showVoice: kind === 'speech' },
    model,
    file: null,
    restored: true,
    lengthOf: () => null,
    consent: { ticked: false, exempt: false },
    ...input,
  });
}

const RECORDING = { workflow: 'speech' as const, speechText: 'Hello there.', voiceMode: 'recording' as const, voicePath: '/voices/me.wav', voiceText: 'Hi, it is me.' };

describe('audioView reasons', () => {

  it('tells a reason of the voice from one of the text', () => {
    expect(view({ ...RECORDING, voiceText: '' }).reasonFromVoice).toBe(true);
    expect(view(RECORDING).reasonFromVoice).toBe(true);
    expect(view({ ...RECORDING, speechText: '' }).reasonFromVoice).toBe(false);
    expect(view({ ...RECORDING, speechText: 'a'.repeat(10_001) }, { consent: { ticked: true, exempt: false } }).reasonFromVoice).toBe(false);
    expect(view({ workflow: 'speech', speechText: 'Hello there.' }).reasonFromVoice).toBe(false);
  });

  it('puts the over-long text last, with its own words', () => {
    const long = 'a'.repeat(10_001);

    expect(view({ ...RECORDING, speechText: long }, { consent: { ticked: true, exempt: false } }).reason).toBe("Shorten the text: it's over 10,000 characters, about 10 minutes of speech.");
    expect(view({ ...RECORDING, speechText: long }).reason).toBe('Tick the box to confirm you may clone this voice.');
    expect(view({ workflow: 'speech', speechText: 'a'.repeat(10_000) }).reason).toBeNull();
  });

  it('counts a character outside the BMP once, as Python does', () => {
    expect(view({ workflow: 'speech', speechText: 'a😀b' }).characters).toBe(3);
    expect(view({ workflow: 'speech', speechText: '😀'.repeat(10_000) }).reason).toBeNull();
  });

  it('says an uploaded recording that did not come back is gone, once the held uploads have been read', () => {
    const gone = { ...RECORDING, voicePath: null, voiceUpload: 'narrator.wav' };

    expect(view(gone, { restored: true }).reason).toBe('Choose the recording again.');
    // Before the read it is not known to be lost: the upload still counts as chosen.
    expect(view(gone, { restored: false }).voice.gone).toBe(false);
    expect(view(gone, { restored: false }).reason).toBe('Tick the box to confirm you may clone this voice.');
  });
});
