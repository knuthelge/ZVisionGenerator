import { describe, expect, it } from 'vitest';

import { AUDIO_MODELS, AUDIO_SETTINGS, SA3_MUSIC } from '../../test-utils/audio';
import { keyName, keyNameOf, keyOptions, keyShortName, musicSummary } from './musicSettings';

describe('keyName', () => {
  it('writes the sharp and flat of the root as ♯ and ♭', () => {
    expect(keyName('F# minor')).toBe('F♯ minor');
    expect(keyName('Eb major')).toBe('E♭ major');
    expect(keyName('A minor')).toBe('A minor');
    expect(keyName('B major')).toBe('B major');
  });

  it('names the key of an id from the vocabulary, and falls back to the id', () => {
    expect(keyNameOf(AUDIO_SETTINGS, 'Am')).toBe('A minor');
    expect(keyNameOf(AUDIO_SETTINGS, 'Db')).toBe('D♭ major');
    expect(keyNameOf(AUDIO_SETTINGS, 'H')).toBe('H');
    expect(keyNameOf(undefined, 'Am')).toBe('Am');
  });

  it('shortens a name for a chip', () => {
    expect(keyShortName('A minor')).toBe('A min');
    expect(keyShortName('C♯ major')).toBe('C♯ maj');
  });
});

describe('musicSummary', () => {
  const yue = AUDIO_MODELS[0];

  it('says Auto until a tempo or a key is set, then what is set', () => {
    expect(musicSummary({ audioTempo: null, audioKey: null, musicSeconds: 0 }, yue, AUDIO_SETTINGS)).toBe('Auto');
    expect(musicSummary({ audioTempo: 92, audioKey: null, musicSeconds: 0 }, yue, AUDIO_SETTINGS)).toBe('92 BPM');
    expect(musicSummary({ audioTempo: null, audioKey: 'F#m', musicSeconds: 0 }, yue, AUDIO_SETTINGS)).toBe('F♯ minor');
    expect(musicSummary({ audioTempo: 92, audioKey: 'Am', musicSeconds: 0 }, yue, AUDIO_SETTINGS)).toBe('92 BPM · A minor');
  });

  it('adds the length for a model that sets it, the default one for 0, and none for a model that does not', () => {
    expect(musicSummary({ audioTempo: null, audioKey: null, musicSeconds: 0 }, SA3_MUSIC, AUDIO_SETTINGS)).toBe('Auto · 30 s');
    expect(musicSummary({ audioTempo: 92, audioKey: 'Am', musicSeconds: 45 }, SA3_MUSIC, AUDIO_SETTINGS)).toBe('92 BPM · A minor · 45 s');
    expect(musicSummary({ audioTempo: 92, audioKey: 'Am', musicSeconds: 45 }, yue, AUDIO_SETTINGS)).toBe('92 BPM · A minor');
  });
});

describe('keyOptions', () => {
  it('lists Auto first and then the 24 keys by name, major before minor', () => {
    const options = keyOptions(AUDIO_SETTINGS);

    expect(options[0]).toEqual({ value: '', label: 'Auto', hint: 'The model picks' });
    expect(options).toHaveLength(25);
    expect(options.find((option) => option.value === 'F#m')).toEqual({ value: 'F#m', label: 'F♯ minor', group: 'Minor' });
    expect(options.find((option) => option.value === 'Db')).toEqual({ value: 'Db', label: 'D♭ major', group: 'Major' });
  });

  it('offers Auto alone while the vocabulary is not at hand', () => {
    expect(keyOptions(undefined).map((option) => option.value)).toEqual(['']);
  });
});
