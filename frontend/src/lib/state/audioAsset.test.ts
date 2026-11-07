import { describe, expect, it } from 'vitest';

import { AUDIO_MODELS, LYRICS, SA3_MUSIC, makeAudioAsset, makeImportedAudio, makeSpeechAsset } from '../../test-utils/audio';
import { audioModelLabel, audioRowOf, cutNote, hasNoWords, lyricLines, seekFraction, takeWords, tileWords, wordBlocks } from './audioAsset';
import type { AudioModelOption } from '$lib/types';

describe('the words of a take', () => {
  it('leaves the [section] lines out of a song\'s lyrics', () => {
    expect(lyricLines(LYRICS)).toEqual(['Streetlights hum a quiet tune', 'Hold the night, don\'t let it go']);
    expect(lyricLines('[verse 2]\n  [bridge]  \nthe words')).toEqual(['the words']);
    expect(lyricLines(null)).toEqual([]);
  });

  it('shows the text of speech and the prompt of a sound as they are', () => {
    expect(tileWords(makeSpeechAsset())).toBe('Welcome back to the workshop.');
    expect(tileWords(makeAudioAsset({ kind: 'sound', prompt: 'Rain on a tin roof', lyrics: null }))).toBe('Rain on a tin roof');
  });

  it('has no words for a file without settings, whose prompt is only its name', () => {
    const imported = makeImportedAudio();
    expect(tileWords(imported)).toBe('');
    expect(hasNoWords(imported)).toBe(true);
    expect(takeWords(imported)).toBe('');
    expect(wordBlocks(imported)).toEqual([]);
  });

  it('gives the viewer a labelled block per part, with the lyrics whole', () => {
    expect(wordBlocks(makeAudioAsset())).toEqual([{ label: 'Style', text: 'Dreamy synth-pop, warm analog pads' }, { label: 'Lyrics', text: LYRICS }]);
    expect(wordBlocks(makeSpeechAsset())).toEqual([{ label: 'Text', text: 'Welcome back to the workshop.' }]);
    expect(wordBlocks(makeAudioAsset({ kind: 'sound', prompt: 'Rain', lyrics: null }))).toEqual([{ label: 'Prompt', text: 'Rain' }]);
  });
});

describe('the cut note', () => {
  it('names the model\'s cap and says how to hear the ending', () => {
    expect(cutNote(makeAudioAsset({ cut: true, duration: 359.9 }), AUDIO_MODELS)).toEqual({
      text: '⚠ Cut at 6:00',
      tip: 'YuE2 3B stops at 6:00. Remove sections to hear its ending.',
    });
  });

  it('falls back to the take\'s length when the model is unknown, and is absent for a take that was not cut', () => {
    expect(cutNote(makeAudioAsset({ cut: true, duration: 120 }), undefined)?.text).toBe('⚠ Cut at 2:00');
    expect(cutNote(makeAudioAsset({ cut: false }), AUDIO_MODELS)).toBeNull();
    expect(cutNote(makeAudioAsset({ cut: true, duration: null }), undefined)).toBeNull();
  });
});

describe('a model of several kinds', () => {
  const soundRow = AUDIO_MODELS[2];
  const musicRow: AudioModelOption = { ...SA3_MUSIC, kind: 'music', limits: { min_seconds: 1, max_seconds: 120, default_seconds: 30 } };
  const rows = [soundRow, musicRow];

  it('is cut at the cap of the kind the take was made as', () => {
    expect(cutNote(makeAudioAsset({ model: 'sa3', kind: 'music', cut: true, duration: 119 }), rows)?.text).toBe('⚠ Cut at 2:00');
    expect(cutNote(makeAudioAsset({ model: 'sa3', kind: 'sound', cut: true, duration: 119 }), rows)?.text).toBe('⚠ Cut at 6:20');
  });

  it('is found by id alone for a take without a kind', () => {
    expect(audioRowOf({ model: 'sa3', kind: null }, rows)).toBe(soundRow);
    expect(audioRowOf({ model: 'sa3', kind: 'speech' }, rows)).toBeUndefined();
  });
});

describe('audioModelLabel', () => {
  it('prefers the server\'s label, then the model list\'s, then the id', () => {
    expect(audioModelLabel(makeAudioAsset({ model_label: 'Server label' }), AUDIO_MODELS)).toBe('Server label');
    expect(audioModelLabel(makeAudioAsset({ model_label: null }), AUDIO_MODELS)).toBe('YuE2 3B');
    expect(audioModelLabel(makeAudioAsset({ model_label: null }), undefined)).toBe('yue2');
  });
});

describe('seekFraction', () => {
  it('maps a pointer to a place in the waveform, clamped to its ends', () => {
    const box = { left: 100, width: 400 };
    expect(seekFraction(300, box)).toBe(0.5);
    expect(seekFraction(50, box)).toBe(0);
    expect(seekFraction(900, box)).toBe(1);
    expect(seekFraction(300, { left: 0, width: 0 })).toBe(0);
  });
});
