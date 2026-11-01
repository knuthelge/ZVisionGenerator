import { beforeEach, describe, expect, it, vi } from 'vitest';

import { AUDIO_ACCEPT, formatTimestamp, isSupportedAudio, startNote, takeAudioFiles, trackOf, trackSet } from './audioTrack';
import { addToast } from './toasts.svelte';
import type { DraftState } from '$lib/types';

vi.mock('./toasts.svelte', () => ({ addToast: vi.fn() }));

describe('formatTimestamp', () => {
  it.each([
    [0, '0:00'],
    [5, '0:05'],
    [90, '1:30'],
    [600, '10:00'],
    [5400, '1:30:00'],
    [1.6, '0:01.6'],
    [59.96, '1:00'],
  ])('writes %s s as %s', (seconds, text) => {
    expect(formatTimestamp(seconds)).toBe(text);
  });
});

describe('startNote', () => {
  it('shows the seconds and the timestamp beside it', () => {
    expect(startNote(90)).toBe('90 → 1:30');
    expect(startNote(6.65)).toBe('6.65 → 0:06.7');
    expect(startNote(0)).toBe('0 → 0:00');
  });
});

describe('audio files', () => {
  it('takes the extensions the server takes, by name', () => {
    for (const name of ['a.wav', 'a.MP3', 'a.m4a', 'a.flac', 'a.ogg']) expect(isSupportedAudio(new File(['x'], name))).toBe(true);
    for (const name of ['notes.txt', 'clip.mp4', 'photo.png']) expect(isSupportedAudio(new File(['x'], name))).toBe(false);
    expect(AUDIO_ACCEPT).toContain('.m4a');
  });
});

describe('the track fields of a tab', () => {
  const state = { audioTrackPath: '/a.mp3', audioTrackUpload: null, audioTrackStart: 4, extendAudioTrackPath: null, extendAudioTrackUpload: 'memo.m4a', extendAudioTrackStart: 9 } as DraftState;

  it('reads each tab\'s own', () => {
    expect(trackOf(state, 'video')).toEqual({ path: '/a.mp3', upload: null, start: 4 });
    expect(trackOf(state, 'extend')).toEqual({ path: null, upload: 'memo.m4a', start: 9 });
  });

  it('is set by a path or an upload marker', () => {
    expect(trackSet({ path: '/a.mp3', upload: null })).toBe(true);
    expect(trackSet({ path: null, upload: 'memo.m4a' })).toBe(true);
    expect(trackSet({ path: null, upload: null })).toBe(false);
    expect(trackSet({ path: '', upload: null })).toBe(false);
  });

});

describe('takeAudioFiles', () => {
  beforeEach(() => vi.mocked(addToast).mockClear());

  it('hands over the first audio file and raises no toast', () => {
    const onfile = vi.fn();
    const song = new File(['x'], 'song.mp3');

    takeAudioFiles([new File(['x'], 'notes.txt'), song, new File(['x'], 'other.wav')], onfile);

    expect(onfile).toHaveBeenCalledExactlyOnceWith(song);
    expect(addToast).not.toHaveBeenCalled();
  });

  it('tells the user when nothing dropped is audio, and ignores an empty drop', () => {
    const onfile = vi.fn();

    takeAudioFiles([new File(['x'], 'photo.png')], onfile);
    takeAudioFiles([], onfile);

    expect(onfile).not.toHaveBeenCalled();
    expect(addToast).toHaveBeenCalledTimes(1);
  });
});
