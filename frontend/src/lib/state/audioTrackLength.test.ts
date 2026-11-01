import { beforeEach, describe, expect, it, vi } from 'vitest';

import { trackKey, trackLengths } from './audioTrackLength.svelte';

const mocks = vi.hoisted(() => ({ inspectAudioTrack: vi.fn(), readFileLength: vi.fn() }));
vi.mock('$lib/api/audioTrack', () => ({ inspectAudioTrack: mocks.inspectAudioTrack }));
vi.mock('./audioLength', () => ({ readFileLength: mocks.readFileLength }));

const flush = () => new Promise((resolve) => setTimeout(resolve, 0));

beforeEach(() => {
  trackLengths.reset();
  mocks.inspectAudioTrack.mockReset();
  mocks.readFileLength.mockReset();
});

describe('trackLengths', () => {
  it('asks the server for a path once and remembers the length', async () => {
    mocks.inspectAudioTrack.mockResolvedValue({ seconds: 4, name: 'song.mp3' });

    trackLengths.ensure('/a/song.mp3', null);
    trackLengths.ensure('/a/song.mp3', null);
    await flush();
    trackLengths.ensure('/a/song.mp3', null);

    expect(mocks.inspectAudioTrack).toHaveBeenCalledExactlyOnceWith('/a/song.mp3');
    expect(trackLengths.lengthOf('path:/a/song.mp3')).toBe(4);
  });

  it('reads an upload in the browser and never asks the server', async () => {
    const file = new File(['x'], 'memo.m4a');
    mocks.readFileLength.mockResolvedValue(3.6);

    trackLengths.ensure('/ignored.mp3', file);
    await flush();

    expect(mocks.inspectAudioTrack).not.toHaveBeenCalled();
    expect(trackLengths.lengthOf(trackKey(null, file))).toBe(3.6);
  });

  it('leaves the length unknown, without asking again, when the server or the browser cannot read the file', async () => {
    mocks.inspectAudioTrack.mockRejectedValue(new Error('unreadable'));

    trackLengths.ensure('/a/gone.mp3', null);
    await flush();
    trackLengths.ensure('/a/gone.mp3', null);

    expect(trackLengths.lengthOf('path:/a/gone.mp3')).toBeNull();
    expect(mocks.inspectAudioTrack).toHaveBeenCalledTimes(1);
  });
});
