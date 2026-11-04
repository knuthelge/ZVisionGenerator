import { beforeEach, describe, expect, it, vi } from 'vitest';

import { audioPeaks } from './audioPeaks.svelte';

const mocks = vi.hoisted(() => ({ getAudioPeaks: vi.fn() }));
vi.mock('$lib/api/gallery', () => ({ getAudioPeaks: mocks.getAudioPeaks }));

const flush = () => new Promise((resolve) => setTimeout(resolve, 0));
const take = (id: string, peaks: number[] | null | undefined = null) => ({ id, media_type: 'audio' as const, peaks });

beforeEach(() => {
  audioPeaks.reset();
  mocks.getAudioPeaks.mockReset();
});

describe('audioPeaks', () => {
  it('draws the peaks a listing carried and never asks for them', async () => {
    const asset = take('a.flac', [1, 2, 3]);

    audioPeaks.ensure(asset);
    await flush();

    expect(audioPeaks.of(asset)).toEqual([1, 2, 3]);
    expect(mocks.getAudioPeaks).not.toHaveBeenCalled();
  });

  it('asks the server once for a take whose listing had none, and draws them when they arrive', async () => {
    mocks.getAudioPeaks.mockResolvedValue([9, 8, 7]);
    const asset = take('a.flac');

    expect(audioPeaks.of(asset)).toEqual([]);
    audioPeaks.ensure(asset);
    audioPeaks.ensure(asset);
    await flush();
    audioPeaks.ensure(asset);

    expect(mocks.getAudioPeaks).toHaveBeenCalledExactlyOnceWith('a.flac');
    expect(audioPeaks.of(asset)).toEqual([9, 8, 7]);
  });

  it('leaves a flat line, without asking again, when the server cannot read the take', async () => {
    mocks.getAudioPeaks.mockRejectedValue(new Error('gone'));
    const asset = take('a.flac');

    audioPeaks.ensure(asset);
    await flush();
    audioPeaks.ensure(asset);

    expect(audioPeaks.of(asset)).toEqual([]);
    expect(mocks.getAudioPeaks).toHaveBeenCalledTimes(1);
  });

  it('does not ask for images or videos', () => {
    audioPeaks.ensure({ id: 'a.png', media_type: 'image', peaks: undefined });

    expect(mocks.getAudioPeaks).not.toHaveBeenCalled();
  });

  it('asks for a few takes at a time and starts the next as each one answers', async () => {
    const answers: Array<(peaks: number[]) => void> = [];
    mocks.getAudioPeaks.mockImplementation(() => new Promise<number[]>((resolve) => answers.push(resolve)));

    for (const id of ['1', '2', '3', '4', '5']) audioPeaks.ensure(take(id));
    expect(mocks.getAudioPeaks).toHaveBeenCalledTimes(3);

    answers[0]([5]);
    await flush();

    expect(mocks.getAudioPeaks).toHaveBeenCalledTimes(4);
    expect(audioPeaks.of(take('1'))).toEqual([5]);
  });
});
