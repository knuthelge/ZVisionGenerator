import { beforeEach, describe, expect, it, vi } from 'vitest';

import { videoFacts } from './videoFacts.svelte';

const mocks = vi.hoisted(() => ({ browseInfo: vi.fn() }));
vi.mock('$lib/api/browse', () => ({ browseInfo: mocks.browseInfo }));

const flush = () => new Promise((resolve) => setTimeout(resolve, 0));

beforeEach(() => {
  videoFacts.reset();
  mocks.browseInfo.mockReset();
});

describe('videoFacts', () => {
  it('has no facts until they arrive, then keeps the rate, frame count and first-frame time', async () => {
    mocks.browseInfo.mockResolvedValue({ width: 768, height: 448, fps: 24, frames: 121, seconds: 5.04, start: 0.021 });

    expect(videoFacts.of('/clips/a.mp4')).toBeNull();
    videoFacts.ensure('/clips/a.mp4');
    await flush();

    expect(videoFacts.of('/clips/a.mp4')).toEqual({ fps: 24, frames: 121, start: 0.021 });
    expect(mocks.browseInfo).toHaveBeenCalledExactlyOnceWith('source_video', '/clips/a.mp4');
  });

  it('asks once per path, and keeps each path apart', async () => {
    mocks.browseInfo.mockImplementation(async (_purpose: string, path: string) => ({ fps: path.endsWith('a.mp4') ? 24 : 30, frames: 100, start: 0 }));

    videoFacts.ensure('/a.mp4');
    videoFacts.ensure('/a.mp4');
    videoFacts.ensure('/b.mp4');
    await flush();
    videoFacts.ensure('/a.mp4');

    expect(mocks.browseInfo).toHaveBeenCalledTimes(2);
    expect(videoFacts.of('/a.mp4')?.fps).toBe(24);
    expect(videoFacts.of('/b.mp4')?.fps).toBe(30);
  });

  it('has none for a video whose request failed, and asks again on the next view', async () => {
    mocks.browseInfo.mockRejectedValueOnce(new Error('gone')).mockResolvedValueOnce({ fps: 24, frames: 100, start: 0 });

    videoFacts.ensure('/a.mp4');
    await flush();
    expect(videoFacts.of('/a.mp4')).toBeNull();

    videoFacts.ensure('/a.mp4');
    await flush();

    expect(mocks.browseInfo).toHaveBeenCalledTimes(2);
    expect(videoFacts.of('/a.mp4')).toEqual({ fps: 24, frames: 100, start: 0 });
  });

  it.each([
    ['no frame rate', { fps: null, frames: 100, start: 0 }],
    ['no frame count', { fps: 24, frames: null, start: 0 }],
    ['no first-frame time', { fps: 24, frames: 100 }],
    ['a first-frame time that is not a number', { fps: 24, frames: 100, start: Number.NaN }],
  ])('has none for an answer with %s', async (_label, info) => {
    mocks.browseInfo.mockResolvedValue(info);

    videoFacts.ensure('/a.mp4');
    await flush();
    videoFacts.ensure('/a.mp4');

    expect(videoFacts.of('/a.mp4')).toBeNull();
    expect(mocks.browseInfo).toHaveBeenCalledTimes(1);
  });
});
