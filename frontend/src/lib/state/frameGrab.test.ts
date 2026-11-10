import { beforeEach, describe, expect, it, vi } from 'vitest';

import { ApiError } from '$lib/api/client';
import type { FramePoint } from '$lib/state/framePoint';
import { toasts } from '$lib/state/toasts.svelte';
import type { GalleryAsset } from '$lib/types';

import { FRAME_ERROR, busyFor, runFrameGrab } from './frameGrab.svelte';

const mocks = vi.hoisted(() => ({ grabFrame: vi.fn() }));
vi.mock('$lib/api/videoFrame', () => ({ grabFrame: mocks.grabFrame }));

const asset = { id: 'out/clip.mp4', filename: 'clip.mp4', media_type: 'video', file_path: '/outputs/out/clip.mp4' } as GalleryAsset;
const point: FramePoint = { time: 3.02, seconds: 3, tooShort: false, atEnd: false };
const saved = { path: '/up/clip_3.0s.png' };

describe('runFrameGrab', () => {
  beforeEach(() => {
    mocks.grabFrame.mockReset();
    toasts.length = 0;
  });

  it('extracts the frame by its time and hands the saved file to apply', async () => {
    mocks.grabFrame.mockResolvedValue(saved);
    const apply = vi.fn();

    await runFrameGrab(asset, point, 'video', apply);

    expect(mocks.grabFrame).toHaveBeenCalledWith('/outputs/out/clip.mp4', 3.02);
    expect(apply).toHaveBeenCalledWith(saved);
    expect(busyFor(asset.id)).toBeNull();
  });

  it('is busy for the target while the extraction runs', async () => {
    let finish: (value: typeof saved) => void = () => undefined;
    mocks.grabFrame.mockReturnValue(new Promise((resolve) => { finish = resolve; }));

    const run = runFrameGrab(asset, point, 'last', vi.fn());
    expect(busyFor(asset.id)).toBe('last');
    expect(busyFor('out/other.mp4')).toBeNull();
    finish(saved);
    await run;

    expect(busyFor(asset.id)).toBeNull();
  });

  it('ignores a second request for the same video while one runs', async () => {
    let finish: (value: typeof saved) => void = () => undefined;
    mocks.grabFrame.mockReturnValue(new Promise((resolve) => { finish = resolve; }));
    const first = vi.fn();
    const second = vi.fn();

    const run = runFrameGrab(asset, point, 'video', first);
    await runFrameGrab(asset, point, 'last', second);
    finish(saved);
    await run;

    expect(mocks.grabFrame).toHaveBeenCalledOnce();
    expect(second).not.toHaveBeenCalled();
    expect(first).toHaveBeenCalledOnce();
  });

  it('shows the server\'s reason as an error toast, applies nothing and clears the busy state', async () => {
    mocks.grabFrame.mockRejectedValue(new ApiError('failed', 500, 'ffmpeg could not read the frame'));
    const apply = vi.fn();

    await runFrameGrab(asset, point, 'video', apply);

    expect(apply).not.toHaveBeenCalled();
    expect(toasts.at(0)).toMatchObject({ message: 'ffmpeg could not read the frame', type: 'error' });
    expect(busyFor(asset.id)).toBeNull();
  });

  it('falls back to a plain reason when the server gave none', async () => {
    mocks.grabFrame.mockRejectedValue(new ApiError('failed', 500, ''));

    await runFrameGrab(asset, point, 'video', vi.fn());

    expect(toasts.at(0)).toMatchObject({ message: FRAME_ERROR, type: 'error' });
  });

  it('does nothing for a video that is not a file on the host', async () => {
    await runFrameGrab({ ...asset, file_path: undefined }, point, 'video', vi.fn());

    expect(mocks.grabFrame).not.toHaveBeenCalled();
  });
});
