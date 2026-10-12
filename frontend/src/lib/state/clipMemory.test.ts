import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { VideoMemoryFitInfo } from '$lib/types';

import { clipMemory } from './clipMemory.svelte';
import { INPUT_DEBOUNCE_MS } from './debounce';
import type { ClipMemoryRequest } from './memoryFit';

const apiMocks = vi.hoisted(() => ({ checkVideoMemoryFit: vi.fn<(body: ClipMemoryRequest) => Promise<VideoMemoryFitInfo>>() }));
vi.mock('$lib/api/videoMemory', () => ({ checkVideoMemoryFit: apiMocks.checkVideoMemoryFit }));

const REQUEST: ClipMemoryRequest = { model: 'ltx-8', ratio: '16:9', size: 'm', frames: 121, low_memory: true, upscale: null };
const TIGHT = { status: 'tight', required_gb: 21, budget_gb: 24 };

function answer(memory_fit: VideoMemoryFitInfo['memory_fit']): VideoMemoryFitInfo {
  return { memory_fit, width: 704, height: 448, frames: 121 };
}

function deferred<T>(): { promise: Promise<T>; resolve: (value: T) => void; reject: (reason: unknown) => void } {
  let resolve!: (value: T) => void;
  let reject!: (reason: unknown) => void;
  const promise = new Promise<T>((done, fail) => { resolve = done; reject = fail; });
  return { promise, resolve, reject };
}

async function settle(): Promise<void> {
  await vi.advanceTimersByTimeAsync(INPUT_DEBOUNCE_MS);
}

describe('clipMemory', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    clipMemory.reset();
    apiMocks.checkVideoMemoryFit.mockReset();
  });

  afterEach(() => {
    clipMemory.reset();
    vi.useRealTimers();
  });

  it('asks the server for the clip once the inputs have been still, and keeps the answer', async () => {
    apiMocks.checkVideoMemoryFit.mockResolvedValue(answer(TIGHT));
    clipMemory.watch(REQUEST);
    expect(apiMocks.checkVideoMemoryFit).not.toHaveBeenCalled();

    await settle();

    expect(apiMocks.checkVideoMemoryFit).toHaveBeenCalledExactlyOnceWith(REQUEST);
    expect(clipMemory.fit).toEqual(TIGHT);
  });

  it('asks only for the last of a burst of changes', async () => {
    apiMocks.checkVideoMemoryFit.mockResolvedValue(answer(TIGHT));
    clipMemory.watch(REQUEST);
    await vi.advanceTimersByTimeAsync(INPUT_DEBOUNCE_MS - 50);
    clipMemory.watch({ ...REQUEST, frames: 145 });
    await vi.advanceTimersByTimeAsync(INPUT_DEBOUNCE_MS - 50);
    clipMemory.watch({ ...REQUEST, frames: 169 });
    await settle();

    expect(apiMocks.checkVideoMemoryFit).toHaveBeenCalledExactlyOnceWith({ ...REQUEST, frames: 169 });
  });

  it('does not ask again for unchanged inputs', async () => {
    apiMocks.checkVideoMemoryFit.mockResolvedValue(answer(TIGHT));
    clipMemory.watch(REQUEST);
    await settle();

    clipMemory.watch({ ...REQUEST });
    await settle();

    expect(apiMocks.checkVideoMemoryFit).toHaveBeenCalledOnce();
  });

  it('does not restart the wait for unchanged inputs', async () => {
    apiMocks.checkVideoMemoryFit.mockResolvedValue(answer(TIGHT));
    clipMemory.watch(REQUEST);
    await vi.advanceTimersByTimeAsync(INPUT_DEBOUNCE_MS - 50);
    clipMemory.watch({ ...REQUEST });
    await vi.advanceTimersByTimeAsync(50);

    expect(apiMocks.checkVideoMemoryFit).toHaveBeenCalledOnce();
  });

  it('drops an answer that arrives after a newer request was made', async () => {
    const first = deferred<VideoMemoryFitInfo>();
    const second = deferred<VideoMemoryFitInfo>();
    apiMocks.checkVideoMemoryFit.mockReturnValueOnce(first.promise).mockReturnValueOnce(second.promise);
    clipMemory.watch(REQUEST);
    await settle();
    clipMemory.watch({ ...REQUEST, frames: 241 });
    await settle();

    second.resolve(answer({ status: 'too_large', required_gb: 40, budget_gb: 24 }));
    await vi.advanceTimersByTimeAsync(0);
    first.resolve(answer(TIGHT));
    await vi.advanceTimersByTimeAsync(0);

    expect(clipMemory.fit).toEqual({ status: 'too_large', required_gb: 40, budget_gb: 24 });
  });

  it('drops an answer to a request made before the inputs changed again', async () => {
    const first = deferred<VideoMemoryFitInfo>();
    apiMocks.checkVideoMemoryFit.mockReturnValueOnce(first.promise).mockResolvedValue(answer(null));
    clipMemory.watch(REQUEST);
    await settle();
    clipMemory.watch({ ...REQUEST, frames: 241 });

    first.resolve(answer(TIGHT));
    await vi.advanceTimersByTimeAsync(0);

    expect(clipMemory.fit).toBeNull();
  });

  it('keeps the previous fit while a changed length is being asked, and replaces it with the answer', async () => {
    apiMocks.checkVideoMemoryFit.mockResolvedValueOnce(answer(TIGHT)).mockResolvedValueOnce(answer(null));
    clipMemory.watch(REQUEST);
    await settle();

    clipMemory.watch({ ...REQUEST, frames: 49 });
    expect(clipMemory.fit).toEqual(TIGHT);
    await settle();

    expect(clipMemory.fit).toBeNull();
  });

  it('forgets the fit at once when the model changes', async () => {
    apiMocks.checkVideoMemoryFit.mockResolvedValue(answer(TIGHT));
    clipMemory.watch(REQUEST);
    await settle();

    clipMemory.watch({ ...REQUEST, model: 'ltx-4' });

    expect(clipMemory.fit).toBeNull();
  });

  it('forgets the fit at once, and cancels what is pending, when there is no clip to check', async () => {
    apiMocks.checkVideoMemoryFit.mockResolvedValue(answer(TIGHT));
    clipMemory.watch(REQUEST);
    await settle();
    clipMemory.watch({ ...REQUEST, frames: 241 });

    clipMemory.watch(null);
    await settle();

    expect(clipMemory.fit).toBeNull();
    expect(apiMocks.checkVideoMemoryFit).toHaveBeenCalledOnce();
  });

  it('asks again for a clip it had forgotten', async () => {
    apiMocks.checkVideoMemoryFit.mockResolvedValue(answer(TIGHT));
    clipMemory.watch(REQUEST);
    await settle();
    clipMemory.watch(null);

    clipMemory.watch(REQUEST);
    await settle();

    expect(apiMocks.checkVideoMemoryFit).toHaveBeenCalledTimes(2);
    expect(clipMemory.fit).toEqual(TIGHT);
  });

  it('shows nothing when the server cannot answer', async () => {
    apiMocks.checkVideoMemoryFit.mockResolvedValueOnce(answer(TIGHT)).mockRejectedValueOnce(new Error('width must be a positive integer.'));
    clipMemory.watch(REQUEST);
    await settle();

    clipMemory.watch({ ...REQUEST, frames: 241 });
    await settle();

    expect(clipMemory.fit).toBeNull();
  });
});
