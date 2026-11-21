import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { JobMemory } from '$lib/types';

import { checkMemory } from '$lib/api/memory';
import { imageMemory, singleMemory } from '../../test-utils/memory';
import { memoryCheck } from './memoryCheck.svelte';
import { INPUT_DEBOUNCE_MS } from './debounce';
import type { MemoryRequest } from './memoryRequest';

const apiMocks = vi.hoisted(() => ({ checkMemory: vi.fn<(body: MemoryRequest) => Promise<JobMemory | null>>() }));
vi.mock('$lib/api/memory', () => ({ checkMemory: apiMocks.checkMemory }));

const REQUEST: MemoryRequest = { media: 'video', kind: 'clip', model: 'ltx-8', ratio: '16:9', size: 'm', frames: 121, low_memory: true, upscale: null };
const IMAGE: MemoryRequest = { media: 'image', kind: 'image', model: 'zit', ratio: '1:1', size: 'm' };
const TIGHT = singleMemory('tight');
const TOO_LARGE = singleMemory('too_large');

function deferred<T>(): { promise: Promise<T>; resolve: (value: T) => void; reject: (reason: unknown) => void } {
  let resolve!: (value: T) => void;
  let reject!: (reason: unknown) => void;
  const promise = new Promise<T>((done, fail) => { resolve = done; reject = fail; });
  return { promise, resolve, reject };
}

async function settle(): Promise<void> {
  await vi.advanceTimersByTimeAsync(INPUT_DEBOUNCE_MS);
}

describe('memoryCheck', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    memoryCheck.watch(null);
    apiMocks.checkMemory.mockReset();
  });

  afterEach(() => {
    memoryCheck.watch(null);
    vi.useRealTimers();
  });

  it('asks the server for the job once the inputs have been still, and keeps the answer', async () => {
    apiMocks.checkMemory.mockResolvedValue(TIGHT);
    memoryCheck.watch(REQUEST);
    expect(apiMocks.checkMemory).not.toHaveBeenCalled();

    await settle();

    expect(checkMemory).toHaveBeenCalledExactlyOnceWith(REQUEST);
    expect(memoryCheck.answer).toEqual(TIGHT);
  });

  it('asks only for the last of a burst of changes', async () => {
    apiMocks.checkMemory.mockResolvedValue(TIGHT);
    memoryCheck.watch(REQUEST);
    await vi.advanceTimersByTimeAsync(INPUT_DEBOUNCE_MS - 50);
    memoryCheck.watch({ ...REQUEST, frames: 145 });
    await vi.advanceTimersByTimeAsync(INPUT_DEBOUNCE_MS - 50);
    memoryCheck.watch({ ...REQUEST, frames: 169 });
    await settle();

    expect(apiMocks.checkMemory).toHaveBeenCalledExactlyOnceWith({ ...REQUEST, frames: 169 });
  });

  it('does not ask again for unchanged inputs, nor restart the wait', async () => {
    apiMocks.checkMemory.mockResolvedValue(TIGHT);
    memoryCheck.watch(REQUEST);
    await vi.advanceTimersByTimeAsync(INPUT_DEBOUNCE_MS - 50);
    memoryCheck.watch({ ...REQUEST });
    await vi.advanceTimersByTimeAsync(50);
    memoryCheck.watch({ ...REQUEST });
    await settle();

    expect(apiMocks.checkMemory).toHaveBeenCalledOnce();
  });

  it('asks again when an image gains an upscale or a start image, since either changes what it needs', async () => {
    apiMocks.checkMemory.mockResolvedValue(TIGHT);
    memoryCheck.watch(IMAGE);
    await settle();
    memoryCheck.watch({ ...IMAGE, upscale: 2 });
    await settle();
    memoryCheck.watch({ ...IMAGE, upscale: 2, image: true });
    await settle();

    expect(apiMocks.checkMemory.mock.calls.map(([body]) => [body.upscale, body.image])).toEqual([[undefined, undefined], [2, undefined], [2, true]]);
  });

  it('drops an answer that arrives after a newer request was made', async () => {
    const first = deferred<JobMemory | null>();
    const second = deferred<JobMemory | null>();
    apiMocks.checkMemory.mockReturnValueOnce(first.promise).mockReturnValueOnce(second.promise);
    memoryCheck.watch(REQUEST);
    await settle();
    memoryCheck.watch({ ...REQUEST, frames: 241 });
    await settle();

    second.resolve(TOO_LARGE);
    await vi.advanceTimersByTimeAsync(0);
    first.resolve(TIGHT);
    await vi.advanceTimersByTimeAsync(0);

    expect(memoryCheck.answer).toEqual(TOO_LARGE);
  });

  it('drops an answer to a request made before the inputs changed again', async () => {
    const first = deferred<JobMemory | null>();
    apiMocks.checkMemory.mockReturnValueOnce(first.promise).mockResolvedValue(null);
    memoryCheck.watch(REQUEST);
    await settle();
    memoryCheck.watch({ ...REQUEST, frames: 241 });

    first.resolve(TIGHT);
    await vi.advanceTimersByTimeAsync(0);

    expect(memoryCheck.answer).toBeNull();
  });

  it('keeps the previous answer while a changed length is being asked, and replaces it with the reply', async () => {
    apiMocks.checkMemory.mockResolvedValueOnce(TIGHT).mockResolvedValueOnce(null);
    memoryCheck.watch(REQUEST);
    await settle();

    memoryCheck.watch({ ...REQUEST, frames: 49 });
    expect(memoryCheck.answer).toEqual(TIGHT);
    await settle();

    expect(memoryCheck.answer).toBeNull();
  });

  it('keeps the previous answer, pending, while a changed model is being asked, and clears pending with the reply', async () => {
    apiMocks.checkMemory.mockResolvedValueOnce(TIGHT).mockResolvedValueOnce(TOO_LARGE);
    memoryCheck.watch(REQUEST);
    await settle();
    expect(memoryCheck.pending).toBe(false);

    memoryCheck.watch({ ...REQUEST, model: 'ltx-4' });
    expect(memoryCheck.answer).toEqual(TIGHT);
    expect(memoryCheck.pending).toBe(true);
    await settle();

    expect(memoryCheck.answer).toEqual(TOO_LARGE);
    expect(memoryCheck.pending).toBe(false);
  });

  it('does not mark the answer pending for a change on the same model and kind', async () => {
    apiMocks.checkMemory.mockResolvedValue(TIGHT);
    memoryCheck.watch(REQUEST);
    await settle();

    memoryCheck.watch({ ...REQUEST, frames: 241 });

    expect(memoryCheck.answer).toEqual(TIGHT);
    expect(memoryCheck.pending).toBe(false);
  });

  it('stays pending until the reply to the latest request, not an earlier one', async () => {
    const first = deferred<JobMemory | null>();
    const second = deferred<JobMemory | null>();
    apiMocks.checkMemory.mockResolvedValueOnce(TIGHT).mockReturnValueOnce(first.promise).mockReturnValueOnce(second.promise);
    memoryCheck.watch(REQUEST);
    await settle();
    memoryCheck.watch({ ...REQUEST, model: 'ltx-4' });
    await settle();
    memoryCheck.watch({ ...REQUEST, model: 'ltx-4', frames: 241 });
    await settle();

    first.resolve(TOO_LARGE);
    await vi.advanceTimersByTimeAsync(0);
    expect(memoryCheck.answer).toEqual(TIGHT);
    expect(memoryCheck.pending).toBe(true);

    second.resolve(TOO_LARGE);
    await vi.advanceTimersByTimeAsync(0);
    expect(memoryCheck.answer).toEqual(TOO_LARGE);
    expect(memoryCheck.pending).toBe(false);
  });

  it('clears the answer and pending when the reply to a changed model fails or is empty', async () => {
    apiMocks.checkMemory.mockResolvedValueOnce(TIGHT).mockRejectedValueOnce(new Error('nope')).mockResolvedValueOnce(TIGHT).mockResolvedValueOnce(null);
    memoryCheck.watch(REQUEST);
    await settle();
    memoryCheck.watch({ ...REQUEST, model: 'ltx-4' });
    expect(memoryCheck.pending).toBe(true);
    await settle();
    expect(memoryCheck.answer).toBeNull();
    expect(memoryCheck.pending).toBe(false);

    memoryCheck.watch({ ...REQUEST, model: 'ltx-2' });
    await settle();
    memoryCheck.watch({ ...REQUEST, model: 'ltx-1' });
    expect(memoryCheck.pending).toBe(true);
    await settle();
    expect(memoryCheck.answer).toBeNull();
    expect(memoryCheck.pending).toBe(false);
  });

  it('clears pending along with the answer when there is nothing to check', async () => {
    apiMocks.checkMemory.mockResolvedValue(TIGHT);
    memoryCheck.watch(REQUEST);
    await settle();
    memoryCheck.watch({ ...REQUEST, model: 'ltx-4' });
    memoryCheck.watch(null);
    expect(memoryCheck.pending).toBe(false);

    memoryCheck.watch({ ...REQUEST, model: 'ltx-4' });
    expect(memoryCheck.pending).toBe(true);
    memoryCheck.watch(null);
    expect(memoryCheck.pending).toBe(false);
    expect(memoryCheck.answer).toBeNull();
  });

  it('shows the answer as pending, asking nothing, while the form cannot say what it would submit, until the next request', async () => {
    apiMocks.checkMemory.mockResolvedValue(TIGHT);
    memoryCheck.watch(IMAGE);
    await settle();
    expect(memoryCheck.pending).toBe(false);

    memoryCheck.hold();
    await settle();

    expect(memoryCheck.pending).toBe(true);
    expect(memoryCheck.answer).toEqual(TIGHT);
    expect(apiMocks.checkMemory).toHaveBeenCalledTimes(1);

    memoryCheck.watch(IMAGE);

    expect(memoryCheck.pending).toBe(false);
    memoryCheck.hold();
    memoryCheck.watch(null);
    expect(memoryCheck.pending).toBe(false);
  });

  it('forgets the answer at once, and cancels what is pending, when there is nothing to check', async () => {
    apiMocks.checkMemory.mockResolvedValue(TIGHT);
    memoryCheck.watch(REQUEST);
    await settle();
    memoryCheck.watch({ ...REQUEST, frames: 241 });

    memoryCheck.watch(null);
    await settle();

    expect(memoryCheck.answer).toBeNull();
    expect(apiMocks.checkMemory).toHaveBeenCalledOnce();
  });

  it('asks again for a job it had forgotten', async () => {
    apiMocks.checkMemory.mockResolvedValue(TIGHT);
    memoryCheck.watch(REQUEST);
    await settle();
    memoryCheck.watch(null);

    memoryCheck.watch(REQUEST);
    await settle();

    expect(apiMocks.checkMemory).toHaveBeenCalledTimes(2);
    expect(memoryCheck.answer).toEqual(TIGHT);
  });

  it('shows nothing when the server cannot answer', async () => {
    apiMocks.checkMemory.mockResolvedValueOnce(TIGHT).mockRejectedValueOnce(new Error('width must be a positive integer.'));
    memoryCheck.watch(REQUEST);
    await settle();

    memoryCheck.watch({ ...REQUEST, frames: 241 });
    await settle();

    expect(memoryCheck.answer).toBeNull();
  });

  it('keeps the previous answer, pending, when the kind of job changes, even for the same model', async () => {
    apiMocks.checkMemory.mockResolvedValue(TIGHT);
    memoryCheck.watch({ ...IMAGE, model: 'ltx-8' });
    await settle();
    expect(memoryCheck.answer).toEqual(TIGHT);

    // Only the tab changed: the image's answer is shown as pending until the clip's arrives.
    memoryCheck.watch({ ...REQUEST, model: 'ltx-8' });
    expect(memoryCheck.answer).toEqual(TIGHT);
    expect(memoryCheck.pending).toBe(true);
    await settle();

    expect(apiMocks.checkMemory).toHaveBeenCalledTimes(2);
    expect(memoryCheck.pending).toBe(false);
  });

  it('keeps the answer across changes made to the same kind of job and model', async () => {
    apiMocks.checkMemory.mockResolvedValue(imageMemory('4'));
    memoryCheck.watch(IMAGE);
    await settle();

    memoryCheck.watch({ ...IMAGE, size: 'l' });

    expect(memoryCheck.answer).toEqual(imageMemory('4'));
  });

  it('asks nothing when only the Quant level changes: the answer already has a row for every level', async () => {
    apiMocks.checkMemory.mockResolvedValue(imageMemory('4'));
    memoryCheck.watch({ ...IMAGE, quantize: 4 });
    await settle();

    memoryCheck.watch({ ...IMAGE, quantize: 8 });
    memoryCheck.watch(IMAGE);
    await settle();

    expect(apiMocks.checkMemory).toHaveBeenCalledOnce();
    expect(memoryCheck.answer).toEqual(imageMemory('4'));
  });

  it('asks nothing when only Low memory changes, and still asks for a changed length', async () => {
    apiMocks.checkMemory.mockResolvedValue(TIGHT);
    memoryCheck.watch(REQUEST);
    await settle();

    memoryCheck.watch({ ...REQUEST, low_memory: false });
    await settle();
    expect(apiMocks.checkMemory).toHaveBeenCalledOnce();

    memoryCheck.watch({ ...REQUEST, low_memory: false, frames: 241 });
    await settle();
    expect(apiMocks.checkMemory).toHaveBeenCalledTimes(2);
    expect(apiMocks.checkMemory).toHaveBeenLastCalledWith({ ...REQUEST, low_memory: false, frames: 241 });
  });

  it('drops an answer to the other kind of job once the kind has changed', async () => {
    const clip = deferred<JobMemory | null>();
    apiMocks.checkMemory.mockReturnValueOnce(clip.promise).mockResolvedValue(null);
    memoryCheck.watch(REQUEST);
    await settle();
    memoryCheck.watch({ ...IMAGE, model: 'ltx-8' });

    clip.resolve(TIGHT);
    await vi.advanceTimersByTimeAsync(0);

    expect(memoryCheck.answer).toBeNull();
  });

  it('treats an answer of null as no answer', async () => {
    apiMocks.checkMemory.mockResolvedValue(null);
    memoryCheck.watch(REQUEST);
    await settle();

    expect(memoryCheck.answer).toBeNull();
  });

  it('refresh asks the same request again, keeping the answer shown and not pending until the reply', async () => {
    const second = deferred<JobMemory | null>();
    apiMocks.checkMemory.mockResolvedValueOnce(TIGHT).mockReturnValueOnce(second.promise);
    memoryCheck.watch(REQUEST);
    await settle();

    memoryCheck.refresh();
    expect(memoryCheck.answer).toEqual(TIGHT);
    expect(memoryCheck.pending).toBe(false);
    await settle();
    expect(apiMocks.checkMemory).toHaveBeenCalledTimes(2);
    expect(apiMocks.checkMemory).toHaveBeenLastCalledWith(REQUEST);
    expect(memoryCheck.answer).toEqual(TIGHT);
    expect(memoryCheck.pending).toBe(false);

    second.resolve(TOO_LARGE);
    await vi.advanceTimersByTimeAsync(0);
    expect(memoryCheck.answer).toEqual(TOO_LARGE);
  });

  it('refresh asks once for several calls inside the debounce, and nothing when there is nothing to check', async () => {
    apiMocks.checkMemory.mockResolvedValue(TIGHT);
    memoryCheck.refresh();
    await settle();
    expect(apiMocks.checkMemory).not.toHaveBeenCalled();

    memoryCheck.watch(REQUEST);
    await settle();
    memoryCheck.refresh();
    memoryCheck.refresh();
    await settle();
    expect(apiMocks.checkMemory).toHaveBeenCalledTimes(2);

    memoryCheck.watch(null);
    memoryCheck.refresh();
    await settle();
    expect(apiMocks.checkMemory).toHaveBeenCalledTimes(2);
  });

  it('refresh uses the latest body when only Quant level or Low memory changed', async () => {
    apiMocks.checkMemory.mockResolvedValue(TIGHT);
    memoryCheck.watch(REQUEST);
    await settle();
    memoryCheck.watch({ ...REQUEST, low_memory: false });
    memoryCheck.refresh();
    await settle();

    expect(apiMocks.checkMemory).toHaveBeenLastCalledWith({ ...REQUEST, low_memory: false });
  });

  it('a reply to a request made before a refresh does not replace the refreshed one', async () => {
    const first = deferred<JobMemory | null>();
    apiMocks.checkMemory.mockReturnValueOnce(first.promise).mockResolvedValueOnce(TOO_LARGE);
    memoryCheck.watch(REQUEST);
    await settle();
    memoryCheck.refresh();
    await settle();
    expect(memoryCheck.answer).toEqual(TOO_LARGE);

    first.resolve(TIGHT);
    await vi.advanceTimersByTimeAsync(0);
    expect(memoryCheck.answer).toEqual(TOO_LARGE);
  });

  it('leaves no answer after a failed reply and does not ask again by itself', async () => {
    apiMocks.checkMemory.mockRejectedValue(new Error('server error'));
    memoryCheck.watch(REQUEST);
    await settle();
    expect(memoryCheck.answer).toBeNull();
    expect(memoryCheck.pending).toBe(false);

    await vi.advanceTimersByTimeAsync(60_000);

    expect(apiMocks.checkMemory).toHaveBeenCalledTimes(1);
    expect(memoryCheck.answer).toBeNull();
  });

  it('asks again when the inputs change after a failed reply', async () => {
    apiMocks.checkMemory.mockRejectedValueOnce(new Error('server error')).mockResolvedValueOnce(TIGHT);
    memoryCheck.watch(REQUEST);
    await settle();
    expect(memoryCheck.answer).toBeNull();

    memoryCheck.watch({ ...REQUEST, frames: 49 });
    await settle();

    expect(apiMocks.checkMemory).toHaveBeenCalledTimes(2);
    expect(memoryCheck.answer).toEqual(TIGHT);
  });
});
