import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { GalleryAsset, JobContext } from '$lib/types';

const submitUpscale = vi.hoisted(() => vi.fn());
const addToast = vi.hoisted(() => vi.fn());
const jobSubmitted = vi.hoisted(() => vi.fn());

vi.mock('$lib/api/gallery', () => ({ submitUpscale }));
vi.mock('$lib/state/toasts.svelte', () => ({ addToast }));
vi.mock('$lib/state/job.svelte', () => ({ jobStore: { jobSubmitted } }));

import { startUpscale } from './upscale';

const asset = { id: 'out/a.png', filename: 'a.png' } as GalleryAsset;

describe('startUpscale', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    submitUpscale.mockResolvedValue({ job_id: 'j', queue_position: null } as JobContext);
  });

  it.each([
    ['seedvr2', 'Upscaling a.png 2×'],
    ['creative', 'Upscaling a.png 2× creative'],
  ] as const)('submits the %s method and announces it', async (method, toast) => {
    const job = await startUpscale(asset, 2, method);

    expect(submitUpscale).toHaveBeenCalledWith('out/a.png', 2, method);
    expect(jobSubmitted).toHaveBeenCalledWith(job);
    expect(addToast).toHaveBeenCalledWith(toast, 'success');
  });

  it('reports a queued upscale by its position', async () => {
    submitUpscale.mockResolvedValue({ job_id: 'j', queue_position: 2 } as JobContext);
    await startUpscale(asset, 4, 'seedvr2');
    expect(addToast).toHaveBeenCalledWith('Upscale of a.png added to the queue as #2.', 'info');
  });

  it('resolves to null and toasts the error when the server refuses', async () => {
    submitUpscale.mockRejectedValue(new Error('nope'));
    expect(await startUpscale(asset, 2, 'creative')).toBeNull();
    expect(addToast).toHaveBeenCalledWith(expect.stringContaining('Upscale failed'), 'error');
    expect(jobSubmitted).not.toHaveBeenCalled();
  });
});
