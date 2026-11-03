import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { GalleryAsset, JobContext, JobDownloadPlan } from '$lib/types';

const submitUpscale = vi.hoisted(() => vi.fn());
const planUpscale = vi.hoisted(() => vi.fn());
const requestConfirm = vi.hoisted(() => vi.fn());
const addToast = vi.hoisted(() => vi.fn());
const jobSubmitted = vi.hoisted(() => vi.fn());

vi.mock('$lib/api/gallery', () => ({ submitUpscale, planUpscale }));
vi.mock('$lib/components/molecules/confirm.svelte', () => ({ requestConfirm }));
vi.mock('$lib/state/toasts.svelte', () => ({ addToast }));
vi.mock('$lib/state/job.svelte', () => ({ jobStore: { jobSubmitted } }));

import { startUpscale } from './upscale';

const asset = { id: 'out/a.png', filename: 'a.png' } as GalleryAsset;
const NOTHING: JobDownloadPlan = { parts: [], total_bytes: 0, space: null, unchecked: false, licences: [] };
const SEEDVR2: JobDownloadPlan = {
  parts: [{ label: 'SeedVR2 7B fp8', model: 'seedvr2-7b', repo: 'org/seedvr2', files: 2, bytes: 9e9 }],
  total_bytes: 9e9,
  space: { path: '/hub', needed_bytes: 9e9, free_bytes: 512e9, enough: true },
  unchecked: false,
  licences: [],
};

describe('startUpscale', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    planUpscale.mockResolvedValue(NOTHING);
    requestConfirm.mockResolvedValue(true);
    submitUpscale.mockResolvedValue({ job_id: 'j', queue_position: null } as JobContext);
  });

  it.each(['seedvr2', 'creative'] as const)('submits the %s method and announces it', async (method) => {
    const job = await startUpscale(asset, 2, method);

    expect(planUpscale).toHaveBeenCalledWith('out/a.png', 2, method);
    expect(submitUpscale).toHaveBeenCalledWith('out/a.png', 2, method);
    expect(jobSubmitted).toHaveBeenCalledWith(job);
    expect(addToast).toHaveBeenCalledWith(expect.any(String), 'success');
  });

  it('asks nothing when no model is missing', async () => {
    await startUpscale(asset, 2, 'seedvr2');
    expect(requestConfirm).not.toHaveBeenCalled();
  });

  it('asks before it downloads a model, then submits when the user agrees', async () => {
    planUpscale.mockResolvedValue(SEEDVR2);

    expect(await startUpscale(asset, 2, 'seedvr2')).not.toBeNull();

    expect(requestConfirm).toHaveBeenCalledTimes(1);
    expect(requestConfirm).toHaveBeenCalledWith(expect.objectContaining({ question: 'Download SeedVR2 7B fp8?', confirmLabel: 'Download' }));
    expect(submitUpscale).toHaveBeenCalledTimes(1);
  });

  it('submits nothing, and says nothing, when the user cancels the download', async () => {
    planUpscale.mockResolvedValue(SEEDVR2);
    requestConfirm.mockResolvedValue(false);

    expect(await startUpscale(asset, 2, 'seedvr2')).toBeNull();

    expect(submitUpscale).not.toHaveBeenCalled();
    expect(addToast).not.toHaveBeenCalled();
    expect(jobSubmitted).not.toHaveBeenCalled();
  });

  it('holds the file while the plan is read, so a second pick of it does nothing', async () => {
    let resolvePlan: (plan: JobDownloadPlan) => void = () => undefined;
    planUpscale.mockReturnValue(new Promise((resolve) => { resolvePlan = resolve; }));

    const first = startUpscale(asset, 2, 'creative');
    expect(await startUpscale(asset, 2, 'creative')).toBeNull();
    expect(planUpscale).toHaveBeenCalledTimes(1);

    resolvePlan(NOTHING);
    expect(await first).not.toBeNull();
    expect(submitUpscale).toHaveBeenCalledTimes(1);

    // Free again once it ended.
    planUpscale.mockResolvedValue(NOTHING);
    expect(await startUpscale(asset, 2, 'creative')).not.toBeNull();
  });

  it('reports a queued upscale by its position', async () => {
    submitUpscale.mockResolvedValue({ job_id: 'j', queue_position: 2 } as JobContext);
    await startUpscale(asset, 4, 'seedvr2');
    expect(addToast).toHaveBeenCalledWith(expect.any(String), 'info');
  });

  it('resolves to null and toasts the error when the server refuses', async () => {
    submitUpscale.mockRejectedValue(new Error('nope'));
    expect(await startUpscale(asset, 2, 'creative')).toBeNull();
    expect(addToast).toHaveBeenCalledWith(expect.any(String), 'error');
    expect(jobSubmitted).not.toHaveBeenCalled();
  });

  it('toasts the error and submits nothing when the plan is refused', async () => {
    planUpscale.mockRejectedValue(new Error('Hugging Face refused org/seedvr2'));

    expect(await startUpscale(asset, 2, 'seedvr2')).toBeNull();

    expect(addToast).toHaveBeenCalledWith(expect.stringContaining('org/seedvr2'), 'error');
    expect(submitUpscale).not.toHaveBeenCalled();
  });
});
