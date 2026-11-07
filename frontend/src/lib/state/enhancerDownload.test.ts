import { beforeEach, describe, expect, it, vi } from 'vitest';

import { ApiError } from '$lib/api/client';
import type { PromptEnhancerContract, RowDownloadPlan } from '$lib/types';

const getDownloadPlan = vi.hoisted(() => vi.fn());
const confirmDownloads = vi.hoisted(() => vi.fn());
vi.mock('$lib/api/models', () => ({ getDownloadPlan }));
vi.mock('./confirmDownloads', () => ({ confirmDownloads }));

import { confirmEnhancerDownload, downloadedTiers, tierMissing } from './enhancerDownload.svelte';

function contract(overrides: Partial<PromptEnhancerContract> = {}): PromptEnhancerContract {
  return { downloaded: false, smart: { downloaded: false, download_size_label: null, unavailable: null }, ...overrides } as PromptEnhancerContract;
}

const PLAN: RowDownloadPlan = { label: 'Fast enhancer model', parts: [{ label: 'Fast enhancer model', model: 'fast', bytes: 2.4e9 }], total_bytes: 2.4e9, space: null, unchecked: false, licences: [] };

beforeEach(() => {
  downloadedTiers.clear();
  getDownloadPlan.mockReset();
  confirmDownloads.mockReset();
  getDownloadPlan.mockResolvedValue(PLAN);
  confirmDownloads.mockResolvedValue(true);
});

describe('tierMissing', () => {
  it('reads each tier from its own flag', () => {
    expect(tierMissing(contract({ downloaded: true }), 'fast')).toBe(false);
    expect(tierMissing(contract({ downloaded: false }), 'fast')).toBe(true);
    expect(tierMissing(contract(), 'smart')).toBe(true);
    expect(tierMissing(contract({ smart: { downloaded: true, download_size_label: null, unavailable: null } }), 'smart')).toBe(false);
  });

  it('stops reporting a tier once it was downloaded during the session', () => {
    downloadedTiers.add('fast');
    expect(tierMissing(contract(), 'fast')).toBe(false);
    expect(tierMissing(contract(), 'smart')).toBe(true);
  });
});

describe('confirmEnhancerDownload', () => {
  it('goes ahead without asking when nothing is missing', async () => {
    await expect(confirmEnhancerDownload(contract({ downloaded: true }), 'fast')).resolves.toEqual({ confirmed: true, error: null });
    expect(getDownloadPlan).not.toHaveBeenCalled();
  });

  it.each([
    ['fast', contract(), 'fast'],
    ['smart', contract(), 'smart'],
    ['fast', contract({ smart: null }), 'custom'],
  ] as const)('asks about the %s model under the Models page row %s', async (tier, enhancer, row) => {
    await confirmEnhancerDownload(enhancer, tier);

    expect(getDownloadPlan).toHaveBeenCalledWith('enhancer', row);
    expect(confirmDownloads).toHaveBeenCalledWith(PLAN);
  });

  it('returns the user\'s answer', async () => {
    confirmDownloads.mockResolvedValue(false);
    await expect(confirmEnhancerDownload(contract(), 'fast')).resolves.toEqual({ confirmed: false, error: null });
  });

  it('still asks, with an unsized plan, when Hugging Face cannot be reached', async () => {
    getDownloadPlan.mockRejectedValue(new ApiError('GET → 503', 503, 'offline'));

    await expect(confirmEnhancerDownload(contract(), 'smart')).resolves.toEqual({ confirmed: true, error: null });

    expect(confirmDownloads).toHaveBeenCalledWith(expect.objectContaining({ unchecked: true, parts: [expect.objectContaining({ label: 'Smart enhancer model', model: 'smart', bytes: null })] }));
  });

  it('says why and does not go ahead when the plan cannot be read for another reason', async () => {
    getDownloadPlan.mockRejectedValue(new Error('boom'));
    await expect(confirmEnhancerDownload(contract(), 'fast')).resolves.toEqual({ confirmed: false, error: 'boom' });
    expect(confirmDownloads).not.toHaveBeenCalled();

    getDownloadPlan.mockRejectedValue('weird');
    await expect(confirmEnhancerDownload(contract(), 'fast')).resolves.toEqual({ confirmed: false, error: 'Could not check the enhancer model download.' });
  });
});
