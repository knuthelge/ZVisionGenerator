import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { JobDownloadPlan } from '$lib/types';

const requestConfirm = vi.hoisted(() => vi.fn());
vi.mock('$lib/components/molecules/confirm.svelte', () => ({ requestConfirm }));

import { confirmDownloads } from './confirmDownloads';
import { uncheckedJobPlan } from './downloadConfirm';

const klein = { label: 'FLUX.2 Klein 9B', model: 'klein9b', bytes: 35.1e9 };
const enhancer = { label: 'Prompt enhancer (Smart)', model: 'org/enhancer', bytes: 9e9 };
const SPACE = { path: '/data', needed_bytes: 44.1e9, free_bytes: 812e9, enough: true };

function plan(overrides: Partial<JobDownloadPlan> = {}): JobDownloadPlan {
  return { parts: [klein], total_bytes: 35.1e9, space: SPACE, unchecked: false, licences: [], ...overrides };
}

describe('confirmDownloads', () => {
  beforeEach(() => {
    requestConfirm.mockReset();
    requestConfirm.mockResolvedValue(true);
  });

  it('answers yes without a dialog when nothing is to be downloaded', async () => {
    await expect(confirmDownloads(plan({ parts: [], total_bytes: 0, space: null }))).resolves.toBe(true);
    expect(requestConfirm).not.toHaveBeenCalled();
  });

  it('asks about the one model by name, with its size and the free space', async () => {
    await confirmDownloads(plan());

    expect(requestConfirm).toHaveBeenCalledTimes(1);
    const request = requestConfirm.mock.calls[0][0];
    expect(request).toMatchObject({
      question: 'Download FLUX.2 Klein 9B?',
      parts: [{ label: 'FLUX.2 Klein 9B', size: '35.1 GB', lines: [] }],
      confirmLabel: 'Download',
      danger: false,
    });
    expect(request.info).toContain('812.0 GB free on /data');
  });

  it('asks one question for several models, in the order given, with the number of parts', async () => {
    await confirmDownloads(plan({ parts: [enhancer, klein], total_bytes: 44.1e9 }));

    const request = requestConfirm.mock.calls[0][0];
    expect(request.question).toBe('Download 2 models?');
    expect(request.parts.map((part: { label: string }) => part.label)).toEqual(['Prompt enhancer (Smart)', 'FLUX.2 Klein 9B']);
    expect(request.info).toContain('in 2 parts');
  });

  it('offers Download anyway when the space is short', async () => {
    await confirmDownloads(plan({ space: { ...SPACE, free_bytes: 10e9, enough: false } }));

    const request = requestConfirm.mock.calls[0][0];
    expect(request.confirmLabel).toBe('Download anyway');
    expect(request.info).toContain('may run out of space');
  });

  it("says it couldn't check the size, as a warning, and offers Download anyway", async () => {
    await confirmDownloads(uncheckedJobPlan('FLUX.2 Klein 9B', 'klein9b'));

    expect(requestConfirm).toHaveBeenCalledWith(
      expect.objectContaining({
        question: 'Download FLUX.2 Klein 9B?',
        parts: [{ label: 'FLUX.2 Klein 9B', size: null, lines: [] }],
        infoTone: 'warning',
        confirmLabel: 'Download anyway',
      }),
    );
    expect(requestConfirm.mock.calls[0][0].info).toContain("Couldn't check the size");
  });
});
