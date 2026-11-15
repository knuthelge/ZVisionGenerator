import { beforeEach, describe, expect, it, vi } from 'vitest';

// A finger: the touch layer is on without a real media query.
const touchLayer = vi.hoisted(() => ({ coarse: true, compact: false, mode: 'wide' }));
vi.mock('$lib/state/layoutMode.svelte', () => ({ layout: touchLayer }));
const confirm = vi.hoisted(() => ({ requestConfirm: vi.fn() }));
vi.mock('./confirm.svelte', () => ({ requestConfirm: confirm.requestConfirm }));

import type { ActiveJobState } from '$lib/types';
import JobCard from './JobCard.svelte';
import { mountComponent, settle } from '../../../test-utils/mount';

function makeJob(overrides: Partial<ActiveJobState> = {}): ActiveJobState {
  return {
    job_id: 'job-touch', workflow: 'txt2img', prompt: 'p', model: 'zit', runs: 3, created_at: '', supported_controls: ['next', 'pause', 'resume', 'repeat', 'quit'],
    status: 'running', phase: 'generate', currentStep: 2, totalSteps: 10, elapsed: 0, remaining: 0, stageName: '', stageIndex: 0, batchLabel: '', batchIndex: 0,
    paused: false, message: '', outputs: [], previewUrl: null, ...overrides,
  };
}

const press = (key: string) => document.dispatchEvent(new KeyboardEvent('keydown', { key, bubbles: true, cancelable: true }));

describe('JobCard on a touch screen', () => {
  beforeEach(() => {
    confirm.requestConfirm.mockReset();
  });

  it('gives every control as an icon button with no key hint', () => {
    const { target } = mountComponent(JobCard, { job: makeJob(), oncancel: vi.fn() });

    for (const name of ['pause-resume', 'next', 'repeat', 'quit']) {
      const button = target.querySelector(`button[data-control="${name}"]`)!;
      expect(button.textContent?.trim()).toBe('');
      expect(button.querySelector('kbd')).toBeNull();
    }
  });

  it('draws the medium ring', () => {
    const { target } = mountComponent(JobCard, { job: makeJob() });
    expect(target.querySelector('[data-testid="ring-hero"]')?.getAttribute('data-size')).toBe('M');
  });

  it('asks before Cancel, keeps the job on "Keep running" and cancels on confirm', async () => {
    const oncancel = vi.fn();
    const { target } = mountComponent(JobCard, { job: makeJob(), oncancel });
    const cancel = target.querySelector<HTMLButtonElement>('button[data-control="quit"]')!;

    confirm.requestConfirm.mockResolvedValueOnce(false);
    cancel.click();
    await settle();
    expect(confirm.requestConfirm).toHaveBeenCalledWith(expect.objectContaining({ question: 'Cancel this job?', confirmLabel: 'Cancel job', cancelLabel: 'Keep running' }));
    expect(oncancel).not.toHaveBeenCalled();

    confirm.requestConfirm.mockResolvedValueOnce(true);
    cancel.click();
    await settle();
    expect(oncancel).toHaveBeenCalledWith('job-touch');
  });

  it('opens the dialog on the first Esc from a hardware keyboard, and never grows an armed button', async () => {
    confirm.requestConfirm.mockResolvedValue(true);
    const oncancel = vi.fn();
    const { target } = mountComponent(JobCard, { job: makeJob(), oncancel });

    press('Escape');
    await settle();

    expect(confirm.requestConfirm).toHaveBeenCalledOnce();
    expect(oncancel).toHaveBeenCalledWith('job-touch');
    expect(target.textContent).not.toContain('Press Esc again');
    expect(target.querySelector('button[data-control="quit"]')?.textContent?.trim()).toBe('');
  });
});
