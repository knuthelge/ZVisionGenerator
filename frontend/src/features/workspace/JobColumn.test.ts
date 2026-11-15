import { flushSync } from 'svelte';
import { describe, expect, it, vi } from 'vitest';

import type { ActiveJobState, JobSnapshot } from '$lib/types';
import JobColumn from './JobColumn.svelte';
import { QUEUE_ROW_PX } from './queueView';
import { mountComponent, settle } from '../../test-utils/mount';
import { reactiveProps } from '../../test-utils/reactiveProps.svelte';

function makeJob(overrides: Partial<ActiveJobState> = {}): ActiveJobState {
  return {
    job_id: 'job-col', workflow: 'txt2img', prompt: 'p', model: 'zit', runs: 1, created_at: '', supported_controls: [], status: 'running', phase: 'generate', currentStep: 2, totalSteps: 10,
    elapsed: 0, remaining: 0, stageName: '', stageIndex: 0, batchLabel: '', batchIndex: 0, paused: false, message: '', outputs: [], previewUrl: null, ...overrides,
  };
}

function queued(id: string): JobSnapshot {
  return { id, job_id: id, job_type: 'Text to Image', workflow: 'txt2img', prompt: `prompt ${id}`, model: 'zit', runs: 1, created_at: '', status: 'queued', event_count: 1, paused: false, settings: {} };
}

/** How far the column's content runs past the column: jsdom has no layout, so the test says. */
function overflowing(px: number): void {
  vi.spyOn(HTMLElement.prototype, 'scrollHeight', 'get').mockReturnValue(600 + px);
  vi.spyOn(HTMLElement.prototype, 'clientHeight', 'get').mockReturnValue(600);
}

const props = (extra: Record<string, unknown> = {}) => ({ job: makeJob(), queued: [queued('a'), queued('b'), queued('c')], onremove: vi.fn(), onload: vi.fn(), onclear: vi.fn(), ...extra });
const toggle = (target: HTMLElement) => target.querySelector<HTMLButtonElement>('button[aria-expanded]')!;

describe('JobColumn', () => {
  it('shows the card and the queue under it', () => {
    overflowing(0);
    const { target } = mountComponent(JobColumn, props());

    expect(target.querySelector('article[aria-label="Active job"]')).not.toBeNull();
    expect(target.querySelectorAll('[data-job-id]')).toHaveLength(3);
  });

  it('keeps the queue open when less than a row would hide, and the column scrolls', async () => {
    overflowing(QUEUE_ROW_PX - 10);
    const { target } = mountComponent(JobColumn, props());
    await settle();

    expect(toggle(target).getAttribute('aria-expanded')).toBe('true');
  });

  it('folds the queue to its head when more than a whole row would hide', async () => {
    overflowing(QUEUE_ROW_PX + 10);
    const { target } = mountComponent(JobColumn, props());
    await settle();

    expect(toggle(target).getAttribute('aria-expanded')).toBe('false');
    expect(target.querySelectorAll('[data-job-id]')).toHaveLength(0);
    expect(target.querySelector('article[aria-label="Active job"]')).not.toBeNull();
  });

  it('lets the user\'s own toggle win for the rest of the job, and measures again for the next', async () => {
    overflowing(QUEUE_ROW_PX + 10);
    const state = reactiveProps(props());
    const { target } = mountComponent(JobColumn, state);
    await settle();
    expect(toggle(target).getAttribute('aria-expanded')).toBe('false');

    toggle(target).click();
    flushSync();
    expect(toggle(target).getAttribute('aria-expanded')).toBe('true');

    state.job = makeJob({ status: 'running', phase: 'load' });
    await settle();
    expect(toggle(target).getAttribute('aria-expanded')).toBe('true');

    state.job = makeJob({ job_id: 'next-job' });
    await settle();
    expect(toggle(target).getAttribute('aria-expanded')).toBe('false');
  });

  it('never folds inside the Job sheet, where the sheet scrolls', async () => {
    overflowing(QUEUE_ROW_PX + 100);
    const { target } = mountComponent(JobColumn, props({ variant: 'sheet' }));
    await settle();

    expect(toggle(target).getAttribute('aria-expanded')).toBe('true');
  });

  it('shows the queue alone when there is no job', () => {
    overflowing(0);
    const { target } = mountComponent(JobColumn, props({ job: null, variant: 'sheet' }));

    expect(target.querySelector('article')).toBeNull();
    expect(target.querySelectorAll('[data-job-id]')).toHaveLength(3);
  });
});
