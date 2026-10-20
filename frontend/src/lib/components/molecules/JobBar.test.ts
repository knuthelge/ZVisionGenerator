// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, mount, unmount } from '../../../../node_modules/svelte/src/index-client.js';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { ActiveJobState } from '$lib/types';
import JobBar from './JobBar.svelte';

function makeJob(overrides: Partial<ActiveJobState> = {}): ActiveJobState {
  return {
    job_id: 'job-bar', workflow: 'txt2img', prompt: 'Bar prompt', model: 'zit', runs: 1, created_at: '2026-04-22T00:00:00Z',
    supported_controls: ['pause', 'resume', 'quit', 'next', 'repeat'], status: 'running', phase: 'generate', currentStep: 12, totalSteps: 32,
    elapsed: 4, remaining: 3, stageName: 'text_to_image', stageIndex: 0, batchLabel: '', batchIndex: 0, paused: false, message: '', outputs: [], previewUrl: null,
    ...overrides,
  };
}

describe('JobBar', () => {
  let target: HTMLDivElement;
  let app: Record<string, unknown> | null = null;

  beforeEach(() => {
    target = document.createElement('div');
    document.body.appendChild(target);
  });

  afterEach(async () => {
    if (app) {
      await unmount(app);
      app = null;
    }
    target.remove();
  });

  it('shows the workflow, the headline and the prompt, and pauses from its button', async () => {
    const onpause = vi.fn().mockResolvedValue(undefined);
    app = flushSync(() => mount(JobBar, { target, props: { job: makeJob(), onpause, ondetails: vi.fn() } }));

    const bar = target.querySelector('[data-testid="job-bar"]') as HTMLElement;
    expect(bar.textContent).toContain('Text to image');
    expect(bar.textContent).toContain('Bar prompt');

    const pause = Array.from(target.querySelectorAll('button')).find((button) => button.textContent?.includes('Pause')) as HTMLButtonElement;
    pause.click();
    await Promise.resolve();
    expect(onpause).toHaveBeenCalledWith('job-bar');
  });

  it('offers Resume while paused', () => {
    app = flushSync(() => mount(JobBar, { target, props: { job: makeJob({ status: 'paused', paused: true }), ondetails: vi.fn() } }));
    expect(Array.from(target.querySelectorAll('button')).some((button) => button.textContent?.includes('Resume'))).toBe(true);
  });

  it('opens the job details straight from ⋯, where Skip, Repeat, Cancel and the queue are', () => {
    const ondetails = vi.fn();
    app = flushSync(() => mount(JobBar, { target, props: { job: makeJob({ runs: 3 }), queuedCount: 2, ondetails } }));

    (target.querySelector('button[aria-label="Job details"]') as HTMLButtonElement).click();
    flushSync();
    expect(ondetails).toHaveBeenCalledOnce();
    expect(document.querySelector('[role="menu"]')).toBeNull();
  });

  it('draws one segment per image for a batch', () => {
    app = flushSync(() => mount(JobBar, { target, props: { job: makeJob({ runs: 4, batchIndex: 1 }), ondetails: vi.fn() } }));
    expect(target.querySelectorAll('.jb-segment')).toHaveLength(4);
  });
});
