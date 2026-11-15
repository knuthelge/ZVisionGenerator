import { flushSync } from 'svelte';
import { describe, expect, it, vi } from 'vitest';

import type { ActiveJobState } from '$lib/types';
import JobBar from './JobBar.svelte';
import { mountComponent } from '../../../test-utils/mount';
import { reactiveProps } from '../../../test-utils/reactiveProps.svelte';

function makeJob(overrides: Partial<ActiveJobState> = {}): ActiveJobState {
  return {
    job_id: 'job-bar', workflow: 'txt2img', prompt: 'Bar prompt', model: 'zit', runs: 1, created_at: '2026-04-22T00:00:00Z',
    supported_controls: ['pause', 'resume', 'quit', 'next', 'repeat'], status: 'running', phase: 'generate', currentStep: 12, totalSteps: 32,
    elapsed: 4, remaining: 3, stageName: 'text_to_image', stageIndex: 0, batchLabel: '', batchIndex: 0, paused: false, message: '', outputs: [], previewUrl: null,
    ...overrides,
  };
}

const bar = (target: HTMLElement) => target.querySelector<HTMLElement>('[data-testid="job-bar"]')!;
const button = (target: HTMLElement, label: string) => target.querySelector<HTMLButtonElement>(`button[aria-label="${label}"]`);

describe('JobBar', () => {
  it('is a group named for the job, with its headline, its step line and its prompt, and pauses from its button', async () => {
    const onpause = vi.fn().mockResolvedValue(undefined);
    const { target } = mountComponent(JobBar, { job: makeJob(), onpause, ondetails: vi.fn() });

    expect(bar(target).getAttribute('role')).toBe('group');
    expect(bar(target).getAttribute('aria-label')).toBe('Running job');
    expect(bar(target).querySelector('[data-testid="job-bar-headline"]')?.textContent).toBe('Generating image');
    expect(bar(target).querySelector('[data-testid="job-bar-sub"]')?.textContent).toContain('Step 12 of 32');
    expect(bar(target).textContent).toContain('Bar prompt');

    button(target, 'Pause')!.click();
    await Promise.resolve();
    expect(onpause).toHaveBeenCalledWith('job-bar');
  });

  it('shows a batch with its short headline and the ETA first on the step line', () => {
    const { target } = mountComponent(JobBar, { job: makeJob({ runs: 4, batchIndex: 2, remaining: 120 }), ondetails: vi.fn() });

    expect(bar(target).querySelector('[data-testid="job-bar-headline"]')?.textContent).toBe('Image 3 of 4');
    expect(bar(target).querySelector('[data-testid="job-bar-sub"] b')?.textContent).toBe('About 2 min left');
  });

  it('draws the job ring, a progress value with a text, and holds Blob\'s place through the snippet', () => {
    const { target } = mountComponent(JobBar, { job: makeJob({ runs: 4, batchIndex: 2, currentStep: 5, totalSteps: 8 }), ondetails: vi.fn() });

    const ring = target.querySelector('[role="progressbar"]')!;
    expect(ring.getAttribute('aria-valuenow')).toBe('65');
    expect(ring.getAttribute('aria-valuetext')).toBe('Image 3 of 4, step 5 of 8, 65 percent of the job');
  });

  it('words a waiting control as the card does', () => {
    const { target } = mountComponent(JobBar, { job: makeJob({ phase: 'load', controlQueued: 'pause' }), ondetails: vi.fn() });
    expect(target.textContent).toContain('Pauses once the model is loaded');
  });

  it('leads an upscale with its source file in place of a prompt', () => {
    const inputs = [{ role: 'source' as const, name: 'portrait.png', kind: 'image' as const }];
    const { target } = mountComponent(JobBar, { job: makeJob({ workflow: 'upscale', prompt: '', inputs }), ondetails: vi.fn() });

    expect(bar(target).textContent).toContain('portrait.png');
    expect(bar(target).textContent).not.toContain('No prompt recorded');
  });

  it('drops the pending notice once the job has ended', () => {
    const props = reactiveProps({ job: makeJob({ controlQueued: 'quit' }), ondetails: vi.fn() });
    const { target } = mountComponent(JobBar, props);
    expect(target.textContent).toContain('Stopping at the next step');

    props.job = { ...props.job, status: 'cancelled' };
    flushSync();
    expect(target.textContent).not.toContain('Stopping at the next step');
  });

  it('offers Resume in place of Pause while paused', () => {
    const { target } = mountComponent(JobBar, { job: makeJob({ status: 'paused', paused: true }), ondetails: vi.fn() });

    expect(button(target, 'Resume')).not.toBeNull();
    expect(button(target, 'Pause')).toBeNull();
  });

  it('opens the job details straight from ⋯, where Skip, Repeat, Cancel and the queue are', () => {
    const ondetails = vi.fn();
    const { target } = mountComponent(JobBar, { job: makeJob({ runs: 3 }), queuedCount: 2, ondetails });

    button(target, 'Job details and queue, 2 queued')!.click();
    flushSync();
    expect(ondetails).toHaveBeenCalledOnce();
    expect(document.querySelector('[role="menu"]')).toBeNull();
  });

  it('shows the queue count on ⋯ only when jobs wait', () => {
    const waiting = mountComponent(JobBar, { job: makeJob(), queuedCount: 3, ondetails: vi.fn() });
    expect(waiting.target.querySelector('[data-testid="queue-count"]')?.textContent).toBe('3');

    const alone = mountComponent(JobBar, { job: makeJob(), ondetails: vi.fn() });
    expect(alone.target.querySelector('[data-testid="queue-count"]')).toBeNull();
    expect(button(alone.target, 'Job details')).not.toBeNull();
  });

  it('names the project the job saves into', () => {
    const { target } = mountComponent(JobBar, { job: makeJob({ save_project: 'Night Bus' }), ondetails: vi.fn() });
    expect(target.querySelector('[data-testid="job-bar-project"]')?.getAttribute('title')).toBe('Saves to Night Bus');
  });

  it('shows the download as the ring while the model downloads', () => {
    const download = { role: 'model' as const, model: 'zit', repo: 'org/zit', part: 1, parts: 2, bytes_done: 250, bytes_total: 1000, bytes_per_sec: null, download_eta_secs: null, shared: false };
    const { target } = mountComponent(JobBar, { job: makeJob({ phase: 'load', download }), ondetails: vi.fn() });

    expect(target.querySelector('[role="progressbar"]')?.getAttribute('aria-valuenow')).toBe('25');
  });

  describe('once the job has ended', () => {
    it('is the last job, with its mark and what it saved, and no controls but ⋯', () => {
      const { target } = mountComponent(JobBar, { job: makeJob({ status: 'failed', runs: 4, batchIndex: 2, outputs: [{ id: 'a' } as never, { id: 'b' } as never] }), ondetails: vi.fn() });

      expect(bar(target).getAttribute('aria-label')).toBe('Last job');
      expect(target.querySelector('[data-testid="ring-mark"]')?.getAttribute('data-mark')).toBe('failed');
      expect(bar(target).textContent).toContain('2 of 4 saved to the Gallery');
      expect(target.querySelector('[role="progressbar"]')).toBeNull();
      expect(button(target, 'Pause')).toBeNull();
      expect(button(target, 'Job details')).not.toBeNull();
    });
  });

  describe('a voice design job', () => {
    const design = (overrides: Partial<ActiveJobState> = {}) => makeJob({
      workflow: 'voice_design', media_type: 'audio', prompt: 'A warm woman in her thirties', runs: 3, currentStep: 0, totalSteps: 0, supported_controls: ['quit'],
      audio: { phase: 'Design', step: 2, total: 3, part: 1, parts: 1 }, ...overrides,
    });

    it('reads Voice 2 of 3 over the description, with a ring at two thirds', () => {
      const { target } = mountComponent(JobBar, { job: design(), ondetails: vi.fn() });

      expect(bar(target).querySelector('[data-testid="job-bar-headline"]')?.textContent).toBe('Voice 2 of 3');
      expect(bar(target).textContent).toContain('A warm woman in her thirties');
      expect(target.querySelector('[role="progressbar"]')?.getAttribute('aria-valuenow')).toBe('66');
    });

    it('has no Pause, only the details button that holds Cancel', () => {
      const { target } = mountComponent(JobBar, { job: design(), ondetails: vi.fn() });

      expect(button(target, 'Pause')).toBeNull();
      expect(button(target, 'Job details')).not.toBeNull();
    });
  });
});
