import { flushSync } from 'svelte';
import { afterEach, describe, expect, it, vi } from 'vitest';

import type { ActiveJobState } from '$lib/types';
import JobBar from './JobBar.svelte';
import JobCard from './JobCard.svelte';
import { mountComponent } from '../../../test-utils/mount';

const MATTE_DOWNLOAD = { role: 'matte' as const, model: 'Background removal', repo: 'org/birefnet', part: 1, parts: 2, bytes_done: 250, bytes_total: 1000, bytes_per_sec: null, download_eta_secs: null, shared: false };

function makeJob(overrides: Partial<ActiveJobState> = {}): ActiveJobState {
  return {
    job_id: 'job-matte', workflow: 'txt2img', prompt: 'A cup', model: 'zit', runs: 1, created_at: '2026-04-22T00:00:00Z', supported_controls: ['quit'],
    status: 'running', phase: 'generate', currentStep: 0, totalSteps: 8, elapsed: 0, remaining: 0, stageName: '', stageIndex: 0, batchLabel: '', batchIndex: 0,
    paused: false, message: '', outputs: [], previewUrl: null, ...overrides,
  };
}

describe('the job card and bar for the background removal stages', () => {
  let component: ReturnType<typeof mountComponent> | null = null;
  afterEach(() => {
    if (component) flushSync(() => void component!.unmount());
    component = null;
  });

  const ring = (root: ParentNode) => root.querySelector('[data-testid="progress-ring"]')!;

  it('names the matte download in the card headline and load row, and the ring is the download', () => {
    component = mountComponent(JobCard, { job: makeJob({ phase: 'load', download: MATTE_DOWNLOAD }) });
    const { target } = component;

    expect(target.querySelector('h3')!.textContent).toBe('Downloading background removal');
    expect(ring(target).getAttribute('aria-valuenow')).toBe('25');
    expect(ring(target).getAttribute('aria-valuetext')).toContain('part 1 of 2');
    expect(target.querySelector('[data-phase="load"]')!.textContent).toContain('Background removal download');
    expect(target.querySelector('[data-testid="job-log"]')!.textContent).toContain('Downloading org/birefnet. Only the first time.');
  });

  it('keeps an enhancer download on the enhance row', () => {
    component = mountComponent(JobCard, { job: makeJob({ phase: 'enhance', download: { ...MATTE_DOWNLOAD, role: 'enhancer' } }) });
    expect(component.target.querySelector('h3')!.textContent).toBe('Downloading the prompt enhancer');
    expect(component.target.querySelector('[data-testid="job-log"]')!.textContent).toContain('Downloading org/birefnet. Only the first time.');
  });

  it('names the matte download in the compact job bar', () => {
    component = mountComponent(JobBar, { job: makeJob({ phase: 'load', download: MATTE_DOWNLOAD }), ondetails: vi.fn() });
    expect(ring(component.target).getAttribute('aria-valuenow')).toBe('25');
    expect(component.target.querySelector('[data-testid="job-bar"]')!.textContent).toContain('Downloading background removal');
  });

  it('reads the cut-out after the steps as removing the background, to the eye', () => {
    const matting: Partial<ActiveJobState> = { stageNames: ['text_to_image', 'matte', 'save'], stageName: 'matte', stageSteps: { text_to_image: 8, matte: 1 }, currentStep: 1, totalSteps: 1, stepPhase: 'image_matte' };
    component = mountComponent(JobCard, { job: makeJob(matting) });
    const { target } = component;

    expect(target.querySelector('[data-testid="job-caption"]')!.textContent).toContain('Removing background…');
    expect(target.textContent).not.toContain('Finishing up');
  });
});
