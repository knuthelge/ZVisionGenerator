import { describe, expect, it } from 'vitest';

import type { ActiveJobState } from '$lib/types';
import JobPreview from './JobPreview.svelte';
import { makeAudioAsset } from '../../../test-utils/audio';
import { mountComponent } from '../../../test-utils/mount';

function makeJob(overrides: Partial<ActiveJobState> = {}): ActiveJobState {
  return {
    id: 'job-1',
    job_id: 'job-1',
    status: 'running',
    phase: 'generate',
    workflow: 'txt2img',
    prompt: 'A lighthouse',
    message: '',
    currentStep: 8,
    totalSteps: 32,
    currentIteration: 1,
    totalIterations: 1,
    elapsed: 4,
    outputs: [],
    previewUrl: null,
    stageName: 'text_to_image',
    meta: '3:4 · m · 32 steps',
    width: 960,
    height: 1280,
    lastEventType: 'step',
    ...overrides,
  } as ActiveJobState;
}

describe('JobPreview', () => {
  it('reserves the output shape and says when the first preview comes', () => {
    const { target } = mountComponent(JobPreview, { job: makeJob({ currentStep: 1 }) });
    const figure = target.querySelector('[data-testid="job-preview"]') as HTMLElement;
    expect(figure.style.aspectRatio.startsWith('0.75')).toBe(true);
    expect(figure.textContent).toContain('First preview after the first step');
    expect(figure.querySelector('img')).toBeNull();
  });

  it('takes the shape of a custom size, not the ratio preset beside it', () => {
    const { target } = mountComponent(JobPreview, { job: makeJob({ meta: '1600x900 · 32 steps', width: 1600, height: 900 }) });
    const figure = target.querySelector('[data-testid="job-preview"]') as HTMLElement;
    expect(figure.style.aspectRatio.startsWith('1.77')).toBe(true);
  });

  it('puts the checkerboard on a finished transparent output shown between images, never on the live preview', () => {
    const output = { id: 'o.png', url: '/media/o.png', thumbnail_url: '/media/o-thumb.png', transparent: true } as ActiveJobState['outputs'][number];
    const paused = { status: 'paused', paused: true, itemFinished: true, outputs: [output] } as Partial<ActiveJobState>;
    const finished = mountComponent(JobPreview, { job: makeJob(paused) });
    expect(finished.target.querySelector('img')?.dataset.transparent).toBe('true');
    expect(finished.target.querySelector('img')?.classList.contains('ui-checkerboard')).toBe(true);

    const opaque = mountComponent(JobPreview, { job: makeJob({ ...paused, outputs: [{ ...output, transparent: false }] }) });
    expect(opaque.target.querySelector('.ui-checkerboard')).toBeNull();

    const live = mountComponent(JobPreview, { job: makeJob({ previewUrl: '/jobs/job-1/preview?v=1', outputs: [output] }) });
    expect(live.target.querySelector('img')?.dataset.transparent).toBe('false');
    expect(live.target.querySelector('.ui-checkerboard')).toBeNull();
  });

  it('shows the live preview once it arrives, with how far along it is', () => {
    const { target } = mountComponent(JobPreview, { job: makeJob({ previewUrl: '/jobs/job-1/preview?v=1', currentStep: 16 }) });
    const img = target.querySelector('img') as HTMLImageElement;
    expect(img.getAttribute('src')).toBe('/jobs/job-1/preview?v=1');
    expect(target.textContent).toContain('50%');
  });

  describe('an audio job', () => {
    const takeJob = (overrides: Partial<ActiveJobState> = {}) => makeJob({ workflow: 'music', media_type: 'audio', width: null, height: null, ...overrides });
    const figure = (target: HTMLElement) => target.querySelector('[data-testid="job-preview"]') as HTMLElement;

    it('shows no image for an audio job', () => {
      const { target } = mountComponent(JobPreview, { job: takeJob() });

      expect(figure(target).querySelector('img')).toBeNull();
    });

    it('shows the waveform of the take that just finished while paused between takes', () => {
      const { target } = mountComponent(JobPreview, { job: takeJob({ status: 'paused', paused: true, itemFinished: true, outputs: [makeAudioAsset()] }) });

      expect(figure(target).querySelectorAll('rect.bar')).toHaveLength(100);
    });
  });
});
