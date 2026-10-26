// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, mount, unmount } from '../../../../node_modules/svelte/src/index-client.js';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import type { ActiveJobState } from '$lib/types';
import JobPreview from './JobPreview.svelte';

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

  it('reserves the output shape and says when the first preview comes', () => {
    app = flushSync(() => mount(JobPreview, { target, props: { job: makeJob({ currentStep: 1 }) } }));
    const figure = target.querySelector('[data-testid="job-preview"]') as HTMLElement;
    expect(figure.style.aspectRatio.startsWith('0.75')).toBe(true);
    expect(figure.textContent).toContain('First preview after the first step');
    expect(figure.querySelector('img')).toBeNull();
  });

  it('takes the shape of a custom size, not the ratio preset beside it', () => {
    app = flushSync(() => mount(JobPreview, { target, props: { job: makeJob({ meta: '1600x900 · 32 steps', width: 1600, height: 900 }) } }));
    const figure = target.querySelector('[data-testid="job-preview"]') as HTMLElement;
    expect(figure.style.aspectRatio.startsWith('1.77')).toBe(true);
  });

  it('shows the live preview once it arrives, with how far along it is', () => {
    app = flushSync(() => mount(JobPreview, { target, props: { job: makeJob({ previewUrl: '/jobs/job-1/preview?v=1', currentStep: 16 }) } }));
    const img = target.querySelector('img') as HTMLImageElement;
    expect(img.getAttribute('src')).toBe('/jobs/job-1/preview?v=1');
    expect(target.textContent).toContain('50%');
  });

  it('says a video has no live preview', () => {
    app = flushSync(() => mount(JobPreview, { target, props: { job: makeJob({ workflow: 'txt2vid', stageName: 'text_to_video' }) } }));
    expect(target.textContent).toContain('No live preview for video');
  });

  it('names the phase while the model loads', () => {
    app = flushSync(() => mount(JobPreview, { target, props: { job: makeJob({ phase: 'load', currentStep: 0, stageName: '' }) } }));
    expect(target.textContent).toContain('Loading the model');
  });
});
