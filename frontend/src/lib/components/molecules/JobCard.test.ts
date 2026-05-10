// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, mount, unmount } from '../../../../node_modules/svelte/src/index-client.js';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { jobStore } from '$lib/state/job.svelte';
import type { ActiveJobState } from '$lib/types';

import JobCard from './JobCard.svelte';
import * as molecules from './index';

describe('JobCard', () => {
  let target: HTMLDivElement;
  let component: Record<string, unknown> | null = null;

  beforeEach(() => {
    target = document.createElement('div');
    document.body.appendChild(target);
    jobStore.clearJob();
  });

  afterEach(() => {
    if (component) {
      unmount(component);
      component = null;
    }
    target.remove();
  });

  function makeJob(overrides: Partial<ActiveJobState> = {}): ActiveJobState {
    return {
      job_id: 'job-card',
      workflow: 'txt2img',
      prompt: 'Card prompt',
      model: 'zit',
      runs: 1,
      created_at: '2026-04-22T00:00:00Z',
      supported_controls: [],
      status: 'running',
      currentStep: 0,
      totalSteps: 10,
      elapsed: 0,
      remaining: 0,
      stageName: '',
      stageIndex: 0,
      batchLabel: '',
      batchIndex: 0,
      paused: false,
      message: '',
      outputs: [],
      ...overrides,
    };
  }

  it('does not re-export the removed ProgressBar placeholder', () => {
    expect('ProgressBar' in molecules).toBe(false);
  });

  it('renders batch counters and remaining time from live job state', () => {
    const jobProps = makeJob({
      runs: 3,
      batchIndex: 1,
      batchLabel: 'Run 2 ready',
      remaining: 125,
      currentStep: 4,
      totalSteps: 20,
    });
    component = mount(JobCard, { target, props: { job: jobProps } });
    flushSync();

    const text = target.textContent ?? '';
    expect(text).toContain(jobProps.batchLabel);
    expect(text).toContain('2 / 3');
    expect(text).toContain('4 / 20');
    expect(text).toContain('02:05');
  });

  it('renders repeat controls and status messages through live callbacks', () => {
    const onrepeat = vi.fn();
    const oncancel = vi.fn();
    const jobProps = makeJob({ supported_controls: ['repeat', 'quit'], message: 'Waiting for operator input.' });
    component = mount(JobCard, {
      target,
      props: { job: jobProps, onrepeat, oncancel },
    });
    flushSync();

    const cancelButton = target.querySelector('button[aria-label="Cancel job"]');
    const repeatButton = Array.from(target.querySelectorAll('button')).find((button) => button !== cancelButton);
    expect(target.querySelector('article')).not.toBeNull();
    expect(cancelButton).not.toBeNull();
    expect(repeatButton).toBeDefined();

    repeatButton!.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    cancelButton!.dispatchEvent(new MouseEvent('click', { bubbles: true }));

    expect(onrepeat).toHaveBeenCalledWith('job-card');
    expect(oncancel).toHaveBeenCalledWith('job-card');
  });

  it('renders only controls listed by the backend', () => {
    component = mount(JobCard, {
      target,
      props: {
        job: makeJob({ supported_controls: ['next', 'quit'] }),
        onnext: vi.fn(),
        oncancel: vi.fn(),
      },
    });
    flushSync();

    expect(target.querySelector('button[aria-label="Cancel job"]')).not.toBeNull();
  });

  it('removes unsupported running controls from the DOM', () => {
    component = mount(JobCard, {
      target,
      props: {
        job: makeJob({ supported_controls: [] }),
        onpause: vi.fn(),
        onresume: vi.fn(),
        onnext: vi.fn(),
        onrepeat: vi.fn(),
        oncancel: vi.fn(),
      },
    });
    flushSync();

    expect(target.querySelector('button[aria-label="Cancel job"]')).toBeNull();
  });

  it('renders completed output previews without requiring a legacy path field', () => {
    component = mount(JobCard, {
      target,
      props: {
        job: makeJob({
          status: 'completed',
          outputs: [
            {
              id: 'outputs/first.png',
              url: '/media/first.png',
              thumbnail_url: '/media/first-thumb.png',
              filename: 'first.png',
              created_at: '2026-04-22T00:00:00Z',
              workflow: 'txt2img',
              prompt: 'First output',
              model: 'zit',
              reuse_workspace_url: '#/workspace?workflow=txt2img',
              media_type: 'image',
            },
            {
              id: 'outputs/second.mp4',
              url: '/media/second.mp4',
              thumbnail_url: '/media/second-thumb.mp4',
              filename: 'second.mp4',
              created_at: '2026-04-22T00:00:01Z',
              workflow: 'txt2vid',
              prompt: 'Second output',
              model: 'ltx-8',
              reuse_workspace_url: '#/workspace?workflow=txt2vid',
              media_type: 'video',
            },
          ],
        }),
      },
    });
    flushSync();

    expect(target.querySelectorAll('a[href^="/media/"]')).toHaveLength(2);
    expect(target.querySelectorAll('img[alt="first.png"]')).toHaveLength(1);
    expect(target.querySelectorAll('video')).toHaveLength(1);
  });
});