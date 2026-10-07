// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, mount, unmount } from '../../../../node_modules/svelte/src/index-client.js';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { jobStore } from '$lib/state/job.svelte';
import type { ActiveJobState } from '$lib/types';

import JobCard from './JobCard.svelte';
import { reactiveProps } from '../../../test-utils/reactiveProps.svelte';
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
      previewUrl: null,
      ...overrides,
    };
  }

  it('labels upscale jobs and shows their notices', () => {
    component = mount(JobCard, { target, props: { job: makeJob({ workflow: 'upscale', notices: ['original settings unknown'] }) } });
    flushSync();

    expect(target.querySelector('h3')?.textContent).toBe('Upscale');
    expect(target.querySelector('[data-testid="job-notices"]')?.textContent).toBe('original settings unknown');
  });

  it('does not re-export the removed ProgressBar placeholder', () => {
    expect('ProgressBar' in molecules).toBe(false);
  });

  it('renders one run counter alongside stage progress and timing', () => {
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
    expect(text).not.toContain(jobProps.batchLabel);
    expect(text.match(/Run 2 of 3/g)).toHaveLength(1);
    expect(text).not.toContain('Batch');
    expect(text).not.toContain('Current run');
    expect(text).toContain('4 / 20');
    expect(text).toContain('02:05');
    expect(target.querySelector('[role="progressbar"]')?.getAttribute('aria-valuenow')).toBe('20');
  });

  it('integrates the active prompt counter above the prompt with separate repeat-run context', () => {
    component = mount(JobCard, { target, props: { job: makeJob({ prompt: 'Current landscape', promptNumber: 2, promptCount: 5, runs: 3, batchIndex: 1 }) } });
    flushSync();
    expect(target.textContent).toContain('Prompt 2 of 5');
    expect(target.textContent).toContain('Run 2 of 3');
    expect(target.querySelector('[aria-live="polite"]')?.textContent?.replace(/\s+/g, ' ').trim()).toBe('Run 2 of 3 · Prompt 2 of 5');
    expect(target.querySelector('[title="Current landscape"]')).not.toBeNull();
  });

  it('keeps single-prompt jobs free of redundant prompt counters', () => {
    component = mount(JobCard, { target, props: { job: makeJob({ promptNumber: 1, promptCount: 1 }) } });
    flushSync();
    expect(target.textContent).not.toContain('Prompt 1 of 1');
  });

  it.each([
    { currentStep: 30, totalSteps: 20, expected: '100' },
    { currentStep: -1, totalSteps: 20, expected: '0' },
    { currentStep: 0, totalSteps: 0, expected: null },
  ])('keeps progress valid when step data is $currentStep / $totalSteps', ({ currentStep, totalSteps, expected }) => {
    component = mount(JobCard, { target, props: { job: makeJob({ currentStep, totalSteps }) } });
    flushSync();
    expect(target.querySelector('[role="progressbar"]')?.getAttribute('aria-valuenow')).toBe(expected);
  });

  it('fills in the step in progress over the time the last step took', () => {
    const now = vi.spyOn(performance, 'now').mockReturnValue(1000);
    const props = reactiveProps({ job: makeJob({ currentStep: 3, totalSteps: 20 }) });
    component = mount(JobCard, { target, props });
    flushSync();
    // No step has been timed yet, so there is nothing to fill.
    expect(target.querySelector('.current-step')).toBeNull();

    now.mockReturnValue(2500);
    props.job = { ...props.job, currentStep: 4 };
    flushSync();
    const current = target.querySelector<HTMLElement>('[role="progressbar"] .current-step');
    expect(current?.dataset.state).toBe('running');
    expect(current?.style.left).toBe('20%');
    expect(current?.style.width).toBe('5%');
    expect(current?.style.getPropertyValue('--step-ms')).toBe('1500ms');

    props.job = { ...props.job, paused: true };
    flushSync();
    expect(target.querySelector<HTMLElement>('.current-step')?.dataset.state).toBe('paused');

    props.job = { ...props.job, paused: false, currentStep: 20 };
    flushSync();
    expect(target.querySelector('.current-step')).toBeNull();
    now.mockRestore();
  });

  it('counts elapsed time up every second between updates and holds it while paused', () => {
    vi.useFakeTimers({ toFake: ['setInterval', 'clearInterval', 'performance'] });
    try {
      const props = reactiveProps({ job: makeJob({ elapsed: 10 }) });
      component = mount(JobCard, { target, props });
      flushSync();
      const elapsed = () => target.querySelector('.job-timing dd')?.textContent;
      expect(elapsed()).toBe('0:10');

      vi.advanceTimersByTime(3000);
      flushSync();
      expect(elapsed()).toBe('0:13');

      // A server update that lags the local count never moves the clock backwards.
      props.job = { ...props.job, elapsed: 12.5 };
      flushSync();
      vi.advanceTimersByTime(1000);
      flushSync();
      expect(elapsed()).toBe('0:14');

      props.job = { ...props.job, paused: true };
      flushSync();
      vi.advanceTimersByTime(5000);
      flushSync();
      expect(elapsed()).toBe('0:14');
    } finally {
      vi.useRealTimers();
    }
  });

  it('shows resume without pause when a running job is paused', () => {
    const onresume = vi.fn();
    component = mount(JobCard, { target, props: {
      job: makeJob({ paused: true, supported_controls: ['pause', 'resume'] }), onresume,
    } });
    flushSync();
    const buttons = Array.from(target.querySelectorAll('button'));
    expect(buttons.map((button) => button.textContent?.trim())).toEqual(['Resume']);
    buttons[0].click();
    expect(onresume).toHaveBeenCalledWith('job-card');
  });

  it.each(['running', 'paused', 'failed', 'cancelled', 'unknown', 'completed'] as const)('keeps the run position separate from the %s status', (status) => {
    component = mount(JobCard, { target, props: { job: makeJob({ runs: 3, batchIndex: 1, status }) } });
    flushSync();
    expect(target.textContent?.match(/Run 2 of 3/g)).toHaveLength(1);
    expect(target.querySelector('.job-status')?.textContent).toContain(status);
    expect(target.textContent).not.toContain('Batch');
  });

  it('keeps large runs compact', () => {
    component = mount(JobCard, { target, props: { job: makeJob({ runs: 100, batchIndex: 40 }) } });
    flushSync();
    expect(target.textContent).toContain('Run 41 of 100');
    expect(target.querySelectorAll('[aria-label="Generation sequence"] [role="listitem"]')).toHaveLength(24);
    expect(target.querySelector('[aria-current="step"]')?.getAttribute('aria-label')).toBe('Run 41: current');
  });

  it('groups prompt segments by run and highlights the current prompt', () => {
    component = mount(JobCard, { target, props: { job: makeJob({ runs: 2, batchIndex: 1, promptCount: 3, promptNumber: 2 }) } });
    flushSync();
    const segments = target.querySelectorAll('[aria-label="Generation sequence"] [role="listitem"]');
    expect(Array.from(segments).map((segment) => segment.getAttribute('data-state'))).toEqual(['previous', 'previous', 'previous', 'previous', 'current', 'waiting']);
    expect(segments[3].classList.contains('run-boundary')).toBe(true);
    expect(segments[4].getAttribute('aria-label')).toBe('Run 2 · Prompt 2: current');
    expect(target.textContent?.match(/Run 2 of 2/g)).toHaveLength(1);
  });

  it('shows one readable stage label without the duplicate running message', () => {
    component = mount(JobCard, { target, props: { job: makeJob({ stageName: 'text_to_image', message: 'Running text to image.' }) } });
    flushSync();
    expect(target.textContent).toContain('Text to image');
    expect(target.textContent).not.toContain('Running text to image.');
    expect(target.textContent).not.toContain('text_to_image');
  });

  it('marks a generation whose prompt was not enhanced, and only when no rewrite is shown', () => {
    component = mount(JobCard, { target, props: { job: makeJob({ enhanceStatus: 'failed' }) } });
    flushSync();
    expect(target.querySelector('[data-enhance-status="failed"]')).not.toBeNull();
    unmount(component);

    component = mount(JobCard, { target, props: { job: makeJob({ enhanceStatus: 'skipped' }) } });
    flushSync();
    expect(target.querySelector('[data-enhance-status="skipped"]')).not.toBeNull();
    unmount(component);

    for (const job of [makeJob({ enhanceStatus: 'off' }), makeJob({ enhanceStatus: 'enhanced', enhancedPrompt: 'A fox in snow.' }), makeJob()]) {
      component = mount(JobCard, { target, props: { job } });
      flushSync();
      expect(target.querySelector('[data-enhance-status]')).toBeNull();
      unmount(component);
    }
    component = null;
  });

  it('immediately shows pending feedback, prevents duplicate clicks, then confirms acceptance', async () => {
    let resolve!: () => void;
    const onpause = vi.fn(() => new Promise<void>((done) => { resolve = done; }));
    component = mount(JobCard, { target, props: { job: makeJob({ supported_controls: ['pause'] }), onpause } });
    flushSync();
    const button = target.querySelector('button')!;
    button.click();
    button.click();
    flushSync();
    expect(onpause).toHaveBeenCalledTimes(1);
    expect(button.disabled).toBe(true);
    expect(button.textContent).toContain('Sending…');
    expect(target.textContent).toContain('Sending pause request…');
    resolve();
    await vi.waitFor(() => {
      flushSync();
      expect(button.disabled).toBe(false);
      expect(target.textContent).toContain('Pause request accepted.');
    });
  });

  it('shows rejected controls and allows retry', async () => {
    const onpause = vi.fn().mockRejectedValue(new Error('This job is no longer running.'));
    component = mount(JobCard, { target, props: { job: makeJob({ supported_controls: ['pause'] }), onpause } });
    flushSync();
    const button = target.querySelector('button')!;
    button.click();
    await vi.waitFor(() => {
      flushSync();
      expect(target.textContent).toContain('Pause failed. This job is no longer running.');
      expect(button.disabled).toBe(false);
    });
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

  it('leaves generated outputs to the history and shows only the live preview', () => {
    const output = {
      id: 'out.png', url: '/media/out.png', thumbnail_url: '/media/out.png', filename: 'out.png',
      created_at: '', workflow: 'txt2img' as const, prompt: '', model: 'zit', reuse_workspace_url: '', media_type: 'image' as const,
    };
    component = mount(JobCard, { target, props: { job: makeJob({ previewUrl: '/jobs/job-card/preview?v=2', outputs: [output] }) } });
    flushSync();

    const figure = target.querySelector('figure') as HTMLElement;
    expect(document.getElementById(figure.getAttribute('aria-labelledby')!)?.textContent).toBe('Live preview');
    expect(target.querySelector('img[alt="out.png"]')).toBeNull();
    expect(target.querySelector('button[aria-label^="View "]')).toBeNull();
  });

  it('shows the live preview only while the job is active', () => {
    const previewAlt = 'img[alt="Live preview of the generation in progress"]';
    component = mount(JobCard, { target, props: { job: makeJob({ previewUrl: '/jobs/job-card/preview?v=2', supported_controls: ['next'] }), onnext: vi.fn() } });
    flushSync();
    const preview = target.querySelector(previewAlt);
    expect(preview?.getAttribute('src')).toBe('/jobs/job-card/preview?v=2');
    // The preview renders below the controls, so Next does not move when a preview appears.
    const next = Array.from(target.querySelectorAll('button')).find((button) => button.textContent?.trim() === 'Next');
    expect(next!.compareDocumentPosition(preview!) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    unmount(component);

    component = mount(JobCard, { target, props: { job: makeJob({ status: 'completed', previewUrl: '/jobs/job-card/preview?v=2' }) } });
    flushSync();
    expect(target.querySelector(previewAlt)).toBeNull();
  });

  describe('job control keys', () => {
    function press(key: string, from: EventTarget = document): KeyboardEvent {
      const event = new KeyboardEvent('keydown', { key, bubbles: true, cancelable: true });
      from.dispatchEvent(event);
      return event;
    }

    function mountCard(job: Partial<ActiveJobState>) {
      const handlers = { onpause: vi.fn(), onresume: vi.fn(), onnext: vi.fn(), onrepeat: vi.fn() };
      component = mount(JobCard, {
        target,
        props: { job: makeJob({ supported_controls: ['pause', 'resume', 'next', 'repeat'], ...job }), ...handlers },
      });
      flushSync();
      return handlers;
    }

    afterEach(() => { document.querySelectorAll('textarea').forEach((el) => el.remove()); });

    it('sends next with N, repeat with R and pause with P', async () => {
      const handlers = mountCard({});
      expect(press('n').defaultPrevented).toBe(true);
      await Promise.resolve();
      expect(handlers.onnext).toHaveBeenCalledWith('job-card');
      press('R');
      await Promise.resolve();
      expect(handlers.onrepeat).toHaveBeenCalledWith('job-card');
      press('p');
      expect(handlers.onpause).toHaveBeenCalledWith('job-card');
    });

    it('resumes a paused job with P', () => {
      const handlers = mountCard({ status: 'paused', paused: true });
      press('p');
      expect(handlers.onresume).toHaveBeenCalledWith('job-card');
      expect(handlers.onpause).not.toHaveBeenCalled();
    });

    it('ignores keys for controls the job does not support, and keys typed into a field', () => {
      const handlers = mountCard({ supported_controls: ['pause'] });
      expect(press('n').defaultPrevented).toBe(false);
      const field = document.createElement('textarea');
      document.body.appendChild(field);
      press('p', field);
      expect(handlers.onnext).not.toHaveBeenCalled();
      expect(handlers.onpause).not.toHaveBeenCalled();
    });
  });

  describe('Escape', () => {
    function escape(from: EventTarget = document): KeyboardEvent {
      const event = new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true });
      from.dispatchEvent(event);
      return event;
    }

    function mountCard(job: Partial<ActiveJobState>): ReturnType<typeof vi.fn> {
      const oncancel = vi.fn();
      component = mount(JobCard, { target, props: { job: makeJob({ supported_controls: ['quit'], ...job }), oncancel } });
      flushSync();
      return oncancel;
    }

    afterEach(() => { document.querySelectorAll('textarea, [aria-modal]').forEach((el) => el.remove()); });

    it('cancels a running job', () => {
      const oncancel = mountCard({});
      expect(escape().defaultPrevented).toBe(true);
      expect(oncancel).toHaveBeenCalledWith('job-card');
    });

    it('does nothing when the job cannot be cancelled', () => {
      const oncancel = mountCard({ supported_controls: [] });
      escape();
      expect(oncancel).not.toHaveBeenCalled();
    });

    it('leaves Escape to a focused field or an open dialog', () => {
      const oncancel = mountCard({});
      const field = document.createElement('textarea');
      document.body.appendChild(field);
      escape(field);
      document.body.insertAdjacentHTML('beforeend', '<div role="dialog" aria-modal="true"></div>');
      escape();
      expect(oncancel).not.toHaveBeenCalled();
    });
  });
});
