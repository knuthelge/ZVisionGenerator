// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, mount, unmount } from '../../../../node_modules/svelte/src/index-client.js';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { jobStore } from '$lib/state/job.svelte';
import type { ActiveJobState, GalleryAsset } from '$lib/types';

import JobCard from './JobCard.svelte';
import { reactiveProps } from '../../../test-utils/reactiveProps.svelte';
import * as molecules from './index';

const IMAGE_CONTROLS = ['next', 'pause', 'resume', 'repeat', 'quit'];

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
      phase: 'generate',
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

  function makeAsset(id: string): GalleryAsset {
    return {
      id, url: `/media/${id}`, thumbnail_url: `/media/thumb/${id}`, filename: id,
      created_at: '', workflow: 'txt2img', prompt: '', model: 'zit', reuse_workspace_url: '', media_type: 'image',
    } as GalleryAsset;
  }

  function render(job: Partial<ActiveJobState>, handlers: Record<string, unknown> = {}): void {
    component = mount(JobCard, { target, props: { job: makeJob(job), ...handlers } });
    flushSync();
  }

  const card = () => target.querySelector<HTMLElement>('.job-card')!;
  const header = () => target.querySelector<HTMLElement>('.job-head')!;
  const log = () => target.querySelector<HTMLElement>('.job-log')!;
  const control = (name: string, within: ParentNode = target) => within.querySelector<HTMLButtonElement>(`button[data-control="${name}"]`);
  const jobBar = () => header().querySelector<HTMLElement>('[aria-label="Job progress"]')!;

  it.each([3, 0])('counts a video upscale in batches from the first one (batch %i of 28) and offers only quit', (currentStep) => {
    render({ workflow: 'upscale', media_type: 'video', stageName: 'upscale', stageNames: ['upscale'], currentStep, totalSteps: 28, remaining: 240, supported_controls: ['quit'] });

    expect(jobBar().getAttribute('aria-valuetext')).toBe(`Batch ${currentStep} of 28`);
    expect(control('quit')).not.toBeNull();
  });

  it('labels the workflow and shows the job notices', () => {
    render({ workflow: 'upscale', notices: ['original settings unknown'] });
    expect(target.querySelector('.job-status')?.textContent).toContain('Upscale');
    expect(log().textContent).toContain('original settings unknown');
  });

  it('does not re-export the removed ProgressBar placeholder', () => {
    expect('ProgressBar' in molecules).toBe(false);
  });

  describe('controls by scope and phase', () => {
    it('puts Pause and Cancel in the header and Skip and Repeat on the task line', () => {
      render({ runs: 4, supported_controls: IMAGE_CONTROLS });
      expect(control('pause-resume', header())).not.toBeNull();
      expect(control('quit', header())).not.toBeNull();
      expect(control('next', header())).toBeNull();
      const taskLine = log().querySelector('.job-item[data-s="current"] .job-line')!;
      expect(control('next', taskLine)).not.toBeNull();
      expect(control('repeat', taskLine)).not.toBeNull();
    });

    it('offers no Skip for a single item, since there is nothing to skip to', () => {
      render({ runs: 1, supported_controls: IMAGE_CONTROLS });
      expect(control('next')).toBeNull();
      expect(control('repeat')).not.toBeNull();
    });

    it('offers only Cancel while the job waits in the queue', () => {
      render({ status: 'queued', phase: 'queue', supported_controls: IMAGE_CONTROLS });
      expect(Array.from(target.querySelectorAll('button')).map((button) => button.dataset.control)).toEqual(['quit']);
    });

    it('offers Pause, Skip this rewrite and Cancel while prompts are enhanced', () => {
      render({ phase: 'enhance', enhanceTotal: 4, stageName: 'enhancing_prompts', currentStep: 1, totalSteps: 4, supported_controls: IMAGE_CONTROLS });
      expect(control('pause-resume', header())).not.toBeNull();
      expect(control('next', log())?.getAttribute('aria-label')).toBe('Skip this rewrite');
      expect(control('repeat')).toBeNull();
    });

    it('offers Pause and Cancel while the model loads', () => {
      render({ phase: 'load', supported_controls: IMAGE_CONTROLS });
      expect(Array.from(target.querySelectorAll('button')).map((button) => button.dataset.control)).toEqual(['pause-resume', 'quit']);
    });

    it('swaps Pause for Resume in the same button and drops the item controls while paused', () => {
      const onresume = vi.fn();
      render({ status: 'paused', paused: true, runs: 3, itemFinished: true, supported_controls: IMAGE_CONTROLS }, { onresume });
      const button = control('pause-resume')!;
      expect(button.getAttribute('aria-label')).toBe('Resume');
      expect(control('next')).toBeNull();
      button.click();
      expect(onresume).toHaveBeenCalledWith('job-card');
    });

    it('renders only controls the backend lists', () => {
      render({ supported_controls: [] }, { onpause: vi.fn(), oncancel: vi.fn() });
      expect(target.querySelector('button')).toBeNull();
    });

    it('warns that skipping during the upscale pass discards the base image too', () => {
      render({
        runs: 2, supported_controls: IMAGE_CONTROLS, stageNames: ['text_to_image', 'upscale'], stageName: 'upscale',
        stageSteps: { text_to_image: 28, upscale: 12 }, currentStep: 5, totalSteps: 12,
      });
      expect(control('next')?.getAttribute('aria-label')).toMatch(/discarded/);
    });
  });

  describe('control feedback', () => {
    it('shows a pending control as on, holds the others but never Cancel, and ignores repeat clicks', async () => {
      let resolve!: () => void;
      const onnext = vi.fn(() => new Promise<void>((done) => { resolve = done; }));
      render({ runs: 3, supported_controls: IMAGE_CONTROLS }, { onnext, onpause: vi.fn(), oncancel: vi.fn() });
      const skip = control('next')!;
      skip.click();
      skip.click();
      flushSync();
      expect(onnext).toHaveBeenCalledTimes(1);
      expect(skip.getAttribute('aria-busy')).toBe('true');
      expect(control('pause-resume')!.disabled).toBe(true);
      expect(control('quit')!.disabled).toBe(false);
      resolve();
      await vi.waitFor(() => {
        flushSync();
        expect(target.querySelector('.job-notice .ui-badge')).not.toBeNull();
      });
    });

    it('keeps a queued pause pending until the job pauses', () => {
      const props = reactiveProps({ job: makeJob({ runs: 3, supported_controls: IMAGE_CONTROLS, controlQueued: 'pause' }) });
      component = mount(JobCard, { target, props });
      flushSync();
      expect(control('pause-resume')!.getAttribute('aria-busy')).toBe('true');
      expect(target.querySelector('.job-notice .ui-badge-warning')).not.toBeNull();

      props.job = { ...props.job, controlQueued: null, status: 'paused', paused: true, itemFinished: true };
      flushSync();
      expect(control('pause-resume')!.getAttribute('aria-label')).toBe('Resume');
      expect(target.querySelector('.job-notice .ui-badge')).toBeNull();
    });

    it('shows a rejected control and allows a retry', async () => {
      const onpause = vi.fn().mockRejectedValue(new Error('This job is no longer running.'));
      render({ supported_controls: IMAGE_CONTROLS }, { onpause });
      control('pause-resume')!.click();
      await vi.waitFor(() => {
        flushSync();
        expect(target.querySelector('.job-notice .ui-badge-error')?.textContent).toContain('This job is no longer running.');
        expect(control('pause-resume')!.disabled).toBe(false);
      });
    });

    it('disables every control and drops the estimate while reconnecting', () => {
      render({ runs: 3, remaining: 600, connection: 'reconnecting', lastEventAt: Date.now(), supported_controls: IMAGE_CONTROLS }, { oncancel: vi.fn() });
      expect(card().dataset.tone).toBe('stale');
      expect(Array.from(target.querySelectorAll('button')).every((button) => button.disabled)).toBe(true);
      expect(target.querySelector('.job-time b')).toBeNull();
    });
  });

  describe('progress', () => {
    it('shows one segment per item, grouped by run, with the current one filling by step', () => {
      render({ runs: 2, batchIndex: 1, promptCount: 3, promptNumber: 2, currentStep: 5, totalSteps: 10 });
      const segments = Array.from(jobBar().querySelectorAll<HTMLElement>('.job-segment'));
      expect(segments.map((segment) => segment.dataset.state)).toEqual(['done', 'done', 'done', 'done', 'now', 'pending']);
      expect(segments[3].classList.contains('run-break')).toBe(true);
      expect(segments[4].querySelector<HTMLElement>('b')?.style.width).toBe('50%');
      expect(jobBar().getAttribute('aria-valuenow')).toBe('4');
    });

    it('turns long jobs into one plain bar', () => {
      render({ runs: 100, batchIndex: 40 });
      expect(jobBar().querySelectorAll('.job-segment')).toHaveLength(0);
      expect(jobBar().getAttribute('aria-valuemax')).toBe('100');
    });

    it('shows the step bar with its percentage for a batch, and folds it into the job bar for a single item', () => {
      render({ runs: 3, currentStep: 7, totalSteps: 28 });
      expect(log().querySelector('[aria-label="Current image"]')?.getAttribute('aria-valuenow')).toBe('7');
      expect(log().textContent).toContain('25%');
      unmount(component);

      render({ runs: 1, currentStep: 7, totalSteps: 28 });
      expect(log().querySelector('[role="progressbar"]')).toBeNull();
      expect(jobBar().getAttribute('aria-valuenow')).toBe('7');
    });

    it.each([
      { currentStep: 30, totalSteps: 20, width: '100%' },
      { currentStep: -1, totalSteps: 20, width: '0%' },
    ])('keeps the fill inside the bar when step data is $currentStep / $totalSteps', ({ currentStep, totalSteps, width }) => {
      render({ currentStep, totalSteps });
      expect(jobBar().querySelector<HTMLElement>('.ui-progress-fill')?.style.width).toBe(width);
    });

    it('fills in the step in progress over the time the last step took, and stops when paused', () => {
      const now = vi.spyOn(performance, 'now').mockReturnValue(1000);
      const props = reactiveProps({ job: makeJob({ runs: 2, currentStep: 3, totalSteps: 20 }) });
      component = mount(JobCard, { target, props });
      flushSync();
      expect(target.querySelector('.step-ghost')).toBeNull();

      now.mockReturnValue(2500);
      props.job = { ...props.job, currentStep: 4 };
      flushSync();
      const ghost = target.querySelector<HTMLElement>('.step-ghost');
      expect(ghost?.style.left).toBe('20%');
      expect(ghost?.style.width).toBe('5%');
      expect(ghost?.style.getPropertyValue('--step-ms')).toBe('1500ms');

      props.job = { ...props.job, paused: true, status: 'paused' };
      flushSync();
      expect(target.querySelector('.step-ghost')).toBeNull();
      now.mockRestore();
    });

    it('shows a bar per stage when the image is generated and then upscaled', () => {
      render({
        stageNames: ['resolve_prompt', 'text_to_image', 'upscale', 'save_image'], stageName: 'upscale',
        stageSteps: { text_to_image: 28, upscale: 12 }, currentStep: 6, totalSteps: 12,
      });
      expect(log().querySelector('[aria-label="Generate stage"]')?.getAttribute('aria-valuenow')).toBe('1');
      expect(log().querySelector('[aria-label="Upscale stage"]')?.getAttribute('aria-valuenow')).toBe('6');
      // Both passes count towards the image: generate done, upscale half way.
      expect(jobBar().querySelector<HTMLElement>('.ui-progress-fill')?.style.width).toBe('75%');
    });

    it('counts elapsed time up every second between updates and holds it while paused', () => {
      vi.useFakeTimers({ toFake: ['setInterval', 'clearInterval', 'performance'] });
      try {
        const props = reactiveProps({ job: makeJob({ elapsed: 10 }) });
        component = mount(JobCard, { target, props });
        flushSync();
        const elapsed = () => target.querySelector('.job-time .num')?.textContent;
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

        props.job = { ...props.job, paused: true, status: 'paused' };
        flushSync();
        vi.advanceTimersByTime(5000);
        flushSync();
        expect(elapsed()).toBe('0:14');
      } finally {
        vi.useRealTimers();
      }
    });
  });

  describe('phases and outcomes', () => {
    it('lists the setup phases while the job prepares, with the current one open', () => {
      render({ phase: 'load', enhanceTotal: 0 });
      const rows = Array.from(log().querySelectorAll<HTMLElement>('.job-item'));
      expect(rows.map((row) => row.dataset.s)).toEqual(['done', 'current', 'pending']);
      expect(card().dataset.tone).toBe('preparing');
    });

    it('pins a failure to the phase it happened in and shows the server reason', () => {
      render({ status: 'failed', phase: 'load', message: 'Not enough memory to load the model.' });
      const rows = Array.from(log().querySelectorAll<HTMLElement>('.job-item'));
      expect(rows.map((row) => row.dataset.s)).toEqual(['done', 'failed', 'pending']);
      expect(log().querySelector('.job-item[data-s="failed"] [role="alert"]')?.textContent).toContain('Not enough memory to load the model.');
      expect(card().dataset.tone).toBe('failed');
    });

    it('leaves outputs to the history: an ended job only counts what it saved', () => {
      render({ status: 'cancelled', runs: 4, batchIndex: 2, itemFinished: true, outputs: [makeAsset('a.png'), makeAsset('b.png'), makeAsset('c.png')] });
      expect(target.querySelector('img')).toBeNull();
      expect(log().textContent).toMatch(/3 of 4/);
      expect(card().dataset.tone).toBe('stopped');
    });

    it('shows the live preview only while the job is active', () => {
      const previewAlt = 'img[alt="Live preview of the generation in progress"]';
      render({ previewUrl: '/jobs/job-card/preview?v=2' });
      expect(target.querySelector(previewAlt)?.getAttribute('src')).toBe('/jobs/job-card/preview?v=2');
      unmount(component);

      render({ status: 'completed', previewUrl: '/jobs/job-card/preview?v=2' });
      expect(target.querySelector(previewAlt)).toBeNull();
    });

    it('shows the image that just finished while paused between images', () => {
      render({ status: 'paused', paused: true, runs: 3, itemFinished: true, outputs: [makeAsset('last.png')], previewUrl: null });
      expect(target.querySelector('img')?.getAttribute('src')).toBe('/media/thumb/last.png');
    });

    it('says the image is finishing up once its last step ran, with the bars still moving', () => {
      render({ runs: 3, currentStep: 28, totalSteps: 28 });
      expect(log().querySelector('.job-step-head')?.textContent).toContain('Finishing up');
      expect(log().querySelector('[aria-label="Current image"]')?.classList.contains('is-live')).toBe(true);
    });

    it('says the clips are being joined once an extension ran its last step, in the log and on the bars', () => {
      render({
        workflow: 'extend', runs: 3, stageNames: ['extend_video', 'join_clips'], stageName: 'join_clips', stageSteps: { extend_video: 8 }, currentStep: 8, totalSteps: 8,
      });
      expect(log().querySelector('.job-step-head')?.textContent).toContain('Joining clips…');
      expect(log().textContent).not.toContain('Finishing up');
      expect(header().querySelector('.ui-area-label')?.textContent).toBe('Extend video');
    });

    it('reads the joining stretch to assistive technology without the ellipsis', () => {
      render({ workflow: 'extend', runs: 3, stageNames: ['extend_video', 'join_clips'], stageName: 'join_clips', stageSteps: { extend_video: 8 }, currentStep: 8, totalSteps: 8 });
      const valuetexts = Array.from(target.querySelectorAll('[role="progressbar"]')).map((bar) => bar.getAttribute('aria-valuetext'));
      expect(valuetexts).toContain('Joining clips');
      expect(valuetexts).not.toContain('Finishing up');
    });

    it('reads the header bar of a single extension as joining clips too', () => {
      render({ workflow: 'extend', runs: 1, stageNames: ['extend_video', 'join_clips'], stageName: 'join_clips', stageSteps: { extend_video: 8 }, currentStep: 8, totalSteps: 8 });
      expect(header().querySelector('[role="progressbar"]')?.getAttribute('aria-valuetext')).toBe('Joining clips');
    });

    it('promises the first preview at 25% and says so when a model renders none', () => {
      render({ runs: 3, currentStep: 2, totalSteps: 28 });
      expect(log().querySelector('.job-media')?.textContent).toContain('25%');
      unmount(component);

      render({ runs: 3, currentStep: 20, totalSteps: 28 });
      expect(log().querySelector('.job-media')?.textContent).toContain('No live preview');
    });

    it('marks a prompt that was not enhanced', () => {
      render({ enhanceStatus: 'skipped' });
      expect(log().querySelector('.job-badges')?.textContent).toContain('Not enhanced');
    });
  });

  describe('job control keys', () => {
    function press(key: string, from: EventTarget = document): KeyboardEvent {
      const event = new KeyboardEvent('keydown', { key, bubbles: true, cancelable: true });
      from.dispatchEvent(event);
      return event;
    }

    function mountCard(job: Partial<ActiveJobState>) {
      const handlers = { onpause: vi.fn(), onresume: vi.fn(), onnext: vi.fn(), onrepeat: vi.fn(), oncancel: vi.fn() };
      render({ runs: 3, supported_controls: IMAGE_CONTROLS, ...job }, handlers);
      return handlers;
    }

    afterEach(() => { document.querySelectorAll('textarea, [aria-modal]').forEach((el) => el.remove()); });

    it.each([
      ['n', 'onnext'],
      ['R', 'onrepeat'],
      ['p', 'onpause'],
    ] as const)('sends the control for %s', (key, handler) => {
      const handlers = mountCard({});
      expect(press(key).defaultPrevented).toBe(true);
      expect(handlers[handler]).toHaveBeenCalledWith('job-card');
    });

    describe('while a review waits', () => {
      const REVIEW_CONTROLS = ['next', 'pause', 'resume', 'repeat', 'approve', 'discard', 'quit'];
      const reviewing = { status: 'paused', paused: true, review: { iteration: 2 }, supported_controls: REVIEW_CONTROLS } as const;

      it('keeps only Stop on the card', () => {
        mountCard({ ...reviewing });
        expect(control('quit')).not.toBeNull();
        for (const name of ['pause-resume', 'next', 'repeat', 'approve', 'discard']) expect(control(name)).toBeNull();
      });

      it.each(['p', 'r', 'n', 'y'])('does nothing for %s', (key) => {
        const handlers = mountCard({ ...reviewing });
        press(key);
        for (const handler of Object.values(handlers)) expect(handler).not.toHaveBeenCalled();
      });

      it('still stops the job with Esc twice', () => {
        const handlers = mountCard({ ...reviewing });
        press('Escape');
        press('Escape');
        expect(handlers.oncancel).toHaveBeenCalledWith('job-card');
      });
    });

    it('resumes a paused job with P', () => {
      const handlers = mountCard({ status: 'paused', paused: true });
      press('p');
      expect(handlers.onresume).toHaveBeenCalledWith('job-card');
      expect(handlers.onpause).not.toHaveBeenCalled();
    });

    it('ignores keys for controls the job does not offer, and keys typed into a field', () => {
      const handlers = mountCard({ supported_controls: ['pause'] });
      expect(press('n').defaultPrevented).toBe(false);
      const field = document.createElement('textarea');
      document.body.appendChild(field);
      press('p', field);
      expect(handlers.onnext).not.toHaveBeenCalled();
      expect(handlers.onpause).not.toHaveBeenCalled();
    });

    it('cancels only on a second Esc, and says so after the first', () => {
      const handlers = mountCard({});
      expect(press('Escape').defaultPrevented).toBe(true);
      flushSync();
      expect(handlers.oncancel).not.toHaveBeenCalled();
      expect(control('quit')!.classList.contains('ui-btn-danger')).toBe(true);
      expect(target.querySelector('.job-notice .ui-badge-error')).not.toBeNull();
      press('Escape');
      expect(handlers.oncancel).toHaveBeenCalledWith('job-card');
    });

    it('disarms Cancel when the second Esc comes too late', () => {
      const now = vi.spyOn(performance, 'now').mockReturnValue(1000);
      const handlers = mountCard({});
      press('Escape');
      now.mockReturnValue(5000);
      press('Escape');
      expect(handlers.oncancel).not.toHaveBeenCalled();
      now.mockRestore();
    });

    it('cancels at once from the Cancel button', () => {
      const handlers = mountCard({});
      control('quit')!.click();
      expect(handlers.oncancel).toHaveBeenCalledWith('job-card');
    });

    it('leaves Escape to a focused field or an open dialog', () => {
      const handlers = mountCard({});
      const field = document.createElement('textarea');
      document.body.appendChild(field);
      press('Escape', field);
      press('Escape', field);
      document.body.insertAdjacentHTML('beforeend', '<div role="dialog" aria-modal="true"></div>');
      press('Escape');
      press('Escape');
      expect(handlers.oncancel).not.toHaveBeenCalled();
    });
  });

  describe('while the model downloads', () => {
    const download = { role: 'model' as const, model: 'zit', repo: 'org/zit', part: 1, parts: 2, bytes_done: 250, bytes_total: 1000, bytes_per_sec: null, download_eta_secs: null, shared: false };

    it('shows the download in the header bar and in the load row', () => {
      render({ phase: 'load', download });
      const bar = target.querySelector<HTMLElement>('.job-head [aria-label="Model download"]')!;
      expect(bar.getAttribute('aria-valuenow')).toBe('0.25');
      expect(log().querySelector('[aria-label="Model download"]')).not.toBeNull();
      expect(log().querySelector('.dl-pct')?.textContent).toBe('25%');
    });

    it('leaves the bars indeterminate while the total is unknown', () => {
      render({ phase: 'load', download: { ...download, bytes_total: null } });
      for (const bar of target.querySelectorAll('[aria-label="Model download"]')) expect(bar.hasAttribute('aria-valuenow')).toBe(false);
    });

    it('shows an enhancer download on the enhance row', () => {
      render({ phase: 'enhance', enhanceTotal: 2, download: { ...download, role: 'enhancer' } });
      expect(log().querySelector('.job-item[data-s="current"] [aria-label="Enhancer download"]')).not.toBeNull();
      expect(log().querySelector('[aria-label="Model download"]')).toBeNull();
    });

    it('drops the download block once the job is stopped', () => {
      render({ phase: 'load', status: 'cancelled', downloadStop: 'detached', download });
      expect(log().querySelector('[aria-label="Model download"]')).toBeNull();
      expect(log().querySelector('.job-item[data-s="stopped"] .job-prompt')).not.toBeNull();
    });
  });
});
