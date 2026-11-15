import { flushSync } from 'svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { jobStore } from '$lib/state/job.svelte';
import type { ActiveJobState, GalleryAsset } from '$lib/types';

import JobCard from './JobCard.svelte';
import { mountComponent } from '../../../test-utils/mount';
import { reactiveProps } from '../../../test-utils/reactiveProps.svelte';

const IMAGE_CONTROLS = ['next', 'pause', 'resume', 'repeat', 'quit'];

describe('JobCard', () => {
  let target: HTMLElement;
  let component: ReturnType<typeof mountComponent> | null = null;

  beforeEach(() => jobStore.clearJob());

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
    component = mountComponent(JobCard, { job: makeJob(job), ...handlers });
    target = component.target;
  }

  const card = () => target.querySelector<HTMLElement>('article[aria-label="Active job"]')!;
  const log = () => target.querySelector<HTMLElement>('[data-testid="job-log"]')!;
  const ring = () => target.querySelector<SVGElement>('[data-testid="progress-ring"]')!;
  const notice = () => target.querySelector<HTMLElement>('[data-testid="job-notice"]');
  const caption = () => target.querySelector<HTMLElement>('[data-testid="job-caption"]');
  const time = () => target.querySelector<HTMLElement>('[data-testid="job-time"]')!;
  const headline = () => target.querySelector('h3')!.textContent;
  const phases = () => Array.from(log().querySelectorAll<HTMLElement>('[data-testid="phase-row"]'));
  const control = (name: string, within: ParentNode = target) => within.querySelector<HTMLButtonElement>(`button[data-control="${name}"]`);

  describe('the hero', () => {
    it('is the ring with the headline, the time and the "now" caption beside it, under the kicker', () => {
      render({ runs: 4, batchIndex: 2, currentStep: 5, totalSteps: 8, remaining: 120, elapsed: 74 });

      expect(card().getAttribute('aria-label')).toBe('Active job');
      expect(card().querySelector('[data-testid="job-kicker"]')?.textContent).toContain('Text to image');
      expect(headline()).toBe('Generating image 3 of 4');
      expect(time().textContent).toContain('About 2 min left');
      expect(time().textContent).toContain('1:14 elapsed');
      expect(caption()?.textContent).toContain('Step 5 of 8');
      expect(ring().getAttribute('aria-valuenow')).toBe('65');
      expect(target.querySelector('[data-testid="ring-stamp"]')?.textContent).toBe('65%');
    });

    it('shortens the headline from 100 images, so the count is never cut', () => {
      render({ runs: 100, batchIndex: 32 });
      expect(headline()).toBe('Image 33 of 100');
    });

    it('keeps the model name whole inside the headline', () => {
      render({ phase: 'load', loadPhase: 'downloading', model_label: 'Z-Image Turbo' });
      expect(card().querySelector('h3 [data-testid="headline-model"]')?.textContent).toBe('Z-Image Turbo');
    });

    it('draws Blob through the snippet it is given, at the ring\'s Blob size', async () => {
      const { createRawSnippet } = await import('svelte');
      const mascot = createRawSnippet((size: () => number) => ({ render: () => `<i data-testid="blob">${size()}</i>` }));
      render({}, { mascot });

      expect(target.querySelector('[data-testid="ring-hero"] [data-testid="blob"]')?.textContent).toBe('112');
    });

    it('leaves the ring empty when Blob is elsewhere', () => {
      render({});
      expect(target.querySelector('svg.mascot')).toBeNull();
      expect(target.querySelector('[data-testid="ring-hero"]')).not.toBeNull();
    });

    it('rests the ring while the job waits, with the worker caption', () => {
      render({ status: 'queued', phase: 'queue', supported_controls: ['quit'] });

      expect(ring().getAttribute('data-kind')).toBe('rest');
      expect(ring().getAttribute('role')).toBeNull();
      expect(caption()?.textContent).toContain('picks it up as soon as it');
      expect(control('quit')).not.toBeNull();
      expect(target.querySelector('[data-testid="controls-divider"]')).toBeNull();
    });

    it('sweeps while the model loads, and says so to assistive technology', () => {
      render({ phase: 'load' });

      expect(ring().getAttribute('data-kind')).toBe('busy');
      expect(ring().getAttribute('aria-busy')).toBe('true');
      expect(target.querySelector('[data-testid="ring-stamp"]')).toBeNull();
    });

    it('counts a video upscale in batches and offers only quit', () => {
      render({ workflow: 'upscale', media_type: 'video', stageName: 'upscale', stageNames: ['upscale'], currentStep: 3, totalSteps: 28, remaining: 240, supported_controls: ['quit'] });

      expect(ring().getAttribute('aria-valuetext')).toContain('batch 3 of 28');
      expect(control('quit')).not.toBeNull();
    });

    it('names the project the job saves into, at the kicker bar\'s end, with "Saves to" for assistive technology', () => {
      render({ save_project: 'Night Bus' });

      const project = card().querySelector('[data-testid="job-kicker"] [data-testid="job-project"]')!;
      expect(project.getAttribute('title')).toBe('Saves to Night Bus');
      expect(project.textContent).toContain('Night Bus');
      expect(project.textContent).toContain('Saves to');
    });

    it('shows no project for a job that saves at the top of the output folder', () => {
      render({ save_project: null });
      expect(target.querySelector('[data-testid="job-project"]')).toBeNull();
    });

    it('shows the job notices', () => {
      render({ workflow: 'upscale', notices: ['original settings unknown'] });
      expect(log().textContent).toContain('original settings unknown');
    });

    it('shows the names of the files the job starts from, under the prompt, with no pictures', () => {
      render({
        workflow: 'edit',
        inputs: [
          { role: 'source', name: 'lighthouse.png', kind: 'image' },
          { role: 'reference', name: 'storm.jpg', kind: 'image' },
          { role: 'reference', name: 'rain.jpg', kind: 'image' },
        ],
      });

      const lines = Array.from(log().querySelectorAll('[data-testid="job-input"]'));
      expect(lines.map((line) => line.getAttribute('data-role'))).toEqual(['source', 'reference']);
      expect(lines[0].textContent).toContain('Source');
      expect(lines[0].textContent).toContain('lighthouse.png');
      expect(card().querySelector('img')).toBeNull();
      expect(lines[1].textContent).toContain('2 references');
      expect(lines[1].textContent).toContain('storm.jpg, rain.jpg');
    });

    it('shows no input lines for a job with no inputs', () => {
      render({});
      expect(log().querySelector('[data-testid="job-input"]')).toBeNull();
    });
  });

  describe('an upscale', () => {
    const upscale = (overrides: Partial<ActiveJobState> = {}): Partial<ActiveJobState> => ({
      workflow: 'upscale', prompt: '', supported_controls: ['quit'], stageName: 'upscale', inputs: [{ role: 'source', name: 'portrait.png', kind: 'image' }], ...overrides,
    });

    it('leads with its source file, with no position label or empty-prompt line', () => {
      render(upscale());

      expect(log().querySelector('[data-testid="job-input"]')?.textContent).toContain('portrait.png');
      expect(log().querySelector('[data-testid="job-position"]')).toBeNull();
      expect(log().querySelector('[data-testid="job-prompt"]')).toBeNull();
      expect(log().textContent).not.toContain('No prompt recorded');
    });

    it('labels the prompt recorded with its source as the source\'s prompt, after the source', () => {
      render(upscale({ prompt: 'A lighthouse at dusk' }));

      const input = log().querySelector('[data-testid="job-input"]')!;
      const label = log().querySelector('[data-testid="job-position"]')!;
      expect(label.textContent).toBe('Source prompt');
      expect(log().querySelector('[data-testid="job-prompt"]')?.textContent).toBe('A lighthouse at dusk');
      expect(input.compareDocumentPosition(label) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    });
  });

  describe('controls by scope and phase', () => {
    it('puts Pause, Skip and Repeat and then Cancel in one row beside the ring, in two groups', () => {
      render({ runs: 4, supported_controls: IMAGE_CONTROLS });

      const row = target.querySelector('[data-testid="job-controls"]')!;
      expect(row.querySelector('[role="group"][aria-label="Job controls"] [data-control="pause-resume"]')).not.toBeNull();
      expect(row.querySelector('[role="group"][aria-label="Current image"] [data-control="next"]')).not.toBeNull();
      expect(row.querySelector('[role="group"][aria-label="Current image"] [data-control="repeat"]')).not.toBeNull();
      expect(control('quit', row)).not.toBeNull();
      expect(row.querySelector('[data-testid="controls-divider"]')).not.toBeNull();
      expect(control('quit', row)!.compareDocumentPosition(control('next', row)!) & Node.DOCUMENT_POSITION_PRECEDING).toBeTruthy();
    });

    it('gives each button its name and key in the tooltip, and the bare key hint without a chip', () => {
      render({ runs: 4, supported_controls: IMAGE_CONTROLS });

      expect(control('next')!.title).toBe('Skip this image (N)');
      expect(control('pause-resume')!.title).toBe('Pause (P)');
      expect(control('quit')!.title).toBe('Cancel job (Esc)');
      expect(control('repeat')!.querySelector('kbd')?.textContent).toBe('R');
      expect(control('repeat')!.textContent?.trim()).toBe('R');
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
      expect(card().textContent).toContain('No controls for image jobs yet');
    });

    it('warns that skipping during the upscale pass discards the base image too', () => {
      render({
        runs: 2, supported_controls: IMAGE_CONTROLS, stageNames: ['text_to_image', 'upscale'], stageName: 'upscale',
        stageSteps: { text_to_image: 28, upscale: 12 }, currentStep: 5, totalSteps: 12,
      });
      expect(control('next')?.getAttribute('aria-label')).toMatch(/discarded/);
    });
  });

  describe('what a screen reader hears', () => {
    it('keeps one polite status area mounted under the hero, empty until there is a notice', () => {
      const props = reactiveProps({ job: makeJob({ runs: 3, supported_controls: IMAGE_CONTROLS }) });
      component = mountComponent(JobCard, props);
      target = component.target;
      const region = target.querySelector('[data-testid="job-notice-region"]')!;
      expect(region.getAttribute('role')).toBe('status');
      expect(region.getAttribute('aria-live')).toBe('polite');
      expect(region.children).toHaveLength(0);

      props.job = { ...props.job, controlQueued: 'pause' };
      flushSync();
      expect(target.querySelector('[data-testid="job-notice-region"]')).toBe(region);
      expect(region.textContent).toContain('Pausing after this image');
    });

    it('announces a failure through the alert region, and never through the polite one', () => {
      render({ status: 'failed', message: 'Out of memory.' });

      expect(target.querySelector('[data-testid="job-alert-region"]')?.getAttribute('role')).toBe('alert');
      expect(target.querySelector('[data-testid="job-alert-region"] [data-testid="job-notice"]')?.textContent).toContain('Out of memory.');
      expect(target.querySelector('[data-testid="job-notice-region"] [data-testid="job-notice"]')).toBeNull();
    });

    it('announces an error only assertively and a plain notice only politely, so nothing is read twice', () => {
      const props = reactiveProps({ job: makeJob({ runs: 3, supported_controls: IMAGE_CONTROLS, controlQueued: 'pause' }), oncancel: vi.fn() });
      component = mountComponent(JobCard, props);
      target = component.target;
      const polite = target.querySelector('[data-testid="job-notice-region"]')!;
      const assertive = target.querySelector('[data-testid="job-alert-region"]')!;

      expect(polite.textContent).toContain('Pausing after this image');
      expect(assertive.textContent).toBe('');

      document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true }));
      flushSync();
      expect(assertive.textContent).toContain('Press Esc again to cancel');
      expect(polite.textContent).toBe('');
    });

    it('has no live region inside another', () => {
      render({ status: 'failed', message: 'Out of memory.' });

      expect(target.querySelectorAll('[role="status"] [role="alert"], [role="alert"] [role="status"], [aria-live] [role="alert"], [role="alert"] [aria-live]')).toHaveLength(0);
    });

    it('says a pause during model loading waits for the model', () => {
      render({ phase: 'load', controlQueued: 'pause', supported_controls: IMAGE_CONTROLS });
      expect(notice()?.textContent).toContain('Pauses once the model is loaded');
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
        expect(notice()?.textContent).toContain('Skipping after this step');
      });
    });

    it('keeps a queued pause pending until the job pauses', () => {
      const props = reactiveProps({ job: makeJob({ runs: 3, supported_controls: IMAGE_CONTROLS, controlQueued: 'pause' }) });
      component = mountComponent(JobCard, props);
      target = component.target;
      expect(control('pause-resume')!.getAttribute('aria-busy')).toBe('true');
      expect(notice()?.textContent).toContain('Pausing after this image');

      props.job = { ...props.job, controlQueued: null, status: 'paused', paused: true, itemFinished: true };
      flushSync();
      expect(control('pause-resume')!.getAttribute('aria-label')).toBe('Resume');
      expect(notice()).toBeNull();
    });

    it('drops the pending pause the moment the pause event arrives, not when the grace runs out', async () => {
      const props = reactiveProps({ job: makeJob({ runs: 3, supported_controls: IMAGE_CONTROLS }), onpause: vi.fn().mockResolvedValue(undefined) });
      component = mountComponent(JobCard, props);
      target = component.target;
      control('pause-resume')!.click();
      await vi.waitFor(() => {
        flushSync();
        expect(notice()?.textContent).toContain('Pausing after this');
      });

      await new Promise((resolve) => setTimeout(resolve, 0));
      flushSync();
      expect(notice()).not.toBeNull();

      // The server's job_paused settled the queued control; the request's own grace is still running.
      props.job = { ...props.job, controlQueued: null, controlSettledAt: Date.now() + 1000, status: 'paused', paused: true, itemFinished: true };
      flushSync();
      expect(notice()).toBeNull();
    });

    it('drops the pending notice once the job has stopped, and shows the reason when it failed', () => {
      const props = reactiveProps({ job: makeJob({ runs: 2, supported_controls: IMAGE_CONTROLS, controlQueued: 'quit' }) });
      component = mountComponent(JobCard, props);
      target = component.target;
      expect(notice()?.textContent).toContain('Stopping at the next step');

      for (const status of ['cancelled', 'completed'] as const) {
        props.job = { ...props.job, status };
        flushSync();
        expect(notice()).toBeNull();
      }
      props.job = { ...props.job, status: 'failed', message: 'Out of memory.' };
      flushSync();
      expect(notice()?.textContent).toContain('Out of memory.');
    });

    it('shows a rejected control and allows a retry', async () => {
      const onpause = vi.fn().mockRejectedValue(new Error('This job is no longer running.'));
      render({ supported_controls: IMAGE_CONTROLS }, { onpause });
      control('pause-resume')!.click();
      await vi.waitFor(() => {
        flushSync();
        expect(notice()?.textContent).toContain('This job is no longer running.');
        expect(control('pause-resume')!.disabled).toBe(false);
      });
    });

    it('disables every control and drops the estimate while reconnecting', () => {
      render({ runs: 3, remaining: 600, connection: 'reconnecting', lastEventAt: Date.now(), supported_controls: IMAGE_CONTROLS }, { oncancel: vi.fn() });
      expect(card().dataset.tone).toBe('stale');
      expect(Array.from(target.querySelectorAll('button')).every((button) => button.disabled)).toBe(true);
      expect(time().textContent).toContain('Last update');
      expect(time().textContent).not.toContain('left');
      expect(notice()?.textContent).toContain('Lost the live connection');
      expect(target.querySelector('[data-testid="ring-mark"]')?.getAttribute('data-mark')).toBe('stale');
    });
  });

  describe('progress', () => {
    it('is one ring with a stamp for a batch and for a single item alike', () => {
      render({ runs: 3, currentStep: 7, totalSteps: 28 });
      expect(target.querySelectorAll('[data-testid="ring-arc"]')).toHaveLength(3);
      expect(target.querySelectorAll('[role="progressbar"]')).toHaveLength(1);
      flushSync(() => void component!.unmount());

      render({ runs: 1, currentStep: 7, totalSteps: 28 });
      expect(target.querySelectorAll('[data-testid="ring-arc"]')).toHaveLength(1);
      expect(ring().getAttribute('aria-valuenow')).toBe('25');
    });

    it('fills in the step in progress over the time the last step took, and stops when paused', () => {
      const now = vi.spyOn(performance, 'now').mockReturnValue(1000);
      const props = reactiveProps({ job: makeJob({ runs: 2, currentStep: 3, totalSteps: 20 }) });
      component = mountComponent(JobCard, props);
      target = component.target;
      expect(target.querySelector('[data-testid="ring-ghost"]')).toBeNull();

      now.mockReturnValue(2500);
      props.job = { ...props.job, currentStep: 4 };
      flushSync();
      const ghost = target.querySelector<SVGElement>('[data-testid="ring-ghost"]');
      expect(ghost?.style.getPropertyValue('--ms')).toBe('1500ms');
      // The spark rides the ghost's front for as long as the step takes.
      expect(target.querySelector('[data-testid="ring-spark"]')?.hasAttribute('data-riding')).toBe(true);

      props.job = { ...props.job, paused: true, status: 'paused' };
      flushSync();
      expect(target.querySelector('[data-testid="ring-ghost"]')).toBeNull();
      expect(target.querySelector('[data-testid="ring-spark"]')).toBeNull();
      now.mockRestore();
    });

    it('counts elapsed time up every second between updates and holds it while paused', () => {
      vi.useFakeTimers({ toFake: ['setInterval', 'clearInterval', 'performance'] });
      try {
        const props = reactiveProps({ job: makeJob({ elapsed: 10 }) });
        component = mountComponent(JobCard, props);
        target = component.target;
        const elapsed = () => time().textContent;
        expect(elapsed()).toContain('0:10 elapsed');

        vi.advanceTimersByTime(3000);
        flushSync();
        expect(elapsed()).toContain('0:13 elapsed');

        // A server update that lags the local count never moves the clock backwards.
        props.job = { ...props.job, elapsed: 12.5 };
        flushSync();
        vi.advanceTimersByTime(1000);
        flushSync();
        expect(elapsed()).toContain('0:14 elapsed');

        props.job = { ...props.job, paused: true, status: 'paused' };
        flushSync();
        vi.advanceTimersByTime(5000);
        flushSync();
        expect(elapsed()).toContain('0:14 elapsed');
      } finally {
        vi.useRealTimers();
      }
    });

    it('reads the joining stretch of an extension in the caption', () => {
      const joining: Partial<ActiveJobState> = { workflow: 'extend', stageNames: ['extend_video', 'join_clips'], stageName: 'join_clips', stageSteps: { extend_video: 8 }, currentStep: 8, totalSteps: 8 };
      render({ ...joining, runs: 3 });
      expect(caption()?.textContent).toContain('Joining clips…');
    });
  });

  describe('phases and outcomes', () => {
    it('pins a failure to the phase it happened in, and gives the server reason as an alert under the hero', () => {
      render({ status: 'failed', phase: 'load', message: 'Not enough memory to load the model.' });
      expect(phases().map((row) => row.dataset.s)).toEqual(['done', 'failed', 'pending']);
      expect(target.querySelector('[data-testid="job-alert-region"] [data-testid="job-notice"]')?.textContent).toContain('Not enough memory to load the model.');
      expect(card().dataset.tone).toBe('failed');
      expect(target.querySelector('[data-testid="ring-mark"]')?.getAttribute('data-mark')).toBe('failed');
    });

    it('swaps the controls for what the job saved, in the same place, once it ended', () => {
      render({ status: 'cancelled', runs: 4, batchIndex: 2, itemFinished: true, outputs: [makeAsset('a.png'), makeAsset('b.png'), makeAsset('c.png')], supported_controls: IMAGE_CONTROLS });

      expect(target.querySelector('img')).toBeNull();
      expect(target.querySelector('[data-testid="job-saved"]')?.textContent).toMatch(/3 of 4 saved to the Gallery/);
      expect(target.querySelector('[data-testid="job-controls"]')).toBeNull();
      expect(time().textContent).toContain('total');
      expect(card().dataset.tone).toBe('stopped');
      expect(target.querySelector('[data-testid="ring-mark"]')?.getAttribute('data-mark')).toBe('stopped');
      expect(caption()?.textContent).toContain('You stopped the job.');
    });

    it('keeps the prompt under its position label after the end, and says nothing extra for Done', () => {
      render({ status: 'completed', runs: 2, batchIndex: 1, itemFinished: true, prompt: 'A lighthouse', outputs: [makeAsset('a'), makeAsset('b')] });

      expect(log().textContent).toContain('A lighthouse');
      expect(caption()).toBeNull();
      expect(headline()).toBe('Finished 2 images');
      expect(target.querySelector('[data-testid="ring-mark"]')?.getAttribute('data-mark')).toBe('completed');
    });

    it('never shows an image inside the card: the stage holds the live preview', () => {
      render({ previewUrl: '/jobs/job-card/preview?v=2' });
      expect(target.querySelector('img')).toBeNull();
    });

    it('folds the setup phases into one scrolling Ready row once generation runs, and puts an enhancer warning first', () => {
      render({ phase: 'generate', enhanceStatus: 'failed', enhanceTotal: 2 });

      const ready = log().querySelector('[data-phase="ready"]')!;
      expect(ready.querySelector('[data-scrolls]')).not.toBeNull();
      const parts = Array.from(ready.querySelectorAll('[data-scrolls] > *')).map((part) => part.textContent);
      expect(parts[0]).toBe('Ready');
      expect(parts[1]).toBe('enhancer failed, original prompt used');
    });

    it('puts the help line under the download row while a model downloads', () => {
      const download = { role: 'model' as const, model: 'zit', repo: 'org/zit', part: 1, parts: 2, bytes_done: 250, bytes_total: 1000, bytes_per_sec: null, download_eta_secs: null, shared: false };
      render({ phase: 'load', download });

      expect(log().querySelector('[data-testid="download-help"]')?.textContent).toBe('Only the first time; later jobs load the saved files.');
      expect(ring().getAttribute('aria-valuenow')).toBe('25');
      expect(caption()?.textContent).toContain('Part 1 of 2');
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

      it('leaves the job keys to the review', () => {
        const handlers = mountCard({ ...reviewing });
        for (const key of ['p', 'r', 'n', 'y']) press(key);
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
      expect(notice()?.textContent).toContain('Press Esc again to cancel');
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



  describe('a voice design job', () => {
    const design = (overrides: Partial<ActiveJobState> = {}): Partial<ActiveJobState> => ({
      workflow: 'voice_design', media_type: 'audio', prompt: 'A warm woman in her thirties', model_label: 'Qwen3-TTS 1.7B VoiceDesign', currentStep: 0, totalSteps: 0, runs: 3, supported_controls: ['quit'],
      audio: { phase: 'Design', step: 2, total: 3, part: 1, parts: 1 }, ...overrides,
    });

    it('is titled by the voice it is on, with the New voice label and no take', () => {
      render(design());

      expect(headline()).toBe('Voice 2 of 3');
      expect(card().querySelector('[data-testid="job-workflow"]')?.textContent).toBe('New voice');
      expect(card().textContent).not.toMatch(/take/i);
    });

    it('draws one arc per voice and counts the voices made in the caption', () => {
      render(design());

      expect(target.querySelectorAll('[data-testid="ring-arc"]')).toHaveLength(3);
      expect(ring().getAttribute('aria-valuenow')).toBe('66');
      expect(caption()?.textContent).toContain('2 of 3 voices made');
    });

    it('offers Cancel only', () => {
      render(design());

      expect(control('quit')).not.toBeNull();
      expect(control('pause-resume')).toBeNull();
    });

    it('logs Make voices with how many are made, over the description, with no step block', () => {
      render(design());

      expect(phases()[0].textContent).toContain('Ready');
      expect(phases()[1].textContent).toContain('Make voices');
      expect(phases()[1].textContent).toContain('2 of 3');
      expect(log().querySelector('[data-testid="job-position"]')?.textContent).toBe('Description');
      expect(log().querySelector('[data-testid="job-prompt"]')?.textContent).toBe('A warm woman in her thirties');
    });
  });

  describe('an audio job', () => {
    const take = (overrides: Partial<ActiveJobState> = {}): Partial<ActiveJobState> => ({
      workflow: 'music', media_type: 'audio', currentStep: 0, totalSteps: 0, runs: 2, supported_controls: IMAGE_CONTROLS, ...overrides,
    });

    it('shows a song at Aurora\'s estimate: no stamp, the phase in the caption and the others in the raw line', () => {
      render(take({ audio: { phase: 'Song tokens', step: 1875, total: null, part: 1, parts: 1 } }));

      expect(target.querySelector('[data-testid="ring-stamp"]')).toBeNull();
      expect(ring().getAttribute('aria-valuenow')).not.toBeNull();
      expect(caption()?.querySelector('[data-testid="caption-lead"]')?.textContent).toMatch(/^\s*Writing the song/);
      expect(caption()?.querySelector('[data-testid="caption-line"]')?.textContent).toMatch(/Planning ✓/);
    });

    it('shows speech by the take\'s parts read, with the job\'s percentage as the stamp', () => {
      render(take({ workflow: 'speech', audio: { phase: 'Speech', step: 1, total: 1, part: 2, parts: 4 } }));

      expect(target.querySelectorAll('[data-testid="ring-arc"]')).toHaveLength(2);
      expect(Number(target.querySelector('[data-testid="ring-stamp"]')!.textContent!.replace('%', ''))).toBeGreaterThan(0);
      expect(caption()?.textContent).toContain('Reading part 2 of 4');
    });

    it('shows a sound as one arc with its step in the caption', () => {
      render(take({ runs: 1, workflow: 'sound', audio: { phase: 'Sampling', step: 5, total: 8, part: 1, parts: 1 } }));

      expect(target.querySelectorAll('[data-testid="ring-arc"]')).toHaveLength(1);
      expect(caption()?.textContent).toContain('Step 5 of 8');
      expect(target.querySelector('[data-testid="ring-stamp"]')?.textContent).toBe('62%');
    });

    it('shows a song\'s finishing stretch in the caption', () => {
      render(take({ audio: { phase: 'Decode', step: 1, total: 1, part: 1, parts: 1 } }));
      expect(caption()?.querySelector('[data-testid="caption-lead"]')?.textContent).toMatch(/^\s*Finishing/);
      expect(caption()?.querySelector('[data-testid="caption-line"]')?.textContent).toMatch(/Rendering ✓/);
    });

    it('keeps no preview slot for audio', () => {
      render(take());
      expect(card().querySelector('figure')).toBeNull();
    });
  });
});
