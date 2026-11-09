import { flushSync } from 'svelte';
import { afterEach, describe, expect, it, vi } from 'vitest';

import type { ActiveJobState } from '$lib/types';
import JobPreview from './JobPreview.svelte';
import { makeAudioAsset } from '../../../test-utils/audio';
import { mountComponent, settle } from '../../../test-utils/mount';
import { reactiveProps } from '../../../test-utils/reactiveProps.svelte';

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

  it('shows the full image of the last output between images, not its thumbnail', () => {
    const output = { id: 'o.png', url: '/media/o.png', thumbnail_url: '/media/o-thumb.png' } as ActiveJobState['outputs'][number];
    const { target } = mountComponent(JobPreview, { job: makeJob({ status: 'paused', paused: true, itemFinished: true, outputs: [output] }) });

    expect(target.querySelector('img')?.getAttribute('src')).toBe('/media/o.png');
  });

  it('shows the still of a video output between takes, not the video file', () => {
    const output = { id: 'o.mp4', url: '/media/o.mp4', thumbnail_url: '/thumbs/o.jpg', media_type: 'video' } as ActiveJobState['outputs'][number];
    const { target } = mountComponent(JobPreview, { job: makeJob({ status: 'paused', paused: true, itemFinished: true, outputs: [output] }) });

    expect(target.querySelector('img')?.getAttribute('src')).toBe('/thumbs/o.jpg');
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

  it('says the model is downloading, not loading, while its files are fetched', () => {
    const download = { role: 'model', model: 'flux', repo: 'org/flux', part: 1, parts: 1, bytes_done: 10, bytes_total: 100, bytes_per_sec: null, download_eta_secs: null, shared: false } as ActiveJobState['download'];
    const loading = mountComponent(JobPreview, { job: makeJob({ phase: 'load' }) });
    const downloading = mountComponent(JobPreview, { job: makeJob({ phase: 'load', download }) });

    expect(loading.target.textContent).toContain('Loading the model');
    expect(downloading.target.textContent).toContain('Downloading the model');
    expect(downloading.target.textContent).not.toContain('Loading the model');
  });

  describe('a voice design job', () => {
    const designJob = (overrides: Partial<ActiveJobState> = {}) => makeJob({
      workflow: 'voice_design', media_type: 'audio', width: null, height: null, runs: 3, audio: { phase: 'Design', step: 2, total: 3, part: 1, parts: 1 }, ...overrides,
    });
    const figure = (target: HTMLElement) => target.querySelector('[data-testid="job-preview"]') as HTMLElement;

    it('is a still waveform tile tagged with where the job is, in the strip shape of a take', () => {
      const { target } = mountComponent(JobPreview, { job: designJob() });

      expect(Number.parseFloat(figure(target).style.aspectRatio)).toBeGreaterThan(1);
      expect(figure(target).querySelector('svg')).not.toBeNull();
      expect(figure(target).querySelector('.job-preview-tag')?.textContent).toBe('Voice 2 of 3');
      expect(figure(target).classList.contains('empty')).toBe(false);
    });

    it('draws no Aurora, whatever the phase', () => {
      for (const overrides of [{}, { audio: null }, { phase: 'load' as const }, { status: 'queued' as const, phase: 'queue' as const }]) {
        const { target } = mountComponent(JobPreview, { job: designJob(overrides) });
        expect(target.querySelector('canvas[data-testid="audio-aurora"]')).toBeNull();
        expect(figure(target).querySelector('img')).toBeNull();
      }
    });

    it('says what the job is doing before any voice is made', () => {
      const { target } = mountComponent(JobPreview, { job: designJob({ audio: null, phase: 'load' }) });

      expect(figure(target).querySelector('.job-preview-tag')?.textContent).toBe('Loading VoiceDesign…');
    });
  });

  describe('an audio job', () => {
    const takeJob = (overrides: Partial<ActiveJobState> = {}) => makeJob({ workflow: 'music', media_type: 'audio', width: null, height: null, ...overrides });
    const figure = (target: HTMLElement) => target.querySelector('[data-testid="job-preview"]') as HTMLElement;

    it('shows no image for an audio job', () => {
      const { target } = mountComponent(JobPreview, { job: takeJob() });

      expect(figure(target).querySelector('img')).toBeNull();
    });

    it('never says there is no live preview while an audio job runs', () => {
      const progress = { audio: { phase: 'Song tokens', step: 250, total: null, part: 1, parts: 1 } } as Partial<ActiveJobState>;
      const noPeaks = { ...makeAudioAsset(), peaks: [] };
      const states: Partial<ActiveJobState>[] = [
        {},
        progress,
        { phase: 'queue', status: 'queued' },
        { phase: 'load' },
        { phase: 'enhance' },
        { audio: null, outputs: [makeAudioAsset()] },
        { audio: null, outputs: [noPeaks] },
        { status: 'paused', paused: true, itemFinished: true, outputs: [noPeaks] },
      ];

      for (const state of states) {
        const { target } = mountComponent(JobPreview, { job: takeJob(state) });

        expect(figure(target).textContent).not.toMatch(/no live preview/i);
      }
    });

    it('shows the waveform of the take that just finished while paused between takes', () => {
      const { target } = mountComponent(JobPreview, { job: takeJob({ status: 'paused', paused: true, itemFinished: true, outputs: [makeAudioAsset()] }) });

      expect(figure(target).querySelectorAll('rect.bar')).toHaveLength(100);
    });

    it('holds the last saved take instead of a frozen Aurora once the job failed or was stopped', () => {
      const running = { audio: { phase: 'Song tokens', step: 900, total: null, part: 1, parts: 1 }, outputs: [makeAudioAsset()] } as Partial<ActiveJobState>;
      const ended: Partial<ActiveJobState>[] = [{ status: 'failed' }, { status: 'cancelled' }];

      for (const state of ended) {
        const { target } = mountComponent(JobPreview, { job: takeJob({ ...running, ...state }) });

        expect(target.querySelector('canvas[data-testid="audio-aurora"]')).toBeNull();
        expect(figure(target).querySelectorAll('rect.bar')).toHaveLength(100);
      }
    });

    it('draws no Aurora for a job that ended with nothing saved', () => {
      const { target } = mountComponent(JobPreview, { job: takeJob({ status: 'failed', audio: { phase: 'Song tokens', step: 900, total: null, part: 1, parts: 1 } }) });

      expect(target.querySelector('canvas[data-testid="audio-aurora"]')).toBeNull();
      expect(figure(target).querySelectorAll('rect.bar')).toHaveLength(0);
    });

    it('gives the same finished take the same caption between takes and as it lands', async () => {
      const take = makeAudioAsset({ duration: 94 });
      const between = mountComponent(JobPreview, { job: takeJob({ audio: null, outputs: [take] }) });
      const landed = mountComponent(JobPreview, { job: takeJob({ audio: { phase: 'Decode', step: 1, total: 1, part: 1, parts: 1 }, itemFinished: true, outputs: [take] }) });
      await settle();

      const caption = (target: HTMLElement) => target.textContent?.match(/\d+:\d{2}/)?.[0];
      expect(caption(between.target)).toBe('1:34');
      expect(caption(landed.target)).toBe('1:34');
      const words = (target: HTMLElement) => figure(target).textContent?.replace(/\s+/g, ' ').trim();
      expect(words(between.target)).toBe(words(landed.target));
    });

    it('plays and pauses the held take from the slot, and plays one take at a time', async () => {
      const play = vi.spyOn(HTMLMediaElement.prototype, 'play').mockImplementation(function (this: HTMLMediaElement) {
        this.dispatchEvent(new Event('play'));
        return Promise.resolve();
      });
      const pause = vi.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(() => undefined);
      const { target } = mountComponent(JobPreview, { job: takeJob({ status: 'paused', paused: true, itemFinished: true, outputs: [makeAudioAsset()] }) });
      const button = () => figure(target).querySelector('.job-preview-play') as HTMLButtonElement;

      expect(figure(target).querySelector('audio')?.getAttribute('src')).toBe('/media/out/song.flac');
      button().click();
      await settle();

      expect(play).toHaveBeenCalledTimes(1);
      expect(button().getAttribute('aria-pressed')).toBe('true');

      figure(target).querySelector('audio')!.dispatchEvent(new Event('pause'));
      await settle();

      expect(button().getAttribute('aria-pressed')).toBe('false');
      pause.mockRestore();
      play.mockRestore();
    });

    it('shows Play, not Pause, once another take is held while one plays', async () => {
      // A real element goes back to paused when its source changes, without a `pause` event.
      const playing = new WeakSet<HTMLMediaElement>();
      vi.spyOn(HTMLMediaElement.prototype, 'play').mockImplementation(function (this: HTMLMediaElement) {
        playing.add(this);
        this.dispatchEvent(new Event('play'));
        return Promise.resolve();
      });
      vi.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(function (this: HTMLMediaElement) {
        if (playing.delete(this)) this.dispatchEvent(new Event('pause'));
      });
      const props = reactiveProps({ job: takeJob({ status: 'paused', paused: true, itemFinished: true, outputs: [makeAudioAsset()] }) });
      const { target } = mountComponent(JobPreview, props);
      const button = () => figure(target).querySelector('.job-preview-play') as HTMLButtonElement;

      button().click();
      await settle();
      expect(button().getAttribute('aria-label')).toBe('Pause the last take');

      props.job = takeJob({ status: 'paused', paused: true, itemFinished: true, outputs: [makeAudioAsset({ id: 'out/other.flac', url: '/media/out/other.flac' })] });
      await settle();

      expect(figure(target).querySelector('audio')?.getAttribute('src')).toBe('/media/out/other.flac');
      expect(button().getAttribute('aria-label')).toBe('Play the last take');
      expect(button().getAttribute('aria-pressed')).toBe('false');
      vi.restoreAllMocks();
    });

    it('plays the take the Aurora has landed on while the job is paused after it', () => {
      const progress = { phase: 'Decode', step: 1, total: 1, part: 1, parts: 1 };
      const paused = mountComponent(JobPreview, { job: takeJob({ audio: progress, status: 'paused', paused: true, itemFinished: true, outputs: [makeAudioAsset()] }) });

      expect(figure(paused.target).querySelectorAll('rect.bar')).toHaveLength(100);
      expect(figure(paused.target).querySelector('audio')?.getAttribute('src')).toBe('/media/out/song.flac');
      expect(figure(paused.target).querySelector('.job-preview-play')).not.toBeNull();
    });

    it('offers no player while the Aurora runs or when the take has no waveform', () => {
      const running = mountComponent(JobPreview, { job: takeJob({ audio: { phase: 'Song tokens', step: 250, total: null, part: 1, parts: 1 } }) });
      const none = mountComponent(JobPreview, { job: takeJob({ status: 'failed' }) });

      expect(running.target.querySelector('.job-preview-play')).toBeNull();
      expect(none.target.querySelector('.job-preview-play')).toBeNull();
    });

    describe('the live preview', () => {
      const running = { audio: { phase: 'Song tokens', step: 250, total: null, part: 1, parts: 1 } } as Partial<ActiveJobState>;
      const ready = { itemFinished: true, outputs: [makeAudioAsset()] } as Partial<ActiveJobState>;
      const canvas = (target: HTMLElement) => target.querySelector('canvas[data-testid="audio-aurora"]');
      const bars = (target: HTMLElement) => target.querySelectorAll('rect.bar').length;

      afterEach(() => {
        vi.useRealTimers();
        vi.unstubAllGlobals();
      });

      const reduceMotion = (reduce: boolean) => vi.stubGlobal('matchMedia', (query: string) => ({ matches: reduce && query.includes('reduce'), addEventListener() {}, removeEventListener() {} }));

      it('shows the canvas from the start of a running job, before any progress event, in every kind', () => {
        for (const workflow of ['music', 'speech', 'sound'] as const) {
          const { target } = mountComponent(JobPreview, { job: takeJob({ workflow }) });

          expect(canvas(target)).not.toBeNull();
          expect(bars(target)).toBe(0);
        }
      });

      it('shows the canvas while the model loads or the prompt is enhanced, but not while the job waits in the queue', () => {
        for (const phase of ['load', 'enhance'] as const) expect(canvas(mountComponent(JobPreview, { job: takeJob({ phase }) }).target)).not.toBeNull();

        expect(canvas(mountComponent(JobPreview, { job: takeJob({ phase: 'queue', status: 'queued' }) }).target)).toBeNull();
      });

      it('keeps the same canvas when the first progress event arrives', async () => {
        const props = reactiveProps({ job: takeJob() });
        const { target } = mountComponent(JobPreview, props);
        const before = canvas(target);

        props.job = takeJob(running);
        await settle();

        expect(before).not.toBeNull();
        expect(canvas(target)).toBe(before);
        expect(bars(target)).toBe(0);
      });

      it('draws a still, with no frame clock, while waiting with reduced motion', async () => {
        vi.useFakeTimers();
        const frames = vi.spyOn(globalThis, 'requestAnimationFrame');
        reduceMotion(true);
        const { target } = mountComponent(JobPreview, { job: takeJob({ phase: 'load' }) });
        await settle(200);

        expect(canvas(target)).not.toBeNull();
        expect(frames).not.toHaveBeenCalled();
      });

      it('replaces the canvas with the waveform of the take when it is ready', async () => {
        vi.useFakeTimers();
        const props = reactiveProps({ job: takeJob(running) });
        const { target } = mountComponent(JobPreview, props);
        await settle(100);
        expect(canvas(target)).not.toBeNull();

        props.job = takeJob({ ...running, ...ready });
        await settle(100);
        await settle(1000);

        expect(canvas(target)).toBeNull();
        expect(bars(target)).toBe(100);
      });

      it('swaps straight to the waveform with reduced motion', async () => {
        reduceMotion(true);
        const props = reactiveProps({ job: takeJob(running) });
        const { target } = mountComponent(JobPreview, props);
        await settle();

        props.job = takeJob({ ...running, ...ready });
        await settle();

        expect(canvas(target)).toBeNull();
        expect(bars(target)).toBe(100);
      });

      it('runs no frame clock with reduced motion, while paused or after the landing', async () => {
        vi.useFakeTimers();
        const frames = vi.spyOn(globalThis, 'requestAnimationFrame');
        reduceMotion(true);
        const reduced = mountComponent(JobPreview, { job: takeJob(running) });
        await settle(200);
        expect(frames).not.toHaveBeenCalled();
        await reduced.unmount();

        reduceMotion(false);
        mountComponent(JobPreview, { job: takeJob({ ...running, status: 'paused', paused: true }) });
        await settle(200);
        expect(frames).not.toHaveBeenCalled();

        const props = reactiveProps({ job: takeJob(running) });
        mountComponent(JobPreview, props);
        await settle(100);
        expect(frames).toHaveBeenCalled();
        props.job = takeJob({ ...running, ...ready });
        await settle(1000);
        frames.mockClear();
        await settle(500);
        expect(frames).not.toHaveBeenCalled();
      });

      it('keeps the waveform of the last take between takes, until the next take reports progress', async () => {
        const props = reactiveProps({ job: takeJob({ ...running, ...ready, audio: null }) });
        const { target } = mountComponent(JobPreview, props);
        await settle();
        expect(bars(target)).toBe(100);

        props.job = takeJob({ ...running, outputs: [makeAudioAsset()], itemFinished: false });
        await settle();

        expect(canvas(target)).not.toBeNull();
        expect(bars(target)).toBe(0);
      });

      it('has no waveform to land on for a take that was skipped, and holds still', async () => {
        vi.useFakeTimers();
        const props = reactiveProps({ job: takeJob({ ...running, outputs: [makeAudioAsset()] }) });
        const { target } = mountComponent(JobPreview, props);
        await settle(100);

        props.job = takeJob({ ...running, outputs: [makeAudioAsset()], itemFinished: true });
        flushSync();
        await settle(1000);

        expect(canvas(target)).not.toBeNull();
        expect(bars(target)).toBe(0);
      });
    });
  });
});
