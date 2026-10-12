// @ts-expect-error Internal Svelte client helpers are the stable mount API in this jsdom test harness.
import { flushSync, mount, unmount } from '../../../../node_modules/svelte/src/index-client.js';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { ActiveJobState } from '$lib/types';

import ReviewStage from './ReviewStage.svelte';
import { reactiveProps } from '../../../test-utils/reactiveProps.svelte';

describe('ReviewStage', () => {
  let target: HTMLDivElement;
  let component: Record<string, unknown> | null = null;
  let onapprove: ReturnType<typeof vi.fn>;
  let ondiscard: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    target = document.createElement('div');
    document.body.appendChild(target);
    onapprove = vi.fn();
    ondiscard = vi.fn();
  });

  afterEach(() => {
    if (component) unmount(component);
    component = null;
    document.body.innerHTML = '';
    vi.restoreAllMocks();
  });

  function makeJob(overrides: Partial<ActiveJobState> = {}): ActiveJobState {
    return {
      job_id: 'job-review', workflow: 'txt2img', prompt: 'P', model: 'zit', runs: 8, created_at: '', status: 'paused', paused: true,
      phase: 'generate', currentStep: 8, totalSteps: 8, elapsed: 0, remaining: 0, stageName: 'review', stageIndex: 1, batchLabel: '',
      batchIndex: 2, message: '', outputs: [], previewUrl: '/jobs/job-review/preview?v=4', review: { iteration: 3 },
      supported_controls: ['approve', 'discard', 'quit'], settings: { upscale: '2' }, ...overrides,
    };
  }

  function render(job: ActiveJobState = makeJob(), extra: Record<string, unknown> = {}): void {
    component = mount(ReviewStage, { target, props: { job, onapprove, ondiscard, ...extra } });
    flushSync();
  }

  const button = (decision: string) => target.querySelector<HTMLButtonElement>(`button[data-decision="${decision}"]`);
  const image = () => target.querySelector<HTMLImageElement>('img');
  function press(key: string, from: EventTarget = document): KeyboardEvent {
    const event = new KeyboardEvent('keydown', { key, bubbles: true, cancelable: true });
    from.dispatchEvent(event);
    flushSync();
    return event;
  }

  it('shows the small render at the preview version the pause bumped', () => {
    render();
    expect(image()?.getAttribute('src')).toBe('/jobs/job-review/preview?v=4');
  });

  it('says which render this is and what it upscales to once the size is known', () => {
    render();
    expect(target.querySelector('.review-line')?.textContent).toBe('Review 3 of 8');
    Object.defineProperty(image()!, 'naturalWidth', { configurable: true, value: 832 });
    Object.defineProperty(image()!, 'naturalHeight', { configurable: true, value: 1216 });
    image()!.dispatchEvent(new Event('load'));
    flushSync();
    expect(target.querySelector('.review-line')?.textContent).toBe('Review 3 of 8 · 832×1216 → 1664×2432');
  });

  it('keeps the decision buttons usable without a preview', () => {
    render(makeJob({ previewUrl: null }));
    expect(image()).toBeNull();
    button('approve')!.click();
    expect(onapprove).toHaveBeenCalledWith('job-review');
  });

  it('offers only the decisions the job supports', () => {
    render(makeJob({ supported_controls: ['quit'] }));
    expect(button('approve')).toBeNull();
    expect(button('discard')).toBeNull();
  });

  it('approves and discards with the buttons, and only once', () => {
    render();
    button('approve')!.click();
    flushSync();
    expect(onapprove).toHaveBeenCalledWith('job-review');
    expect(button('approve')!.disabled).toBe(true);
    expect(button('discard')!.disabled).toBe(true);
    button('discard')!.click();
    press('n');
    expect(ondiscard).not.toHaveBeenCalled();
    expect(onapprove).toHaveBeenCalledTimes(1);
  });

  it.each([
    ['y', 'approve'],
    ['Y', 'approve'],
    ['n', 'discard'],
  ] as const)('decides with %s', (key, action) => {
    render();
    expect(press(key).defaultPrevented).toBe(true);
    expect(action === 'approve' ? onapprove : ondiscard).toHaveBeenCalledWith('job-review');
    expect(action === 'approve' ? ondiscard : onapprove).not.toHaveBeenCalled();
  });

  it('ignores other keys, held keys and keys typed into a field', () => {
    render();
    expect(press('p').defaultPrevented).toBe(false);
    expect(press('r').defaultPrevented).toBe(false);
    const field = document.createElement('textarea');
    document.body.appendChild(field);
    press('y', field);
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'n', ctrlKey: true, bubbles: true, cancelable: true }));
    expect(onapprove).not.toHaveBeenCalled();
    expect(ondiscard).not.toHaveBeenCalled();
  });

  it('is a labelled, non-modal dialog, so page shortcuts keep working', () => {
    render();
    const dialog = target.querySelector('[role="dialog"]')!;
    expect(dialog.getAttribute('aria-label')).toBe('Review');
    expect(dialog.getAttribute('aria-modal')).not.toBe('true');
  });

  it('names the buttons without the key hint', () => {
    render();
    expect(button('approve')!.getAttribute('aria-label')).toBe('Approve');
    expect(button('discard')!.getAttribute('aria-label')).toBe('Discard');
    for (const hint of target.querySelectorAll('kbd')) expect(hint.getAttribute('aria-hidden')).toBe('true');
  });

  it('moves focus to Approve when the review appears, and not out of a field being typed in', () => {
    render();
    expect(document.activeElement).toBe(button('approve'));
    unmount(component);
    component = null;

    const field = document.createElement('textarea');
    document.body.appendChild(field);
    field.focus();
    render();
    expect(document.activeElement).toBe(field);
  });

  it('leaves focus alone while a dialog is open, so Enter cannot approve a render nobody has seen', () => {
    document.body.insertAdjacentHTML('beforeend', '<div role="dialog" aria-modal="true"><button id="viewer-close">Close</button></div>');
    document.getElementById('viewer-close')!.focus();
    render();
    expect(document.activeElement).toBe(document.getElementById('viewer-close'));
    unmount(component);
    component = null;

    document.body.innerHTML = '';
    target = document.createElement('div');
    document.body.appendChild(target);
    document.body.insertAdjacentHTML('beforeend', '<div role="dialog" aria-modal="true"></div>');
    render();
    expect(document.activeElement).not.toBe(button('approve'));
  });

  it('leaves focus on a menu item being navigated', () => {
    document.body.insertAdjacentHTML('beforeend', '<div role="menu"><button role="menuitem" id="menu-item">Reuse</button></div>');
    document.getElementById('menu-item')!.focus();
    render();
    expect(document.activeElement).toBe(document.getElementById('menu-item'));
  });

  it('moves focus to Approve when nothing has it', () => {
    (document.activeElement as HTMLElement | null)?.blur();
    expect(document.activeElement).toBe(document.body);
    render();
    expect(document.activeElement).toBe(button('approve'));
  });

  it('does not trap focus', () => {
    render();
    const outside = document.createElement('button');
    document.body.appendChild(outside);
    outside.focus();
    flushSync();
    expect(document.activeElement).toBe(outside);
  });

  it('offers nothing but the render, its line, Approve and Discard', () => {
    render();
    expect(image()).not.toBeNull();
    expect(target.querySelector('.review-line')?.textContent).toContain('Review 3 of 8');
    const labels = [...target.querySelectorAll('button')].map((element) => element.getAttribute('aria-label'));
    expect(labels).toEqual(['Approve', 'Discard']);
    expect(target.querySelector('.review-notice')).toBeNull();
  });

  it('waits while the connection is down', () => {
    render(makeJob({ connection: 'reconnecting' }));
    expect(button('approve')!.disabled).toBe(true);
    press('y');
    expect(onapprove).not.toHaveBeenCalled();
  });

  it('shows why the server refused a decision and lets the user try again', async () => {
    onapprove.mockRejectedValueOnce(new Error('A decision was already made.'));
    render();
    button('approve')!.click();
    await Promise.resolve();
    await Promise.resolve();
    flushSync();
    expect(target.textContent).toContain('A decision was already made.');
    expect(button('approve')!.disabled).toBe(false);
  });

  it('allows a decision for the next review of the same job', () => {
    const props = reactiveProps({ job: makeJob(), onapprove, ondiscard });
    component = mount(ReviewStage, { target, props });
    flushSync();
    button('approve')!.click();
    flushSync();
    expect(button('approve')!.disabled).toBe(true);
    props.job = makeJob({ previewUrl: '/jobs/job-review/preview?v=9', review: { iteration: 4 } });
    flushSync();
    expect(button('approve')!.disabled).toBe(false);
  });
  it('allows a decision for the next review when neither render has a preview', () => {
    const props = reactiveProps({ job: makeJob({ previewUrl: null }), onapprove, ondiscard });
    component = mount(ReviewStage, { target, props });
    flushSync();
    button('approve')!.click();
    flushSync();
    expect(button('approve')!.disabled).toBe(true);
    props.job = makeJob({ previewUrl: null, review: { iteration: 4 } });
    flushSync();
    expect(button('approve')!.disabled).toBe(false);
  });

  describe('for a video', () => {
    const videoJob = (overrides: Partial<ActiveJobState> = {}) => makeJob({
      workflow: 'txt2vid', previewUrl: null, review: { iteration: 2 }, settings: { upscale: '2', audio: ['on', 'false'] }, ...overrides,
    });
    const video = () => target.querySelector<HTMLVideoElement>('video');
    const playback = () => target.querySelector<HTMLButtonElement>('button[data-playback]');
    const size = (element: HTMLVideoElement, width: number, height: number) => {
      Object.defineProperty(element, 'videoWidth', { configurable: true, value: width });
      Object.defineProperty(element, 'videoHeight', { configurable: true, value: height });
      element.dispatchEvent(new Event('resize'));
      flushSync();
    };

    it('loads the render from the review route, paused on its first frame', () => {
      render(videoJob());
      expect(video()?.getAttribute('src')).toBe('/jobs/job-review/review?v=2');
      expect(video()?.autoplay).toBe(false);
      expect(video()?.paused).toBe(true);
      expect(image()).toBeNull();
    });

    it('keeps the native controls and the sound when Audio is on', () => {
      render(videoJob());
      expect(video()?.controls).toBe(true);
      expect(video()?.muted).toBe(false);
      expect(playback()).toBeNull();
    });

    it.each([
      ['Audio off', { audio: 'false' }],
      ['no Audio setting', {}],
    ])('mutes the video and swaps the native controls for play and pause with %s', (_name, extra) => {
      const play = vi.spyOn(HTMLMediaElement.prototype, 'play').mockResolvedValue();
      const pause = vi.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(() => {});
      try {
        render(videoJob({ settings: { upscale: '2', ...extra } }));
        expect(video()?.muted).toBe(true);
        expect(video()?.controls).toBe(false);
        expect(playback()?.textContent).toContain('Play');

        playback()!.click();
        expect(play).toHaveBeenCalledTimes(1);
        Object.defineProperty(video()!, 'paused', { configurable: true, value: false });
        video()!.dispatchEvent(new Event('play'));
        flushSync();
        expect(playback()?.textContent).toContain('Pause');
        playback()!.click();
        expect(pause).toHaveBeenCalledTimes(1);
      } finally {
        play.mockRestore();
        pause.mockRestore();
      }
    });

    it('reads the video size from the loaded video and doubles it for the upscaled size', () => {
      render(videoJob());
      expect(target.querySelector('.review-line')?.textContent).toBe('Review 2 of 8');
      size(video()!, 704, 448);
      expect(target.querySelector('.review-line')?.textContent).toBe('Review 2 of 8 · 704×448 → 1408×896');
    });

    it('shows the size of the next video, not the last one, once it is reviewed', () => {
      const props = reactiveProps({ job: videoJob(), onapprove, ondiscard });
      component = mount(ReviewStage, { target, props });
      flushSync();
      size(video()!, 704, 448);
      props.job = videoJob({ review: { iteration: 3 } });
      flushSync();
      expect(video()?.getAttribute('src')).toBe('/jobs/job-review/review?v=3');
      expect(target.querySelector('.review-line')?.textContent).toBe('Review 3 of 8');
    });

    it('approves and discards a video with the buttons and with Y and N, and says video', () => {
      render(videoJob());
      expect(button('approve')!.title).toContain('video');
      press('y');
      expect(onapprove).toHaveBeenCalledWith('job-review');
      expect(button('discard')!.disabled).toBe(true);
    });

    it('stops the player and drops its source when the review closes, so the browser stops buffering', () => {
      const pause = vi.spyOn(HTMLMediaElement.prototype, 'pause');
      const load = vi.spyOn(HTMLMediaElement.prototype, 'load');
      render(videoJob());
      const player = video()!;
      expect(player.getAttribute('src')).toBe('/jobs/job-review/review?v=2');
      unmount(component);
      component = null;
      expect(pause).toHaveBeenCalled();
      expect(player.hasAttribute('src')).toBe(false);
      expect(load).toHaveBeenCalled();
    });

    it('keeps the decision buttons usable when the video cannot load', () => {
      render(videoJob());
      video()!.dispatchEvent(new Event('error'));
      flushSync();
      expect(video()).toBeNull();
      expect(target.textContent).toContain('not available to show');
      button('discard')!.click();
      expect(ondiscard).toHaveBeenCalledWith('job-review');
    });

    it('allows a decision for the next video of the same job', () => {
      const props = reactiveProps({ job: videoJob(), onapprove, ondiscard });
      component = mount(ReviewStage, { target, props });
      flushSync();
      button('approve')!.click();
      flushSync();
      expect(button('approve')!.disabled).toBe(true);
      props.job = videoJob({ review: { iteration: 3 } });
      flushSync();
      expect(button('approve')!.disabled).toBe(false);
    });
  });
});
