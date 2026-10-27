import { flushSync } from 'svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { ActiveJobState } from '$lib/types';

import ReviewStage from './ReviewStage.svelte';
import { mountComponent } from '../../../test-utils/mount';
import { reactiveProps } from '../../../test-utils/reactiveProps.svelte';

describe('ReviewStage', () => {
  let target: HTMLElement;
  let component: ReturnType<typeof mountComponent> | null = null;
  let onapprove: ReturnType<typeof vi.fn>;
  let ondiscard: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    onapprove = vi.fn();
    ondiscard = vi.fn();
  });

  afterEach(() => vi.restoreAllMocks());

  function makeJob(overrides: Partial<ActiveJobState> = {}): ActiveJobState {
    return {
      job_id: 'job-review', workflow: 'txt2img', prompt: 'P', model: 'zit', runs: 8, created_at: '', status: 'paused', paused: true,
      phase: 'generate', currentStep: 8, totalSteps: 8, elapsed: 0, remaining: 0, stageName: 'review', stageIndex: 1, batchLabel: '',
      batchIndex: 2, message: '', outputs: [], previewUrl: '/jobs/job-review/preview?v=4', review: { iteration: 3 },
      supported_controls: ['approve', 'discard', 'quit'], settings: { upscale: '2' }, ...overrides,
    };
  }

  function render(job: ActiveJobState = makeJob(), extra: Record<string, unknown> = {}): void {
    component = mountComponent(ReviewStage, { job, onapprove, ondiscard, ...extra });
    target = component.target;
  }

  const videoJob = (overrides: Partial<ActiveJobState> = {}) => makeJob({
    workflow: 'txt2vid', previewUrl: null, review: { iteration: 2 }, settings: { upscale: '2', audio: ['on', 'false'] }, ...overrides,
  });

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

  it('says what the render upscales to once its size is known', () => {
    render();
    Object.defineProperty(image()!, 'naturalWidth', { configurable: true, value: 832 });
    Object.defineProperty(image()!, 'naturalHeight', { configurable: true, value: 1216 });
    image()!.dispatchEvent(new Event('load'));
    flushSync();
    expect(target.querySelector('.review-line')?.textContent).toContain('832×1216 → 1664×2432');
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

  it('moves focus to Approve when the review appears and nothing has focus', () => {
    (document.activeElement as HTMLElement | null)?.blur();
    render();
    expect(document.activeElement).toBe(button('approve'));
  });

  it('leaves focus on a field being typed in, an open dialog or a menu item, so Enter cannot approve a render nobody has seen', async () => {
    const field = document.createElement('textarea');
    document.body.appendChild(field);
    field.focus();
    render();
    expect(document.activeElement).toBe(field);
    await component!.unmount();
    field.remove();

    document.body.insertAdjacentHTML('beforeend', '<div role="dialog" aria-modal="true"><button id="viewer-close">Close</button></div>');
    document.getElementById('viewer-close')!.focus();
    render();
    expect(document.activeElement).toBe(document.getElementById('viewer-close'));
    await component!.unmount();
    document.getElementById('viewer-close')!.closest('[role="dialog"]')!.remove();

    document.body.insertAdjacentHTML('beforeend', '<div role="menu"><button role="menuitem" id="menu-item">Reuse</button></div>');
    document.getElementById('menu-item')!.focus();
    render();
    expect(document.activeElement).toBe(document.getElementById('menu-item'));
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

  it.each([
    ['an image', () => makeJob(), () => makeJob({ previewUrl: '/jobs/job-review/preview?v=9', review: { iteration: 4 } })],
    ['an image without a preview', () => makeJob({ previewUrl: null }), () => makeJob({ previewUrl: null, review: { iteration: 4 } })],
    ['a video', () => videoJob(), () => videoJob({ review: { iteration: 3 } })],
  ])('allows a decision for the next review of the same job, for %s', (_name, first, next) => {
    const props = reactiveProps({ job: first(), onapprove, ondiscard });
    component = mountComponent(ReviewStage, props);
    target = component.target;
    button('approve')!.click();
    flushSync();
    expect(button('approve')!.disabled).toBe(true);
    props.job = next();
    flushSync();
    expect(button('approve')!.disabled).toBe(false);
  });

  describe('for a video', () => {
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
      size(video()!, 704, 448);
      expect(target.querySelector('.review-line')?.textContent).toContain('704×448 → 1408×896');
    });

    it('shows the size of the next video, not the last one, once it is reviewed', () => {
      const props = reactiveProps({ job: videoJob(), onapprove, ondiscard });
      component = mountComponent(ReviewStage, props);
      target = component.target;
      size(video()!, 704, 448);
      props.job = videoJob({ review: { iteration: 3 } });
      flushSync();
      expect(video()?.getAttribute('src')).toBe('/jobs/job-review/review?v=3');
      expect(target.querySelector('.review-line')?.textContent).not.toContain('704×448');
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
      flushSync(() => void component!.unmount());
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

  });
});
