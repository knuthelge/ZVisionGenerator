import { flushSync } from 'svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { ActionMenuItem } from './ActionMenu.svelte';
import { FRAME_BUSY_LABEL } from './frameMenu';
import MediaPlayer from './MediaPlayer.svelte';
import type { FramePoint } from '$lib/state/framePoint';
import { EXTEND_SHORT_REASON, stepTarget } from '$lib/state/framePoint';
import { mediaVolume } from '$lib/state/mediaVolume.svelte';
import { videoFacts } from '$lib/state/videoFacts.svelte';
import { videoSprite } from '$lib/state/videoSprite.svelte';
import { installFakeMedia, type FakeMedia } from '../../../test-utils/media';
import { mountComponent, settle } from '../../../test-utils/mount';
import { PEAKS } from '../../../test-utils/audio';

// The pointer and the width follow the screen; a test makes it a finger or a phone without a real media query.
const touchLayer = vi.hoisted(() => ({ coarse: false, compact: false }));
vi.mock('$lib/state/layoutMode.svelte', async (importOriginal) => {
  const actual = await importOriginal<typeof import('$lib/state/layoutMode.svelte')>();
  return {
    ...actual,
    layout: new Proxy(actual.layout, {
      get: (target, key) => (key === 'coarse' && touchLayer.coarse ? true : key === 'compact' && touchLayer.compact ? true : Reflect.get(target, key)),
    }),
  };
});

// iPhone and iPad Safari ignore a page's volume; a test says whether this browser does.
const volumeSupport = vi.hoisted(() => ({ works: true }));
vi.mock('./mediaVolumeProbe', async (importOriginal) => ({ ...(await importOriginal<typeof import('./mediaVolumeProbe')>()), canSetVolume: () => volumeSupport.works }));

const browse = vi.hoisted(() => ({ browseInfo: vi.fn() }));
vi.mock('$lib/api/browse', async (importOriginal) => ({ ...(await importOriginal<typeof import('$lib/api/browse')>()), browseInfo: browse.browseInfo }));

// 6 s at 24 fps: 144 frames, the last is frame 143.
const FACTS = { fps: 24, frames: 144, start: 0 };
const LAST = FACTS.frames - 1;
const frameTime = (frame: number): number => stepTarget(frame, FACTS);
const BOX = { left: 100, top: 700, width: 400, height: 46, right: 500, bottom: 746, x: 100, y: 700, toJSON: () => ({}) };
/** The x on the 400 px timeline, from x = 100, that points at *frame*. */
const xOf = (frame: number): number => 100 + (frame / LAST) * 400;

describe('MediaPlayer', () => {
  let media: FakeMedia;
  let target: HTMLElement;
  let points: (FramePoint | null)[];

  beforeEach(() => {
    media = installFakeMedia();
    // jsdom draws nothing; its canvas only logs that it cannot.
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(null);
    videoFacts.reset();
    videoSprite.reset();
    mediaVolume.reset();
    browse.browseInfo.mockReset().mockResolvedValue({ ...FACTS, seconds: 6 });
    points = [];
  });
  afterEach(() => {
    touchLayer.coarse = false;
    touchLayer.compact = false;
    volumeSupport.works = true;
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  const timeline = (): HTMLElement => target.querySelector<HTMLElement>('[data-testid="media-timeline"]')!;
  const element = (): HTMLVideoElement => target.querySelector<HTMLVideoElement>('video')!;
  const audioElement = (): HTMLAudioElement => target.querySelector<HTMLAudioElement>('audio')!;
  const root = (): HTMLElement => target.querySelector<HTMLElement>('[data-media-player]')!;
  const button = (name: string): HTMLButtonElement => target.querySelector<HTMLButtonElement>(`button[aria-label="${name}"]`)!;
  /** The big Play over the middle of the picture. */
  const bigPlay = (): HTMLButtonElement | null => target.querySelector<HTMLButtonElement>('[data-testid="media-big-play"]');
  /** The Play in the row, beside Mute. */
  const rowPlay = (): HTMLButtonElement => target.querySelector<HTMLButtonElement>('[aria-label="Playback controls"] button[aria-label="Play"], [aria-label="Playback controls"] button[aria-label="Pause"]')!;
  const controls = (): HTMLElement | null => target.querySelector<HTMLElement>('[aria-label="Playback controls"]');
  const volumeSlider = (): HTMLInputElement | null => target.querySelector<HTMLInputElement>('input[data-volume]');
  const status = (): string => target.querySelector('[role="status"]')!.textContent ?? '';
  /** The time chip: the current time, then the total. */
  const timeText = (): string => `${target.querySelector('.now')?.textContent} | ${target.querySelector('.total')?.textContent?.replace('/', '').trim()}`;

  /** Mount a video player, wait for its frame facts, and give it its metadata (6 s). */
  async function mountVideo(props: Record<string, unknown> = {}): Promise<HTMLVideoElement> {
    const page = mountComponent(MediaPlayer, { src: '/media/clip.mp4', kind: 'video', size: 'viewer', label: 'clip.mp4', filePath: '/out/clip.mp4', onpoint: (point: FramePoint | null) => points.push(point), ...props });
    target = page.target;
    await settle();
    media.load(element(), 6);
    flushSync();
    timeline().getBoundingClientRect = () => BOX;
    return element();
  }

  async function mountAudio(props: Record<string, unknown> = {}): Promise<HTMLAudioElement> {
    const page = mountComponent(MediaPlayer, { src: '/media/take.flac', kind: 'audio', size: 'viewer', label: 'take.flac', peaks: PEAKS, duration: 100, ...props });
    target = page.target;
    media.load(audioElement(), 100);
    flushSync();
    timeline().getBoundingClientRect = () => BOX;
    return audioElement();
  }

  function pointer(type: string, x: number, y = 720, init: PointerEventInit = {}): void {
    timeline().dispatchEvent(new PointerEvent(type, { clientX: x, clientY: y, pointerType: 'mouse', pointerId: 1, button: 0, bubbles: true, ...init }));
    flushSync();
  }

  function key(name: string, init: KeyboardEventInit = {}, on: Element = timeline()): KeyboardEvent {
    const event = new KeyboardEvent('keydown', { key: name, bubbles: true, cancelable: true, ...init });
    on.dispatchEvent(event);
    flushSync();
    return event;
  }

  /** Let the seek in flight land, as the browser does. */
  const land = (el: HTMLMediaElement = element()): void => { media.land(el); flushSync(); };

  /** Press Play and report that the element started. */
  function play(el: HTMLMediaElement = element()): void {
    button('Play').click();
    flushSync();
    expect(el.paused).toBe(false);
  }

  describe('as a video', () => {
    it('has no native controls, plays inline and never starts by itself', async () => {
      const video = await mountVideo({ poster: '/thumbs/clip.jpg' });

      expect(video.controls).toBe(false);
      expect(video.hasAttribute('playsinline')).toBe(true);
      expect(video.autoplay).toBe(false);
      expect(media.played).toEqual([]);
    });

    it('shows the poster until the video has its own first frame, then the frame itself', async () => {
      const page = mountComponent(MediaPlayer, { src: '/media/clip.mp4', kind: 'video', size: 'viewer', label: 'clip.mp4', poster: '/thumbs/clip.jpg' });
      target = page.target;
      await settle();
      expect(element().getAttribute('poster')).toBe('/thumbs/clip.jpg');

      media.load(element(), 6);
      element().dispatchEvent(new Event('loadeddata'));
      flushSync();

      expect(element().hasAttribute('poster')).toBe(false);
    });

    it('carries the player marker on its root and puts the focus stop on the timeline, which is a slider and never a range input', async () => {
      await mountVideo();

      expect(root().hasAttribute('tabindex')).toBe(false);
      expect(timeline().getAttribute('role')).toBe('slider');
      expect(timeline().tabIndex).toBe(0);
      expect(timeline().tagName).not.toBe('INPUT');
      expect(volumeSlider()?.type).toBe('range');
    });

    it('names the timeline and reads its value as seconds', async () => {
      await mountVideo();

      expect(timeline().getAttribute('aria-label')).toBe('Playback position');
      expect(timeline().getAttribute('aria-valuemin')).toBe('0');
      expect(timeline().getAttribute('aria-valuemax')).toBe('6');
      expect(timeline().getAttribute('aria-valuetext')).toBe('0.0 s');
      expect(timeText()).toBe('0.0 s | 6.0 s');
    });

    it('loads only its metadata at first, on the stage and in the pane too', async () => {
      expect((await mountVideo({ size: 'stage' })).getAttribute('preload')).toBe('metadata');
      expect((await mountVideo({ size: 'pane' })).getAttribute('preload')).toBe('metadata');
    });

    it('asks for the frame facts of the file once', async () => {
      await mountVideo();

      expect(browse.browseInfo).toHaveBeenCalledWith('source_video', '/out/clip.mp4');
      expect(browse.browseInfo).toHaveBeenCalledTimes(1);
    });
  });

  describe('the timeline', () => {
    it('seeks to the middle of the frame under the pointer, and never changes playing or paused', async () => {
      const video = await mountVideo();

      pointer('pointerdown', xOf(72));
      pointer('pointerup', xOf(72));

      expect(media.seeks(video)).toEqual([frameTime(72)]);
      expect(video.paused).toBe(true);
      expect(media.played).toEqual([]);
    });

    it('shows the picture live while dragging, one seek in flight, the latest target next', async () => {
      const video = await mountVideo();

      pointer('pointerdown', xOf(10));
      pointer('pointermove', xOf(40));
      pointer('pointermove', xOf(80));
      pointer('pointermove', xOf(120));
      // Only the first seek was asked of the browser while it was still busy.
      expect(media.seeks(video)).toEqual([frameTime(10)]);

      land();
      expect(media.seeks(video)).toEqual([frameTime(10), frameTime(120)]);
      land();
      pointer('pointerup', xOf(120));
      expect(media.seeks(video)).toHaveLength(2);
    });

    it('never uses fastSeek', async () => {
      const video = await mountVideo();
      const fastSeek = vi.fn();
      (video as unknown as { fastSeek: () => void }).fastSeek = fastSeek;

      pointer('pointerdown', xOf(50));
      pointer('pointermove', xOf(60));

      expect(fastSeek).not.toHaveBeenCalled();
    });

    it('moves the playhead, the time and the slider to the target before the seek lands', async () => {
      await mountVideo();

      pointer('pointerdown', xOf(72));

      expect(element().seeking).toBe(true);
      expect(timeline().getAttribute('aria-valuetext')).toBe('3.0 s');
      expect(target.querySelector('.now')?.textContent).toBe('3.0 s');
    });

    it('pauses a playing video for the drag and plays it again on release, which never shows as a pause', async () => {
      const video = await mountVideo();
      play();
      expect(status()).toBe('Playing');

      pointer('pointerdown', xOf(30));
      expect(video.paused).toBe(true);
      expect(button('Pause')).not.toBeNull();
      expect(status()).toBe('Playing');

      pointer('pointerup', xOf(30));
      expect(video.paused).toBe(false);
      expect(media.played).toHaveLength(2);
    });

    it('lets a drag that starts on a paused video leave it paused', async () => {
      const video = await mountVideo();

      pointer('pointerdown', xOf(30));
      pointer('pointerup', xOf(30));

      expect(video.paused).toBe(true);
      expect(media.played).toEqual([]);
    });

    it('slows the scrub when the pointer slides up more than 48 px, with a label, and takes it away when it comes back', async () => {
      const video = await mountVideo();

      pointer('pointerdown', xOf(72), 720);
      expect(target.querySelector('.fine')).toBeNull();
      pointer('pointermove', xOf(72), 720 - 49);
      expect(target.querySelector('.fine')?.textContent).toBe('Fine scrub: one frame at a time');

      // 40 px right at an eighth of the speed is about a frame.
      land();
      pointer('pointermove', xOf(72) + 40, 720 - 49);
      land();
      const slowed = media.seeks(video).at(-1)!;
      expect(slowed - frameTime(72)).toBeGreaterThan(0);
      expect(slowed - frameTime(72)).toBeLessThan(4 / FACTS.fps);

      pointer('pointermove', xOf(72) + 40, 720 - 10);
      expect(target.querySelector('.fine')).toBeNull();
    });

    it('is not grabbable until the metadata is in', async () => {
      const page = mountComponent(MediaPlayer, { src: '/media/clip.mp4', kind: 'video', size: 'viewer', label: 'clip.mp4', filePath: '/out/clip.mp4' });
      target = page.target;
      await settle();
      timeline().getBoundingClientRect = () => BOX;

      expect(timeline().getAttribute('aria-disabled')).toBe('true');
      pointer('pointerdown', xOf(50));

      expect(media.seeks(element())).toEqual([]);
    });

    it('ignores a right click', async () => {
      const video = await mountVideo();

      pointer('pointerdown', xOf(50), 720, { button: 2 });

      expect(media.seeks(video)).toEqual([]);
    });
  });

  describe('keys', () => {
    it('steps one frame while paused, five with Shift, from the shown frame, in the middle of it', async () => {
      const video = await mountVideo();

      key('ArrowRight');
      expect(media.seeks(video).at(-1)).toBe(frameTime(1));
      land();
      key('ArrowRight', { shiftKey: true });
      expect(media.seeks(video).at(-1)).toBe(frameTime(6));
      land();
      key('ArrowLeft');
      expect(media.seeks(video).at(-1)).toBe(frameTime(5));
    });

    it('steps from the frame it is going to, so a held key keeps moving', async () => {
      const video = await mountVideo();

      key('ArrowRight');
      key('ArrowRight');
      key('ArrowRight');
      land();
      land();

      expect(media.seeks(video)).toEqual([frameTime(1), frameTime(3)]);
    });

    it('does not step before the first frame or past the last', async () => {
      const video = await mountVideo();

      key('ArrowLeft');
      land();
      expect(media.seeks(video).at(-1)).toBe(frameTime(0));

      key('End');
      land();
      key('ArrowRight');
      expect(media.seeks(video).at(-1)).toBe(frameTime(LAST));
    });

    it('jumps to a tenth of the clip with a digit, on the exact frame, and 0 goes to the first frame', async () => {
      const video = await mountVideo();

      expect(key('5').defaultPrevented).toBe(true);
      land();
      // Frame round(0.5 * 143) of 144.
      expect(media.seeks(video).at(-1)).toBe(frameTime(72));
      key('9');
      land();
      expect(media.seeks(video).at(-1)).toBe(frameTime(Math.round(0.9 * LAST)));
      key('0');
      land();
      expect(media.seeks(video).at(-1)).toBe(frameTime(0));
    });

    it('jumps while playing without pausing, and shows the frame menu for the point once paused', async () => {
      const video = await mountVideo({ actions: [{ kind: 'item', id: 'extend', label: 'Extend', onselect: vi.fn() }] });
      play(video);

      key('3');
      land();
      expect(video.paused).toBe(false);

      video.pause();
      video.dispatchEvent(new Event('pause'));
      key('6');
      land();
      expect(target.querySelector('[data-testid="frame-menu"]')?.getAttribute('aria-label')).toBe(`Use this frame, ${(Math.round(0.6 * LAST) / FACTS.fps).toFixed(1)} s`);
    });

    it('leaves a digit typed in a field to the field', async () => {
      const video = await mountVideo();
      const field = document.createElement('input');
      field.type = 'text';
      root().appendChild(field);
      const seeks = (): number => media.seeks(video).length;
      const before = seeks();

      expect(key('5', {}, field).defaultPrevented).toBe(false);

      expect(seeks()).toBe(before);
    });

    it('does not take a digit with ⌘, Ctrl or Alt', async () => {
      const video = await mountVideo();
      const before = media.seeks(video).length;

      for (const modifier of ['altKey', 'ctrlKey', 'metaKey']) expect(key('5', { [modifier]: true }).defaultPrevented).toBe(false);

      expect(media.seeks(video)).toHaveLength(before);
    });

    it('seeks one second while playing, and five with Shift', async () => {
      const video = await mountVideo();
      play();
      media.tick(video, 2);
      flushSync();

      key('ArrowRight');
      expect(media.seeks(video).at(-1)).toBe(3);
      land();
      key('ArrowLeft', { shiftKey: true });
      expect(media.seeks(video).at(-1)).toBe(0);
    });

    it('goes to the first frame with Home and the last with End', async () => {
      const video = await mountVideo();

      key('End');
      expect(media.seeks(video).at(-1)).toBe(frameTime(LAST));
      land();
      key('Home');
      expect(media.seeks(video).at(-1)).toBe(frameTime(0));
    });

    it('plays and pauses with Space, and handles the key', async () => {
      const video = await mountVideo();

      expect(key(' ').defaultPrevented).toBe(true);
      expect(video.paused).toBe(false);
      key(' ');
      expect(video.paused).toBe(true);
    });

    it('leaves Space to a focused button, which activates itself', async () => {
      const video = await mountVideo();

      const event = key(' ', {}, button('Mute'));

      expect(event.defaultPrevented).toBe(false);
      expect(video.paused).toBe(true);
    });

    it('sets the volume with the arrows up and down, and unmutes', async () => {
      const video = await mountVideo();
      key('ArrowDown');
      key('ArrowDown');
      expect(video.volume).toBeCloseTo(0.8);
      expect(mediaVolume.volume).toBeCloseTo(0.8);

      button('Mute').click();
      flushSync();
      expect(video.muted).toBe(true);
      key('ArrowUp');
      expect(video.muted).toBe(false);
      expect(video.volume).toBeCloseTo(0.9);
    });

    it('moves the volume, not the clip, with left and right on the volume slider', async () => {
      const video = await mountVideo();
      const slider = target.querySelector<HTMLElement>('[data-volume]')!;

      key('ArrowLeft', {}, slider);

      expect(video.volume).toBeCloseTo(0.9);
      expect(media.seeks(video)).toEqual([]);
      expect(slider.getAttribute('aria-valuetext')).toBe('90%');
    });

    it('takes no key it does not handle, so the page still gets them', async () => {
      await mountVideo();

      for (const name of ['e', 'k', 'm', 'f', 'x', 'i', 'PageUp', 'PageDown', 'Escape', 'Delete']) {
        expect(key(name).defaultPrevented).toBe(false);
      }
      expect(key('ArrowRight', { altKey: true }).defaultPrevented).toBe(false);
    });

    it('does not move before the metadata is in', async () => {
      const page = mountComponent(MediaPlayer, { src: '/media/clip.mp4', kind: 'video', size: 'viewer', label: 'clip.mp4', filePath: '/out/clip.mp4' });
      target = page.target;
      await settle();

      key('ArrowRight');
      key('End');

      expect(media.seeks(element())).toEqual([]);
    });
  });

  describe('the selection', () => {
    it('is none at the first frame and while playing, and the frame once paused after it', async () => {
      const video = await mountVideo();
      expect(points.at(-1) ?? null).toBeNull();

      pointer('pointerdown', xOf(72));
      pointer('pointerup', xOf(72));
      expect(points.at(-1)).toMatchObject({ seconds: 3, atEnd: false, tooShort: false });

      play(video);
      expect(points.at(-1)).toBeNull();
    });

    it('follows the target while a seek is in flight', async () => {
      await mountVideo();

      pointer('pointerdown', xOf(10));
      expect(element().seeking).toBe(true);

      expect(points.at(-1)).toMatchObject({ seconds: 10 / 24, tooShort: true });
    });

    it('is the last frame, for Use as only, when playback ends', async () => {
      const video = await mountVideo();
      play(video);

      media.finish(video);
      flushSync();

      expect(points.at(-1)).toMatchObject({ seconds: LAST / 24, atEnd: true, tooShort: false });
    });

    it('is the last frame when a seek lands on it, and clears when the player goes', async () => {
      const page = mountComponent(MediaPlayer, { src: '/media/clip.mp4', kind: 'video', size: 'viewer', label: 'clip.mp4', filePath: '/out/clip.mp4', onpoint: (point: FramePoint | null) => points.push(point) });
      target = page.target;
      await settle();
      media.load(element(), 6);
      flushSync();
      key('End');
      land();
      expect(points.at(-1)).toMatchObject({ seconds: LAST / 24, atEnd: true });

      await page.unmount();

      expect(points.at(-1)).toBeNull();
    });

    it('is none without frame facts, and for a media error', async () => {
      browse.browseInfo.mockRejectedValue(new Error('no facts'));
      const video = await mountVideo();
      key('ArrowRight');
      land();
      expect(points.at(-1) ?? null).toBeNull();

      video.dispatchEvent(new Event('error'));
      flushSync();
      expect(points.at(-1) ?? null).toBeNull();
    });

    describe('with the frame the browser reports', () => {
      beforeEach(() => { media = installFakeMedia({ frameCallbacks: true }); });

      /** Play to *shown* with the clock *ahead* frames beyond it, then pause: the clock is ahead of the picture, as a decoder leaves it. */
      async function pausedAhead(shown: number, ahead: number): Promise<HTMLVideoElement> {
        const video = await mountVideo();
        play(video);
        media.present(video, shown / FACTS.fps);
        media.tick(video, (shown + ahead + 0.5) / FACTS.fps);
        flushSync();
        video.pause();
        flushSync();
        return video;
      }

      it('is the frame on screen after a pause, not the one the clock reached', async () => {
        await pausedAhead(88, 2);

        expect(points.at(-1)).toMatchObject({ seconds: 88 / 24 });
        expect(points.at(-1)?.time).toBeCloseTo(88 / FACTS.fps, 6);
        expect(target.querySelector('.now')?.textContent).toBe('3.7 s');
      });

      it('follows the picture while it plays, and clears the selection', async () => {
        const video = await mountVideo();
        play(video);

        media.present(video, 40 / FACTS.fps);
        media.tick(video, 42.5 / FACTS.fps);
        flushSync();

        expect(points.at(-1)).toBeNull();
        expect(target.querySelector('.now')?.textContent).toBe('1.7 s');
      });

      it('steps one frame from the picture, not from the clock', async () => {
        const video = await pausedAhead(88, 2);

        key('ArrowRight');

        expect(media.seeks(video).at(-1)).toBe(frameTime(89));
        land(video);
        expect(points.at(-1)).toMatchObject({ seconds: 89 / 24 });
      });

      it('ignores a picture that is shown on the way to a seek', async () => {
        const video = await pausedAhead(88, 2);

        key('ArrowRight');
        media.present(video, 88 / FACTS.fps);
        land(video);

        expect(points.at(-1)).toMatchObject({ seconds: 89 / 24 });
      });

      it('lands on the seek target until the picture of the seek is reported, then on that picture', async () => {
        const video = await pausedAhead(88, 2);

        key('ArrowRight');
        land(video);
        media.present(video, 89 / FACTS.fps);
        flushSync();

        expect(points.at(-1)).toMatchObject({ seconds: 89 / 24 });
        expect(points.at(-1)?.time).toBeCloseTo(89 / FACTS.fps, 6);
      });
    });

    it('is the frame the clock reached when the browser reports no pictures', async () => {
      const video = await mountVideo();
      play(video);
      media.tick(video, 90.5 / FACTS.fps);
      flushSync();

      video.pause();
      flushSync();

      expect(points.at(-1)).toMatchObject({ seconds: 90 / 24 });
    });

    it('is never a point for audio', async () => {
      const take = await mountAudio({ onpoint: (point: FramePoint | null) => points.push(point) });

      key('ArrowRight');
      land(take);

      expect(points.filter((point) => point !== null)).toEqual([]);
    });
  });

  describe('when the file cannot be played', () => {
    it('shows an alert in place of the time and the controls, and keeps the rest line', async () => {
      const video = await mountVideo();

      video.dispatchEvent(new Event('error'));
      flushSync();

      expect(target.querySelector('[role="alert"]')?.textContent).toContain('This file can’t be played.');
      expect(target.querySelector('[role="alert"]')?.textContent).not.toContain('browser');
      expect(controls()).toBeNull();
      expect(bigPlay()).toBeNull();
    });

    it('leaves no timeline behind it', async () => {
      const video = await mountVideo();

      video.dispatchEvent(new Event('error'));
      flushSync();

      expect(target.querySelector('[data-testid="media-timeline"]')).toBeNull();
    });
  });

  describe('the chrome', () => {
    function shown(): string | null {
      return root().getAttribute('data-chrome');
    }

    it('stays while paused, and fades after 2.6 s of quiet while playing', async () => {
      const video = await mountVideo();
      vi.useFakeTimers();
      expect(shown()).toBe('shown');

      play(video);
      await vi.advanceTimersByTimeAsync(2599);
      expect(shown()).toBe('shown');
      await vi.advanceTimersByTimeAsync(2);
      flushSync();

      expect(shown()).toBe('hidden');
    });

    it('comes back on a mouse move, and the countdown starts again', async () => {
      const video = await mountVideo();
      vi.useFakeTimers();
      play(video);
      await vi.advanceTimersByTimeAsync(3000);
      flushSync();
      expect(shown()).toBe('hidden');

      root().dispatchEvent(new PointerEvent('pointermove', { clientX: 300, clientY: 300, pointerType: 'mouse', bubbles: true }));
      flushSync();
      expect(shown()).toBe('shown');
      await vi.advanceTimersByTimeAsync(2700);
      flushSync();
      expect(shown()).toBe('hidden');
    });

    it('comes back on a key the player handles, and when the video is paused', async () => {
      const video = await mountVideo();
      vi.useFakeTimers();
      play(video);
      await vi.advanceTimersByTimeAsync(3000);
      flushSync();

      key('ArrowUp');
      expect(shown()).toBe('shown');
      await vi.advanceTimersByTimeAsync(3000);
      flushSync();
      key(' ');
      expect(shown()).toBe('shown');
      await vi.advanceTimersByTimeAsync(10_000);
      expect(shown()).toBe('shown');
    });

    it('does not fade while a scrub is under way', async () => {
      const video = await mountVideo();
      vi.useFakeTimers();
      play(video);

      pointer('pointerdown', xOf(30));
      await vi.advanceTimersByTimeAsync(5000);
      flushSync();

      expect(shown()).toBe('shown');
    });

    it('stays on a tap or a click on the picture, which plays and pauses instead of hiding it', async () => {
      const video = await mountVideo();

      root().dispatchEvent(new PointerEvent('pointerdown', { pointerType: 'touch', bubbles: true }));
      video.click();
      flushSync();
      expect(video.paused).toBe(false);
      expect(shown()).toBe('shown');
      video.click();
      flushSync();
      expect(video.paused).toBe(true);
      expect(shown()).toBe('shown');

      root().dispatchEvent(new PointerEvent('pointerdown', { pointerType: 'mouse', bubbles: true }));
      video.click();
      flushSync();
      expect(video.paused).toBe(false);
    });

    it('never fades in the pane', async () => {
      const video = await mountVideo({ size: 'pane' });
      vi.useFakeTimers();

      play(video);
      await vi.advanceTimersByTimeAsync(10_000);

      expect(shown()).toBe('shown');
    });
  });

  describe('the loupe', () => {
    class FakeImage {
      naturalWidth = 1760;
      naturalHeight = 990;
      onload: (() => void) | null = null;
      onerror: (() => void) | null = null;
      set src(_value: string) { queueMicrotask(() => this.onload?.()); }
    }

    it('shows the time at the pointer on a hover, and the sprite tile once the sheet has loaded', async () => {
      vi.stubGlobal('Image', FakeImage);
      await mountVideo();
      root().getBoundingClientRect = () => ({ ...BOX, left: 0, width: 1000, top: 0, height: 760, right: 1000, bottom: 760 });

      pointer('pointermove', xOf(72));
      expect(target.querySelector('[data-testid="media-loupe"] .loupe-time')?.textContent).toBe('3.0 s');
      expect(target.querySelector('.loupe-picture')).toBeNull();

      await settle();
      const picture = target.querySelector<HTMLElement>('.loupe-picture')!;
      expect(picture.style.backgroundImage).toContain('/api/browse/sprite');
      // Tile 72 of 144 frames at 121 tiles is round(72 / 143 * 120) = 60: column 5, row 5.
      expect(picture.style.backgroundPosition).toContain(`${-5 * 168}px`);
      vi.unstubAllGlobals();
    });

    it('asks for a contact sheet that failed to build once per player, not on every hover', async () => {
      const made: unknown[] = [];
      class FailingImage {
        onload: (() => void) | null = null;
        onerror: (() => void) | null = null;
        constructor() { made.push(this); }
        set src(_value: string) { queueMicrotask(() => this.onerror?.()); }
      }
      vi.stubGlobal('Image', FailingImage);
      await mountVideo();
      root().getBoundingClientRect = () => ({ ...BOX, left: 0, width: 1000, top: 0, height: 760, right: 1000, bottom: 760 });

      for (let hover = 0; hover < 3; hover++) {
        pointer('pointermove', xOf(72));
        await settle();
        pointer('pointerleave', xOf(72));
        await settle();
      }
      pointer('pointerdown', xOf(60));
      pointer('pointerup', xOf(60));
      await settle();

      expect(made).toHaveLength(1);
      // The next time the clip is opened asks again.
      await mountVideo();
      root().getBoundingClientRect = () => ({ ...BOX, left: 0, width: 1000, top: 0, height: 760, right: 1000, bottom: 760 });
      pointer('pointermove', xOf(72));
      await settle();
      expect(made).toHaveLength(2);
      vi.unstubAllGlobals();
    });

    it('goes when the pointer leaves, and is hidden while an ordinary scrub is under way', async () => {
      await mountVideo();

      pointer('pointermove', xOf(50));
      expect(target.querySelector('[data-testid="media-loupe"]')).not.toBeNull();
      pointer('pointerleave', xOf(50));
      expect(target.querySelector('[data-testid="media-loupe"]')).toBeNull();

      pointer('pointermove', xOf(50));
      pointer('pointerdown', xOf(50));
      expect(target.querySelector('[data-testid="media-loupe"]')).toBeNull();
    });

    it('has none in the pane, and none for a take', async () => {
      await mountVideo({ size: 'pane' });
      pointer('pointermove', xOf(50));
      expect(target.querySelector('[data-testid="media-loupe"]')).toBeNull();

      await mountAudio();
      pointer('pointermove', 300);
      expect(target.querySelector('[data-testid="media-loupe"]')).toBeNull();
    });

    describe('in slow mode', () => {
      it('shows after a seek has been in flight for 150 ms during a scrub, and stays until the scrub ends and the picture has landed', async () => {
        await mountVideo();
        vi.useFakeTimers();

        pointer('pointerdown', xOf(60));
        await vi.advanceTimersByTimeAsync(149);
        expect(target.querySelector('[data-testid="media-loupe"]')).toBeNull();
        await vi.advanceTimersByTimeAsync(2);
        flushSync();
        expect(target.querySelector('[data-testid="media-loupe"]')).not.toBeNull();

        // Landing does not take it away while the pointer is still down: no flicker within one gesture.
        land();
        expect(target.querySelector('[data-testid="media-loupe"]')).not.toBeNull();

        pointer('pointerup', xOf(60));
        pointer('pointerleave', xOf(60));
        expect(target.querySelector('[data-testid="media-loupe"]')).toBeNull();
      });

      it('waits for the picture to land after the scrub has ended', async () => {
        await mountVideo();
        vi.useFakeTimers();

        pointer('pointerdown', xOf(60));
        await vi.advanceTimersByTimeAsync(200);
        flushSync();
        pointer('pointerup', xOf(60));
        pointer('pointerleave', xOf(60));
        expect(target.querySelector('[data-testid="media-loupe"]')).not.toBeNull();

        land();
        expect(target.querySelector('[data-testid="media-loupe"]')).toBeNull();
      });

      it('does not show for a seek that lands quickly', async () => {
        await mountVideo();
        vi.useFakeTimers();

        pointer('pointerdown', xOf(60));
        await vi.advanceTimersByTimeAsync(100);
        land();
        await vi.advanceTimersByTimeAsync(500);
        flushSync();

        expect(target.querySelector('[data-testid="media-loupe"]')).toBeNull();
      });

      it('shows for a held arrow key too', async () => {
        await mountVideo();
        vi.useFakeTimers();

        key('ArrowRight');
        await vi.advanceTimersByTimeAsync(200);
        flushSync();

        expect(target.querySelector('[data-testid="media-loupe"]')?.querySelector('.loupe-time')?.textContent).toBe('0.0 s');
        timeline().dispatchEvent(new KeyboardEvent('keyup', { key: 'ArrowRight', bubbles: true }));
        land();
        expect(target.querySelector('[data-testid="media-loupe"]')).toBeNull();
      });
    });
  });

  describe('the loading and stalled states', () => {
    it('shows a spinner in place of the centre glyph after 400 ms with no metadata', async () => {
      vi.useFakeTimers();
      const page = mountComponent(MediaPlayer, { src: '/media/clip.mp4', kind: 'video', size: 'viewer', label: 'clip.mp4', filePath: '/out/clip.mp4' });
      target = page.target;
      await vi.advanceTimersByTimeAsync(399);
      expect(target.querySelector('.centre .spinner')).toBeNull();

      await vi.advanceTimersByTimeAsync(2);
      flushSync();

      expect(target.querySelector('.centre .spinner')).not.toBeNull();
    });

    it('shows the spinner when playback stalls for 400 ms, and takes it away when it resumes', async () => {
      const video = await mountVideo();
      play(video);
      vi.useFakeTimers();

      video.dispatchEvent(new Event('waiting'));
      await vi.advanceTimersByTimeAsync(401);
      flushSync();
      expect(target.querySelector('.centre .spinner')).not.toBeNull();

      video.dispatchEvent(new Event('playing'));
      flushSync();
      expect(target.querySelector('.centre .spinner')).toBeNull();
    });
  });

  describe('the centre button and flash', () => {
    it('shows whenever the clip is paused, wherever it is, and goes while it plays', async () => {
      const video = await mountVideo();
      expect(bigPlay()).not.toBeNull();

      pointer('pointerdown', xOf(72));
      pointer('pointerup', xOf(72));
      land();
      expect(bigPlay()).not.toBeNull();

      play(video);
      expect(bigPlay()).toBeNull();

      video.pause();
      video.dispatchEvent(new Event('pause'));
      key('End');
      land();
      expect(bigPlay()).not.toBeNull();
      expect(media.seeks(video).at(-1)).toBe(frameTime(LAST));
    });

    it('shows at every size of the player, and on a finger as on a mouse', async () => {
      vi.spyOn(Element.prototype, 'clientWidth', 'get').mockImplementation(function (this: Element) { return this.classList.contains('player') ? 300 : 0; });
      vi.spyOn(Element.prototype, 'clientHeight', 'get').mockImplementation(function (this: Element) { return this.classList.contains('player') ? 90 : 0; });
      touchLayer.coarse = true;
      await mountVideo({ size: 'stage' });

      expect(bigPlay()).not.toBeNull();
    });

    it('is the middle of the picture, whatever the controls take', async () => {
      vi.spyOn(Element.prototype, 'clientWidth', 'get').mockImplementation(function (this: Element) { return this.classList.contains('player') ? 800 : 0; });
      vi.spyOn(Element.prototype, 'clientHeight', 'get').mockImplementation(function (this: Element) { return this.classList.contains('player') ? 600 : 0; });
      await mountVideo({ naturalSize: { width: 800, height: 450 } });

      expect(root().style.getPropertyValue('--centre-top')).toBe('300px');
    });

    it('plays again from the start at the end', async () => {
      const video = await mountVideo();
      key('End');
      land();

      bigPlay()!.click();
      flushSync();

      expect(media.seeks(video).at(-1)).toBe(frameTime(0));
      expect(video.paused).toBe(false);
    });

    it('flashes the new state on each play and pause, and not at rest', async () => {
      await mountVideo();
      expect(target.querySelector('.flash')).toBeNull();

      play();
      expect(target.querySelector('.flash')).not.toBeNull();
    });

    it('does not flash under reduced motion', async () => {
      vi.stubGlobal('matchMedia', (query: string) => ({ matches: query.includes('reduce'), addEventListener: vi.fn(), removeEventListener: vi.fn() }));
      await mountVideo();

      play();

      expect(target.querySelector('.flash')).toBeNull();
      vi.unstubAllGlobals();
    });
  });

  describe('the announcements', () => {
    it('says Playing and where it paused, and nothing while it plays or scrubs', async () => {
      const video = await mountVideo();
      expect(status()).toBe('');

      play(video);
      expect(status()).toBe('Playing');
      media.tick(video, 3.0208);
      flushSync();
      expect(status()).toBe('Playing');

      button('Pause').click();
      flushSync();
      expect(status()).toBe('Paused at 3.0 s');
    });
  });

  describe('volume and mute', () => {
    it('starts muted when asked, as the stage does, and the mute button says so', async () => {
      const video = await mountVideo({ size: 'stage', muted: true });

      expect(video.muted).toBe(true);
      expect(button('Unmute')).not.toBeNull();
    });

    it('carries the volume and mute of one player to the next', async () => {
      await mountVideo();
      key('ArrowDown');
      button('Mute').click();
      flushSync();

      const next = await mountVideo();

      expect(next.volume).toBeCloseTo(0.9);
      expect(next.muted).toBe(true);
    });

    it('has no volume slider in the pane, and still a mute button', async () => {
      await mountVideo({ size: 'pane' });

      expect(target.querySelector('[data-volume]')).toBeNull();
      expect(button('Mute')).not.toBeNull();
    });

    it('sets the volume from the slider, which reads as a percentage', async () => {
      const video = await mountVideo();
      const slider = volumeSlider()!;

      slider.value = '25';
      slider.dispatchEvent(new Event('input', { bubbles: true }));
      flushSync();

      expect(video.volume).toBeCloseTo(0.25);
      expect(volumeSlider()!.getAttribute('aria-valuetext')).toBe('25%');
      expect(volumeSlider()!.getAttribute('aria-label')).toBe('Volume');
    });

    it('unmutes when the slider is moved up from silence', async () => {
      const video = await mountVideo();
      button('Mute').click();
      flushSync();
      expect(video.muted).toBe(true);

      volumeSlider()!.value = '40';
      volumeSlider()!.dispatchEvent(new Event('input', { bubbles: true }));
      flushSync();

      expect(video.muted).toBe(false);
      expect(video.volume).toBeCloseTo(0.4);
    });

    it('has no slider where the browser ignores a page\'s volume, and keeps Mute alone', async () => {
      volumeSupport.works = false;
      const video = await mountVideo();

      expect(volumeSlider()).toBeNull();
      expect(button('Mute')).not.toBeNull();
      button('Mute').click();
      flushSync();
      expect(video.muted).toBe(true);
      expect(button('Unmute')).not.toBeNull();
    });

    it('leaves the volume keys to the page where there is no slider, as on such a browser', async () => {
      volumeSupport.works = false;
      const video = await mountVideo();

      const event = key('ArrowDown');

      expect(event.defaultPrevented).toBe(false);
      expect(video.volume).toBe(1);
      expect(mediaVolume.volume).toBe(1);
    });

    it('leaves the volume keys to the page in the pane, which has no volume control', async () => {
      const video = await mountVideo({ size: 'pane' });

      expect(key('ArrowUp').defaultPrevented).toBe(false);
      expect(key('ArrowDown').defaultPrevented).toBe(false);
      expect(video.volume).toBe(1);
      expect(mediaVolume.volume).toBe(1);
    });

    it('brings the slider up for a moment while the up and down keys change the volume', async () => {
      vi.useFakeTimers();
      await mountVideo();
      expect(root().hasAttribute('data-volume-shown')).toBe(false);

      key('ArrowUp');
      expect(root().hasAttribute('data-volume-shown')).toBe(true);
      await vi.advanceTimersByTimeAsync(1500);
      flushSync();

      expect(root().hasAttribute('data-volume-shown')).toBe(false);
    });

    it('can be reached with the keys: the slider is in the tab order, after Mute', async () => {
      await mountVideo();

      const order = Array.from(controls()!.querySelectorAll<HTMLElement>('button, input')).map((el) => el.getAttribute('aria-label'));

      expect(order).toEqual(['Play', 'Mute', 'Volume', 'Fullscreen']);
      expect(volumeSlider()!.tabIndex).toBe(0);
    });
  });

  describe('fullscreen', () => {
    it('has a button for a video outside the pane, which fullscreens the player and not the video', async () => {
      await mountVideo();
      const request = vi.fn().mockResolvedValue(undefined);
      root().requestFullscreen = request;

      button('Fullscreen').click();

      expect(request).toHaveBeenCalledOnce();
    });

    it('has none in the pane, and none for a take', async () => {
      await mountVideo({ size: 'pane' });
      expect(button('Fullscreen')).toBeNull();

      await mountAudio();
      expect(button('Fullscreen')).toBeNull();
    });

    it('leaves fullscreen from the same button', async () => {
      await mountVideo();
      const exit = vi.fn().mockResolvedValue(undefined);
      Object.defineProperty(document, 'fullscreenElement', { configurable: true, get: () => root() });
      document.exitFullscreen = exit;
      document.dispatchEvent(new Event('fullscreenchange'));
      flushSync();

      expect(root().getAttribute('data-fullscreen')).toBe('true');
      button('Exit fullscreen').click();

      expect(exit).toHaveBeenCalledOnce();
      Reflect.deleteProperty(document, 'fullscreenElement');
    });
  });

  describe('touch and swipe', () => {
    it('opts the timeline and the cluster out of the swipe in the viewer, and not the root, so the picture still pages', async () => {
      await mountVideo();

      expect(root().hasAttribute('data-own-drag')).toBe(false);
      expect(timeline().hasAttribute('data-own-drag')).toBe(true);
      expect(controls()?.hasAttribute('data-own-drag')).toBe(true);
      expect(element().closest('[data-own-drag]')).toBeNull();
    });

    it('opts the whole player out on the stage', async () => {
      await mountVideo({ size: 'stage' });

      expect(root().hasAttribute('data-own-drag')).toBe(true);
    });
  });

  describe('as audio', () => {
    it('is the same player with its waveform as the timeline, and no fullscreen', async () => {
      const take = await mountAudio();

      expect(take.controls).toBe(false);
      expect(take.autoplay).toBe(false);
      expect(timeline().getAttribute('aria-label')).toBe('Playback position');
      expect(timeline().getAttribute('aria-valuetext')).toBe('0:00');
      expect(timeText()).toBe('0:00 | 1:40');
      expect(button('Fullscreen')).toBeNull();
      expect(target.querySelector('.ambient')).toBeNull();
    });

    it('reads a take under a minute in seconds', async () => {
      await mountAudio({ duration: 8 });
      media.load(audioElement(), 8);
      flushSync();

      expect(timeText()).toBe('0.0 s | 8.0 s');
    });

    it('seeks by a click on the waveform, drags, and keeps what it was doing', async () => {
      const take = await mountAudio();

      pointer('pointerdown', 300);
      expect(media.seeks(take)).toEqual([50]);
      land(take);
      pointer('pointermove', 400);
      expect(media.seeks(take).at(-1)).toBe(75);
      pointer('pointerup', 400);

      expect(take.paused).toBe(true);
    });

    it('moves one second with the arrows, always, and five with Shift', async () => {
      const take = await mountAudio();

      key('ArrowRight');
      land(take);
      key('ArrowRight', { shiftKey: true });
      land(take);
      key('ArrowLeft');

      expect(media.seeks(take)).toEqual([1, 6, 5]);
    });

    it('jumps to a tenth of the take with a digit', async () => {
      const take = await mountAudio();

      key('5');
      land(take);
      key('0');

      expect(media.seeks(take)).toEqual([50, 0]);
    });

    it('goes to the start and the end with Home and End', async () => {
      const take = await mountAudio();

      key('End');
      expect(media.seeks(take).at(-1)).toBe(100);
    });

    it('labels the slowed scrub without a frame', async () => {
      await mountAudio();

      pointer('pointerdown', 300, 720);
      pointer('pointermove', 300, 600);

      expect(target.querySelector('.fine')?.textContent).toBe('Fine scrub');
    });

    it('shows Play over its waveform while paused, in the viewer and on the stage alike', async () => {
      await mountAudio();
      expect(bigPlay()).not.toBeNull();

      await mountAudio({ size: 'stage' });
      expect(bigPlay()).not.toBeNull();
    });

    it('has the big Play under a finger too, and Play and Mute in its row, with no fullscreen', async () => {
      touchLayer.coarse = true;
      await mountAudio();

      expect(bigPlay()).not.toBeNull();
      expect(Array.from(controls()!.querySelectorAll('button')).map((el) => el.getAttribute('aria-label'))).toEqual(['Play', 'Mute']);
    });

    it('reads its time in the same chip as a video, which is never a menu button', async () => {
      await mountVideo();
      const videoChip = target.querySelector('.time-chip');
      const take = await mountAudio({ actions: [{ kind: 'item', id: 'extend', label: 'Extend', onselect: vi.fn() }] });
      const takeChip = target.querySelector('.time-chip');
      key('ArrowRight');
      land(take);

      expect(takeChip).not.toBeNull();
      expect(takeChip!.tagName).toBe(videoChip!.tagName);
      expect(target.querySelector('[aria-haspopup]')).toBeNull();
      expect(timeText()).toBe('0:01 | 1:40');
    });

    it('takes the big Play away while it plays, and brings it back on pause', async () => {
      const take = await mountAudio();

      play(take);
      expect(bigPlay()).toBeNull();
      take.pause();
      take.dispatchEvent(new Event('pause'));
      flushSync();

      expect(bigPlay()).not.toBeNull();
    });

    it('keeps its time and controls in view while it plays: nothing lies behind them to uncover', async () => {
      const take = await mountAudio();
      vi.useFakeTimers();
      play(take);
      await vi.advanceTimersByTimeAsync(5000);
      flushSync();

      expect(root().getAttribute('data-chrome')).toBe('shown');
      expect(root().hasAttribute('data-fades')).toBe(false);
    });
  });

  describe('one take at a time', () => {
    it('pauses the one that was playing when another starts', async () => {
      const first = await mountVideo();
      play(first);
      const firstTarget = target;
      const second = await mountVideo();
      media.paused.length = 0;

      play(second);

      expect(media.paused).toContain(first);
      expect(firstTarget).not.toBe(target);
    });

    it('does not pause a player that has left when another starts', async () => {
      const page = mountComponent(MediaPlayer, { src: '/media/a.mp4', kind: 'video', size: 'viewer', label: 'a.mp4' });
      const first = page.target.querySelector('video')!;
      first.dispatchEvent(new Event('play'));
      await page.unmount();
      media.paused.length = 0;

      const second = await mountVideo();
      play(second);

      expect(media.paused).not.toContain(first);
    });

    it('pauses when the player goes', async () => {
      const page = mountComponent(MediaPlayer, { src: '/media/a.mp4', kind: 'video', size: 'viewer', label: 'a.mp4' });
      const el = page.target.querySelector('video')!;
      media.load(el, 6);
      flushSync();
      el.play();
      media.paused.length = 0;

      await page.unmount();

      expect(media.paused).toContain(el);
    });
  });

  describe('the picture', () => {
    it('gives the timeline the focus on a mouse click, so the arrows step the clip afterwards', async () => {
      const video = await mountVideo();
      (document.activeElement as HTMLElement | null)?.blur();

      video.click();
      flushSync();

      expect(document.activeElement).toBe(timeline());
      expect(video.paused).toBe(false);
      key('ArrowRight', {}, document.activeElement!);
      expect(media.seeks(video).length).toBeGreaterThan(0);
    });

    it('gives the timeline the focus when the big centre button is pressed, as a click on the picture does', async () => {
      const video = await mountVideo();
      (document.activeElement as HTMLElement | null)?.blur();

      target.querySelector<HTMLButtonElement>('button.centre')!.click();
      flushSync();

      expect(document.activeElement).toBe(timeline());
      expect(video.paused).toBe(false);
    });

    it('gives the timeline the focus on a tap too, so Space works on an iPad at once', async () => {
      const video = await mountVideo();
      root().dispatchEvent(new PointerEvent('pointerdown', { pointerType: 'touch', bubbles: true }));
      (document.activeElement as HTMLElement | null)?.blur();

      video.click();
      flushSync();

      expect(document.activeElement).toBe(timeline());
      expect(video.paused).toBe(false);
      key(' ', {}, document.activeElement!);
      expect(video.paused).toBe(true);
    });

    it('gives the timeline the focus when the row\'s Play is tapped, which a tap leaves on no button on an iPad', async () => {
      const video = await mountVideo();
      root().dispatchEvent(new PointerEvent('pointerdown', { pointerType: 'touch', bubbles: true }));
      (document.activeElement as HTMLElement | null)?.blur();

      rowPlay().dispatchEvent(new MouseEvent('click', { bubbles: true, detail: 1 }));
      flushSync();

      expect(document.activeElement).toBe(timeline());
      expect(video.paused).toBe(false);
    });

    it('leaves the focus on the row\'s Play after a key press on it, and after a mouse click', async () => {
      await mountVideo();

      rowPlay().focus();
      rowPlay().dispatchEvent(new MouseEvent('click', { bubbles: true, detail: 0 }));
      flushSync();
      expect(document.activeElement).toBe(rowPlay());

      root().dispatchEvent(new PointerEvent('pointerdown', { pointerType: 'mouse', bubbles: true }));
      rowPlay().dispatchEvent(new MouseEvent('click', { bubbles: true, detail: 1 }));
      flushSync();
      expect(document.activeElement).toBe(rowPlay());
    });

    describe('a tap or a swipe', () => {
      /** Press on the picture at *from* and release it at *to*, as a finger does; the browser then sends the click if it was a tap. */
      function touchOn(video: HTMLVideoElement, from: [number, number], to: [number, number]): void {
        video.dispatchEvent(new PointerEvent('pointerdown', { clientX: from[0], clientY: from[1], pointerType: 'touch', bubbles: true }));
        video.dispatchEvent(new MouseEvent('click', { clientX: to[0], clientY: to[1], bubbles: true }));
        flushSync();
      }

      it('plays and pauses on a press that stays where it began', async () => {
        const video = await mountVideo();

        touchOn(video, [200, 200], [204, 203]);
        expect(video.paused).toBe(false);
        touchOn(video, [200, 200], [200, 200]);

        expect(video.paused).toBe(true);
      });

      it('does nothing on a press that moved, which is a swipe for the viewer to page on', async () => {
        const video = await mountVideo();

        touchOn(video, [300, 200], [180, 204]);

        expect(video.paused).toBe(true);
        expect(media.played).toEqual([]);
      });

      it('does not carry a swipe over to the tap after it', async () => {
        const video = await mountVideo();
        touchOn(video, [300, 200], [180, 204]);

        touchOn(video, [200, 200], [200, 200]);

        expect(video.paused).toBe(false);
      });
    });
  });

  describe('focus rings', () => {
    const ring = (): string | null => root().getAttribute('data-input');
    const keydown = (name: string, init: KeyboardEventInit = {}, on: EventTarget = document.body): void => {
      on.dispatchEvent(new KeyboardEvent('keydown', { key: name, bubbles: true, cancelable: true, ...init }));
      flushSync();
    };

    it('does not show after Space plays and pauses, which is not navigation', async () => {
      const video = await mountVideo();
      root().dispatchEvent(new PointerEvent('pointerdown', { pointerType: 'mouse', bubbles: true }));
      video.click();
      flushSync();
      expect(document.activeElement).toBe(timeline());

      keydown(' ', {}, timeline());
      expect(video.paused).toBe(true);
      keydown(' ', {}, timeline());
      expect(video.paused).toBe(false);

      expect(ring()).toBe('pointer');
    });

    it('does not show after Space with no focus in a player either', async () => {
      await mountVideo();
      root().dispatchEvent(new PointerEvent('pointerdown', { pointerType: 'touch', bubbles: true }));

      keydown(' ');

      expect(ring()).toBe('pointer');
    });

    it('shows after Tab, Shift+Tab, the arrows, Home and End, and goes at the next pointer press', async () => {
      await mountVideo();
      root().dispatchEvent(new PointerEvent('pointerdown', { pointerType: 'mouse', bubbles: true }));
      expect(ring()).toBe('pointer');

      for (const [name, init] of [['Tab', {}], ['Tab', { shiftKey: true }], ['ArrowRight', {}], ['Home', {}], ['End', {}]] as [string, KeyboardEventInit][]) {
        document.dispatchEvent(new PointerEvent('pointerdown', { pointerType: 'mouse', bubbles: true }));
        flushSync();
        expect(ring()).toBe('pointer');
        keydown(name, init);
        expect(ring()).toBe('keys');
      }
    });

    it('stays after Space once a navigation key has shown it', async () => {
      await mountVideo();
      keydown('Tab');

      keydown(' ', {}, timeline());

      expect(ring()).toBe('keys');
    });
  });

  describe('Space with the focus nowhere in a player', () => {
    const spaceOn = (on: EventTarget = document.body): KeyboardEvent => {
      const event = new KeyboardEvent('keydown', { key: ' ', bubbles: true, cancelable: true });
      on.dispatchEvent(event);
      flushSync();
      return event;
    };

    it('plays and pauses the player last pressed, and leaves the focus where it was', async () => {
      const video = await mountVideo();
      root().dispatchEvent(new PointerEvent('pointerdown', { pointerType: 'touch', bubbles: true }));
      (document.activeElement as HTMLElement | null)?.blur();

      const event = spaceOn();
      expect(event.defaultPrevented).toBe(true);
      expect(video.paused).toBe(false);
      expect(document.activeElement).not.toBe(timeline());
      spaceOn();

      expect(video.paused).toBe(true);
    });

    it('leaves a player that was never pressed alone', async () => {
      const video = await mountVideo();

      const event = spaceOn();

      expect(event.defaultPrevented).toBe(false);
      expect(video.paused).toBe(true);
    });

    it('leaves Space to a button, a link and a field', async () => {
      const video = await mountVideo();
      root().dispatchEvent(new PointerEvent('pointerdown', { pointerType: 'touch', bubbles: true }));
      const outside = [document.createElement('button'), document.createElement('a'), document.createElement('input')];
      outside[1].setAttribute('href', '#top');
      for (const node of outside) document.body.appendChild(node);

      for (const node of outside) spaceOn(node);

      expect(video.paused).toBe(true);
      for (const node of outside) node.remove();
    });

    it('leaves Space to a checkbox, a switch and a radio, which it flips instead of the video', async () => {
      const video = await mountVideo();
      root().dispatchEvent(new PointerEvent('pointerdown', { pointerType: 'touch', bubbles: true }));
      const checkbox = document.createElement('input');
      checkbox.type = 'checkbox';
      const toggle = document.createElement('input');
      toggle.type = 'checkbox';
      toggle.setAttribute('role', 'switch');
      const radio = document.createElement('input');
      radio.type = 'radio';
      const controls = [checkbox, toggle, radio];
      for (const node of controls) document.body.appendChild(node);

      for (const node of controls) expect(spaceOn(node).defaultPrevented).toBe(false);

      expect(video.paused).toBe(true);
      for (const node of controls) node.remove();
    });

    it('plays from an element the focus rests on without being a tab stop', async () => {
      const video = await mountVideo();
      root().dispatchEvent(new PointerEvent('pointerdown', { pointerType: 'touch', bubbles: true }));
      const rest = document.createElement('div');
      rest.tabIndex = -1;
      document.body.appendChild(rest);

      const event = spaceOn(rest);

      expect(event.defaultPrevented).toBe(true);
      expect(video.paused).toBe(false);
      rest.remove();
    });

    it('goes to the player pressed last, not the one pressed before', async () => {
      const first = await mountVideo();
      const firstRoot = root();
      const second = await mountVideo();
      firstRoot.dispatchEvent(new PointerEvent('pointerdown', { pointerType: 'touch', bubbles: true }));
      root().dispatchEvent(new PointerEvent('pointerdown', { pointerType: 'touch', bubbles: true }));

      spaceOn();

      expect(second.paused).toBe(false);
      expect(first.paused).toBe(true);
    });

    it('stays out of a modal the player is not in', async () => {
      const video = await mountVideo();
      root().dispatchEvent(new PointerEvent('pointerdown', { pointerType: 'touch', bubbles: true }));
      const modal = document.createElement('div');
      modal.setAttribute('aria-modal', 'true');
      document.body.appendChild(modal);

      spaceOn();

      expect(video.paused).toBe(true);
      modal.remove();
    });
  });

  describe('the tab order', () => {
    /** The player's stops in the order Tab visits them. */
    function stops(): string[] {
      return Array.from(root().querySelectorAll<HTMLElement>('button, input, [role="slider"], a[href]')).map((el) => el.getAttribute('aria-label') ?? el.textContent ?? '');
    }

    it('puts the big Play before the timeline, where it shows', async () => {
      await mountVideo();

      expect(stops().slice(0, 2)).toEqual(['Play', 'Playback position']);
    });

    it('reaches the cluster after the timeline on the stage too', async () => {
      await mountVideo({ size: 'stage' });

      expect(stops()).toEqual(['Play', 'Playback position', 'Play', 'Mute', 'Volume', 'Fullscreen']);
    });

    it('runs through the timeline, then the frame menu\'s button, then the row\'s controls', async () => {
      await mountVideo({ actions: [{ kind: 'item', id: 'extend', label: 'Extend', onselect: vi.fn() }] });
      pointer('pointerdown', xOf(72));
      pointer('pointerup', xOf(72));

      expect(stops()).toEqual(['Play', 'Playback position', 'Use this frame, 3.0 s', 'Play', 'Mute', 'Volume', 'Fullscreen']);
    });
  });

  describe('the frame menu', () => {
    const selects = { extend: vi.fn(), 'frame-first': vi.fn() };
    beforeEach(() => { selects.extend.mockReset(); selects['frame-first'].mockReset(); });
    // A row as the builder gives it: a reason shows beside the label in place of the note, and a busy row says so.
    const item = (base: { id: string; label: string; description: string; onselect: () => void }, { reason, busy }: { reason?: string; busy?: boolean } = {}): ActionMenuItem => ({
      kind: 'item', ...base, label: busy ? 'Extracting the frame…' : base.label, description: reason ? undefined : base.description, hint: reason, disabled: Boolean(reason) || busy === true, busy: busy === true,
    });
    // The caller words the rows; the menu shows them as given.
    const EXTEND_NOTE = 'Keeps 3.0 s of 6.0 s, drops what came after';
    const FIRST_NOTE = 'Start a new clip on this frame';
    const rows = (overrides: { reason?: string; busy?: boolean }[] = []) => [
      item({ id: 'extend', label: 'Extend from 3.0 s', description: EXTEND_NOTE, onselect: selects.extend }, overrides[0]),
      item({ id: 'frame-first', label: 'First frame', description: FIRST_NOTE, onselect: selects['frame-first'] }, overrides[1]),
    ];
    /** The time chip while paused: the menu's button. */
    const chip = (): HTMLButtonElement | null => target.querySelector<HTMLButtonElement>('[data-testid="frame-menu"]');
    /** The menu lives in the page's body, as every popover does. */
    const menu = (): HTMLElement | null => document.querySelector<HTMLElement>('[role="menu"]');
    const row = (id: string): HTMLElement => menu()!.querySelector<HTMLElement>(`[data-action="${id}"]`)!;

    async function pausedAt(frame: number, props: Record<string, unknown> = {}): Promise<HTMLVideoElement> {
      const video = await mountVideo({ actions: rows(), ...props });
      pointer('pointerdown', xOf(frame));
      pointer('pointerup', xOf(frame));
      return video;
    }

    describe('the time chip', () => {
      it('is the time of the paused frame and a button for what it offers', async () => {
        await pausedAt(72);

        expect(chip()?.getAttribute('aria-label')).toBe('Use this frame, 3.0 s');
        expect(chip()?.textContent).toContain('3.0 s');
        expect(chip()?.getAttribute('aria-haspopup')).toBe('menu');
        expect(chip()?.getAttribute('aria-expanded')).toBe('false');
        expect(chip()?.hasAttribute('data-own-drag')).toBe(true);
      });

      it('shows the time and the clip\'s length alone while the clip plays, with no button', async () => {
        const video = await mountVideo({ actions: rows() });
        play(video);

        expect(chip()).toBeNull();
        expect(timeText()).toBe('0.0 s | 6.0 s');
        expect(target.querySelector('[aria-haspopup]')).toBeNull();
      });

      it('is a plain readout at the first frame, and without rows', async () => {
        await mountVideo({ actions: rows() });
        expect(chip()).toBeNull();
        expect(timeText()).toBe('0.0 s | 6.0 s');

        await mountVideo({ actions: [] });
        pointer('pointerdown', xOf(72));
        pointer('pointerup', xOf(72));

        expect(chip()).toBeNull();
        expect(timeText()).toBe('3.0 s | 6.0 s');
      });

      it('keeps the readout\'s words when it becomes the button, and takes them back when the clip plays', async () => {
        const video = await pausedAt(72);
        expect(timeText()).toBe('3.0 s | 6.0 s');

        play(video);

        expect(timeText()).toBe('3.0 s | 6.0 s');
        expect(chip()).toBeNull();
      });

      it('is on the stage too, and not on the file browser\'s pane, which passes no rows', async () => {
        await pausedAt(72, { size: 'stage' });
        expect(chip()).not.toBeNull();

        await pausedAt(72, { size: 'pane', actions: [] });
        expect(chip()).toBeNull();
      });

      it('goes while the timeline is scrubbed and returns on release', async () => {
        await pausedAt(72);

        pointer('pointerdown', xOf(100));
        expect(chip()).toBeNull();
        pointer('pointerup', xOf(100));

        expect(chip()?.getAttribute('aria-label')).toBe('Use this frame, 4.2 s');
      });

      it('moves in place with a step, and reads the first and the last frame', async () => {
        await pausedAt(72);
        const first = chip();

        key('ArrowRight');
        land();
        expect(chip()).toBe(first);
        expect(chip()?.getAttribute('aria-label')).toBe('Use this frame, 3.0 s');

        key('End');
        land();
        expect(chip()?.getAttribute('aria-label')).toBe('Use this frame, 6.0 s');
        pointer('pointerdown', xOf(10));
        pointer('pointerup', xOf(10));
        land();
        expect(chip()?.getAttribute('aria-label')).toBe('Use this frame, 0.4 s');
      });

      it('does not flicker the chip or the play glyph when a scrub of a playing video lets go', async () => {
        const video = await mountVideo({ actions: rows() });
        play(video);
        const flash = target.querySelector('.flash');

        pointer('pointerdown', xOf(40));
        // The element answers the resume a moment later than the release.
        video.play = vi.fn(() => Promise.resolve()) as unknown as typeof video.play;
        pointer('pointerup', xOf(40));

        expect(root().getAttribute('data-state')).toBe('playing');
        expect(chip()).toBeNull();
        Object.defineProperty(video, 'paused', { configurable: true, value: false });
        video.dispatchEvent(new Event('play'));
        flushSync();
        expect(root().getAttribute('data-state')).toBe('playing');
        expect(chip()).toBeNull();
        expect(target.querySelector('.flash')).toBe(flash);
      });

      it('is left out of a tiny player, which has no room for it beside the big Play', async () => {
        vi.spyOn(Element.prototype, 'clientWidth', 'get').mockImplementation(function (this: Element) { return this.classList.contains('player') ? 300 : 0; });
        vi.spyOn(Element.prototype, 'clientHeight', 'get').mockImplementation(function (this: Element) { return this.classList.contains('player') ? 90 : 0; });
        await pausedAt(72, { size: 'stage' });

        expect(chip()).toBeNull();
      });

      it('is not offered in fullscreen', async () => {
        await pausedAt(72);
        Object.defineProperty(document, 'fullscreenElement', { configurable: true, get: () => root() });
        document.dispatchEvent(new Event('fullscreenchange'));
        flushSync();

        expect(chip()).toBeNull();
        Reflect.deleteProperty(document, 'fullscreenElement');
      });

      it('shows busy while a frame is extracted, and says so for a screen reader', async () => {
        await pausedAt(72, { actions: rows([{}, { busy: true }]) });

        expect(chip()?.getAttribute('aria-busy')).toBe('true');
        expect(chip()?.querySelector('.spinner')).not.toBeNull();
        expect(status()).toBe(FRAME_BUSY_LABEL);
        chip()!.click();
        flushSync();
        expect(row('frame-first').textContent).toContain(FRAME_BUSY_LABEL);
        expect(row('frame-first').getAttribute('aria-busy')).toBe('true');
        expect(row('frame-first').getAttribute('aria-disabled')).toBe('true');
      });
    });

    describe('the menu', () => {
      it('opens on a click with a row for each action, headed by the frame, and a row runs its action', async () => {
        await pausedAt(72);

        chip()!.click();
        flushSync();
        expect(menu()?.getAttribute('aria-label')).toBe('3.0 s, this frame');
        expect(chip()?.getAttribute('aria-expanded')).toBe('true');
        expect(Array.from(menu()!.querySelectorAll('[data-action]')).map((item) => item.getAttribute('data-action'))).toEqual(['extend', 'frame-first']);
        expect(row('extend').textContent).toContain(EXTEND_NOTE);
        row('frame-first').click();
        flushSync();

        expect(selects['frame-first']).toHaveBeenCalledOnce();
        expect(menu()).toBeNull();
      });

      it('opens above the time button, as its arrow points', async () => {
        await pausedAt(72);
        const anchor = { left: 100, top: 500, width: 130, height: 28, right: 230, bottom: 528, x: 100, y: 500, toJSON: () => ({}) };
        chip()!.getBoundingClientRect = () => anchor;
        chip()!.click();
        flushSync();
        await settle();
        const top = (): number => parseFloat(menu()!.style.top);
        vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockImplementation(function (this: HTMLElement) {
          return this === menu() ? ({ left: 0, top: 0, width: 200, height: 120, right: 200, bottom: 120, x: 0, y: 0, toJSON: () => ({}) } as DOMRect) : this === chip() ? (anchor as DOMRect) : new DOMRect();
        });
        chip()!.click();
        flushSync();
        chip()!.click();
        flushSync();
        await settle();

        expect(top()).toBeLessThan(anchor.top);
        expect(top() + 120).toBeLessThanOrEqual(anchor.top);
      });

      it('shows a row it can not use with its reason in place, and a click does nothing', async () => {
        await pausedAt(10, { actions: rows([{ reason: EXTEND_SHORT_REASON }]) });
        chip()!.click();
        flushSync();

        expect(row('extend').getAttribute('aria-disabled')).toBe('true');
        expect(row('extend').textContent).toContain(EXTEND_SHORT_REASON);
        expect(row('extend').textContent).not.toContain(EXTEND_NOTE);
        row('extend').click();

        expect(selects.extend).not.toHaveBeenCalled();
      });

      it('reads the end as the last frame of the clip', async () => {
        const video = await mountVideo({ actions: rows() });
        key('End');
        land();
        chip()!.click();
        flushSync();

        expect(menu()?.getAttribute('aria-label')).toBe('6.0 s, the last frame');
        expect(video.paused).toBe(true);
      });

      it('closes on play, on a scrub, and when the selection goes', async () => {
        const video = await pausedAt(72);
        chip()!.click();
        flushSync();
        play(video);
        expect(menu()).toBeNull();

        video.pause();
        video.dispatchEvent(new Event('pause'));
        flushSync();
        chip()!.click();
        flushSync();
        expect(menu()).not.toBeNull();
        pointer('pointerdown', xOf(100));
        expect(menu()).toBeNull();
        pointer('pointerup', xOf(100));
        chip()!.click();
        flushSync();
        pointer('pointerdown', xOf(0));
        pointer('pointerup', xOf(0));
        land();
        expect(menu()).toBeNull();
        expect(chip()).toBeNull();
      });

      it('stays open when a step moves the frame, and updates its heading', async () => {
        await pausedAt(72);
        chip()!.click();
        flushSync();
        expect(menu()!.textContent).toContain('3.0 s, this frame');

        key('ArrowRight', { shiftKey: true });
        land();

        expect(menu()).not.toBeNull();
        expect(menu()!.textContent).toContain('3.2 s, this frame');
      });

      it('closes on a click on the picture, which does nothing else', async () => {
        const video = await pausedAt(72);
        chip()!.click();
        flushSync();

        video.click();
        flushSync();

        expect(menu()).toBeNull();
        expect(video.paused).toBe(true);
        expect(chip()).not.toBeNull();
      });

      it('takes the big Play out of the way while it is open', async () => {
        await pausedAt(72);
        expect(bigPlay()).not.toBeNull();

        chip()!.click();
        flushSync();
        expect(bigPlay()).toBeNull();
        chip()!.click();
        flushSync();

        expect(bigPlay()).not.toBeNull();
      });

      it('closes on Escape and gives the focus to the timeline, where Space plays', async () => {
        const video = await pausedAt(72);
        chip()!.dispatchEvent(new MouseEvent('click', { bubbles: true, detail: 0 }));
        await settle();
        expect(document.activeElement).toBe(row('extend'));

        document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true }));
        await settle();

        expect(menu()).toBeNull();
        expect(document.activeElement).toBe(timeline());
        const space = key(' ', {}, document.activeElement!);

        expect(space.defaultPrevented).toBe(true);
        expect(video.paused).toBe(false);
        expect(menu()).toBeNull();
      });

      it('steps the frame with the arrows while the menu has the focus, and stays open on the new point', async () => {
        await pausedAt(72);
        chip()!.dispatchEvent(new MouseEvent('click', { bubbles: true, detail: 0 }));
        await settle();
        expect(document.activeElement).toBe(row('extend'));
        const anchored = menu()!.style.left;

        const forward = key('ArrowRight', { shiftKey: true }, document.activeElement!);
        land();

        expect(forward.defaultPrevented).toBe(true);
        expect(menu()).not.toBeNull();
        expect(menu()!.getAttribute('aria-label')).toBe('3.2 s, this frame');
        expect(chip()?.getAttribute('aria-label')).toBe('Use this frame, 3.2 s');
        expect(menu()!.style.left).toBe(anchored);
        expect(document.activeElement).toBe(row('extend'));

        const back = key('ArrowLeft', { shiftKey: true }, document.activeElement!);
        land();

        expect(back.defaultPrevented).toBe(true);
        expect(menu()!.getAttribute('aria-label')).toBe('3.0 s, this frame');
      });

      it('lets the menu keep its own keys, and does not step on them', async () => {
        await pausedAt(72);
        chip()!.dispatchEvent(new MouseEvent('click', { bubbles: true, detail: 0 }));
        await settle();

        const down = key('ArrowDown', {}, document.activeElement!);
        land();

        expect(down.defaultPrevented).toBe(true);
        expect(document.activeElement).toBe(row('frame-first'));
        expect(chip()?.getAttribute('aria-label')).toBe('Use this frame, 3.0 s');
        key('Tab', {}, document.activeElement!);
        expect(menu()).toBeNull();
      });

      it('opened by the keyboard, moves the focus to the first row that can be used', async () => {
        await pausedAt(10, { actions: rows([{ reason: EXTEND_SHORT_REASON }]) });

        chip()!.dispatchEvent(new MouseEvent('click', { bubbles: true, detail: 0 }));
        await settle();

        expect(document.activeElement).toBe(row('frame-first'));
      });

      it('is a sheet on a phone, headed by the frame alone, with the rows and no notes', async () => {
        touchLayer.compact = true;
        await pausedAt(72, { actions: rows([{ reason: EXTEND_SHORT_REASON }]) });

        chip()!.click();
        flushSync();
        await settle();

        expect(document.querySelector('[data-testid="action-sheet"]')).not.toBeNull();
        expect(row('frame-first').textContent).not.toContain(FIRST_NOTE);
        expect(row('extend').textContent).toContain(EXTEND_SHORT_REASON);
        expect(document.querySelector('[data-testid="action-sheet"] .action-menu-heading')).toBeNull();
      });
    });
  });

  describe('the controls on small and touch players', () => {
    function sizeOf(width: number, height: number): void {
      vi.spyOn(Element.prototype, 'clientWidth', 'get').mockImplementation(function (this: Element) { return this.classList.contains('player') ? width : 0; });
      vi.spyOn(Element.prototype, 'clientHeight', 'get').mockImplementation(function (this: Element) { return this.classList.contains('player') ? height : 0; });
    }

    it('has the big Play over the file browser\'s picture under a finger while it is paused, and not while it plays', async () => {
      touchLayer.coarse = true;
      const video = await mountVideo({ size: 'pane' });
      expect(bigPlay()).not.toBeNull();

      play(video);

      expect(bigPlay()).toBeNull();
    });

    it('has the big Play under a finger over a video that fades, wherever it is paused', async () => {
      touchLayer.coarse = true;
      await mountVideo();

      pointer('pointerdown', xOf(72));
      pointer('pointerup', xOf(72));

      expect(bigPlay()).not.toBeNull();
    });

    it('is tiny when it is a small square, with no room for the menu', async () => {
      sizeOf(300, 90);
      await mountVideo({ size: 'stage' });

      expect(root().hasAttribute('data-tiny')).toBe(true);
    });

    it('is not tiny from the height of a short player up', async () => {
      sizeOf(800, 264);
      await mountVideo();
      expect(root().hasAttribute('data-tiny')).toBe(false);

      sizeOf(1280, 720);
      await mountVideo();
      expect(root().hasAttribute('data-tiny')).toBe(false);
    });
  });

  describe('where the controls lie on the picture', () => {
    function sizeOf(width: number, height: number): void {
      vi.spyOn(Element.prototype, 'clientWidth', 'get').mockImplementation(function (this: Element) { return this.classList.contains('player') ? width : 0; });
      vi.spyOn(Element.prototype, 'clientHeight', 'get').mockImplementation(function (this: Element) { return this.classList.contains('player') ? height : 0; });
    }
    const prop = (name: string): number => parseFloat(root().style.getPropertyValue(name));

    it('lays the controls on the picture\'s own bottom edge, as wide as the picture', async () => {
      sizeOf(1000, 600);
      vi.spyOn(HTMLVideoElement.prototype, 'videoWidth', 'get').mockImplementation(() => 480);
      vi.spyOn(HTMLVideoElement.prototype, 'videoHeight', 'get').mockImplementation(() => 848);
      await mountVideo({ size: 'stage' });

      // The portrait picture fits the 600 px height at about 340 px wide: a band of about 330 px on each side.
      const bottom = target.querySelector<HTMLElement>('[data-testid="media-timeline"]')!.parentElement!;
      expect(parseFloat(bottom.style.getPropertyValue('--deck-w'))).toBeCloseTo(339.6, 0);
      expect(parseFloat(bottom.style.getPropertyValue('--deck-left'))).toBeCloseTo(330.2, 0);
      expect(parseFloat(bottom.style.getPropertyValue('--deck-bottom'))).toBe(0);
    });

    it('lies on the edge of a letterboxed picture, in the band under it', async () => {
      sizeOf(1000, 800);
      vi.spyOn(HTMLVideoElement.prototype, 'videoWidth', 'get').mockImplementation(() => 1000);
      vi.spyOn(HTMLVideoElement.prototype, 'videoHeight', 'get').mockImplementation(() => 500);
      await mountVideo();

      const bottom = target.querySelector<HTMLElement>('[data-testid="media-timeline"]')!.parentElement!;
      expect(parseFloat(bottom.style.getPropertyValue('--deck-bottom'))).toBe(150);
    });

    it('is laid out from the gallery\'s size before the file is read, so nothing moves when it arrives', async () => {
      sizeOf(1000, 800);
      const page = mountComponent(MediaPlayer, { src: '/media/clip.mp4', kind: 'video', size: 'viewer', label: 'clip.mp4', naturalSize: { width: 1000, height: 500 } });
      target = page.target;
      await settle();

      expect(root().hasAttribute('data-placed')).toBe(true);
      expect([prop('--pic-w'), prop('--pic-h'), prop('--pic-top')]).toEqual([1000, 500, 150]);

      // The file's own size, when it comes, is the same size and puts nothing elsewhere.
      vi.spyOn(HTMLVideoElement.prototype, 'videoWidth', 'get').mockImplementation(() => 1000);
      vi.spyOn(HTMLVideoElement.prototype, 'videoHeight', 'get').mockImplementation(() => 500);
      media.load(element(), 6);
      flushSync();

      expect([prop('--pic-w'), prop('--pic-h'), prop('--pic-top')]).toEqual([1000, 500, 150]);
    });

    it('has no laid-out picture until a size is known, and fades it in once read', async () => {
      sizeOf(1000, 800);
      const page = mountComponent(MediaPlayer, { src: '/media/clip.mp4', kind: 'video', size: 'viewer', label: 'clip.mp4' });
      target = page.target;
      await settle();

      expect(root().hasAttribute('data-sized')).toBe(false);
      expect(root().hasAttribute('data-placed')).toBe(false);
    });

    it('shrinks a clip no larger than its own size to it, and fills the pane with a pane\'s clip', async () => {
      sizeOf(1000, 800);
      await mountVideo({ naturalSize: { width: 400, height: 300 } });
      expect(prop('--pic-w')).toBe(400);

      await mountVideo({ size: 'pane', naturalSize: { width: 400, height: 300 } });
      expect(root().hasAttribute('data-placed')).toBe(false);
    });
  });

  describe('what the viewer downloads', () => {
    it('holds to the metadata at first and loads the rest once the page has settled', async () => {
      vi.useFakeTimers();
      const video = await mountVideo();
      expect(video.getAttribute('preload')).toBe('metadata');

      await settle(299);
      expect(video.getAttribute('preload')).toBe('metadata');
      await settle(1);
      expect(video.getAttribute('preload')).toBe('auto');
    });

    it('never loads the rest of a clip that is paged past before it has settled', async () => {
      vi.useFakeTimers();
      const first = mountComponent(MediaPlayer, { src: '/media/a.mp4', kind: 'video', size: 'viewer', label: 'a.mp4' });
      const video = first.target.querySelector('video')!;
      await settle(200);
      await first.unmount();
      await settle(1000);

      expect(video.getAttribute('preload')).toBe('metadata');
    });

    it('loads the rest at once on the first play', async () => {
      vi.useFakeTimers();
      const video = await mountVideo();

      play(video);
      flushSync();

      expect(video.getAttribute('preload')).toBe('auto');
    });

    it('loads the rest at once on the first scrub', async () => {
      vi.useFakeTimers();
      const video = await mountVideo();

      pointer('pointerdown', 300);

      expect(video.getAttribute('preload')).toBe('auto');
    });

    it('leaves a take on its metadata', async () => {
      vi.useFakeTimers();
      const take = await mountAudio();
      await settle(1000);

      expect(take.getAttribute('preload')).toBe('metadata');
    });

    it('lets go of its file when it goes: paused, with no source, and the browser told to drop it', async () => {
      const page = mountComponent(MediaPlayer, { src: '/media/a.mp4', kind: 'video', size: 'viewer', label: 'a.mp4' });
      const video = page.target.querySelector('video')!;
      media.load(video, 6);
      flushSync();
      video.play();
      media.paused.length = 0;
      expect(video.getAttribute('src')).toBe('/media/a.mp4');

      await page.unmount();

      expect(media.paused).toContain(video);
      expect(video.hasAttribute('src')).toBe(false);
      expect(media.reloaded).toContain(video);
    });

    it('lets go of a take the same way', async () => {
      const page = mountComponent(MediaPlayer, { src: '/media/t.flac', kind: 'audio', size: 'viewer', label: 't.flac', peaks: PEAKS, duration: 100 });
      const take = page.target.querySelector('audio')!;

      await page.unmount();

      expect(take.hasAttribute('src')).toBe(false);
      expect(media.reloaded).toContain(take);
    });
  });

  describe('the timeline drawing', () => {
    let clears: ReturnType<typeof vi.fn>;

    beforeEach(() => {
      vi.useFakeTimers();
      clears = vi.fn();
      const context = new Proxy({}, { get: (_target, name) => (name === 'clearRect' ? clears : vi.fn()) });
      vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(context as unknown as CanvasRenderingContext2D);
      // jsdom measures nothing: the timeline is 400 by 46 px.
      vi.spyOn(Element.prototype, 'clientWidth', 'get').mockImplementation(function (this: Element) { return this.classList.contains('timeline') ? 400 : 0; });
      vi.spyOn(Element.prototype, 'clientHeight', 'get').mockImplementation(function (this: Element) { return this.classList.contains('timeline') ? 46 : 0; });
    });

    /** Run *frames* display frames of 16 ms. */
    async function frames(count: number): Promise<void> {
      for (let i = 0; i < count; i += 1) await vi.advanceTimersByTimeAsync(16);
    }

    it('does not redraw a video that plays without a visible change', async () => {
      const video = await mountVideo();
      play(video);
      await frames(5);
      clears.mockClear();

      await frames(60);

      expect(clears).not.toHaveBeenCalled();
    });

    it('draws a video again when the playhead moves a pixel, and not for a move inside one', async () => {
      const video = await mountVideo();
      play(video);
      await frames(5);
      clears.mockClear();

      media.tick(video, frameTime(72));
      await frames(3);
      const moved = clears.mock.calls.length;
      media.tick(video, frameTime(72) + 0.001);
      await frames(3);

      expect(moved).toBe(1);
      expect(clears).toHaveBeenCalledTimes(1);
    });

    it('draws a video for the pointer, the bloom and a drag', async () => {
      await mountVideo();
      await frames(5);
      clears.mockClear();

      pointer('pointermove', xOf(30));
      await frames(60);
      const hovered = clears.mock.calls.length;
      pointer('pointermove', xOf(100));
      await frames(60);
      const hoveredElsewhere = clears.mock.calls.length;
      pointer('pointerdown', xOf(50));
      await frames(3);
      const dragged = clears.mock.calls.length;

      expect(hovered).toBeGreaterThan(1);
      expect(hoveredElsewhere).toBeGreaterThan(hovered);
      expect(dragged).toBeGreaterThan(hoveredElsewhere);
    });

    it('draws a take only when the playhead passes a bar', async () => {
      const take = await mountAudio();
      play(take);
      await frames(5);
      clears.mockClear();

      media.tick(take, 0.2);
      await frames(5);
      expect(clears).not.toHaveBeenCalled();

      media.tick(take, 3);
      await frames(5);
      expect(clears).toHaveBeenCalledTimes(1);

      media.tick(take, 3.4);
      await frames(5);
      expect(clears).toHaveBeenCalledTimes(1);
    });
  });

  describe('a take in the viewer', () => {
    it('is a panel-less row above its words: no picture, no ambient light, no fade of its controls', async () => {
      const take = await mountAudio();
      vi.useFakeTimers();

      play(take);
      await vi.advanceTimersByTimeAsync(3000);
      flushSync();

      expect(target.querySelector('.picture')).toBeNull();
      expect(target.querySelector('canvas.ambient')).toBeNull();
      expect(root().getAttribute('data-chrome')).toBe('shown');
    });
  });

  describe('the screen', () => {
    it('draws the ambient light only for a video in the viewer', async () => {
      await mountVideo();
      expect(target.querySelector('canvas.ambient')).not.toBeNull();

      await mountVideo({ size: 'stage' });
      expect(target.querySelector('canvas.ambient')).toBeNull();
    });

    describe('when the picture covers the player', () => {
      const ambient = (): HTMLCanvasElement => target.querySelector<HTMLCanvasElement>('canvas.ambient')!;
      const glow = (): HTMLElement => target.querySelector<HTMLElement>('.glow')!;
      let picture = { width: 1280, height: 720 };

      beforeEach(() => {
        picture = { width: 1280, height: 720 };
        vi.spyOn(Element.prototype, 'clientWidth', 'get').mockImplementation(function (this: Element) { return this.classList.contains('player') ? 1280 : 0; });
        vi.spyOn(Element.prototype, 'clientHeight', 'get').mockImplementation(function (this: Element) { return this.classList.contains('player') ? 720 : 0; });
        vi.spyOn(HTMLVideoElement.prototype, 'videoWidth', 'get').mockImplementation(() => picture.width);
        vi.spyOn(HTMLVideoElement.prototype, 'videoHeight', 'get').mockImplementation(() => picture.height);
      });

      it('hides the light and stops painting it', async () => {
        vi.useFakeTimers();
        const drawImage = vi.fn();
        vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({ drawImage } as unknown as CanvasRenderingContext2D);
        const video = await mountVideo();
        Object.defineProperty(video, 'readyState', { configurable: true, value: 4 });
        drawImage.mockClear();

        play(video);
        vi.advanceTimersByTime(1000);

        expect(glow().hidden).toBe(true);
        expect(drawImage).not.toHaveBeenCalled();
      });

      it('shows the light and paints it when the picture leaves bands', async () => {
        vi.useFakeTimers();
        picture = { width: 960, height: 720 };
        const drawImage = vi.fn();
        vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({ drawImage } as unknown as CanvasRenderingContext2D);
        const video = await mountVideo();
        Object.defineProperty(video, 'readyState', { configurable: true, value: 4 });
        drawImage.mockClear();

        play(video);
        vi.advanceTimersByTime(1000);

        expect(glow().hidden).toBe(false);
        expect(drawImage).toHaveBeenCalled();
      });

      it('draws the picture where it sits and its edges outward into the bands, in a canvas as wide as the player and its reach', async () => {
        vi.useFakeTimers();
        picture = { width: 960, height: 720 };
        const drawImage = vi.fn();
        vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({ drawImage } as unknown as CanvasRenderingContext2D);
        const video = await mountVideo();
        Object.defineProperty(video, 'readyState', { configurable: true, value: 4 });
        drawImage.mockClear();

        play(video);
        vi.advanceTimersByTime(20);

        // 1280 x 720 and 104 px of reach on each side: 1488 x 928 CSS pixels in 32 columns of about 46.5 px.
        expect([ambient().width, ambient().height]).toEqual([32, 20]);
        const [picture1, left, right, top, bottom, onto] = drawImage.mock.calls;
        const scratch = left[0];
        // The picture is read once, into a scratch canvas of the light's own few pixels, and the bands are drawn outward from that copy.
        expect(picture1).toEqual([video, 6, 3, 21, 15]);
        expect(scratch).toBeInstanceOf(HTMLCanvasElement);
        expect(scratch).not.toBe(ambient());
        // The left band is the picture's first column, the right band its last, then the top and bottom rows of what is drawn.
        expect(left).toEqual([scratch, 6, 3, 1, 15, 0, 3, 6, 15]);
        expect(right).toEqual([scratch, 26, 3, 1, 15, 27, 3, 5, 15]);
        expect(top).toEqual([scratch, 0, 3, 32, 1, 0, 0, 32, 3]);
        expect(bottom).toEqual([scratch, 0, 17, 32, 1, 0, 18, 32, 2]);
        // The finished light is laid over the last one.
        expect(onto).toEqual([scratch, 0, 0]);
      });

      it('reaches past the player by its reach, so the soft edge of the canvas is out of view', async () => {
        await mountVideo();

        expect(ambient().style.getPropertyValue('--ambient-reach')).toBe('104px');
      });

      it('lays a soft mask round the picture\'s box on the light, and none while it is hidden', async () => {
        picture = { width: 960, height: 720 };
        await mountVideo({ pane: { width: 1408, height: 760 } });
        expect(glow().style.maskImage).toMatch(/^url\("data:image\/svg\+xml,/);

        picture = { width: 1280, height: 720 };
        await mountVideo();

        expect(glow().hidden).toBe(true);
        expect(glow().style.maskImage).toBe('none');
      });

      describe('while the clip plays', () => {
        beforeEach(() => { media = installFakeMedia({ frameCallbacks: true }); });

        /** A canvas context that records how much of each new picture the light takes in. */
        function recordingContext(): { drawImage: ReturnType<typeof vi.fn>; shares: number[] } {
          const shares: number[] = [];
          const drawImage = vi.fn();
          const context = { drawImage, set globalAlpha(value: number) { shares.push(value); } } as unknown as CanvasRenderingContext2D;
          vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(context);
          return { drawImage, shares };
        }

        it('repaints with each picture the video shows, and not on a timer', async () => {
          picture = { width: 960, height: 720 };
          const { drawImage } = recordingContext();
          const video = await mountVideo();
          Object.defineProperty(video, 'readyState', { configurable: true, value: 4 });
          play(video);
          vi.useFakeTimers();
          drawImage.mockClear();

          vi.advanceTimersByTime(1000);
          expect(drawImage).not.toHaveBeenCalled();

          media.present(video, 0.5);
          expect(drawImage).toHaveBeenCalled();
          drawImage.mockClear();
          media.present(video, 0.54);
          expect(drawImage).toHaveBeenCalled();
        });

        it('takes in about a third of each new picture, so the colours glide rather than step', async () => {
          picture = { width: 960, height: 720 };
          const { shares } = recordingContext();
          const video = await mountVideo();
          Object.defineProperty(video, 'readyState', { configurable: true, value: 4 });
          play(video);
          shares.length = 0;

          media.present(video, 0.5);

          expect(shares).toContain(0.3);
        });

        it('paints a seek or a pause whole at once, so the light matches the picture', async () => {
          picture = { width: 960, height: 720 };
          const { shares } = recordingContext();
          const video = await mountVideo();
          Object.defineProperty(video, 'readyState', { configurable: true, value: 4 });
          shares.length = 0;

          video.dispatchEvent(new Event('canplay'));
          flushSync();

          expect(shares).toContain(1);
        });
      });

      it('follows the picture when its shape changes', async () => {
        const video = await mountVideo();
        expect(glow().hidden).toBe(true);

        picture = { width: 960, height: 720 };
        video.dispatchEvent(new Event('resize'));
        flushSync();
        expect(glow().hidden).toBe(false);

        picture = { width: 1280, height: 720 };
        video.dispatchEvent(new Event('resize'));
        flushSync();
        expect(glow().hidden).toBe(true);
      });

      it('keeps the light off for a sliver of a pixel or two', async () => {
        picture = { width: 1278, height: 720 };
        await mountVideo();

        expect(glow().hidden).toBe(true);
      });

      it('fills the viewer\'s pane, padding and all, rather than the player\'s own box', async () => {
        await mountVideo({ pane: { width: 1408, height: 760 } });

        // The pane is 64 px wider on each side and 20 px taller above and below.
        expect([glow().style.left, glow().style.top, glow().style.width, glow().style.height]).toEqual(['-64px', '-20px', '1408px', '760px']);
        // The picture fills the player, but the pane's padding is a band, so the light shows.
        expect(glow().hidden).toBe(false);
      });

      it('hides the light only when the picture covers the whole pane', async () => {
        await mountVideo({ pane: { width: 1280, height: 720 } });

        expect(glow().hidden).toBe(true);
      });

      it('lights a clip smaller than the player all round, shown at its own size', async () => {
        picture = { width: 640, height: 360 };
        await mountVideo();

        expect(glow().hidden).toBe(false);
      });

      it('fills the player itself in fullscreen, where the pane is not the player\'s surround', async () => {
        const video = await mountVideo({ pane: { width: 1408, height: 760 } });
        Object.defineProperty(document, 'fullscreenElement', { configurable: true, get: () => root() });
        document.dispatchEvent(new Event('fullscreenchange'));
        flushSync();

        expect([glow().style.left, glow().style.top, glow().style.width, glow().style.height]).toEqual(['0px', '0px', '1280px', '720px']);
        void video;
        Object.defineProperty(document, 'fullscreenElement', { configurable: true, get: () => null });
      });
    });

    it('repaints the ambient light only while the video plays', async () => {
      vi.useFakeTimers();
      vi.spyOn(Element.prototype, 'clientWidth', 'get').mockImplementation(function (this: Element) { return this.classList.contains('player') ? 1280 : 0; });
      vi.spyOn(Element.prototype, 'clientHeight', 'get').mockImplementation(function (this: Element) { return this.classList.contains('player') ? 720 : 0; });
      vi.spyOn(HTMLVideoElement.prototype, 'videoWidth', 'get').mockImplementation(() => 960);
      vi.spyOn(HTMLVideoElement.prototype, 'videoHeight', 'get').mockImplementation(() => 720);
      const drawImage = vi.fn();
      vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({ drawImage } as unknown as CanvasRenderingContext2D);
      const video = await mountVideo();
      Object.defineProperty(video, 'readyState', { configurable: true, value: 4 });
      drawImage.mockClear();

      vi.advanceTimersByTime(1000);
      expect(drawImage).not.toHaveBeenCalled();

      play(video);
      vi.advanceTimersByTime(1000);
      expect(drawImage).toHaveBeenCalled();
    });

    it('is a group named for the file', async () => {
      await mountVideo();

      expect(root().getAttribute('role')).toBe('group');
      expect(root().getAttribute('aria-label')).toBe('clip.mp4');
    });
  });
});
