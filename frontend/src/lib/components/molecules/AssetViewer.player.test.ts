import { flushSync } from 'svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { GalleryAsset } from '$lib/types';
import AssetViewer from './AssetViewer.svelte';
import { CHORD_TIMEOUT_MS } from '$lib/keyboard';
import { mediaVolume } from '$lib/state/mediaVolume.svelte';
import { toasts } from '$lib/state/toasts.svelte';
import { videoFacts } from '$lib/state/videoFacts.svelte';
import { makeAudioAsset } from '../../../test-utils/audio';
import { installFakeMedia, type FakeMedia } from '../../../test-utils/media';
import { mountComponent, settle } from '../../../test-utils/mount';
import { reactiveProps } from '../../../test-utils/reactiveProps.svelte';

const browse = vi.hoisted(() => ({ browseInfo: vi.fn() }));
vi.mock('$lib/api/browse', async (importOriginal) => ({ ...(await importOriginal<typeof import('$lib/api/browse')>()), browseInfo: browse.browseInfo }));

// The touch layer follows the pointer; a test switches it on without a real media query.
const touchLayer = vi.hoisted(() => ({ coarse: false }));
vi.mock('$lib/state/layoutMode.svelte', async (importOriginal) => {
  const actual = await importOriginal<typeof import('$lib/state/layoutMode.svelte')>();
  return { ...actual, layout: new Proxy(actual.layout, { get: (target, key) => (key === 'coarse' && touchLayer.coarse ? true : Reflect.get(target, key)) }) };
});

function makeAsset(overrides: Partial<GalleryAsset> = {}): GalleryAsset {
  return {
    id: 'out/clip-a.mp4',
    url: '/media/out/clip-a.mp4',
    thumbnail_url: '/thumbs/out/clip-a.mp4',
    filename: 'clip-a.mp4',
    created_at: '2026-04-30T12:00:00Z',
    project: null,
    workflow: 'txt2vid',
    prompt: 'A clip',
    model: 'ltx',
    media_type: 'video',
    file_path: '/outputs/out/clip-a.mp4',
    seed: 1,
    has_reusable_config: true,
    reuse_workspace_url: '#/workspace?workflow=txt2vid',
    ...overrides,
  };
}

const clipB = makeAsset({ id: 'out/clip-b.mp4', url: '/media/out/clip-b.mp4', filename: 'clip-b.mp4', file_path: '/outputs/out/clip-b.mp4' });
const image = makeAsset({ id: 'out/still.png', url: '/media/out/still.png', thumbnail_url: '/media/out/still.png', filename: 'still.png', media_type: 'image', file_path: '/outputs/out/still.png' });

/** Dispatch a touch event with one finger at (x, y). */
function touch(node: Element, type: string, x: number, y: number): void {
  const point = { clientX: x, clientY: y, target: node } as unknown as Touch;
  const event = new Event(type, { bubbles: true, cancelable: true }) as TouchEvent;
  Object.defineProperty(event, 'touches', { value: type === 'touchend' ? [] : [point] });
  Object.defineProperty(event, 'changedTouches', { value: [point] });
  node.dispatchEvent(event);
}

describe('AssetViewer with the media player', () => {
  let media: FakeMedia;

  beforeEach(() => {
    localStorage.clear();
    media = installFakeMedia();
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(null);
    videoFacts.reset();
    mediaVolume.reset();
    toasts.length = 0;
    browse.browseInfo.mockReset().mockResolvedValue({ fps: 24, frames: 144, start: 0 });
  });
  afterEach(() => {
    vi.restoreAllMocks();
    touchLayer.coarse = false;
    document.documentElement.removeAttribute('data-input');
  });

  function mountViewer(props: Record<string, unknown> = {}): ReturnType<typeof mountComponent> {
    const assets = (props.assets as GalleryAsset[] | undefined) ?? [makeAsset(), clipB];
    return mountComponent(AssetViewer, { assets, currentId: assets[0].id, open: true, onclose: vi.fn(), onnavigate: vi.fn(), onreuse: vi.fn(), onextend: vi.fn(), ...props });
  }

  const video = () => document.querySelector<HTMLVideoElement>('video')!;
  const timeline = () => document.querySelector<HTMLElement>('[data-media-player] [role="slider"][aria-label="Playback position"]')!;
  const viewer = () => document.querySelector<HTMLElement>('[data-testid="asset-viewer"]')!;
  const closeButton = () => document.querySelector<HTMLButtonElement>('.viewer-btn[aria-label="Close viewer"]')!;

  /** Open the viewer and give the player its metadata, as the browser would. */
  async function openedVideo(props: Record<string, unknown> = {}): Promise<HTMLVideoElement> {
    mountViewer(props);
    await settle();
    media.load(video(), 6);
    flushSync();
    return video();
  }

  /** Mount a viewer on two videos whose parent follows the page it is asked for, as the Gallery does. */
  function pagedViewer(): void {
    const props = reactiveProps({ assets: [makeAsset(), clipB], currentId: 'out/clip-a.mp4', open: true, onclose: vi.fn(), onnavigate: (_index: number): void => undefined });
    props.onnavigate = (index: number) => { props.currentId = props.assets[index].id; };
    mountComponent(AssetViewer, props);
  }

  function press(key: string, init: KeyboardEventInit = {}, on: EventTarget = document): KeyboardEvent {
    const event = new KeyboardEvent('keydown', { key, bubbles: true, cancelable: true, ...init });
    on.dispatchEvent(event);
    flushSync();
    return event;
  }

  describe('a digit with the focus in the player', () => {
    const upscale = { factors: [], seedvr2: { options: [{ factor: 2 as const, width: 640, height: 384, allowed: true as const, reason: null, warning: null }], model: 'SeedVR2 7B fp8', downloaded: true, download_size_label: null } };

    it('jumps the playhead to that tenth of the clip', async () => {
      const el = await openedVideo({ assets: [makeAsset({ upscale }), clipB], onupscale: vi.fn() });
      timeline().focus();

      press('5', {}, timeline());

      expect(media.seeks(el).at(-1)).toBeGreaterThan(2.9);
      expect(media.seeks(el).at(-1)).toBeLessThan(3.1);
    });

    it('finishes X then 2 as the upscale shortcut and does not move the playhead', async () => {
      const onupscale = vi.fn();
      const el = await openedVideo({ assets: [makeAsset({ upscale }), clipB], onupscale });
      timeline().focus();
      const before = media.seeks(el).length;

      press('x', {}, timeline());
      const digit = press('2', {}, timeline());

      expect(onupscale).toHaveBeenCalledWith(expect.objectContaining({ id: 'out/clip-a.mp4' }), 2, 'seedvr2');
      expect(digit.defaultPrevented).toBe(true);
      expect(media.seeks(el)).toHaveLength(before);
      // The chord is done: the next digit is the player's again.
      press('2', {}, timeline());
      expect(media.seeks(el).length).toBe(before + 1);
    });

    it('finishes X then 2 with the focus outside the player as well', async () => {
      const onupscale = vi.fn();
      const el = await openedVideo({ assets: [makeAsset({ upscale }), clipB], onupscale });
      const before = media.seeks(el).length;

      press('x');
      press('2');

      expect(onupscale).toHaveBeenCalledTimes(1);
      expect(media.seeks(el)).toHaveLength(before);
    });

    it('lets any other digit after X reach the player, which jumps to it', async () => {
      const onupscale = vi.fn();
      const el = await openedVideo({ assets: [makeAsset({ upscale }), clipB], onupscale });
      timeline().focus();

      press('x', {}, timeline());
      const digit = press('5', {}, timeline());

      expect(onupscale).not.toHaveBeenCalled();
      expect(digit.defaultPrevented).toBe(true);
      expect(media.seeks(el).at(-1)).toBeGreaterThan(2.9);
      expect(media.seeks(el).at(-1)).toBeLessThan(3.1);
    });

    it('drops the chord when X waits too long, so the digit jumps', async () => {
      vi.useFakeTimers({ toFake: ['Date'] });
      try {
        const onupscale = vi.fn();
        const el = await openedVideo({ assets: [makeAsset({ upscale }), clipB], onupscale });
        timeline().focus();
        const before = media.seeks(el).length;

        press('x', {}, timeline());
        vi.setSystemTime(Date.now() + CHORD_TIMEOUT_MS + 1);
        press('2', {}, timeline());

        expect(onupscale).not.toHaveBeenCalled();
        expect(media.seeks(el)).toHaveLength(before + 1);
      } finally {
        vi.useRealTimers();
      }
    });
  });

  describe('a video', () => {
    it('plays in the app\'s player in place of the browser\'s controls', async () => {
      const el = await openedVideo();

      expect(el.controls).toBe(false);
      expect(el.getAttribute('src')).toBe('/media/out/clip-a.mp4');
      expect(document.querySelector('[data-media-player]')?.getAttribute('data-size')).toBe('viewer');
    });

    it('builds a new player for each video it pages to', async () => {
      const props = reactiveProps({ assets: [makeAsset(), clipB], currentId: 'out/clip-a.mp4', open: true, onclose: vi.fn(), onnavigate: vi.fn() });
      mountComponent(AssetViewer, props);
      await settle();
      const first = video();

      props.currentId = 'out/clip-b.mp4';
      flushSync();

      expect(video()).not.toBe(first);
      expect(video().getAttribute('src')).toBe('/media/out/clip-b.mp4');
      expect(document.querySelectorAll('video')).toHaveLength(1);
    });

    it('opens with the focus on Close, so the arrows page until the player is engaged', async () => {
      const onnavigate = vi.fn();
      const el = await openedVideo({ onnavigate });
      await settle();

      expect(document.activeElement).toBe(closeButton());
      press('ArrowRight', {}, document.activeElement!);
      expect(onnavigate).toHaveBeenCalledWith(1);
      expect(media.seeks(el)).toHaveLength(0);
    });

    it('plays on Space while the focus is still on Close, and moves the focus to the timeline', async () => {
      const onclose = vi.fn();
      const el = await openedVideo({ onclose });
      await settle();

      const event = press(' ', {}, closeButton());

      expect(event.defaultPrevented).toBe(true);
      expect(media.played).toContain(el);
      expect(document.activeElement).toBe(timeline());
      expect(onclose).not.toHaveBeenCalled();
    });

    it('plays a take on Space while the focus is still on Close, as a video does', async () => {
      const onclose = vi.fn();
      mountViewer({ assets: [makeAudioAsset()], onclose });
      await settle();

      const event = press(' ', {}, closeButton());

      expect(event.defaultPrevented).toBe(true);
      expect(media.played).toContain(document.querySelector('audio'));
      expect(document.activeElement).toBe(timeline());
      expect(onclose).not.toHaveBeenCalled();
    });

    it('leaves Enter on Close to close the viewer', async () => {
      const el = await openedVideo();
      await settle();

      const event = press('Enter', {}, closeButton());

      expect(event.defaultPrevented).toBe(false);
      expect(media.played).not.toContain(el);
    });

    it('leaves Space on Close to the button for an image', async () => {
      mountViewer({ assets: [image, makeAsset()] });
      await settle();

      expect(press(' ', {}, closeButton()).defaultPrevented).toBe(false);
    });

    it('lists the keys a clicked video takes in its shortcuts, and only for a video', async () => {
      await openedVideo();
      press('?');
      expect(document.querySelector('#asset-viewer-shortcuts')?.textContent).toContain('step a frame');
    });

    it('steps the clip with the arrows once the timeline has the focus', async () => {
      const onnavigate = vi.fn();
      const el = await openedVideo({ onnavigate });
      await settle();

      timeline().focus();
      press('ArrowRight', {}, timeline());

      expect(onnavigate).not.toHaveBeenCalled();
      expect(media.seeks(el)).toHaveLength(1);
    });

    it('gives the timeline the focus when the picture is clicked, so the arrows step a frame instead of paging', async () => {
      const onnavigate = vi.fn();
      const el = await openedVideo({ onnavigate });
      closeButton().focus();

      el.click();
      flushSync();
      expect(document.activeElement).toBe(timeline());
      press('ArrowRight', {}, document.activeElement!);

      expect(onnavigate).not.toHaveBeenCalled();
      expect(media.seeks(el).length).toBeGreaterThan(0);
    });

    it('puts the centre button, then the timeline, then the cluster in the tab order of the player', async () => {
      await openedVideo();

      const stops = Array.from(document.querySelectorAll<HTMLElement>('[data-media-player] button, [data-media-player] [role="slider"]')).map((el) => el.getAttribute('aria-label') ?? '');

      expect(stops.slice(0, 3)).toEqual(['Play', 'Playback position', 'Play']);
      expect(stops.indexOf('Playback position')).toBeLessThan(stops.indexOf('Mute'));
    });

    it('opens an image with the focus on Close', async () => {
      mountViewer({ assets: [image, makeAsset()] });
      await settle();

      expect(document.activeElement).toBe(closeButton());
    });

    it('opens a take with the focus on Close', async () => {
      mountViewer({ assets: [makeAudioAsset()] });
      await settle();

      expect(document.activeElement).toBe(closeButton());
    });

    it('leaves the focus where it was when it was outside the player', async () => {
      pagedViewer();
      await settle();
      const close = closeButton();
      close.focus();

      press('ArrowRight', {}, close);
      await settle();

      expect(document.querySelector('video')?.getAttribute('src')).toBe('/media/out/clip-b.mp4');
      expect(document.activeElement).not.toBe(timeline());
    });
  });

  describe('keys', () => {
    it('leaves Space, the arrows, Home and End to the player while it has focus, so the viewer does not page', async () => {
      const onnavigate = vi.fn();
      await openedVideo({ onnavigate });

      for (const key of [' ', 'ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Home', 'End']) press(key, {}, timeline());

      expect(onnavigate).not.toHaveBeenCalled();
    });

    it('pages with the arrows, Home and End when the focus is elsewhere', async () => {
      const onnavigate = vi.fn();
      await openedVideo({ onnavigate });
      (document.activeElement as HTMLElement).blur();

      press('ArrowRight');
      press('End');

      expect(onnavigate).toHaveBeenCalledWith(1);
      expect(onnavigate).toHaveBeenCalledTimes(2);
    });

    it('still runs the viewer\'s shortcuts with the focus in the player', async () => {
      const onreuse = vi.fn();
      await openedVideo({ onreuse });

      press('r', {}, timeline());

      expect(onreuse).toHaveBeenCalledOnce();
    });

    it('extends with E from the player', async () => {
      const onextend = vi.fn();
      await openedVideo({ onextend, assets: [makeAsset({ extend: { available: true, reason: null } })] });

      press('e', {}, timeline());

      expect(onextend).toHaveBeenCalledOnce();
    });

    it('keeps X then 2 for upscale with the focus in the player', async () => {
      const onupscale = vi.fn();
      const upscale = {
        factors: [],
        seedvr2: { options: [{ factor: 2 as const, width: 1536, height: 896, allowed: true as const, reason: null, warning: null }], model: 'SeedVR2 7B fp8', downloaded: true, download_size_label: null },
      };
      await openedVideo({ onupscale, assets: [makeAsset({ upscale })] });

      press('x', {}, timeline());
      press('2', {}, timeline());

      expect(onupscale).toHaveBeenCalledWith(expect.objectContaining({ media_type: 'video' }), 2, 'seedvr2');
    });

    it('leaves Space to a focused button other than Close', async () => {
      const el = await openedVideo();
      const details = document.querySelector<HTMLButtonElement>('.viewer-btn[data-action="details"]')!;
      details.focus();

      const event = press(' ', {}, details);

      expect(event.defaultPrevented).toBe(false);
      expect(el.paused).toBe(true);
    });
  });

  describe('touch', () => {
    it('pages on a swipe that starts on the picture', async () => {
      touchLayer.coarse = true;
      const onnavigate = vi.fn();
      await openedVideo({ onnavigate });

      touch(video(), 'touchstart', 300, 200);
      touch(video(), 'touchmove', 220, 202);
      touch(video(), 'touchmove', 20, 204);
      touch(video(), 'touchend', 20, 204);

      expect(onnavigate).toHaveBeenCalledWith(1);
    });

    it('does not page on a drag that starts on the timeline or the controls', async () => {
      touchLayer.coarse = true;
      const onnavigate = vi.fn();
      await openedVideo({ onnavigate });

      for (const node of [timeline(), document.querySelector('.cluster')!, document.querySelector('.centre')!]) {
        touch(node, 'touchstart', 300, 200);
        touch(node, 'touchmove', 220, 202);
        touch(node, 'touchmove', 20, 204);
        touch(node, 'touchend', 20, 204);
      }

      expect(onnavigate).not.toHaveBeenCalled();
    });

    it('lets the player\'s own tap play the video, and leaves the viewer\'s chrome as it is', async () => {
      touchLayer.coarse = true;
      vi.useFakeTimers();
      mountViewer();
      await vi.advanceTimersByTimeAsync(0);
      media.load(video(), 6);
      flushSync();
      const player = document.querySelector<HTMLElement>('[data-media-player]')!;

      player.dispatchEvent(new PointerEvent('pointerdown', { pointerType: 'touch', bubbles: true }));
      touch(video(), 'touchstart', 100, 100);
      touch(video(), 'touchend', 100, 100);
      video().click();
      await vi.advanceTimersByTimeAsync(400);
      flushSync();

      expect(video().paused).toBe(false);
      expect(viewer().getAttribute('data-chrome')).toBe('shown');
      vi.useRealTimers();
    });

    it('brings the bar and the strip back on a tap on a video, when a tap on an earlier asset hid them', async () => {
      touchLayer.coarse = true;
      vi.useFakeTimers();
      const props = reactiveProps({ assets: [image, makeAsset()], currentId: 'out/still.png', open: true, onclose: vi.fn(), onnavigate: (_index: number): void => undefined });
      props.onnavigate = (index: number) => { props.currentId = props.assets[index].id; };
      mountComponent(AssetViewer, props);
      const stage = document.querySelector('.viewer-stage')!;
      touch(stage, 'touchstart', 100, 100);
      touch(stage, 'touchend', 100, 100);
      await vi.advanceTimersByTimeAsync(400);
      flushSync();
      expect(viewer().getAttribute('data-chrome')).toBe('hidden');

      props.currentId = 'out/clip-a.mp4';
      flushSync();
      const player = document.querySelector<HTMLElement>('[data-media-player]')!;
      player.dispatchEvent(new PointerEvent('pointerdown', { pointerType: 'touch', bubbles: true }));
      touch(video(), 'touchstart', 100, 100);
      touch(video(), 'touchend', 100, 100);
      flushSync();

      expect(viewer().getAttribute('data-chrome')).toBe('shown');
      vi.useRealTimers();
    });

    it('still toggles the chrome of an image on a tap', async () => {
      touchLayer.coarse = true;
      vi.useFakeTimers();
      mountViewer({ assets: [image] });
      const stage = document.querySelector('.viewer-stage')!;

      touch(stage, 'touchstart', 100, 100);
      touch(stage, 'touchend', 100, 100);
      await vi.advanceTimersByTimeAsync(400);
      flushSync();

      expect(viewer().getAttribute('data-chrome')).toBe('hidden');
      vi.useRealTimers();
    });
  });

  describe('the chrome', () => {
    it('fades the player\'s controls while the video plays and leaves the title, Close, arrows and strip alone', async () => {
      const el = await openedVideo();
      vi.useFakeTimers();
      const player = document.querySelector<HTMLElement>('[data-media-player]')!;

      el.play();
      flushSync();
      await vi.advanceTimersByTimeAsync(2700);
      flushSync();

      expect(player.getAttribute('data-chrome')).toBe('hidden');
      expect(viewer().getAttribute('data-chrome')).toBe('shown');
      vi.useRealTimers();
    });
  });

  describe('a take', () => {
    it('plays in the same player, as a waveform above its words', async () => {
      mountViewer({ assets: [makeAudioAsset()] });
      await settle();

      const player = document.querySelector('[data-media-player]');
      expect(player?.getAttribute('data-kind')).toBe('audio');
      expect(player?.getAttribute('data-size')).toBe('viewer');
      expect(document.querySelector('.viewer-audio [data-media-player]')).not.toBeNull();
    });
  });
});
