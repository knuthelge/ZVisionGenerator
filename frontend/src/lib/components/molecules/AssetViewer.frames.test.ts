import { flushSync } from 'svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { GalleryAsset } from '$lib/types';
import AssetViewer from './AssetViewer.svelte';
import { EXTEND_SHORT_REASON, stepTarget } from '$lib/state/framePoint';
import { mediaVolume } from '$lib/state/mediaVolume.svelte';
import { toasts } from '$lib/state/toasts.svelte';
import { videoFacts } from '$lib/state/videoFacts.svelte';
import { installFakeMedia, type FakeMedia } from '../../../test-utils/media';
import { mountComponent, settle } from '../../../test-utils/mount';
import { reactiveProps } from '../../../test-utils/reactiveProps.svelte';

const browse = vi.hoisted(() => ({ browseInfo: vi.fn() }));
vi.mock('$lib/api/browse', async (importOriginal) => ({ ...(await importOriginal<typeof import('$lib/api/browse')>()), browseInfo: browse.browseInfo }));

// 6 s at 24 fps: 144 frames, the last is frame 143.
const FACTS = { fps: 24, frames: 144, start: 0 };
const LAST = FACTS.frames - 1;
const BOX = { left: 100, top: 700, width: 400, height: 46, right: 500, bottom: 746, x: 100, y: 700, toJSON: () => ({}) };
const xOf = (frame: number): number => 100 + (frame / LAST) * 400;

function makeVideo(overrides: Partial<GalleryAsset> = {}): GalleryAsset {
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
    extend: { available: true, reason: null },
    ...overrides,
  };
}

// Where the viewer stands, one per row of the state table that a paused video can be in.
type Standing = { name: string; row: string; stand: (api: Api) => void; point: null | { frame: number; atEnd: boolean; tooShort: boolean } };
interface Api {
  seekTo: (frame: number) => void;
  play: () => void;
  finish: () => void;
}

describe('AssetViewer, a paused frame', () => {
  let media: FakeMedia;
  const onextend = vi.fn();
  const onuseframe = vi.fn();

  beforeEach(() => {
    localStorage.clear();
    media = installFakeMedia();
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(null);
    videoFacts.reset();
    mediaVolume.reset();
    toasts.length = 0;
    onextend.mockReset();
    onuseframe.mockReset();
    browse.browseInfo.mockReset().mockResolvedValue({ ...FACTS, seconds: 6 });
  });
  afterEach(() => {
    vi.restoreAllMocks();
    Reflect.deleteProperty(document, 'fullscreenElement');
  });

  const video = () => document.querySelector<HTMLVideoElement>('video')!;
  const timeline = () => document.querySelector<HTMLElement>('[data-media-player] [role="slider"][aria-label="Playback position"]')!;
  const menuButton = () => document.querySelector<HTMLButtonElement>('[data-testid="frame-menu"]');
  const menu = () => document.querySelector<HTMLElement>('[role="menu"]');
  const menuRow = (id: string) => menu()?.querySelector<HTMLElement>(`[data-action="${id}"]`) ?? null;

  function press(key: string, init: KeyboardEventInit = {}, on: EventTarget = document): KeyboardEvent {
    const event = new KeyboardEvent('keydown', { key, bubbles: true, cancelable: true, ...init });
    on.dispatchEvent(event);
    flushSync();
    return event;
  }

  function pointer(type: string, x: number): void {
    timeline().dispatchEvent(new PointerEvent(type, { clientX: x, clientY: 720, pointerType: 'mouse', pointerId: 1, button: 0, bubbles: true }));
    flushSync();
  }

  const api: Api = {
    seekTo(frame) {
      pointer('pointerdown', xOf(frame));
      pointer('pointerup', xOf(frame));
      media.land(video());
      flushSync();
    },
    play() {
      video().play();
      flushSync();
    },
    finish() {
      media.finish(video());
      flushSync();
    },
  };

  /** Open a viewer on the clip, give its player the metadata, and wait for the frame facts. */
  async function open(props: Record<string, unknown> = {}, asset: GalleryAsset = makeVideo()): Promise<void> {
    mountComponent(AssetViewer, { assets: [asset], currentId: asset.id, open: true, onclose: vi.fn(), onnavigate: vi.fn(), onextend, onuseframe, ...props });
    await settle();
    media.load(video(), 6);
    flushSync();
    timeline().getBoundingClientRect = () => BOX;
  }

  // Rows 1 to 6 of the state table, each reached the way a user reaches it.
  const STANDINGS: Standing[] = [
    { name: 'not yet played, at 0', row: '1', stand: () => undefined, point: null },
    { name: 'playing', row: '2', stand: ({ seekTo, play }) => { seekTo(72); play(); }, point: null },
    { name: 'paused mid-clip', row: '3', stand: ({ seekTo }) => seekTo(72), point: { frame: 72, atEnd: false, tooShort: false } },
    { name: 'paused before the 25th frame', row: '4', stand: ({ seekTo }) => seekTo(10), point: { frame: 10, atEnd: false, tooShort: true } },
    { name: 'paused on the last frame', row: '5', stand: ({ seekTo }) => seekTo(LAST), point: { frame: LAST, atEnd: true, tooShort: false } },
    { name: 'ended', row: '6', stand: ({ play, finish }) => { play(); finish(); }, point: { frame: LAST, atEnd: true, tooShort: false } },
  ];

  describe.each(STANDINGS)('$name (row $row)', ({ stand, point }) => {
    const selection = point !== null;
    const reads = !point || point.atEnd ? 'Extend the clip' : `Extend from ${(point.frame / FACTS.fps).toFixed(1)} s`;

    it(selection ? 'shows the menu button and, opened, the menu\'s three rows' : 'shows no menu button', async () => {
      await open();
      stand(api);

      if (!selection) {
        expect(menuButton()).toBeNull();
        return;
      }
      expect(menuButton()).not.toBeNull();
      menuButton()!.click();
      flushSync();
      expect(Array.from(menu()!.querySelectorAll('[data-action]')).map((row) => row.getAttribute('data-action'))).toEqual(['extend', 'frame-first', 'frame-last']);
      expect(menuRow('extend')!.textContent).toContain(reads);
      expect(menuRow('extend')!.getAttribute('aria-disabled')).toBe(point.tooShort ? 'true' : null);
      expect(menuRow('extend')!.textContent?.includes(EXTEND_SHORT_REASON)).toBe(point.tooShort);
      expect(menuRow('frame-first')!.getAttribute('aria-disabled')).toBeNull();
      expect(menuRow('frame-last')!.getAttribute('aria-disabled')).toBeNull();
    });

    it('runs the menu\'s Extend row from the frame, and refuses it when the frame is too early', async () => {
      await open();
      stand(api);
      if (!selection) return;
      menuButton()!.click();
      flushSync();

      menuRow('extend')!.click();

      if (point.tooShort) {
        expect(onextend).not.toHaveBeenCalled();
      } else {
        expect(onextend).toHaveBeenCalledWith(expect.objectContaining({ id: 'out/clip-a.mp4' }), expect.objectContaining({ seconds: point.frame / FACTS.fps, atEnd: point.atEnd }));
      }
    });

    it('takes the frame as the first or last frame from the menu', async () => {
      await open();
      stand(api);
      if (!selection) return;
      menuButton()!.click();
      flushSync();

      menuRow('frame-first')!.click();
      menuRow('frame-last')!.click();

      expect(onuseframe).toHaveBeenNthCalledWith(1, expect.anything(), expect.objectContaining({ seconds: point.frame / FACTS.fps }), 'video');
      expect(onuseframe).toHaveBeenNthCalledWith(2, expect.anything(), expect.objectContaining({ seconds: point.frame / FACTS.fps }), 'last');
    });

    it('keeps E as Extend of the whole clip and the menu as it was', async () => {
      await open();
      stand(api);

      press('e');

      expect(onextend).toHaveBeenCalledWith(expect.objectContaining({ id: 'out/clip-a.mp4' }));
      expect(onextend.mock.calls[0]).toHaveLength(1);
      press('1');
      press('2');
      expect(onuseframe).not.toHaveBeenCalled();
    });
  });

  describe('with the frame menu open', () => {
    // The menu sits in the page's body, outside the player, and takes the focus: the viewer's keys must not fire from it.
    async function openMenu(props: Record<string, unknown> = {}): Promise<HTMLElement> {
      const onnavigate = vi.fn();
      await open({ onnavigate, ...props });
      api.seekTo(72);
      menuButton()!.dispatchEvent(new MouseEvent('click', { bubbles: true, detail: 0 }));
      await settle();
      expect(document.activeElement).toBe(menuRow('extend'));
      return document.activeElement as HTMLElement;
    }

    it('steps the frame with the arrows and stays open, instead of paging to the next asset', async () => {
      const onnavigate = vi.fn();
      const focused = await openMenu({ onnavigate, assets: [makeVideo(), makeVideo({ id: 'out/clip-b.mp4', filename: 'clip-b.mp4' })] });
      const before = menuButton()!.getAttribute('aria-label');

      const event = press('ArrowRight', { shiftKey: true }, focused);
      media.land(video());
      flushSync();

      expect(event.defaultPrevented).toBe(true);
      expect(onnavigate).not.toHaveBeenCalled();
      expect(menu()).not.toBeNull();
      expect(menuButton()!.getAttribute('aria-label')).not.toBe(before);
    });

    it('does not run the viewer shortcuts E, R and Delete', async () => {
      const onreuse = vi.fn();
      const ondelete = vi.fn();
      const focused = await openMenu({ onreuse, ondelete });

      for (const key of ['e', 'r', 'Delete', 'Backspace', 'd', 'c', 'i']) press(key, {}, focused);

      expect(onextend).not.toHaveBeenCalled();
      expect(onreuse).not.toHaveBeenCalled();
      expect(ondelete).not.toHaveBeenCalled();
      expect(menu()).not.toBeNull();
    });

    it('closes on Escape, and the focus is in the timeline so Space plays', async () => {
      const focused = await openMenu();

      press('Escape', {}, focused);
      await settle();

      expect(menu()).toBeNull();
      expect(document.activeElement).toBe(timeline());
      press(' ', {}, timeline());
      expect(video().paused).toBe(false);
    });
  });

  describe('a model that cannot use a frame', () => {
    const useAs = { video: { disabled: true, hint: 'LTX can\'t start from an image' }, last: { disabled: true, hint: 'LTX has no last frame' } };

    it('keeps the rows with the reason in place, where a click does nothing', async () => {
      await open({ useAs });
      api.seekTo(72);
      menuButton()!.click();
      flushSync();

      expect(menuRow('frame-first')!.getAttribute('aria-disabled')).toBe('true');
      expect(menuRow('frame-first')!.textContent).toContain('LTX can\'t start from an image');
      expect(menuRow('frame-last')!.textContent).toContain('LTX has no last frame');
      menuRow('frame-first')!.click();
      menuRow('frame-last')!.click();

      expect(onuseframe).not.toHaveBeenCalled();
    });
  });

  describe('Extend that is not available', () => {
    it('reads Extend, disabled with its reason, and never says from', async () => {
      await open({}, makeVideo({ extend: { available: false, reason: 'An upscaled video can\'t be extended' } }));
      api.seekTo(72);
      menuButton()!.click();
      flushSync();

      expect(menuRow('extend')!.getAttribute('aria-disabled')).toBe('true');
      expect(menuRow('extend')!.textContent).toContain('Extend');
      expect(menuRow('extend')!.textContent).not.toContain('Extend from');
      expect(menuRow('extend')!.textContent).toContain('An upscaled video can\'t be extended');
      expect(menuRow('frame-first')!.getAttribute('aria-disabled')).toBeNull();
    });

    it('has no menu button without a handler for any row', async () => {
      await open({ onextend: undefined, onuseframe: undefined });
      api.seekTo(72);

      expect(menuButton()).toBeNull();
    });

    it('offers only Extend where there is no file to take a frame from', async () => {
      await open({}, makeVideo({ file_path: undefined }));
      api.seekTo(72);

      expect(menuButton()).toBeNull();
    });
  });

  describe('without frame facts', () => {
    it('shows no menu button, and E still extends the whole clip', async () => {
      browse.browseInfo.mockRejectedValue(new Error('no facts'));
      await open();
      api.seekTo(72);

      expect(menuButton()).toBeNull();
      press('e');

      expect(onextend.mock.calls[0]).toHaveLength(1);
    });
  });

  describe('fullscreen', () => {
    it('puts the menu button away', async () => {
      await open();
      api.seekTo(72);
      expect(menuButton()).not.toBeNull();
      Object.defineProperty(document, 'fullscreenElement', { configurable: true, get: () => document.querySelector('[data-media-player]') });
      document.dispatchEvent(new Event('fullscreenchange'));
      flushSync();

      expect(menuButton()).toBeNull();
    });
  });

  describe('paging', () => {
    it('clears the frame at once, so the next clip starts without a menu button', async () => {
      const next = makeVideo({ id: 'out/clip-b.mp4', url: '/media/out/clip-b.mp4', filename: 'clip-b.mp4', file_path: '/outputs/out/clip-b.mp4' });
      const props = { assets: [makeVideo(), next], currentId: 'out/clip-a.mp4', open: true, onclose: vi.fn(), onnavigate: vi.fn(), onextend, onuseframe };
      const live = reactiveProps({ ...props, onnavigate: (_index: number): void => undefined });
      live.onnavigate = (index: number) => { live.currentId = live.assets[index].id; };
      mountComponent(AssetViewer, live);
      await settle();
      media.load(video(), 6);
      flushSync();
      timeline().getBoundingClientRect = () => BOX;
      api.seekTo(72);
      expect(menuButton()).not.toBeNull();

      live.currentId = 'out/clip-b.mp4';
      flushSync();

      expect(menuButton()).toBeNull();
    });
  });

  it('keeps the frame target\'s time on the point it hands over', async () => {
    await open();
    api.seekTo(72);
    menuButton()!.click();
    flushSync();

    menuRow('frame-last')!.click();

    expect(onuseframe.mock.calls[0][1].time).toBeCloseTo(stepTarget(72, FACTS), 6);
  });
});
