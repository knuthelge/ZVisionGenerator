import { flushSync } from 'svelte';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { EXTEND_SHORT_REASON } from '$lib/state/framePoint';
import { runFrameGrab } from '$lib/state/frameGrab.svelte';
import { mediaVolume } from '$lib/state/mediaVolume.svelte';
import { videoFacts } from '$lib/state/videoFacts.svelte';
import type { GalleryAsset } from '$lib/types';

import AssetTile from './AssetTile.svelte';
import { installFakeMedia, type FakeMedia } from '../../../test-utils/media';
import { mountComponent, settle } from '../../../test-utils/mount';

const mocks = vi.hoisted(() => ({ browseInfo: vi.fn(), grabFrame: vi.fn() }));
vi.mock('$lib/api/browse', async (importOriginal) => ({ ...(await importOriginal<typeof import('$lib/api/browse')>()), browseInfo: mocks.browseInfo }));
vi.mock('$lib/api/videoFrame', () => ({ grabFrame: mocks.grabFrame }));

// 6 s at 24 fps: 144 frames, the last is frame 143.
const FACTS = { fps: 24, frames: 144, start: 0 };
const LAST = FACTS.frames - 1;
const BOX = { left: 100, top: 700, width: 400, height: 46, right: 500, bottom: 746, x: 100, y: 700, toJSON: () => ({}) };
const xOf = (frame: number): number => 100 + (frame / LAST) * 400;

const clip: GalleryAsset = {
  id: 'out/c.mp4', url: '/media/out/c.mp4', thumbnail_url: '/thumbs/c.jpg', filename: 'c.mp4', created_at: '2026-10-03T12:00:00Z', project: null, workflow: 'txt2vid', prompt: 'A clip', model: 'ltx',
  media_type: 'video', file_path: '/outputs/out/c.mp4', has_reusable_config: true, reuse_workspace_url: '', extend: { available: true, reason: null },
};

describe('AssetTile, the stage video', () => {
  let media: FakeMedia;
  const onextend = vi.fn();
  const onuseframe = vi.fn();

  beforeEach(() => {
    media = installFakeMedia();
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(null);
    videoFacts.reset();
    mediaVolume.reset();
    onextend.mockReset();
    onuseframe.mockReset();
    mocks.browseInfo.mockReset().mockResolvedValue({ ...FACTS, seconds: 6 });
    mocks.grabFrame.mockReset();
  });

  const timeline = () => document.querySelector<HTMLElement>('[data-testid="media-timeline"]')!;
  const more = () => document.querySelector<HTMLButtonElement>('button[aria-label="More actions for c.mp4"]')!;

  async function stage(props: Record<string, unknown> = {}): Promise<void> {
    mountComponent(AssetTile, { asset: clip, density: 'stage', onextend, onuseframe, ...props });
    await settle();
    const video = document.querySelector('video')!;
    media.load(video, 6);
    flushSync();
    timeline().getBoundingClientRect = () => BOX;
  }

  function seekTo(frame: number): void {
    for (const type of ['pointerdown', 'pointerup']) {
      timeline().dispatchEvent(new PointerEvent(type, { clientX: xOf(frame), clientY: 720, pointerType: 'mouse', pointerId: 1, button: 0, bubbles: true }));
    }
    media.land(document.querySelector('video')!);
    flushSync();
  }

  /** The menu's items: label, then whether it is disabled. */
  async function menu(): Promise<{ items: string[]; item: (id: string) => HTMLElement | null }> {
    more().click();
    await settle();
    const root = document.querySelector('[role="menu"]')!;
    return {
      items: Array.from(root.querySelectorAll('.action-menu-heading, [role="menuitem"]')).map((el) => el.textContent?.trim() ?? ''),
      item: (id) => root.querySelector<HTMLElement>(`[data-action="${id}"]`),
    };
  }

  it('offers plain Extend and no Use as before the video has been paused on a frame', async () => {
    await stage();

    const { items, item } = await menu();

    expect(items).not.toContain('Use as');
    expect(item('extend')?.textContent?.trim()).toBe('Extend');
    expect(item('frame-first')).toBeNull();
  });

  it('offers Use as and Extend from the frame it is paused on', async () => {
    await stage();
    seekTo(72);

    const { items, item } = await menu();

    expect(items[0]).toBe('Use as');
    expect(items[1]).toMatch(/^First frame/);
    expect(items[2]).toMatch(/^Last frame/);
    expect(item('extend')?.textContent).toContain('Extend from 3.0 s');
    item('frame-last')!.click();
    flushSync();
    expect(onuseframe).toHaveBeenCalledWith(expect.objectContaining({ id: 'out/c.mp4' }), expect.objectContaining({ seconds: 3 }), 'last');
  });

  it('refuses Extend on a frame too early, with the reason as the item\'s description, and still offers Use as', async () => {
    await stage();
    seekTo(10);

    const { item } = await menu();

    expect(item('extend')?.getAttribute('aria-disabled')).toBe('true');
    expect(item('extend')?.textContent).toContain(EXTEND_SHORT_REASON);
    expect(item('frame-first')?.getAttribute('aria-disabled')).toBeNull();
  });

  const menuButton = () => document.querySelector<HTMLButtonElement>('[data-testid="frame-menu"]');
  const menuRow = (id: string) => document.querySelector<HTMLElement>(`[role="menu"] [data-action="${id}"]`);

  it('shows the menu button on the frame it is paused on, and none at the start or while it plays', async () => {
    await stage();
    expect(menuButton()).toBeNull();

    seekTo(72);
    expect(menuButton()?.textContent).toContain('3.0 s');

    document.querySelector('video')!.play();
    flushSync();
    expect(menuButton()).toBeNull();
  });

  it('opens the frame menu from its button with the rows the menu offers, and runs them on the frame', async () => {
    await stage();
    seekTo(72);
    menuButton()!.click();
    flushSync();

    expect(Array.from(document.querySelectorAll('[role="menu"] [data-action]')).map((row) => row.getAttribute('data-action'))).toEqual(['extend', 'frame-first', 'frame-last']);
    expect(menuRow('extend')?.textContent).toContain('Extend from 3.0 s');
    menuRow('extend')!.click();
    menuRow('frame-first')!.click();

    expect(onextend).toHaveBeenCalledWith(expect.objectContaining({ id: 'out/c.mp4' }), expect.objectContaining({ seconds: 3 }));
    expect(onuseframe).toHaveBeenCalledWith(expect.objectContaining({ id: 'out/c.mp4' }), expect.objectContaining({ seconds: 3 }), 'video');
  });

  it('refuses Extend on a frame too early in the frame menu too, and keeps the frame rows', async () => {
    await stage();
    seekTo(10);
    menuButton()!.click();
    flushSync();

    expect(menuRow('extend')?.getAttribute('aria-disabled')).toBe('true');
    expect(menuRow('extend')?.textContent).toContain(EXTEND_SHORT_REASON);
    expect(menuRow('frame-last')?.getAttribute('aria-disabled')).toBeNull();
  });

  it('shows no menu button when the tile has no handler for any row', async () => {
    await stage({ onextend: undefined, onuseframe: undefined });
    seekTo(72);

    expect(menuButton()).toBeNull();
  });

  it('keeps the ⋯ menu as it was beside the menu button', async () => {
    await stage();
    seekTo(72);

    const { items } = await menu();

    expect(items[0]).toBe('Use as');
    expect(items[1]).toMatch(/^First frame/);
    expect(items[2]).toMatch(/^Last frame/);
    expect(menuButton()).not.toBeNull();
  });

  it('lets go of the frame when it plays', async () => {
    await stage();
    seekTo(72);
    document.querySelector('video')!.play();
    flushSync();

    const { items } = await menu();

    expect(items).not.toContain('Use as');
  });

  it('shows a spinner on a closed ⋯ while a frame is extracted, and the icon again after', async () => {
    await stage();
    seekTo(72);
    let finish: (value: unknown) => void = () => undefined;
    mocks.grabFrame.mockReturnValue(new Promise((resolve) => { finish = resolve; }));

    const run = runFrameGrab(clip, { time: 3, seconds: 3, tooShort: false, atEnd: false }, 'video', vi.fn());
    flushSync();
    expect(more().getAttribute('aria-busy')).toBe('true');
    expect(more().querySelector('.spinner')).not.toBeNull();

    finish({ path: '/up/c_3.0s.png' });
    await run;
    flushSync();

    expect(more().getAttribute('aria-busy')).toBe('false');
    expect(more().querySelector('.spinner')).toBeNull();
  });
});
