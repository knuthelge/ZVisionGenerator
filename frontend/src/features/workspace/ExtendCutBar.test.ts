import { flushSync } from 'svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { videoSprite } from '$lib/state/videoSprite.svelte';
import { sourceInfo } from '$lib/state/extend.fixtures';
import { mountComponent, settle } from '../../test-utils/mount';
import ExtendCutBar from './ExtendCutBar.svelte';

vi.mock('$lib/api/browse', async () => (await import('../../test-utils/workspaceMocks')).browseApiMocks);

// 144 frames at 24 fps (6.0 s), the first frame at 0 s: frame f is at f / 24 s.
const info = sourceInfo({ fps: 24, frame_count: 144, video_start: 0, extend_frames: 48 });
const PATH = '/clips/walk.mp4';
const BOX = { left: 100, top: 0, width: 400, height: 48, right: 500, bottom: 48, x: 100, y: 0, toJSON: () => ({}) } as DOMRect;
// The pixel of the bar where the handle keeps `keep` frames (the last frame is at the right edge).
const xOf = (keep: number) => BOX.left + ((keep - 1) / 143) * BOX.width;

describe('ExtendCutBar', () => {
  let onchange = vi.fn();

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  beforeEach(() => {
    videoSprite.reset();
    onchange = vi.fn();
  });

  function show(props: Record<string, unknown> = {}) {
    // Defined, not spread, so a getter and setter pair (a binding) stays one.
    const all = { info, path: PATH, extendFrom: null, busy: false, onchange };
    Object.defineProperties(all, Object.getOwnPropertyDescriptors(props));
    const mounted = mountComponent(ExtendCutBar, all);
    const bar = mounted.target.querySelector<HTMLElement>('[data-testid="extend-cut-bar"]')!;
    bar.getBoundingClientRect = () => BOX;
    return { ...mounted, bar };
  }

  const keeps = () => document.body.querySelector('[data-testid="extend-keeps"]')?.textContent;
  const reason = () => document.body.querySelector('[data-testid="extend-cut-reason"]')?.textContent;

  function pointer(bar: HTMLElement, type: string, x: number, init: PointerEventInit = {}): void {
    bar.dispatchEvent(new PointerEvent(type, { clientX: x, clientY: 20, pointerType: 'mouse', pointerId: 1, button: 0, bubbles: true, ...init }));
    flushSync();
  }

  function key(bar: HTMLElement, type: 'keydown' | 'keyup', name: string, init: KeyboardEventInit = {}): KeyboardEvent {
    const event = new KeyboardEvent(type, { key: name, bubbles: true, cancelable: true, ...init });
    bar.dispatchEvent(event);
    flushSync();
    return event;
  }

  it('is a slider named Continue from, with the frames as its range', () => {
    const { bar } = show();

    expect(bar.getAttribute('role')).toBe('slider');
    expect(bar.getAttribute('aria-label')).toBe('Continue from');
    expect(bar.getAttribute('tabindex')).toBe('0');
    expect(bar.getAttribute('aria-valuemin')).toBe('25');
    expect(bar.getAttribute('aria-valuemax')).toBe('144');
    expect(bar.getAttribute('aria-valuenow')).toBe('144');
    expect(bar.getAttribute('aria-valuetext')).toBe('Whole clip');
  });

  it('keeps the whole clip with no shade at the end, and shades what is dropped at a point', () => {
    const whole = show();
    expect(keeps()).toBe('Keeps the whole clip');
    expect(whole.target.querySelector('[data-testid="extend-cut-dropped"]')).toBeNull();

    const point = show({ extendFrom: 76 / 24 });
    expect(point.bar.getAttribute('aria-valuetext')).toBe('3.2 s of 6.0 s');
    expect(point.target.querySelector('[data-testid="extend-cut-dropped"]')).not.toBeNull();
    expect(point.target.querySelector<HTMLElement>('[data-testid="extend-cut-dropped"]')?.style.left).toBe(`${(76 / 143) * 100}%`);
  });

  it('reads Keeps with the point and the length of the clip', () => {
    show({ extendFrom: 76 / 24 });
    expect(keeps()).toBe('Keeps 3.2 s of 6.0 s');
  });

  it('moves the handle on a press and sets the point on release, not before', () => {
    const { bar } = show();

    pointer(bar, 'pointerdown', xOf(77));
    expect(keeps()).toBe('Keeps 3.2 s of 6.0 s');
    expect(onchange).not.toHaveBeenCalled();

    pointer(bar, 'pointermove', xOf(100));
    expect(keeps()).toBe('Keeps 4.1 s of 6.0 s');
    expect(onchange).not.toHaveBeenCalled();

    pointer(bar, 'pointerup', xOf(100));
    expect(onchange).toHaveBeenCalledTimes(1);
    expect(onchange.mock.calls[0][0]).toBeCloseTo(99 / 24, 6);
  });

  it('reports the frames it keeps while dragging, and clears them on release', () => {
    let dragKeep: number | null = null;
    const { bar } = show({ get dragKeep() { return dragKeep; }, set dragKeep(value: number | null) { dragKeep = value; } });

    pointer(bar, 'pointerdown', xOf(60));
    expect(dragKeep).toBe(60);
    pointer(bar, 'pointerup', xOf(60));
    expect(dragKeep).toBeNull();
  });

  it('keeps the whole clip when the handle is dragged to the end', () => {
    const { bar } = show({ extendFrom: 76 / 24 });

    pointer(bar, 'pointerdown', xOf(77));
    pointer(bar, 'pointermove', BOX.right + 40);
    pointer(bar, 'pointerup', BOX.right + 40);

    expect(onchange).toHaveBeenCalledWith(null);
  });

  it('stops at 25 frames and says why, and says nothing elsewhere', () => {
    const { bar } = show();
    expect(reason()).toBe('');

    pointer(bar, 'pointerdown', BOX.left);
    expect(bar.getAttribute('aria-valuenow')).toBe('25');
    expect(reason()).toContain('25 frames');
    pointer(bar, 'pointerup', BOX.left);
    expect(onchange.mock.calls[0][0]).toBeCloseTo(24 / 24, 6);
  });

  it('lets a finger scroll the page: a touch that is cancelled moves nothing', () => {
    const { bar } = show();

    pointer(bar, 'pointerdown', xOf(60), { pointerType: 'touch' });
    pointer(bar, 'pointercancel', xOf(60), { pointerType: 'touch' });

    expect(onchange).not.toHaveBeenCalled();
    expect(keeps()).toBe('Keeps the whole clip');
  });

  it('takes a tap of a finger as a press where it landed', () => {
    const { bar } = show();

    pointer(bar, 'pointerdown', xOf(77), { pointerType: 'touch' });
    pointer(bar, 'pointerup', xOf(77), { pointerType: 'touch' });

    expect(onchange.mock.calls[0][0]).toBeCloseTo(76 / 24, 6);
  });

  it('moves a frame at a time by the arrow keys and sets the point when the key lifts', () => {
    const { bar } = show({ extendFrom: 76 / 24 });

    expect(key(bar, 'keydown', 'ArrowLeft').defaultPrevented).toBe(true);
    key(bar, 'keydown', 'ArrowLeft');
    expect(onchange).not.toHaveBeenCalled();
    expect(bar.getAttribute('aria-valuenow')).toBe('75');

    key(bar, 'keyup', 'ArrowLeft');
    expect(onchange).toHaveBeenCalledTimes(1);
    expect(onchange.mock.calls[0][0]).toBeCloseTo(74 / 24, 6);
  });

  it('moves five frames with Shift', () => {
    const { bar } = show({ extendFrom: 76 / 24 });

    key(bar, 'keydown', 'ArrowRight', { shiftKey: true });
    key(bar, 'keyup', 'ArrowRight');

    expect(onchange.mock.calls[0][0]).toBeCloseTo(81 / 24, 6);
  });

  it('goes to 25 frames on Home and to the whole clip on End', () => {
    const { bar } = show({ extendFrom: 76 / 24 });

    key(bar, 'keydown', 'Home');
    key(bar, 'keyup', 'Home');
    expect(onchange.mock.calls[0][0]).toBeCloseTo(24 / 24, 6);

    key(bar, 'keydown', 'End');
    key(bar, 'keyup', 'End');
    expect(onchange.mock.calls[1][0]).toBeNull();
  });

  it('does not go past 25 frames or the end', () => {
    const { bar } = show({ extendFrom: 24 / 24 });
    key(bar, 'keydown', 'ArrowLeft', { shiftKey: true });
    expect(bar.getAttribute('aria-valuenow')).toBe('25');
    key(bar, 'keyup', 'ArrowLeft');

    key(bar, 'keydown', 'End');
    key(bar, 'keydown', 'ArrowRight');
    expect(bar.getAttribute('aria-valuenow')).toBe('144');
  });

  it('leaves the other keys to the page', () => {
    const { bar } = show();

    expect(key(bar, 'keydown', 'Tab').defaultPrevented).toBe(false);
    expect(key(bar, 'keydown', 'ArrowLeft', { ctrlKey: true }).defaultPrevented).toBe(false);
  });

  it('does not move while a job is being submitted', () => {
    const { bar } = show({ busy: true });

    pointer(bar, 'pointerdown', xOf(60));
    key(bar, 'keydown', 'ArrowLeft');

    expect(onchange).not.toHaveBeenCalled();
    expect(bar.getAttribute('aria-disabled')).toBe('true');
    expect(bar.getAttribute('tabindex')).toBe('-1');
  });

  it('draws its cells from the contact sheet once it has loaded, and is flat before', async () => {
    class SheetImage {
      naturalWidth = 1760;
      naturalHeight = 990;
      onload: (() => void) | null = null;
      onerror: (() => void) | null = null;
      set src(_url: string) { queueMicrotask(() => this.onload?.()); }
    }
    vi.stubGlobal('Image', SheetImage);
    const { target } = show();
    const cells = () => Array.from(target.querySelectorAll<HTMLElement>('.cell'));

    expect(cells().length).toBeGreaterThanOrEqual(4);
    expect(cells().every((cell) => cell.style.backgroundImage === '')).toBe(true);

    await settle();

    expect(cells().every((cell) => cell.style.backgroundImage.includes('/sprite/source_video/clips/walk.mp4'))).toBe(true);
    // Tile 0 (the first frame) is the top row of the sheet; the last cell's tile (120) is its bottom row.
    expect(cells()[0].style.backgroundPosition).toMatch(/ 0px$/);
    expect(cells().at(-1)!.style.backgroundPosition).not.toMatch(/ 0px$/);
  });

  it('asks for the contact sheet of the source once', () => {
    const ensure = vi.spyOn(videoSprite, 'ensure');
    show();
    show();

    expect(ensure).toHaveBeenCalledWith(PATH);
  });
});
