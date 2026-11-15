import { beforeEach, describe, expect, it, vi } from 'vitest';
import { flushSync } from 'svelte';

import type { Crop } from '$lib/state/adjustGeometry';
import { mountComponent } from '../../../../test-utils/mount';
import CropFrame from './CropFrame.svelte';

const FRAME = { width: 800, height: 600 };

function pointer(node: Element, type: string, x: number, y: number, pointerId = 1): void {
  const event = new MouseEvent(type, { bubbles: true, cancelable: true, clientX: x, clientY: y, button: 0 });
  Object.defineProperty(event, 'pointerType', { value: 'mouse' });
  Object.defineProperty(event, 'pointerId', { value: pointerId });
  node.dispatchEvent(event);
  flushSync();
}

describe('CropFrame', () => {
  let onchange: ReturnType<typeof vi.fn>;
  let oncommit: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    onchange = vi.fn();
    oncommit = vi.fn();
  });

  function mountFrame(props: Record<string, unknown> = {}): void {
    mountComponent(CropFrame, { frame: FRAME, crop: { x: 100, y: 100, width: 400, height: 300 } satisfies Crop, ratio: null, scale: 0.5, onchange, oncommit, ...props });
  }

  const box = (): HTMLElement => document.querySelector<HTMLElement>('[data-testid="adjust-crop-box"]')!;
  const handle = (name: string): HTMLElement => document.querySelector<HTMLElement>(`.crop-handle[data-h="${name}"]`)!;
  const handleNames = (): string[] => [...document.querySelectorAll<HTMLElement>('.crop-handle')].map((node) => node.dataset.h!);

  it('is a focusable group named Crop frame, with the handles hidden from assistive tech', () => {
    mountFrame({ describedBy: 'help' });
    expect(box().getAttribute('role')).toBe('group');
    expect(box().getAttribute('aria-label')).toBe('Crop frame');
    expect(box().getAttribute('aria-describedby')).toBe('help');
    expect(box().tabIndex).toBe(0);
    expect(document.querySelector('.crop-handles')!.getAttribute('aria-hidden')).toBe('true');
  });

  it('places the box in CSS pixels from the crop and the scale', () => {
    mountFrame();
    expect(box().style.left).toBe('50px');
    expect(box().style.top).toBe('50px');
    expect(box().style.width).toBe('200px');
    expect(box().style.height).toBe('150px');
  });

  it('shows four corner handles and four edge handles in Free, and corners only with a locked ratio', () => {
    mountFrame();
    expect(handleNames().sort()).toEqual(['e', 'n', 'ne', 'nw', 's', 'se', 'sw', 'w']);
  });

  it('shows corner handles only when a ratio is locked', () => {
    mountFrame({ ratio: 4 / 3 });
    expect(handleNames().sort()).toEqual(['ne', 'nw', 'se', 'sw']);
  });

  it('moves the box by the pointer travel divided by the scale, clamped to the frame', () => {
    mountFrame();
    pointer(box(), 'pointerdown', 100, 100);
    expect(box().hasAttribute('data-dragging')).toBe(true);
    pointer(box(), 'pointermove', 150, 120);
    expect(onchange).toHaveBeenLastCalledWith({ x: 200, y: 140, width: 400, height: 300 });
    pointer(box(), 'pointermove', 5000, 5000);
    expect(onchange).toHaveBeenLastCalledWith({ x: 400, y: 300, width: 400, height: 300 });
    pointer(box(), 'pointerup', 5000, 5000);
    expect(oncommit).toHaveBeenCalledOnce();
    expect(box().hasAttribute('data-dragging')).toBe(false);
  });

  it('resizes from a handle with the far side fixed', () => {
    mountFrame();
    pointer(handle('se'), 'pointerdown', 0, 0);
    pointer(handle('se'), 'pointermove', 50, 25);
    expect(onchange).toHaveBeenLastCalledWith({ x: 100, y: 100, width: 500, height: 350 });
  });

  it('resizes by an edge in Free', () => {
    mountFrame();
    pointer(handle('e'), 'pointerdown', 0, 0);
    pointer(handle('e'), 'pointermove', -50, 80);
    expect(onchange).toHaveBeenLastCalledWith({ x: 100, y: 100, width: 300, height: 300 });
  });

  it('keeps the shape when a corner is dragged with a locked ratio', () => {
    mountFrame({ ratio: 4 / 3 });
    pointer(handle('se'), 'pointerdown', 0, 0);
    pointer(handle('se'), 'pointermove', 100, 100);
    const crop = onchange.mock.lastCall![0] as Crop;
    expect(crop.width / crop.height).toBeCloseTo(4 / 3, 1);
    expect(crop.x).toBe(100);
    expect(crop.y).toBe(100);
  });

  it('never goes below 16 source pixels, or the size that keeps 44 screen pixels', () => {
    mountFrame();
    pointer(handle('se'), 'pointerdown', 0, 0);
    pointer(handle('se'), 'pointermove', -5000, -5000);
    // 44 / 0.5 = 88 source pixels.
    expect(onchange).toHaveBeenLastCalledWith({ x: 100, y: 100, width: 88, height: 88 });
  });

  it('restores the start box when the pointer is cancelled', () => {
    mountFrame();
    pointer(box(), 'pointerdown', 100, 100);
    pointer(box(), 'pointermove', 150, 120);
    pointer(box(), 'pointercancel', 150, 120);
    expect(onchange).toHaveBeenLastCalledWith({ x: 100, y: 100, width: 400, height: 300 });
    expect(oncommit).not.toHaveBeenCalled();
  });

  it('ignores another pointer while one drags', () => {
    mountFrame();
    pointer(box(), 'pointerdown', 100, 100, 1);
    pointer(box(), 'pointermove', 150, 120, 2);
    expect(onchange).not.toHaveBeenCalled();
  });

  it('moves by max(1, round(min(W, H) / 100)) pixels per arrow key, ten times that with Shift', () => {
    mountFrame();
    const right = new KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true, cancelable: true });
    box().dispatchEvent(right);
    expect(right.defaultPrevented).toBe(true);
    expect(onchange).toHaveBeenLastCalledWith({ x: 106, y: 100, width: 400, height: 300 });
    box().dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowDown', shiftKey: true, bubbles: true }));
    expect(onchange).toHaveBeenLastCalledWith({ x: 100, y: 160, width: 400, height: 300 });
    expect(oncommit).toHaveBeenCalledTimes(2);
  });

  it('does not report a nudge that cannot move, and leaves other keys alone', () => {
    mountFrame({ crop: { x: 0, y: 0, width: 800, height: 600 } });
    box().dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowLeft', bubbles: true }));
    expect(onchange).not.toHaveBeenCalled();
    const tab = new KeyboardEvent('keydown', { key: 'Tab', bubbles: true, cancelable: true });
    box().dispatchEvent(tab);
    expect(tab.defaultPrevented).toBe(false);
  });

  it('treats a missing crop as the whole frame', () => {
    mountFrame({ crop: null });
    expect(box().style.width).toBe('400px');
    expect(box().style.height).toBe('300px');
  });

  it('does not move while disabled', () => {
    mountFrame({ disabled: true });
    pointer(box(), 'pointerdown', 100, 100);
    pointer(box(), 'pointermove', 150, 120);
    box().dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true }));
    expect(onchange).not.toHaveBeenCalled();
  });

  it('makes the handle hit areas bigger on a touch screen', () => {
    mountFrame({ coarse: true });
    expect(document.querySelector('.crop-layer')!.hasAttribute('data-coarse')).toBe(true);
  });
});
