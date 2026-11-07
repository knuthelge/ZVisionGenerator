import { afterEach, describe, expect, it, vi } from 'vitest';

import { placePopover, popover } from './popover';

function box(left: number, top: number, width: number, height: number) {
  return { left, top, width, height, right: left + width, bottom: top + height };
}

describe('placePopover', () => {
  const viewport = { width: 1000, height: 800 };

  it('opens below the anchor, aligned to its start or end edge', () => {
    expect(placePopover(box(100, 100, 80, 26), box(0, 0, 320, 200), viewport, 'start')).toEqual({ top: 132, left: 100 });
    expect(placePopover(box(500, 100, 28, 28), box(0, 0, 220, 200), viewport, 'end')).toEqual({ top: 134, left: 308 });
  });

  it('flips above the anchor when it would run off the bottom', () => {
    expect(placePopover(box(100, 700, 80, 26), box(0, 0, 320, 300), viewport, 'start')).toEqual({ top: 394, left: 100 });
  });

  it('stays inside the viewport horizontally', () => {
    expect(placePopover(box(900, 100, 80, 26), box(0, 0, 320, 200), viewport, 'start').left).toBe(672);
    expect(placePopover(box(10, 100, 28, 28), box(0, 0, 220, 200), viewport, 'end').left).toBe(8);
  });
});

describe('popover', () => {
  it('closes on a click outside, but not on a click or scroll inside a nested popover', () => {
    const anchor = document.createElement('button');
    const panel = document.createElement('div');
    const menu = document.createElement('div');
    menu.dataset.popoverNested = '';
    const item = document.createElement('button');
    menu.appendChild(item);
    document.body.append(anchor, menu);
    const onclose = vi.fn();
    const handle = popover(panel, { anchor, onclose });

    item.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true }));
    menu.dispatchEvent(new Event('scroll'));
    expect(onclose).not.toHaveBeenCalled();
    document.body.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true }));
    expect(onclose).toHaveBeenCalledTimes(1);

    handle.destroy();
    anchor.remove();
    menu.remove();
    panel.remove();
  });

  it('closes on a resize or scroll unless it follows the viewport, as a dropdown list does', () => {
    const anchor = document.createElement('button');
    document.body.append(anchor);
    const closing = document.createElement('div');
    const following = document.createElement('div');
    const closeOnResize = vi.fn();
    const keepOnResize = vi.fn();
    const first = popover(closing, { anchor, onclose: closeOnResize });
    const second = popover(following, { anchor, followViewport: true, onclose: keepOnResize });

    window.dispatchEvent(new Event('resize'));
    document.dispatchEvent(new Event('scroll'));
    expect(closeOnResize).toHaveBeenCalledTimes(2);
    expect(keepOnResize).not.toHaveBeenCalled();

    first.destroy();
    second.destroy();
    anchor.remove();
  });

  describe('when its size changes', () => {
    const observers: { callback: () => void; observe: ReturnType<typeof vi.fn>; disconnect: ReturnType<typeof vi.fn> }[] = [];

    class FakeResizeObserver {
      observe = vi.fn();
      disconnect = vi.fn();
      constructor(callback: () => void) {
        observers.push({ callback, observe: this.observe, disconnect: this.disconnect });
      }
    }

    afterEach(() => {
      observers.length = 0;
      vi.unstubAllGlobals();
    });

    it('places a popover that grew after opening again, flipped above its anchor', () => {
      vi.stubGlobal('ResizeObserver', FakeResizeObserver);
      vi.stubGlobal('innerHeight', 600);
      vi.stubGlobal('innerWidth', 1000);
      const anchor = document.createElement('button');
      anchor.getBoundingClientRect = () => box(100, 500, 80, 26) as DOMRect;
      const panel = document.createElement('div');
      let height = 40;
      panel.getBoundingClientRect = () => box(0, 0, 200, height) as DOMRect;
      document.body.append(anchor);
      const handle = popover(panel, { anchor, onclose: vi.fn() });

      expect(panel.style.top).toBe('532px');
      height = 300;
      observers[0].callback();
      expect(panel.style.top).toBe('194px');

      height = 40;
      observers[0].callback();
      expect(panel.style.top).toBe('532px');

      handle.destroy();
      anchor.remove();
    });

    it('stops watching the size when it is destroyed', () => {
      vi.stubGlobal('ResizeObserver', FakeResizeObserver);
      const anchor = document.createElement('button');
      const panel = document.createElement('div');
      document.body.append(anchor);
      const handle = popover(panel, { anchor, onclose: vi.fn() });

      expect(observers[0].observe).toHaveBeenCalledWith(panel);
      expect(observers[0].disconnect).not.toHaveBeenCalled();
      handle.destroy();
      expect(observers[0].disconnect).toHaveBeenCalledTimes(1);

      anchor.remove();
    });
  });
});
