import { afterEach, describe, expect, it, vi } from 'vitest';
import { lockAxis, pan, releaseVelocity, rubberBand, type PanOptions } from './pan';

describe('lockAxis', () => {
  it('follows the larger movement and gives a tie to scrolling', () => {
    expect(lockAxis(5, 2)).toBe('x');
    expect(lockAxis(-1, 4)).toBe('y');
    expect(lockAxis(3, 3)).toBe('y');
  });
});

describe('releaseVelocity', () => {
  it('reads the speed of the last moments, not the average of the drag', () => {
    const samples = [
      { x: 0, y: 0, at: 0 },
      { x: 0, y: 10, at: 400 },
      { x: 0, y: 30, at: 440 },
      { x: 0, y: 70, at: 480 },
    ];
    expect(releaseVelocity(samples, 'y')).toBeCloseTo(60 / 80);
  });

  it('is zero without movement over time', () => {
    expect(releaseVelocity([], 'x')).toBe(0);
    expect(releaseVelocity([{ x: 4, y: 0, at: 10 }], 'x')).toBe(0);
  });
});

describe('rubberBand', () => {
  it('gives less and less, never past its limit, keeping the direction', () => {
    expect(rubberBand(10, 60)).toBeLessThan(10);
    expect(rubberBand(1000, 60)).toBeLessThan(60);
    expect(rubberBand(-40, 60)).toBeLessThan(0);
  });
});

/** Dispatch a touch event with one finger at (x, y). */
function touch(node: Element, type: string, x: number, y: number, fingers = 1): TouchEvent {
  const point = { clientX: x, clientY: y, target: node } as unknown as Touch;
  const event = new Event(type, { bubbles: true, cancelable: true }) as TouchEvent;
  Object.defineProperty(event, 'touches', { value: type === 'touchend' ? [] : Array(fingers).fill(point) });
  Object.defineProperty(event, 'changedTouches', { value: [point] });
  node.dispatchEvent(event);
  return event;
}

describe('pan', () => {
  let node: HTMLElement;
  let stop: (() => void) | null = null;

  afterEach(() => {
    stop?.();
    node?.remove();
  });

  function attach(overrides: Partial<PanOptions> = {}) {
    node = document.createElement('div');
    document.body.append(node);
    const options = { axes: ['y'] as const, onmove: vi.fn(), onend: vi.fn(), oncancel: vi.fn(), ...overrides };
    stop = pan(node, options).destroy;
    return options;
  }

  it('claims a drag on its first movement and cancels that movement, so the page cannot start scrolling', () => {
    const options = attach();
    touch(node, 'touchstart', 100, 100);
    const first = touch(node, 'touchmove', 100, 101);
    expect(first.defaultPrevented).toBe(true);
    touch(node, 'touchmove', 100, 160);
    touch(node, 'touchend', 100, 160);
    expect(options.onmove).toHaveBeenLastCalledWith({ axis: 'y', offset: 60 });
    expect(options.onend).toHaveBeenCalledWith(expect.objectContaining({ axis: 'y', offset: 60 }));
  });

  it('leaves a movement along another axis to the page for the whole touch', () => {
    const options = attach();
    touch(node, 'touchstart', 100, 100);
    const first = touch(node, 'touchmove', 110, 101);
    const later = touch(node, 'touchmove', 100, 200);
    touch(node, 'touchend', 100, 200);
    expect(first.defaultPrevented).toBe(false);
    expect(later.defaultPrevented).toBe(false);
    expect(options.onmove).not.toHaveBeenCalled();
    expect(options.onend).not.toHaveBeenCalled();
  });

  it('leaves a drag the surface declines to the content', () => {
    const options = attach({ claim: ({ dy }) => dy > 0 });
    touch(node, 'touchstart', 100, 100);
    expect(touch(node, 'touchmove', 100, 98).defaultPrevented).toBe(false);
    touch(node, 'touchend', 100, 60);
    expect(options.onend).not.toHaveBeenCalled();
  });

  it('gives a claimed drag up when a second finger lands', () => {
    const options = attach();
    touch(node, 'touchstart', 100, 100);
    touch(node, 'touchmove', 100, 120);
    touch(node, 'touchstart', 100, 120, 2);
    touch(node, 'touchend', 100, 130);
    expect(options.oncancel).toHaveBeenCalledTimes(1);
    expect(options.onend).not.toHaveBeenCalled();
  });
});
