import { describe, expect, it } from 'vitest';

import { placePopover } from './popover';

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
