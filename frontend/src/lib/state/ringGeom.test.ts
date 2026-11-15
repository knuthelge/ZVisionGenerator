import { describe, expect, it } from 'vitest';

import { RING_SPECS, arcLayout, baseGap, ghostPlace, headAngles, markDrop, ringRadius } from './ringGeom';

const circumference = (size: keyof typeof RING_SPECS) => 2 * Math.PI * ringRadius(RING_SPECS[size].box, RING_SPECS[size].stroke);

describe('baseGap', () => {
  it.each([[144, 5], [128, 5], [70, 3], [14, 1]])('is the ring\'s own gap for a %d px box', (box, gap) => {
    expect(baseGap(box)).toBe(gap);
  });
});

describe('arcLayout', () => {
  it('is one unbroken ring for one arc', () => {
    const layout = arcLayout(433, 1, 4, 144);
    expect(layout).toEqual({ pitch: 433, gap: 0, dash: 433, offset: 0 });
  });

  it.each([
    ['L', 2, 212, 5],
    ['L', 4, 103, 5],
    ['M', 2, 187, 5],
    ['S', 4, 48, 3],
    ['XS', 2, 14, 1],
  ] as const)('keeps the ring\'s own gap for a few arcs (%s, %d arcs)', (size, count, visible, gap) => {
    const { box, stroke } = RING_SPECS[size];
    const layout = arcLayout(circumference(size), count, stroke, box);
    expect(layout.gap).toBe(gap);
    // The visible arc is the dash plus its two round caps.
    expect(layout.dash + stroke).toBeCloseTo(visible + 0, -1);
  });

  it('shrinks the gap to a quarter of the slot as the count grows, with no threshold', () => {
    const { box, stroke } = RING_SPECS.L;
    const gaps = [2, 8, 24, 60, 100].map((count) => arcLayout(circumference('L'), count, stroke, box).gap);
    expect(gaps[0]).toBe(5);
    for (let k = 1; k < gaps.length; k += 1) expect(gaps[k]).toBeLessThanOrEqual(gaps[k - 1]);
    const slot = circumference('L') / 100;
    expect(gaps[4]).toBeCloseTo(slot / 4, 5);
  });

  it('turns arcs into touching dots and never a negative dash', () => {
    const { box, stroke } = RING_SPECS.L;
    for (const count of [100, 200, 500]) {
      const layout = arcLayout(circumference('L'), count, stroke, box);
      expect(layout.dash).toBeGreaterThanOrEqual(0);
    }
    expect(arcLayout(circumference('L'), 500, stroke, box).dash).toBe(0);
  });

  it('starts the first arc half a gap and one cap in', () => {
    const { box, stroke } = RING_SPECS.L;
    const layout = arcLayout(circumference('L'), 4, stroke, box);
    expect(layout.offset).toBe((layout.gap + stroke) / 2);
  });
});

describe('markDrop', () => {
  it('moves the mark from the box\'s foot onto the ring\'s 6 o\'clock', () => {
    const { box, stroke, mark } = RING_SPECS.L;
    // The mark's centre starts mark/2 above the box's foot and must land on the arc's lowest point.
    const lowest = box / 2 + ringRadius(box, stroke);
    expect(box - mark / 2 + markDrop(box, stroke, mark)).toBeCloseTo(lowest, 6);
  });
});

describe('ghostPlace', () => {
  it('draws the ghost on the arc while one step is long enough to see', () => {
    expect(ghostPlace('L', 1, 1 / 8, true)).toBe('ring');
    expect(ghostPlace('L', 4, 1 / 8, true)).toBe('ring');
  });

  it('moves it to the stamp when one step is under 4 px of arc', () => {
    expect(ghostPlace('L', 100, 1 / 8, true)).toBe('stamp');
    expect(ghostPlace('S', 24, 1 / 30, true)).toBe('stamp');
  });

  it('has nowhere to show without a stamp', () => {
    expect(ghostPlace('L', 100, 1 / 8, false)).toBeNull();
  });
});

describe('headAngles', () => {
  const { box, stroke } = RING_SPECS.L;

  it('puts the head of a single arc at its fill, clockwise from 12 o\'clock', () => {
    expect(headAngles(box, stroke, [{ fill: 0.25 }], 0, null)!.from).toBeCloseTo(90, 5);
    expect(headAngles(box, stroke, [{ fill: 0.5 }], 0, null)!.from).toBeCloseTo(180, 5);
    expect(headAngles(box, stroke, [{ fill: 0 }], 0, null)!.from).toBe(0);
  });

  it('puts the head of a batch in its own arc\'s slot', () => {
    const quarter = headAngles(box, stroke, [{ fill: 1 }, { fill: 1 }, { fill: 0.5 }, { fill: 0 }], 2, null)!;

    // The third of four slots starts at 180 degrees; its half-filled arc ends a little before the middle of the slot.
    expect(quarter.from).toBeGreaterThan(180);
    expect(quarter.from).toBeLessThan(270);
    expect(quarter.to).toBe(quarter.from);
  });

  it('runs the head on to the ghost\'s front', () => {
    const arcs = [{ fill: 1 }, { fill: 0.5 }];
    const plain = headAngles(box, stroke, arcs, 1, null)!;
    const ghosted = headAngles(box, stroke, arcs, 1, { arc: 1, from: 0.5, share: 0.25 })!;

    expect(ghosted.from).toBe(plain.from);
    expect(ghosted.to).toBeGreaterThan(ghosted.from);
  });

  it('has no head without an arc to be the head', () => {
    expect(headAngles(box, stroke, [], null, null)).toBeNull();
    expect(headAngles(box, stroke, [{ fill: 1 }], 3, null)).toBeNull();
    expect(headAngles(box, stroke, [{ fill: 1 }], null, null)).toBeNull();
  });
});
