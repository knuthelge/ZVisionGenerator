import { describe, expect, it } from 'vitest';

import { clampToSpec, nudgeValue, scrubValue, snapToSpec } from './scrub';

describe('scrub value math', () => {
  const guidance = { step: 0.1, min: 0, max: 10 };

  it('snaps to the step precision and clamps to bounds', () => {
    expect(clampToSpec(3.14159, guidance)).toBe(3.1);
    expect(clampToSpec(-2, guidance)).toBe(0);
    expect(clampToSpec(12, guidance)).toBe(10);
  });

  it('moves one step per pointer interval while dragging', () => {
    expect(scrubValue(3.5, 8, guidance, 4)).toBe(3.7);
    expect(scrubValue(3.5, -12, guidance, 4)).toBe(3.2);
    expect(scrubValue(9.9, 400, guidance, 4)).toBe(10);
  });

  it('nudges by one step, or ten with the big modifier', () => {
    const steps = { step: 1, min: 1, max: 60 };
    expect(nudgeValue(10, 1, steps)).toBe(11);
    expect(nudgeValue(10, -1, steps, true)).toBe(1);
    expect(nudgeValue(55, 1, steps, true)).toBe(60);
  });

  it('nudges an empty value up from the minimum', () => {
    expect(nudgeValue(null, 1, { step: 0.01, min: 0, max: 1 })).toBe(0.01);
  });

  it('snaps typed values to the step and bounds', () => {
    expect(snapToSpec(1000, { step: 16, min: 256, max: 2048 })).toBe(1008);
    expect(snapToSpec(2.5, { step: 1, min: 1, max: 60 })).toBe(3);
    expect(snapToSpec(0, { step: 1, min: 1, max: 100 })).toBe(1);
    expect(snapToSpec(1.234, { step: 0.05, min: 0.5, max: 2 })).toBe(1.25);
  });
});
