import { describe, expect, it } from 'vitest';

import { fitRect, flightKeyframes, usableRect } from './viewerFlight';

describe('viewer flight', () => {
  it('fits a box of the given aspect inside the stage, centred', () => {
    const stage = { left: 10, top: 20, width: 400, height: 300 };
    expect(fitRect(stage, 2)).toEqual({ left: 10, top: 70, width: 400, height: 200 });
    expect(fitRect(stage, 0.5)).toEqual({ left: 135, top: 20, width: 150, height: 300 });
    expect(fitRect(stage, 0)).toEqual({ left: 60, top: 20, width: 300, height: 300 });
  });

  it('starts the element over the tile and ends it in place, moving only the transform', () => {
    const from = { left: 20, top: 500, width: 100, height: 50 };
    const to = { left: 120, top: 100, width: 400, height: 200 };
    expect(flightKeyframes(from, to)).toEqual([
      { transform: 'translate(-100px, 400px) scale(0.25, 0.25)' },
      { transform: 'translate(0px, 0px) scale(1, 1)' },
    ]);
  });

  it('flies only to and from a box that is on screen and not collapsed', () => {
    const viewport = { width: 390, height: 844 };
    expect(usableRect({ left: 10, top: 10, width: 100, height: 100 }, viewport)).toBe(true);
    expect(usableRect({ left: 10, top: 10, width: 0, height: 0 }, viewport)).toBe(false);
    expect(usableRect({ left: 10, top: 900, width: 100, height: 100 }, viewport)).toBe(false);
    expect(usableRect(null, viewport)).toBe(false);
  });
});
