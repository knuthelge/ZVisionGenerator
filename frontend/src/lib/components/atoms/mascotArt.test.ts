import { describe, expect, it } from 'vitest';

import { TILE_STEP_COUNT, tileCells, tileStep } from './mascotArt';

describe('tileStep', () => {
  it('moves from the first step to the last as the job progresses', () => {
    const steps = [0, 0.2, 0.4, 0.6, 0.84].map((progress) => tileStep(progress, 0.85));

    expect(steps[0]).toBe(0);
    expect(steps).toEqual([...steps].sort((a, b) => a - b));
    expect(steps.at(-1)).toBe(TILE_STEP_COUNT - 1);
  });

  it('stays on the last step once the job is past the point where he adds the finishing touches', () => {
    expect(tileStep(1, 0.85)).toBe(TILE_STEP_COUNT - 1);
  });
});

describe('tileCells', () => {
  it('draws the same picture for the same step', () => {
    expect(tileCells(3)).toEqual(tileCells(3));
  });

  it('looks different at each stage of the denoise, and different again once finished', () => {
    const pictures = [0, 4, 7].map((step) => JSON.stringify(tileCells(step)));

    expect(new Set([...pictures, JSON.stringify(tileCells(null))]).size).toBe(4);
  });
});
