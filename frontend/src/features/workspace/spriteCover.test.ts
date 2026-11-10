import { describe, expect, it } from 'vitest';

import { barCellCount, cellFrame } from './spriteCover';

describe('the cells of the bar', () => {
  it('fit about one per 66 px, at least four', () => {
    expect(barCellCount(0)).toBe(4);
    expect(barCellCount(336)).toBe(5);
    expect(barCellCount(660)).toBe(10);
  });

  it('spread over the clip from its first frame to its last', () => {
    expect([0, 1, 2, 3, 4].map((index) => cellFrame(index, 5, 100))).toEqual([0, 25, 50, 75, 100]);
    expect(cellFrame(0, 1, 100)).toBe(0);
  });
});
