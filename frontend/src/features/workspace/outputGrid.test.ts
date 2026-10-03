import { describe, expect, it } from 'vitest';

import { fitOutputGrid } from './outputGrid';

describe('fitOutputGrid', () => {
  it('uses the column count that gives the largest cells', () => {
    expect(fitOutputGrid(2, 1000, 600, 12, 160)).toEqual({ columns: 2, cellSize: 494 });
    // Ties keep fewer columns, so four outputs form a 2×2 block.
    expect(fitOutputGrid(4, 1000, 600, 12, 160)).toEqual({ columns: 2, cellSize: 294 });
  });

  it('keeps a minimum cell size and lets many outputs scroll', () => {
    expect(fitOutputGrid(40, 1000, 400, 12, 160)).toEqual({ columns: 5, cellSize: 160 });
  });

  it('falls back safely before the box is measured', () => {
    expect(fitOutputGrid(3, 0, 0, 12, 160)).toEqual({ columns: 1, cellSize: 160 });
  });
});
