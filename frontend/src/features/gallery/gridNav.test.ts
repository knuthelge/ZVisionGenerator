import { describe, expect, it } from 'vitest';

import { gridColumns, moveInGrid } from './gridNav';

describe('gridColumns', () => {
  it('counts the items on the first row', () => {
    expect(gridColumns([0, 0, 0, 120, 120, 120, 240])).toBe(3);
  });

  it('treats a single row as all columns and an empty grid as one', () => {
    expect(gridColumns([0, 0])).toBe(2);
    expect(gridColumns([])).toBe(1);
  });
});

describe('moveInGrid', () => {
  // 3 columns, 7 items: rows [0 1 2] [3 4 5] [6]
  it.each([
    [4, 'ArrowLeft', 3],
    [4, 'ArrowRight', 5],
    [4, 'ArrowUp', 1],
    [4, 'ArrowDown', null],
    [3, 'ArrowDown', 6],
    [0, 'ArrowLeft', null],
    [1, 'ArrowUp', null],
    [4, 'Enter', null],
  ])('moves from %i with %s to %s', (index, key, expected) => {
    expect(moveInGrid(index, key, 3, 7)).toBe(expected);
  });
});
