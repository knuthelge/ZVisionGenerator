/** Columns and square cell size for laying out a run's outputs. */
export interface OutputGridLayout {
  columns: number;
  cellSize: number;
}

/**
 * Pick the column count that gives the largest square cells while every output fits the box.
 *
 * When even the best fit would shrink cells below `minCell`, cells stay at `minCell` and the
 * grid scrolls vertically instead.
 */
export function fitOutputGrid(count: number, width: number, height: number, gap: number, minCell: number): OutputGridLayout {
  if (count <= 0 || width <= 0 || height <= 0) return { columns: 1, cellSize: minCell };
  let best: OutputGridLayout = { columns: 1, cellSize: 0 };
  for (let columns = 1; columns <= count; columns += 1) {
    const rows = Math.ceil(count / columns);
    const cellSize = Math.floor(Math.min((width - gap * (columns - 1)) / columns, (height - gap * (rows - 1)) / rows));
    if (cellSize > best.cellSize) best = { columns, cellSize };
  }
  if (best.cellSize >= minCell) return best;
  const columns = Math.max(1, Math.min(count, Math.floor((width + gap) / (minCell + gap))));
  return { columns, cellSize: minCell };
}
