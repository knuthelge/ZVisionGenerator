/** Count the columns of a wrapped grid from its items' top offsets, in DOM order. */
export function gridColumns(tops: readonly number[]): number {
  if (tops.length === 0) return 1;
  const firstRow = tops.findIndex((top) => top !== tops[0]);
  return firstRow === -1 ? tops.length : firstRow;
}

/** Return the grid index an arrow key moves to, or null when the key does not move or would leave the grid. */
export function moveInGrid(index: number, key: string, columns: number, count: number): number | null {
  const step: Record<string, number> = { ArrowLeft: -1, ArrowRight: 1, ArrowUp: -columns, ArrowDown: columns };
  if (!(key in step)) return null;
  const next = index + step[key];
  return next >= 0 && next < count ? next : null;
}
