/** The tiles each cell of the bar shows. */
const CELL_WIDTH = 66;
const MIN_CELLS = 4;

/** Return how many cells fill a bar *width* px wide: about one per 66 px, at least four. */
export function barCellCount(width: number): number {
  return Math.max(MIN_CELLS, Math.round(width / CELL_WIDTH));
}

/** Return the frame cell *index* of *count* shows: evenly spread over the clip, the first on its first frame and the last on its last. */
export function cellFrame(index: number, count: number, lastFrame: number): number {
  return count <= 1 ? 0 : Math.round((index / (count - 1)) * lastFrame);
}
