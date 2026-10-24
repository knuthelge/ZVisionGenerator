/** Which way a finger can page from the shown asset. */
export interface PagerMoves {
  prev: boolean;
  next: boolean;
}

/** What a sideways drag does when the finger lifts. */
export type PagerOutcome = 'prev' | 'next' | 'stay';

/** What a vertical drag does when the finger lifts. */
export type PullOutcome = 'close' | 'details' | 'stay';

/** A drag this fraction of the pane's width pages, as does a flick. */
const PAGE_FRACTION = 0.25;
const FLICK_PX_PER_MS = 0.35;
const FLICK_MIN_PX = 16;
/** A pull down this far closes; a pull up this far opens the details. */
export const CLOSE_PULL_PX = 80;
export const DETAILS_PULL_PX = 60;

/** Return where a sideways drag settles: the next or previous pane when it went far or fast enough that way. */
export function pagerOutcome(offset: number, velocity: number, paneWidth: number, moves: PagerMoves): PagerOutcome {
  const far = Math.abs(offset) >= paneWidth * PAGE_FRACTION;
  const flicked = Math.abs(offset) >= FLICK_MIN_PX && Math.abs(velocity) >= FLICK_PX_PER_MS && Math.sign(velocity) === Math.sign(offset);
  if (!far && !flicked) return 'stay';
  if (offset < 0) return moves.next ? 'next' : 'stay';
  return moves.prev ? 'prev' : 'stay';
}

/** Return what a vertical drag does: down far or fast closes, up far enough opens the details. */
export function pullOutcome(offset: number, velocity: number, canOpenDetails: boolean): PullOutcome {
  const flickedDown = offset >= FLICK_MIN_PX && velocity >= FLICK_PX_PER_MS;
  if (offset >= CLOSE_PULL_PX || flickedDown) return 'close';
  if (canOpenDetails && offset <= -DETAILS_PULL_PX) return 'details';
  return 'stay';
}

/** The image's transform while it is pulled down: it follows the finger and shrinks a little as it goes. */
export function pullTransform(offset: number): string {
  const scale = Math.max(0.6, 1 - Math.max(0, offset) / 800);
  return `translate(0px, ${offset}px) scale(${scale.toFixed(3)})`;
}

/** The backdrop's opacity while the image is pulled down: it clears as the image nears the page. */
export function pullBackdropOpacity(offset: number): number {
  return Math.max(0, Math.min(1, 1 - Math.max(0, offset) / 400));
}
