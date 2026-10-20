import type { PanEnd } from '$lib/actions/pan';

/** What a drag on the stage does when it ends. */
export type StageSwipe = 'older' | 'newer' | 'gallery' | 'stay';

/** Where the stage can go from here. */
export interface StageMoves {
  /** An earlier output exists (a swipe left). */
  older: boolean;
  /** A later output exists (a swipe right). */
  newer: boolean;
  /** A swipe up opens the Gallery (a phone). */
  gallery: boolean;
}

/** A sideways drag this far steps through history, as does a flick. */
const STEP_PX = 48;
const FLICK_PX_PER_MS = 0.35;
const FLICK_MIN_PX = 16;
/** A pull up this far opens the Gallery, as does a flick. */
export const GALLERY_PULL_PX = 80;
/** Sideways trackpad scrolling, in wheel pixels, that steps once. */
const WHEEL_STEP_PX = 60;

/** Return whether a pull up has gone far enough that letting go opens the Gallery. */
export function galleryPullReady(offset: number): boolean {
  return offset <= -GALLERY_PULL_PX;
}

/** Return what a drag on the stage does when the finger lifts. Left is older, as the history strip runs. */
export function stageSwipeOutcome({ axis, offset, velocity }: PanEnd, moves: StageMoves): StageSwipe {
  if (axis === 'y') {
    const flicked = offset <= -FLICK_MIN_PX && velocity <= -FLICK_PX_PER_MS * 1.5;
    return moves.gallery && (galleryPullReady(offset) || flicked) ? 'gallery' : 'stay';
  }
  const far = Math.abs(offset) >= STEP_PX;
  const flicked = Math.abs(offset) >= FLICK_MIN_PX && Math.abs(velocity) >= FLICK_PX_PER_MS && Math.sign(velocity) === Math.sign(offset);
  if (!far && !flicked) return 'stay';
  if (offset < 0) return moves.older ? 'older' : 'stay';
  return moves.newer ? 'newer' : 'stay';
}

/** Return the history step for sideways trackpad scrolling summed over one gesture: 1 older, -1 newer, 0 not yet. */
export function wheelStep(totalDeltaX: number): -1 | 0 | 1 {
  if (totalDeltaX >= WHEEL_STEP_PX) return 1;
  if (totalDeltaX <= -WHEEL_STEP_PX) return -1;
  return 0;
}
