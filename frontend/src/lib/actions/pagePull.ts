/** Which way a pull runs: up opens the page beyond a surface's foot, down the page beyond its top. */
export type PullDirection = 'up' | 'down';

/** A pull this far goes to the other page, as does a flick. */
export const PAGE_PULL_PX = 80;
const FLICK_PX_PER_MS = 0.35 * 1.5;
const FLICK_MIN_PX = 16;

/** Return how far a drag has gone in the pull direction, in px; a drag the other way is 0. */
export function pullDistance(offset: number, direction: PullDirection): number {
  return Math.max(0, direction === 'up' ? -offset : offset);
}

/** Return how far a pull is toward the threshold, from 0 to 1, for a hint that fades in with the pull. */
export function pullProgress(offset: number, direction: PullDirection): number {
  return Math.min(1, pullDistance(offset, direction) / PAGE_PULL_PX);
}

/** Return whether a pull has gone far enough that letting go changes page. */
export function pagePullReady(offset: number, direction: PullDirection): boolean {
  return pullDistance(offset, direction) >= PAGE_PULL_PX;
}

/** Return whether a pull that just ended changes page: it went far enough, or it ended in a flick in the pull direction. */
export function pagePullCommits(end: { offset: number; velocity: number }, direction: PullDirection): boolean {
  const speed = direction === 'up' ? -end.velocity : end.velocity;
  return pagePullReady(end.offset, direction) || (pullDistance(end.offset, direction) >= FLICK_MIN_PX && speed >= FLICK_PX_PER_MS);
}
