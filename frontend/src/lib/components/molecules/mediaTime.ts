import { formatClock } from '$lib/state/clock';

/** An audio take of this many seconds or longer reads as `1:05 / 3:21`; everything shorter, and every video, reads in seconds. */
export const CLOCK_FROM_SECONDS = 60;

/** Return the label of a time in seconds with one decimal: `3.2 s`. The same form the Extend labels use. */
export function secondsLabel(seconds: number): string {
  return `${(Number.isFinite(seconds) ? Math.max(0, seconds) : 0).toFixed(1)} s`;
}

/** Return whether a player of this kind and length reads its times as a clock. */
export function readsAsClock(kind: 'video' | 'audio', total: number): boolean {
  return kind === 'audio' && total >= CLOCK_FROM_SECONDS;
}

/** Return the label of a time: a clock for a long take, else seconds. */
export function timeLabel(seconds: number, clock: boolean): string {
  return clock ? formatClock(seconds) : secondsLabel(seconds);
}
