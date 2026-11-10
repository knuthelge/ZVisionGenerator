/** Sliding the pointer up by more than this many pixels from where it pressed slows the scrub. */
export const FINE_LIFT_PX = 48;

/** The most a slowed scrub moves per pixel, as a share of normal speed. */
export const FINE_MAX_RATE = 0.125;

/** A slowed scrub never moves faster than one frame per this many pixels, however long the clip. */
export const FINE_FRAME_PX = 6;

/** Where a drag along a timeline stands: the fraction it points at (0 to 1), and what later moves count from. */
export interface Scrub {
  fraction: number;
  /** Whether the pointer is lifted past `FINE_LIFT_PX`, which slows the scrub. */
  fine: boolean;
  /** The pointer's last x. */
  x: number;
  /** The x and fraction the position counts from: the press at first, then wherever the speed last changed. */
  anchorX: number;
  anchorFraction: number;
  /** The y where the pointer pressed; the lift is measured from it. */
  pressY: number;
}

/** The track a drag runs along: its width in pixels, and the last frame of the clip (`null` for audio, which has none). */
export interface ScrubTrack {
  width: number;
  lastFrame: number | null;
}

function clampFraction(value: number): number {
  return Math.min(1, Math.max(0, value));
}

/** Return the share of normal speed a slowed scrub moves at: `min(0.125, width / (lastFrame * 6))`, so a long clip never goes faster than a frame per 6 px. */
export function fineRate(track: ScrubTrack): number {
  if (track.lastFrame === null || track.lastFrame <= 0) return FINE_MAX_RATE;
  return Math.min(FINE_MAX_RATE, track.width / (track.lastFrame * FINE_FRAME_PX));
}

/** Return the scrub of a press at (*x*, *y*) that points at *fraction* of the track. */
export function startScrub(x: number, y: number, fraction: number): Scrub {
  const clamped = clampFraction(fraction);
  return { fraction: clamped, fine: false, x, anchorX: x, anchorFraction: clamped, pressY: y };
}

/**
 * Return the scrub after the pointer moves to (*x*, *y*).
 *
 * The position counts from the anchor. Crossing the lift line in either direction moves the anchor to the pointer's last
 * position and fraction, so the playhead never jumps when the speed changes.
 */
export function moveScrub(scrub: Scrub, x: number, y: number, track: ScrubTrack): Scrub {
  const fine = scrub.pressY - y > FINE_LIFT_PX;
  const anchored = fine === scrub.fine ? scrub : { ...scrub, fine, anchorX: scrub.x, anchorFraction: scrub.fraction };
  if (!(track.width > 0)) return { ...anchored, x };
  const rate = fine ? fineRate(track) : 1;
  const fraction = clampFraction(anchored.anchorFraction + ((x - anchored.anchorX) / track.width) * rate);
  return { ...anchored, fraction, x };
}
