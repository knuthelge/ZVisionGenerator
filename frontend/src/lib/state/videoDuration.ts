import type { NumberSpec } from '$lib/actions/scrub';

/** Round to the nearest integer, halves to even, as Python's `round()` does. */
function roundHalfEven(value: number): number {
  const rounded = Math.round(value);
  return value % 1 === 0.5 && rounded % 2 !== 0 ? rounded - 1 : rounded;
}

/**
 * Round a frame count to the nearest one the model accepts, as the server does.
 *
 * Valid counts are `alignment * k + 1` (LTX: 9, 17, 25, …); a clip is never shorter than one step.
 */
export function alignFrameCount(frames: number, alignment: number): number {
  const steps = Math.max(1, roundHalfEven((frames - 1) / alignment));
  return alignment * steps + 1;
}

/** Convert a clip length in seconds to the nearest frame count the model accepts. */
export function secondsToFrames(seconds: number, fps: number, alignment: number): number {
  return alignFrameCount(seconds * fps + 1, alignment);
}

/** Return the clip length in seconds for a frame count (121 frames at 24 fps is 5 s). */
export function framesToSeconds(frames: number, fps: number): number {
  return Math.max(0, frames - 1) / fps;
}

/** Return the seconds field's bounds: one frame-alignment step per nudge, no upper limit. */
export function durationSpec(fps: number, alignment: number): NumberSpec {
  const step = alignment / fps;
  return { step, min: step };
}
