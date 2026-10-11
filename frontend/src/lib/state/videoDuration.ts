import type { NumberSpec } from '$lib/actions/scrub';

/** Round to the nearest integer, halves to even, as Python's `round()` does. */
export function roundHalfEven(value: number): number {
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

/** Round seconds to two places for a duration field (6.0417 is 6.04). */
export function roundSeconds(seconds: number): number {
  return Math.round(seconds * 100) / 100;
}

/** Return the seconds field's bounds: one frame-alignment step per nudge, no upper limit. */
export function durationSpec(fps: number, alignment: number): NumberSpec {
  const step = alignment / fps;
  return { step, min: step };
}

/** Frames an extension adds come in multiples of this (the model's temporal compression). */
export const EXTEND_FRAME_STEP = 8;
/** Seconds an extension adds when no length was chosen. */
export const DEFAULT_EXTEND_SECONDS = 2;

/** Round a frame count to the nearest multiple of 8, at least 8; an extension is not aligned to `8k + 1`. */
export function snapExtendFrames(frames: number): number {
  if (!Number.isFinite(frames)) return EXTEND_FRAME_STEP;
  return Math.max(EXTEND_FRAME_STEP, roundHalfEven(frames / EXTEND_FRAME_STEP) * EXTEND_FRAME_STEP);
}

/** Convert a length in seconds to the new frames at the source's frame rate: a multiple of 8, ties to even, as the server does. */
export function extendFramesFromSeconds(seconds: number, fps: number): number {
  return snapExtendFrames(seconds * fps);
}

/** Return the frames a 2 s extension adds at the source's frame rate. */
export function defaultExtendFrames(fps: number): number {
  return extendFramesFromSeconds(DEFAULT_EXTEND_SECONDS, fps);
}

/** The finished clip's length after an extension: source plus new frames. */
export interface ExtendTotal {
  frames: number;
  seconds: number;
}

/** Return the finished clip's length: the source's frames plus the new ones, at the source's frame rate. */
export function extendTotal(sourceFrames: number, extendFrames: number, fps: number): ExtendTotal {
  const frames = sourceFrames + extendFrames;
  return { frames, seconds: frames / fps };
}
