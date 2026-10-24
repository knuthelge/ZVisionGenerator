import type { NumberSpec } from '$lib/actions/scrub';
import type { FrameRule } from '$lib/types';

/** Round to the nearest integer, halves to even, as Python's `round()` does. */
export function roundHalfEven(value: number): number {
  const rounded = Math.round(value);
  return value % 1 === 0.5 && rounded % 2 !== 0 ? rounded - 1 : rounded;
}

/**
 * Move a frame count to one the model accepts, as the server does.
 *
 * Valid counts are `alignment * k + offset` for `k >= 1` (LTX: 9, 17, 25, …; H3: 22, 39, 56, …), so a clip is never shorter
 * than one step. A `nearest` rule rounds halves to even; an `up` rule takes the next valid count.
 */
export function alignFrames(frames: number, rule: FrameRule): number {
  const quotient = (frames - rule.offset) / rule.alignment;
  const steps = rule.round === 'up' ? Math.ceil(quotient) : roundHalfEven(quotient);
  return rule.alignment * Math.max(1, steps) + rule.offset;
}

/** Convert a clip length in seconds to the frame count the model accepts. An `up` rule counts whole frames before the first one is added. */
export function secondsToFrames(seconds: number, fps: number, rule: FrameRule): number {
  return alignFrames(rule.round === 'up' ? Math.round(seconds * fps) + 1 : seconds * fps + 1, rule);
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
export function durationSpec(fps: number, rule: Pick<FrameRule, 'alignment'>): NumberSpec {
  const step = rule.alignment / fps;
  return { step, min: step };
}

/** Frames an extension adds come in multiples of this unless the model says otherwise (LTX's temporal compression). */
export const EXTEND_FRAME_STEP = 8;
/** Seconds an extension adds when no length was chosen. */
export const DEFAULT_EXTEND_SECONDS = 2;

/** Round a frame count to the nearest multiple of *step*, at least *step*; an extension is not aligned to `step * k + offset`. */
export function snapExtendFrames(frames: number, step: number = EXTEND_FRAME_STEP): number {
  if (!Number.isFinite(frames)) return step;
  return Math.max(step, roundHalfEven(frames / step) * step);
}

/** Convert a length in seconds to the new frames at the source's frame rate: a multiple of *step*, ties to even, as the server does. */
export function extendFramesFromSeconds(seconds: number, fps: number, step: number = EXTEND_FRAME_STEP): number {
  return snapExtendFrames(seconds * fps, step);
}

/** Return the frames a 2 s extension adds at the source's frame rate. */
export function defaultExtendFrames(fps: number, step: number = EXTEND_FRAME_STEP): number {
  return extendFramesFromSeconds(DEFAULT_EXTEND_SECONDS, fps, step);
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
