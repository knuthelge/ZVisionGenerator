import type { ExtendSourceInfo } from '$lib/types';
import { EXTEND_MIN_FRAMES, frameAt, videoFactsOf, type VideoFacts } from './framePoint';
import { extendTotal, type ExtendTotal } from './videoDuration';

/** Seconds on the wire are rounded to the microsecond, as the server records them. */
const WIRE_DIGITS = 1e6;

function round6(seconds: number): number {
  return Math.round(seconds * WIRE_DIGITS) / WIRE_DIGITS;
}

/** What a source's point to extend from means for the clip: the frames it keeps, the length it gives and where its sound carries on. */
export interface ExtendCutView {
  /** The frames the continued clip starts with: the point's frame and every one before it. */
  keepFrames: number;
  /** True when the whole clip is kept, which is every clip too short to cut. */
  whole: boolean;
  /** The point's own time in the clip, `frame / fps`, as the labels show it. */
  keepSeconds: number;
  /** The clip's length, `frames / fps`. */
  clipSeconds: number;
  /** The seconds that name the kept last frame on the server's clock, so one frame has one preview URL; for the whole clip, its last frame. */
  thumbAt: number;
  /** Where in its audio track the kept clip's sound ends; null for a source that followed no track. */
  trackStart: number | null;
  /** The finished clip's frames and seconds (the kept frames plus the new ones); null until the new frames are known. */
  total: ExtendTotal | null;
}

/** Return the frame facts of a source, or `null` when the server could not read its rate, frames or first-frame time. */
export function extendFacts(info: ExtendSourceInfo): VideoFacts | null {
  return videoFactsOf(info.fps, info.frame_count, info.video_start);
}

/** Return the seconds, on the original's clock, that name the frame *keepFrames* frames in; `null` when that keeps the whole clip. */
export function extendFromAt(keepFrames: number, facts: VideoFacts): number | null {
  if (keepFrames >= facts.frames) return null;
  return round6(facts.start + (Math.max(keepFrames, EXTEND_MIN_FRAMES) - 1) / facts.fps);
}

/** Return how many frames a point keeps, from the first frame to the one it lies in; a point too early still keeps the least a cut may. */
function keptFrames(extendFrom: number | null, facts: VideoFacts): number {
  if (extendFrom === null || !Number.isFinite(extendFrom)) return facts.frames;
  return Math.min(facts.frames, Math.max(frameAt(extendFrom, facts) + 1, EXTEND_MIN_FRAMES));
}

/**
 * Return what continuing *info*'s clip from *extendFrom* means, or `null` when the source's frames are not known.
 *
 * A point at or past the last frame, or none, is the whole clip. A point before the 25th frame keeps 25. The point is read
 * as the server reads it, by `frameAt`.
 *
 * @param extendFrames The new frames to add, for the total; the source's default when absent.
 */
export function extendCutView(info: ExtendSourceInfo, extendFrom: number | null, extendFrames: number | null = info.extend_frames): ExtendCutView | null {
  const facts = extendFacts(info);
  if (!facts) return null;
  const keepFrames = keptFrames(extendFrom, facts);
  const removedFrames = facts.frames - keepFrames;
  const whole = removedFrames === 0;
  const trackEnd = info.audio_track_path ? (info.audio_track_end ?? 0) : null;
  return {
    keepFrames,
    whole,
    keepSeconds: (whole ? facts.frames : keepFrames - 1) / facts.fps,
    clipSeconds: facts.frames / facts.fps,
    thumbAt: round6(facts.start + (keepFrames - 1) / facts.fps),
    trackStart: trackEnd === null ? null : whole ? trackEnd : round6(Math.max(0, trackEnd - removedFrames / facts.fps)),
    total: extendFrames === null ? null : extendTotal(keepFrames, extendFrames, facts.fps),
  };
}

/**
 * Return the point a source can use: `null` for the whole clip, a point before the 25th frame moved to it, else the point as given.
 * A source whose frames are not known leaves it as it is.
 */
export function settledExtendFrom(info: ExtendSourceInfo, extendFrom: number | null): number | null {
  const facts = extendFacts(info);
  if (!facts || extendFrom === null) return extendFrom;
  const keepFrames = keptFrames(extendFrom, facts);
  if (keepFrames >= facts.frames) return null;
  return frameAt(extendFrom, facts) + 1 < keepFrames ? extendFromAt(keepFrames, facts) : extendFrom;
}

/** Return the line above the bar: `Keeps 3.2 s of 6.0 s`, or `Keeps the whole clip` at the end. Labels use `frame / fps` to one decimal. */
export function keepsLabel(view: Pick<ExtendCutView, 'whole' | 'keepSeconds' | 'clipSeconds'>): string {
  return view.whole ? 'Keeps the whole clip' : `Keeps ${view.keepSeconds.toFixed(1)} s of ${view.clipSeconds.toFixed(1)} s`;
}

/** What the bar says at the 25-frame stop. */
export const CUT_STOP_REASON = 'The handle stops at 25 frames: Extend needs that much video before the point.';

/** Whether the source is long enough to cut: it has more than `EXTEND_MIN_FRAMES` frames and its frame facts are known. */
export function extendCuttable(info: ExtendSourceInfo | null): boolean {
  const facts = info ? extendFacts(info) : null;
  return facts !== null && facts.frames > EXTEND_MIN_FRAMES;
}
