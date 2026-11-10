import { secondsLabel } from '$lib/components/molecules/mediaTime';

/** The frames of a clip: the rate, how many there are, and the seconds from the file start to the first one. */
export interface VideoFacts {
  fps: number;
  frames: number;
  start: number;
}

/** Return the frame facts of a clip, or `null` when the rate, the frame count or the first-frame time is missing or unusable. */
export function videoFactsOf(fps: number | null | undefined, frames: number | null | undefined, start: number | null | undefined): VideoFacts | null {
  if (!fps || fps <= 0 || !frames || frames < 1 || start == null || !Number.isFinite(start)) return null;
  return { fps, frames, start };
}

/** What a player knows about its video at one moment. `time` is the position the picture is at, or going to during a seek. */
export interface MediaSnapshot {
  paused: boolean;
  ended: boolean;
  time: number;
  error: boolean;
  duration: number;
}

/** The frame a paused video is on, as the actions of the viewer and the stage read it. */
export interface FramePoint {
  /** Seconds to send to the server: the position itself, or the middle of the last frame at the end. */
  time: number;
  /** The frame's own time in the clip, `frame / fps`, for every label. */
  seconds: number;
  /** True when an Extend from here would keep fewer than `EXTEND_MIN_FRAMES` frames. Never true at the end. */
  tooShort: boolean;
  /** True on the last frame, or when playback has ended. Use as acts on the frame; Extend acts on the whole clip. */
  atEnd: boolean;
}

/** The fewest frames an extended clip may keep. */
export const EXTEND_MIN_FRAMES = 25;

export const EXTEND_SHORT_REASON = 'Too early to extend: it needs 25 frames before the point.';

// Absorbs the float error of a time computed as `start + frame / fps`, so it lands on its own frame. Equal to `_FRAME_EPSILON` in `utils/video_point.py`.
const FRAME_EPSILON = 1e-3;

/** Return the index of the frame a time lies in, clamped to the clip. The same rule as `frame_at` in `utils/video_point.py`. */
export function frameAt(seconds: number, facts: VideoFacts): number {
  const frame = Math.floor((seconds - facts.start) * facts.fps + FRAME_EPSILON);
  return Math.min(Math.max(frame, 0), facts.frames - 1);
}

/** Return the time in the middle of *frame*, which a player seeks to so the browser lands on that frame whatever its rounding. */
export function stepTarget(frame: number, facts: VideoFacts): number {
  return facts.start + (frame + 0.5) / facts.fps;
}

/** Return the frame the player is paused on, or `null` when nothing is selected: playing, at the first frame, a media error, or no frame facts yet. */
export function framePoint(snapshot: MediaSnapshot, facts: VideoFacts | null): FramePoint | null {
  if (!facts || snapshot.error || !(snapshot.duration > 0)) return null;
  if (!snapshot.paused && !snapshot.ended) return null;
  const last = facts.frames - 1;
  const frame = snapshot.ended ? last : frameAt(snapshot.time, facts);
  if (snapshot.ended || (facts.frames > 1 && frame === last)) {
    return { time: stepTarget(last, facts), seconds: last / facts.fps, tooShort: false, atEnd: true };
  }
  if (frame === 0) return null;
  return { time: snapshot.time, seconds: frame / facts.fps, tooShort: frame + 1 < EXTEND_MIN_FRAMES, atEnd: false };
}

/** Return the label of a point's time: `3.2 s`. */
export function pointLabel(point: Pick<FramePoint, 'seconds'>): string {
  return secondsLabel(point.seconds);
}
