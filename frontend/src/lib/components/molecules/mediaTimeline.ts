/**
 * Draws the player's timeline on a 2D canvas: a hairline with the frames rising out of it under the pointer for a video,
 * and a mirrored waveform for a take. The geometry is plain functions of numbers; the two `draw` functions only paint it.
 * All lengths are CSS pixels: the caller scales the canvas by the device pixel ratio.
 */

/** The colours the canvas paints with, resolved from the design tokens once at mount. */
export interface TimelineColors {
  rest: string;
  played: string;
  tick: string;
  /** A dark line drawn under the hairline, so it stays visible on a bright frame. */
  shade: string;
  playhead: string;
  bloomRest: string;
  waveRest: string;
}

/** What a video timeline shows at one moment. */
export interface TimelineView {
  width: number;
  height: number;
  /** The playhead, 0 to 1. */
  fraction: number;
  /** The clip's length in seconds, for the second ticks. */
  duration: number;
  /** The clip's frame count, 0 when not known yet. */
  frames: number;
  scrubbing: boolean;
  /** How far the frames have risen, 0 to 1. */
  bloom: number;
  /** The x the bloom is centred on: the pointer, or the playhead during a keyboard or touch scrub. */
  focus: number;
}

/** One frame risen out of the hairline. */
export interface BloomBar {
  x: number;
  height: number;
  played: boolean;
}

/** The hairline sits this far above the foot of the canvas. */
const BASE_GAP = 12;
const HAIRLINE = 2;
const BLOOM_SPREAD = 74;
const BLOOM_HEIGHT = 30;
const BLOOM_MIN = 0.4;
const BLOOM_EASE = 0.14;
const BLOOM_SNAP = 0.002;

/** The waveform's bars: width, gap and the share of the height a silent bar keeps. */
const WAVE_BAR = 4;
const WAVE_GAP = 3;
const WAVE_FLOOR = 0.05;
const WAVE_MIN_PX = 2;
/** What a take whose peaks could not be read draws: a low, even line (the same as the tile waveform). */
const FLAT_PEAK = 16;

/** Return how many frames rise out of the hairline: all of them up to `max(24, width / 5)`. */
export function bloomBarCount(frames: number, width: number): number {
  const cap = Math.max(24, Math.floor(width / 5));
  return frames > 0 ? Math.min(frames, cap) : cap;
}

/** Return the bars of the bloom: each centred on its slice of the width, as tall as a Gaussian of its distance from the focus. */
export function bloomBars(view: Pick<TimelineView, 'width' | 'frames' | 'fraction' | 'focus' | 'bloom'>): BloomBar[] {
  if (view.bloom <= 0 || view.width <= 0) return [];
  const count = bloomBarCount(view.frames, view.width);
  const playhead = view.fraction * view.width;
  const bars: BloomBar[] = [];
  for (let i = 0; i < count; i += 1) {
    const x = ((i + 0.5) / count) * view.width;
    const distance = x - view.focus;
    const height = view.bloom * Math.exp(-(distance * distance) / (2 * BLOOM_SPREAD * BLOOM_SPREAD)) * BLOOM_HEIGHT;
    if (height >= BLOOM_MIN) bars.push({ x, height, played: x <= playhead });
  }
  return bars;
}

/** Return how far the bloom has risen one animation frame later: 14% of the gap, snapping to the target near it, at once under reduced motion. */
export function easeBloom(current: number, target: number, reduced: boolean): number {
  if (reduced) return target;
  const next = current + (target - current) * BLOOM_EASE;
  return Math.abs(target - next) < BLOOM_SNAP ? target : next;
}

/** Return the x of each whole second of a clip, from the first. */
export function secondTicks(duration: number, width: number): number[] {
  if (!(duration > 0) || !(width > 0)) return [];
  const ticks: number[] = [];
  for (let second = 1; second < duration; second += 1) ticks.push((second / duration) * width);
  return ticks;
}

function fillRound(ctx: CanvasRenderingContext2D, x: number, y: number, width: number, height: number, radius: number): void {
  ctx.beginPath();
  if (typeof ctx.roundRect === 'function') ctx.roundRect(x, y, width, height, radius);
  else ctx.rect(x, y, width, height);
  ctx.fill();
}

/** Paint the hairline timeline of a video: the line, its played part, the second ticks, the bloom and the playhead. */
export function drawVideoTimeline(ctx: CanvasRenderingContext2D, view: TimelineView, colors: TimelineColors, showPlayhead = true): void {
  const { width, height } = view;
  ctx.clearRect(0, 0, width, height);
  const base = height - BASE_GAP;
  const playheadX = view.fraction * width;

  ctx.fillStyle = colors.shade;
  ctx.fillRect(0, base + HAIRLINE, width, 1);
  ctx.fillStyle = colors.rest;
  ctx.fillRect(0, base, width, HAIRLINE);
  if (!showPlayhead) return;
  ctx.fillStyle = colors.played;
  ctx.fillRect(0, base, playheadX, HAIRLINE);

  ctx.fillStyle = colors.tick;
  for (const x of secondTicks(view.duration, width)) ctx.fillRect(x, base + 5, 1, 4);

  for (const bar of bloomBars(view)) {
    ctx.fillStyle = bar.played ? colors.played : colors.bloomRest;
    fillRound(ctx, bar.x - 1, base + HAIRLINE - bar.height, 2, bar.height, 1);
  }

  const grow = view.scrubbing;
  ctx.save();
  ctx.shadowColor = colors.played;
  ctx.shadowBlur = grow ? 12 : 6;
  ctx.fillStyle = colors.playhead;
  fillRound(ctx, Math.min(Math.max(playheadX, 1.5), width - 1.5) - 1.5, grow ? base - 17 : base - 7, 3, grow ? 26 : 16, 1.5);
  ctx.restore();
}

/** Return the height of bar *index* of *count* as a value from 0 to 255: the tallest of the peaks its slice of the take covers. */
export function waveValue(peaks: readonly number[], index: number, count: number): number {
  if (peaks.length === 0) return FLAT_PEAK;
  const start = Math.floor((index * peaks.length) / count);
  const end = Math.max(start + 1, Math.floor(((index + 1) * peaks.length) / count));
  let tallest = 0;
  for (let i = start; i < Math.min(end, peaks.length); i += 1) tallest = Math.max(tallest, peaks[i]);
  return Math.min(255, Math.max(0, tallest));
}

/** Return how many bars fit across *width*: 4 px wide with a 3 px gap. */
export function waveBarCount(width: number): number {
  return Math.max(1, Math.floor((width + WAVE_GAP) / (WAVE_BAR + WAVE_GAP)));
}

/** Return how many of a take's bars the playhead has passed, counted from the left: a bar is played once its middle is. */
export function wavePlayedBars(width: number, fraction: number): number {
  const count = waveBarCount(width);
  const left = (width - (count * (WAVE_BAR + WAVE_GAP) - WAVE_GAP)) / 2;
  const reached = (fraction * width - left - WAVE_BAR / 2) / (WAVE_BAR + WAVE_GAP);
  return reached < 0 ? 0 : Math.min(count, Math.floor(reached) + 1);
}

/** Paint a take's waveform: mirrored bars around the middle, the played ones in the accent colour. Progress is the colour alone. */
export function drawWaveform(ctx: CanvasRenderingContext2D, view: Pick<TimelineView, 'width' | 'height' | 'fraction'>, peaks: readonly number[], colors: TimelineColors): void {
  const { width, height } = view;
  ctx.clearRect(0, 0, width, height);
  const count = waveBarCount(width);
  const span = count * (WAVE_BAR + WAVE_GAP) - WAVE_GAP;
  const left = (width - span) / 2;
  const played = wavePlayedBars(width, view.fraction);
  for (let i = 0; i < count; i += 1) {
    const x = left + i * (WAVE_BAR + WAVE_GAP);
    const bar = Math.max(WAVE_MIN_PX, Math.max(WAVE_FLOOR, waveValue(peaks, i, count) / 255) * height);
    ctx.fillStyle = i < played ? colors.played : colors.waveRest;
    fillRound(ctx, x, (height - bar) / 2, WAVE_BAR, bar, 2);
  }
}
