/**
 * Pure crop, rotate and flip geometry for Adjust.
 *
 * The displayed frame is `FlipH^flip(RotCW^q(upright source))` with `q = rotate / 90`. Its size is the upright
 * `W × H`, swapped when `q` is odd. A crop is integer pixels of the displayed frame. The server applies the same
 * order (rotate, flip, crop), so these definitions must stay in step with it.
 */

/** Clockwise quarter turns, in degrees. */
export type Rotate = 0 | 90 | 180 | 270;

/** A crop box in whole pixels of the displayed frame. */
export interface Crop {
  x: number;
  y: number;
  width: number;
  height: number;
}

/** The geometry of an adjusted image. `crop` is null for the whole frame. */
export interface Geometry {
  rotate: Rotate;
  flip: boolean;
  crop: Crop | null;
}

export type AspectPreset = 'free' | 'original' | '1:1' | '4:5' | '3:2' | '16:9' | '9:16';

/** A crop handle: an edge or a corner. */
export type Handle = 'n' | 's' | 'e' | 'w' | 'ne' | 'nw' | 'se' | 'sw';

/** The chosen shape chip and its width / height ratio (null for Free). */
export interface Aspect {
  preset: AspectPreset;
  ratio: number | null;
}

export interface Size {
  width: number;
  height: number;
}

/** A point `[x, y]`. */
export type Point = readonly [number, number];

/** CSS that shows the unadjusted `<img>` as the displayed region. */
export interface CssFrame {
  /** `transform` for the `<img>`, which is laid out at its natural size with `transform-origin: 0 0` at the container's top-left. */
  transform: string;
  /** Size in CSS pixels of the visible region: the container, with `overflow: hidden`. */
  width: number;
  height: number;
  /** `clip-path` for the `<img>`, in its untransformed pixels (`none` for the whole frame). */
  clip: string;
}

export type CropRegion = 'frame' | 'crop';

const PRESET_RATIOS: Record<Exclude<AspectPreset, 'free' | 'original'>, number> = {
  '1:1': 1,
  '4:5': 4 / 5,
  '3:2': 3 / 2,
  '16:9': 16 / 9,
  '9:16': 9 / 16,
};

/** The quarter-turn count of a rotation. */
function quarterTurns(rotate: Rotate): number {
  return rotate / 90;
}

function clamp(value: number, lo: number, hi: number): number {
  return Math.min(Math.max(value, lo), hi);
}

/** Return the displayed frame's size for an upright `W × H` image. */
export function frameSize(rotate: Rotate, W: number, H: number): Size {
  return quarterTurns(rotate) % 2 === 1 ? { width: H, height: W } : { width: W, height: H };
}

/** Rotate a crop a quarter turn clockwise inside a frame of the given (pre-rotation) size. */
function cropRight(c: Crop, frame: Size): Crop {
  return { x: frame.height - c.y - c.height, y: c.x, width: c.height, height: c.width };
}

/** Rotate a crop a quarter turn anticlockwise inside a frame of the given (pre-rotation) size. */
function cropLeft(c: Crop, frame: Size): Crop {
  return { x: c.y, y: frame.width - c.x - c.width, width: c.height, height: c.width };
}

/** Rotate the displayed frame a quarter turn, carrying the crop along. */
export function rotateGeometry(g: Geometry, direction: 'left' | 'right', W: number, H: number): Geometry {
  const right = direction === 'right';
  const step = right !== g.flip ? 1 : -1;
  const rotate = ((((quarterTurns(g.rotate) + step) % 4) + 4) % 4) * 90;
  const frame = frameSize(g.rotate, W, H);
  const crop = g.crop ? (right ? cropRight(g.crop, frame) : cropLeft(g.crop, frame)) : null;
  return { rotate: rotate as Rotate, flip: g.flip, crop };
}

/** Mirror the displayed frame left to right, carrying the crop along. */
export function flipGeometry(g: Geometry, W: number, H: number): Geometry {
  const frame = frameSize(g.rotate, W, H);
  const crop = g.crop ? { ...g.crop, x: frame.width - g.crop.x - g.crop.width } : null;
  return { rotate: g.rotate, flip: !g.flip, crop };
}

/** Return the width / height ratio of a shape chip, or null for Free. */
export function presetRatio(preset: AspectPreset, frame: Size): number | null {
  if (preset === 'free') return null;
  if (preset === 'original') return frame.width / frame.height;
  return PRESET_RATIOS[preset];
}

/** Whether Swap does anything: it is off for Free and for square shapes. */
export function canSwapRatio(aspect: Aspect): boolean {
  return aspect.ratio !== null && Math.abs(aspect.ratio - 1) > 1e-9;
}

/** Invert the ratio. 16:9 and 9:16 trade chips; every other chip stays selected. */
export function swapRatio(aspect: Aspect): Aspect {
  if (!canSwapRatio(aspect) || aspect.ratio === null) return aspect;
  if (aspect.preset === '16:9') return { preset: '9:16', ratio: PRESET_RATIOS['9:16'] };
  if (aspect.preset === '9:16') return { preset: '16:9', ratio: PRESET_RATIOS['16:9'] };
  return { preset: aspect.preset, ratio: 1 / aspect.ratio };
}

/** Label a shape chip, showing the swapped form ("5:4", "2:3") when the ratio was inverted. */
export function aspectLabel(aspect: Aspect): string {
  const { preset, ratio } = aspect;
  if (preset === 'free') return 'Free';
  if (preset === 'original') return 'Original';
  const base = PRESET_RATIOS[preset];
  if (ratio !== null && base !== 1 && Math.abs(ratio - 1 / base) < Math.abs(ratio - base)) {
    return preset.split(':').reverse().join(':');
  }
  return preset;
}

/** Return the largest box of `ratio` in the frame, centred on `crop` (or the frame), clamped and in whole pixels. */
export function fitAspect(crop: Crop | null, ratio: number, frame: Size): Crop {
  const widthLimited = ratio >= frame.width / frame.height;
  const width = Math.max(1, widthLimited ? frame.width : Math.round(frame.height * ratio));
  const height = Math.max(1, widthLimited ? Math.round(frame.width / ratio) : frame.height);
  const cx = crop ? crop.x + crop.width / 2 : frame.width / 2;
  const cy = crop ? crop.y + crop.height / 2 : frame.height / 2;
  return {
    x: clamp(Math.round(cx - width / 2), 0, Math.max(0, frame.width - width)),
    y: clamp(Math.round(cy - height / 2), 0, Math.max(0, frame.height - height)),
    width: Math.min(width, frame.width),
    height: Math.min(height, frame.height),
  };
}

/** The sign (-1, 0, +1) a handle moves each axis with. */
function handleSigns(handle: Handle): [number, number] {
  const sx = handle.includes('e') ? 1 : handle.includes('w') ? -1 : 0;
  const sy = handle.includes('s') ? 1 : handle.includes('n') ? -1 : 0;
  return [sx, sy];
}

/** Resize one axis of a free drag: the opposite edge stays put. Returns `[origin, length]`. */
function resizeAxis(
  origin: number,
  length: number,
  delta: number,
  sign: number,
  limit: number,
  minSize: number,
): [number, number] {
  if (sign === 0) return [origin, length];
  const anchor = sign > 0 ? origin : origin + length;
  const room = sign > 0 ? limit - anchor : anchor;
  const moved = (sign > 0 ? origin + length : origin) + delta;
  const size = clamp(Math.round((moved - anchor) * sign), Math.min(minSize, room), room);
  return [sign > 0 ? anchor : anchor - size, size];
}

/** Resize from a corner with a locked ratio: the opposite corner stays put. */
function dragLocked(
  start: Crop,
  sx: number,
  sy: number,
  dx: number,
  dy: number,
  ratio: number,
  frame: Size,
  minSize: number,
): Crop {
  const ax = sx > 0 ? start.x : start.x + start.width;
  const ay = sy > 0 ? start.y : start.y + start.height;
  const wRaw = ((sx > 0 ? start.x + start.width : start.x) + dx - ax) * sx;
  const hRaw = ((sy > 0 ? start.y + start.height : start.y) + dy - ay) * sy;
  const maxW = sx > 0 ? frame.width - ax : ax;
  const maxH = sy > 0 ? frame.height - ay : ay;
  // Project the pointer onto the diagonal of the ratio, then stay inside the frame and above the minimum.
  const t = (wRaw * ratio + hRaw) / (ratio * ratio + 1);
  const hMax = Math.max(1, Math.floor(Math.min(maxH, maxW / ratio) + 1e-9));
  const hMin = Math.min(Math.ceil(Math.max(minSize, minSize / ratio)), hMax);
  let height = clamp(Math.round(t), hMin, hMax);
  let width = Math.round(height * ratio);
  while (width > maxW && height > 1) {
    height -= 1;
    width = Math.round(height * ratio);
  }
  return { x: sx > 0 ? ax : ax - width, y: sy > 0 ? ay : ay - height, width, height };
}

/**
 * Return the crop after dragging a handle by `(dx, dy)` source pixels from `start`.
 *
 * It always works from the start state, so a drag never drifts. Edges resize only in Free (`ratio` null); a locked
 * ratio resizes from corners only. The result stays inside the frame and at least `minSize` on each side.
 */
export function dragCrop(
  start: Crop,
  handle: Handle,
  dx: number,
  dy: number,
  ratio: number | null,
  frame: Size,
  minSize: number,
): Crop {
  const [sx, sy] = handleSigns(handle);
  if (ratio === null) {
    const [x, width] = resizeAxis(start.x, start.width, dx, sx, frame.width, minSize);
    const [y, height] = resizeAxis(start.y, start.height, dy, sy, frame.height, minSize);
    return { x, y, width, height };
  }
  if (sx === 0 || sy === 0) return start;
  return dragLocked(start, sx, sy, dx, dy, ratio, frame, minSize);
}

/** Return the crop moved by `(dx, dy)` source pixels from `start`, kept inside the frame. */
export function moveCrop(start: Crop, dx: number, dy: number, frame: Size): Crop {
  return {
    ...start,
    x: clamp(Math.round(start.x + dx), 0, Math.max(0, frame.width - start.width)),
    y: clamp(Math.round(start.y + dy), 0, Math.max(0, frame.height - start.height)),
  };
}

const ARROWS: Record<string, [number, number]> = {
  ArrowLeft: [-1, 0],
  ArrowRight: [1, 0],
  ArrowUp: [0, -1],
  ArrowDown: [0, 1],
};

/** Return the crop moved by an arrow key: `max(1, round(min(Wf, Hf) / 100))` pixels, ten times that with Shift. */
export function nudgeCrop(crop: Crop, key: string, shift: boolean, frame: Size): Crop {
  const dir = ARROWS[key];
  if (!dir) return crop;
  const step = Math.max(1, Math.round(Math.min(frame.width, frame.height) / 100)) * (shift ? 10 : 1);
  return moveCrop(crop, dir[0] * step, dir[1] * step, frame);
}

/** Return null when the crop is missing or covers the whole frame, else the crop. */
export function normalizedCrop(crop: Crop | null, frame: Size): Crop | null {
  if (!crop) return null;
  const whole = crop.x === 0 && crop.y === 0 && crop.width === frame.width && crop.height === frame.height;
  return whole ? null : crop;
}

/** The displayed region as a crop: the crop when asked for, else the whole frame. */
function regionCrop(g: Geometry, frame: Size, region: CropRegion): Crop {
  return region === 'crop' && g.crop ? g.crop : { x: 0, y: 0, width: frame.width, height: frame.height };
}

/**
 * Return the source UVs (0..1, origin top-left) of the displayed region's corners: top-left, top-right,
 * bottom-right, bottom-left.
 */
export function quadCorners(g: Geometry, W: number, H: number, region: CropRegion): [Point, Point, Point, Point] {
  const frame = frameSize(g.rotate, W, H);
  const r = regionCrop(g, frame, region);
  const q = quarterTurns(g.rotate);
  const toSource = (fx: number, fy: number): Point => {
    let u = g.flip ? 1 - fx / frame.width : fx / frame.width;
    let v = fy / frame.height;
    for (let i = 0; i < q; i += 1) [u, v] = [v, 1 - u];
    return [u, v];
  };
  return [
    toSource(r.x, r.y),
    toSource(r.x + r.width, r.y),
    toSource(r.x + r.width, r.y + r.height),
    toSource(r.x, r.y + r.height),
  ];
}

/** A 2D affine matrix `[a, b, c, d, e, f]` in CSS `matrix()` order. */
type Matrix = [number, number, number, number, number, number];

/** Return `m1 ∘ m2`: apply `m2` first. */
function compose(m1: Matrix, m2: Matrix): Matrix {
  const [a1, b1, c1, d1, e1, f1] = m1;
  const [a2, b2, c2, d2, e2, f2] = m2;
  return [
    a1 * a2 + c1 * b2,
    b1 * a2 + d1 * b2,
    a1 * c2 + c1 * d2,
    b1 * c2 + d1 * d2,
    a1 * e2 + c1 * f2 + e1,
    b1 * e2 + d1 * f2 + f1,
  ];
}

/** The matrix taking upright source pixels to pixels of the displayed frame. */
function frameMatrix(g: Geometry, W: number, H: number): Matrix {
  const rotations: Record<Rotate, Matrix> = {
    0: [1, 0, 0, 1, 0, 0],
    90: [0, 1, -1, 0, H, 0],
    180: [-1, 0, 0, -1, W, H],
    270: [0, -1, 1, 0, 0, W],
  };
  const turned = rotations[g.rotate];
  const frame = frameSize(g.rotate, W, H);
  return g.flip ? compose([-1, 0, 0, 1, frame.width, 0], turned) : turned;
}

/** Format a number for CSS: rounded, and never "-0". */
function css(value: number): string {
  const rounded = Math.round(value * 1e6) / 1e6;
  return String(rounded === 0 ? 0 : rounded);
}

/**
 * Return CSS that shows the unadjusted `<img>` as the displayed region, for browsers without WebGL.
 *
 * `box` is the room available; the region is scaled to fit it. Lay the `<img>` out at its natural `W × H` inside a
 * container of the returned `width × height` with `overflow: hidden`, and apply `transform` and `clip` to the `<img>`.
 */
export function cssFrameTransform(g: Geometry, W: number, H: number, box: Size, region: CropRegion): CssFrame {
  const frame = frameSize(g.rotate, W, H);
  const r = regionCrop(g, frame, region);
  const scale = Math.min(box.width / r.width, box.height / r.height);
  const view: Matrix = [scale, 0, 0, scale, -r.x * scale, -r.y * scale];
  const m = compose(view, frameMatrix(g, W, H));
  // The region in source pixels, for the clip. Corners 0 and 2 are opposite, so they bound the region.
  const [tl, , br] = quadCorners(g, W, H, region);
  const left = Math.min(tl[0], br[0]) * W;
  const right = Math.max(tl[0], br[0]) * W;
  const top = Math.min(tl[1], br[1]) * H;
  const bottom = Math.max(tl[1], br[1]) * H;
  const whole = left <= 0 && top <= 0 && right >= W && bottom >= H;
  return {
    transform: `matrix(${m.map(css).join(', ')})`,
    width: r.width * scale,
    height: r.height * scale,
    clip: whole ? 'none' : `inset(${css(top)}px ${css(W - right)}px ${css(H - bottom)}px ${css(left)}px)`,
  };
}
