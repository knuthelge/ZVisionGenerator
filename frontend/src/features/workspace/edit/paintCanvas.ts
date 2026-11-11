import type { Rect, Stroke } from './editOps';
import { toScreen, type Size, type View } from './paintView';

/** The mask is kept at no more than this many pixels, which is what iOS allows a canvas; it is scaled up when a job is sent. */
const MASK_MAX_PIXELS = 4_000_000;

/** The scale that keeps a mask of `width × height` within `MASK_MAX_PIXELS`. */
export function maskScale(width: number, height: number): number {
  return Math.min(1, Math.sqrt(MASK_MAX_PIXELS / Math.max(1, width * height)));
}

/** The colours the painter draws with, read from the design tokens (canvas code cannot use `var()`). */
export interface Palette {
  fill: string;
  edge: string;
  checkerLight: string;
  checkerDark: string;
}

/** Read the painter's colours from the tokens in force at `element`. */
export function readPalette(element: Element): Palette {
  const style = getComputedStyle(element);
  const token = (name: string): string => style.getPropertyValue(name).trim();
  return {
    fill: token('--color-primary-main'),
    edge: token('--color-primary-ink'),
    checkerLight: token('--color-checker-light'),
    checkerDark: token('--color-checker-dark'),
  };
}

/** A 2D context of a canvas, or null where canvases are not drawn (a test environment). */
function context2d(canvas: HTMLCanvasElement): CanvasRenderingContext2D | null {
  try {
    return canvas.getContext('2d', { willReadFrequently: false });
  } catch {
    return null;
  }
}

/**
 * Draw one stroke, in full or from its point `from` on, onto a mask whose pixels are `scale` times the picture's. A stroke is white at full
 * alpha: the paint is the alpha, and a canvas painted completely is an opaque white PNG, which the server reads as fully painted.
 */
export function drawStroke(ctx: CanvasRenderingContext2D, stroke: Stroke, scale: number, from = 0): void {
  const { points, size, erase } = stroke;
  if (points.length < 2) return;
  ctx.save();
  ctx.globalCompositeOperation = erase ? 'destination-out' : 'source-over';
  ctx.strokeStyle = '#fff';
  ctx.fillStyle = '#fff';
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.lineWidth = size * scale;
  const start = Math.max(0, from - 2);
  if (from === 0) {
    ctx.beginPath();
    ctx.arc(points[0] * scale, points[1] * scale, (size * scale) / 2, 0, Math.PI * 2);
    ctx.fill();
  }
  if (points.length - start > 2) {
    ctx.beginPath();
    ctx.moveTo(points[start] * scale, points[start + 1] * scale);
    for (let index = start + 2; index < points.length; index += 2) ctx.lineTo(points[index] * scale, points[index + 1] * scale);
    ctx.stroke();
  }
  ctx.restore();
}

/** Draw all strokes onto a cleared mask, over `base` (the paint the strokes sit on) when there is one. */
export function drawStrokes(ctx: CanvasRenderingContext2D, strokes: readonly Stroke[], scale: number, width: number, height: number, base: CanvasImageSource | null = null): void {
  ctx.clearRect(0, 0, width, height);
  if (base) ctx.drawImage(base, 0, 0, width, height);
  for (const stroke of strokes) drawStroke(ctx, stroke, scale);
}

/** The scratch canvases the overlay draws through, kept between frames. */
interface OverlayScratch {
  silhouette: HTMLCanvasElement;
  ring: HTMLCanvasElement;
  fill: HTMLCanvasElement;
}

export function makeScratch(): OverlayScratch {
  return { silhouette: document.createElement('canvas'), ring: document.createElement('canvas'), fill: document.createElement('canvas') };
}

const RING_OFFSETS: readonly (readonly [number, number])[] = [[1, 0], [-1, 0], [0, 1], [0, -1], [0.7, 0.7], [-0.7, 0.7], [0.7, -0.7], [-0.7, -0.7]];

/** The width of the mask's edge, in device pixels at a device pixel ratio of 1. */
const EDGE_PX = 2;
/** How much of the fill shows through: 55%. */
const FILL_ALPHA = 0.55;

/**
 * Draw the paint over the picture: the mask as a teal fill at 55% with a 2 px dark edge outside it, and the 2 px edge at the join
 * of an extension. The view's transform is applied here, so the edge stays 2 px at every zoom.
 */
export function drawOverlay(args: {
  canvas: HTMLCanvasElement;
  scratch: OverlayScratch;
  mask: HTMLCanvasElement | null;
  /** The picture's size, in image pixels. */
  image: Size;
  view: View;
  fit: number;
  area: Size;
  dpr: number;
  palette: Palette;
  /** The new area of an extension, in image pixels; its join with the picture gets the edge. */
  join: { rect: Rect; side: 'top' | 'right' | 'bottom' | 'left' } | null;
}): void {
  const { canvas, scratch, mask, image, view, fit, area, dpr, palette, join } = args;
  const ctx = context2d(canvas);
  if (!ctx) return;
  const pw = Math.max(1, Math.round(area.w * dpr));
  const ph = Math.max(1, Math.round(area.h * dpr));
  if (canvas.width !== pw || canvas.height !== ph) {
    canvas.width = pw;
    canvas.height = ph;
  }
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, pw, ph);

  const origin = toScreen(view, fit, area, { x: 0, y: 0 });
  const s = fit * view.z * dpr;

  if (mask) {
    for (const layer of [scratch.silhouette, scratch.ring, scratch.fill]) {
      if (layer.width !== pw || layer.height !== ph) {
        layer.width = pw;
        layer.height = ph;
      }
    }
    const sil = scratch.silhouette.getContext('2d');
    const ring = scratch.ring.getContext('2d');
    const fill = scratch.fill.getContext('2d');
    if (sil && ring && fill) {
      sil.setTransform(1, 0, 0, 1, 0, 0);
      sil.clearRect(0, 0, pw, ph);
      sil.imageSmoothingEnabled = true;
      sil.imageSmoothingQuality = 'high';
      sil.drawImage(mask, origin.x * dpr, origin.y * dpr, image.w * s, image.h * s);

      ring.globalCompositeOperation = 'source-over';
      ring.clearRect(0, 0, pw, ph);
      const reach = EDGE_PX * dpr;
      for (const [dx, dy] of RING_OFFSETS) ring.drawImage(scratch.silhouette, dx * reach, dy * reach);
      ring.globalCompositeOperation = 'source-in';
      ring.fillStyle = palette.edge;
      ring.fillRect(0, 0, pw, ph);
      ring.globalCompositeOperation = 'destination-out';
      ring.drawImage(scratch.silhouette, 0, 0);
      ring.globalCompositeOperation = 'source-over';

      fill.globalCompositeOperation = 'source-over';
      fill.clearRect(0, 0, pw, ph);
      fill.drawImage(scratch.silhouette, 0, 0);
      fill.globalCompositeOperation = 'source-in';
      fill.fillStyle = palette.fill;
      fill.fillRect(0, 0, pw, ph);
      fill.globalCompositeOperation = 'source-over';

      ctx.drawImage(scratch.ring, 0, 0);
      ctx.globalAlpha = FILL_ALPHA;
      ctx.drawImage(scratch.fill, 0, 0);
      ctx.globalAlpha = 1;
    }
  }

  if (join) {
    // The edge lies on the new area's side of the join, so the picture itself stays clear.
    const px = EDGE_PX * dpr;
    const left = origin.x * dpr + join.rect.x * s;
    const top = origin.y * dpr + join.rect.y * s;
    const w = join.rect.w * s;
    const h = join.rect.h * s;
    ctx.fillStyle = palette.edge;
    if (join.side === 'top') ctx.fillRect(left, top + h - px, w, px);
    else if (join.side === 'bottom') ctx.fillRect(left, top, w, px);
    else if (join.side === 'left') ctx.fillRect(left + w - px, top, px, h);
    else ctx.fillRect(left, top, px, h);
  }
}

/** Whether a loaded picture has transparent pixels (a cut-out), looked for on a small copy. Null where it cannot be read. */
export function detectCutout(image: HTMLImageElement): boolean | null {
  try {
    const long = 64;
    const scale = Math.min(1, long / Math.max(image.naturalWidth, image.naturalHeight));
    const canvas = document.createElement('canvas');
    canvas.width = Math.max(1, Math.round(image.naturalWidth * scale));
    canvas.height = Math.max(1, Math.round(image.naturalHeight * scale));
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) return null;
    ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
    const { data } = ctx.getImageData(0, 0, canvas.width, canvas.height);
    for (let index = 3; index < data.length; index += 4) if (data[index] < 250) return true;
    return false;
  } catch {
    return null;
  }
}

/** The picture of a mask's PNG; null where the browser cannot decode it (or has no bitmaps, as a test environment). */
export async function decodeMask(file: File | null): Promise<ImageBitmap | null> {
  if (file === null || typeof createImageBitmap !== 'function') return null;
  try {
    return await createImageBitmap(file);
  } catch {
    return null;
  }
}

/** Draw a mask as the teal fill over a small canvas that shows the picture contained in a box of `box × box` pixels. */
export function drawThumbMask(canvas: HTMLCanvasElement, mask: ImageBitmap | null, box: number, dpr: number, fill: string): void {
  const pixels = Math.round(box * dpr);
  // Resizing a canvas clears it, so no mask needs no context at all.
  canvas.width = pixels;
  canvas.height = pixels;
  if (mask === null || mask.width <= 0 || mask.height <= 0) return;
  const ctx = context2d(canvas);
  if (!ctx) return;
  const scale = Math.min(pixels / mask.width, pixels / mask.height);
  const w = Math.max(1, Math.round(mask.width * scale));
  const h = Math.max(1, Math.round(mask.height * scale));
  const layer = document.createElement('canvas');
  layer.width = w;
  layer.height = h;
  const lctx = layer.getContext('2d');
  if (!lctx) return;
  lctx.drawImage(mask, 0, 0, w, h);
  lctx.globalCompositeOperation = 'source-in';
  lctx.fillStyle = fill;
  lctx.fillRect(0, 0, w, h);
  ctx.globalAlpha = FILL_ALPHA;
  ctx.drawImage(layer, (pixels - w) / 2, (pixels - h) / 2);
  ctx.globalAlpha = 1;
}

/**
 * Encode a mask canvas as the PNG a job uploads, and measure it. The strokes are antialiased; the file keeps a pixel as painted when
 * it is at least half covered, so its edge is crisp and no faint fringe widens the paint. `share` is the painted part of the mask,
 * 0 to 1. Null when nothing is painted, or where the browser cannot read or encode the canvas.
 */
export async function maskUpload(mask: HTMLCanvasElement): Promise<{ file: File; share: number } | null> {
  try {
    const source = mask.getContext('2d', { willReadFrequently: true });
    if (!source) return null;
    const image = source.getImageData(0, 0, mask.width, mask.height);
    const { data } = image;
    let painted = 0;
    for (let index = 3; index < data.length; index += 4) {
      const on = data[index] >= 128;
      data[index] = on ? 255 : 0;
      if (on) painted += 1;
    }
    if (painted === 0) return null;
    const crisp = document.createElement('canvas');
    crisp.width = mask.width;
    crisp.height = mask.height;
    crisp.getContext('2d')?.putImageData(image, 0, 0);
    const blob = await new Promise<Blob | null>((resolve) => crisp.toBlob(resolve, 'image/png'));
    return blob ? { file: new File([blob], 'mask.png', { type: 'image/png' }), share: painted / (mask.width * mask.height) } : null;
  } catch {
    return null;
  }
}
