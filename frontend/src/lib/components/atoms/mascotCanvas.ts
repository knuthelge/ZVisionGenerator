/**
 * Canvas effects for the mascot's arrivals: he is sampled in like an image (Denoise in), and flies
 * between spots as a stream of pixels (Latent stream). Each runs on a temporary overlay, from one raster
 * of his SVG, and only for the length of a transition.
 */
import { clamp, hash, lerp, mix, noiseRGB, type RGB } from './mascotArt';

export const DENOISE_IN_MS = 800;
/** The most the stream's cells rise above the straight line between the spots, as a multiple of `hop`. */
export const STREAM_RISE = 1.3;

const DENOISE_STEP_MS = 85;
const DENOISE_SNAP_MS = 120;
/** Grid resolution per sampler step and the share of noise left after it. */
const DENOISE_GRID = [7, 10, 13, 18, 24, 34, 50, 75] as const;
const DENOISE_SIGMA = [1, 0.86, 0.72, 0.58, 0.44, 0.3, 0.17, 0.07] as const;
/** Cells per side of the pixel stream. */
const STREAM_CELLS = 16;

const sine = (t: number): number => 0.5 - 0.5 * Math.cos(Math.PI * t);
const rgba = (c: RGB, a: number): string => `rgba(${Math.round(c[0])},${Math.round(c[1])},${Math.round(c[2])},${a.toFixed(3)})`;

/** Report whether this browser can rasterise the mascot and draw the overlay. jsdom cannot. */
export function canvasSupported(): boolean {
  return typeof document !== 'undefined' && typeof Image !== 'undefined' && typeof CanvasRenderingContext2D !== 'undefined';
}

/** One raster of the mascot, and the size in CSS px it stands for. */
export interface MascotSprite {
  width: number;
  height: number;
  canvas: HTMLCanvasElement;
}

/** Where a sprite is drawn, in page (viewport) coordinates. */
export interface Placement {
  sprite: MascotSprite;
  x: number;
  y: number;
  width: number;
  height: number;
}

/** A sampler step: a grid of cells, each his raster at that resolution mixed with noise. */
interface DenoiseStep {
  n: number;
  cells: { gx: number; gy: number; fill: string }[];
}

interface StreamCell {
  gx: number;
  gy: number;
  col: RGB;
  /** The noise colour the cell turns toward in mid-flight. */
  noise: RGB;
  h1: number;
  h3: number;
}

function makeCanvas(width: number, height: number): HTMLCanvasElement {
  const canvas = document.createElement('canvas');
  canvas.width = Math.max(1, Math.round(width));
  canvas.height = Math.max(1, Math.round(height));
  return canvas;
}

/** Average the raster down to an n x n grid of RGBA values. */
function sample(sprite: MascotSprite, n: number): Uint8ClampedArray {
  const small = makeCanvas(n, n);
  const ctx = small.getContext('2d', { willReadFrequently: true })!;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(sprite.canvas, 0, 0, n, n);
  return ctx.getImageData(0, 0, n, n).data;
}

/** Work out the eight sampler steps of Denoise in once, before it plays: the noise is the costly part. */
export function denoiseSteps(sprite: MascotSprite): DenoiseStep[] {
  return DENOISE_GRID.map((grid, step) => {
    const n = Math.min(grid, Math.floor(Math.min(sprite.width, sprite.height) / 2));
    const data = sample(sprite, n);
    const sigma = DENOISE_SIGMA[step]!;
    const cells: DenoiseStep['cells'] = [];
    for (let gy = 0; gy < n; gy += 1) {
      for (let gx = 0; gx < n; gx += 1) {
        const i = (gy * n + gx) * 4;
        const ta = data[i + 3]! / 255;
        // Noise is thickest around his body and dies away toward the corners.
        const e = (((gx + 0.5) / n - 0.5) / 0.34) ** 2 + (((gy + 0.5) / n - 0.57) / 0.36) ** 2;
        const m = e > 1.7 ? 0 : Math.exp(-e * 1.3);
        const an = m > 0 ? clamp(m * 1.5 - 0.25 + (hash(gx * 7 + n, gy * 7, step * 13 + 5) - 0.5) * 0.9) : 0;
        const alpha = ta * (1 - sigma) + an * sigma;
        if (alpha < 0.03) continue;
        const noise = noiseRGB(gx * 3 + n, gy * 5, step * 29 + 5);
        const w1 = ta * (1 - sigma);
        const w2 = an * sigma;
        const colour: RGB = [0, 1, 2].map((j) => (data[i + j]! * w1 + noise[j]! * w2) / (w1 + w2 || 1)) as unknown as RGB;
        cells.push({ gx, gy, fill: rgba(colour, alpha) });
      }
    }
    return { n, cells };
  });
}

/** Cut his raster into the pixel stream's 16 x 16 cells, once, before it plays. */
export function streamCells(sprite: MascotSprite): StreamCell[] {
  const n = STREAM_CELLS;
  const data = sample(sprite, n);
  const cells: StreamCell[] = [];
  for (let gy = 0; gy < n; gy += 1) {
    for (let gx = 0; gx < n; gx += 1) {
      const i = (gy * n + gx) * 4;
      if (data[i + 3]! < 120) continue;
      cells.push({ gx, gy, col: [data[i]!, data[i + 1]!, data[i + 2]!], noise: noiseRGB(gx, gy, 12), h1: hash(gx, gy, 1), h3: hash(gx, gy, 3) });
    }
  }
  return cells;
}

const BAKED_PROPERTIES = ['opacity', 'transform', 'transform-origin', 'transform-box', 'stroke-width'] as const;

/**
 * Serialise the mascot's SVG as a standalone image: the pose each part has right now is written
 * into its style, parts that are hidden are dropped, and the mood effects are left out.
 */
export function bakeMascotMarkup(svg: SVGSVGElement, width: number, height: number): string {
  const clone = svg.cloneNode(true) as SVGSVGElement;
  const sources = svg.querySelectorAll('*');
  const copies = clone.querySelectorAll('*');
  sources.forEach((source, i) => {
    const copy = copies[i];
    if (!copy) return;
    if (source.classList.contains('fx')) {
      copy.remove();
      return;
    }
    if (!source.hasAttribute('class')) return;
    const style = getComputedStyle(source);
    if (style.display === 'none' || style.opacity === '0') {
      copy.remove();
      return;
    }
    for (const property of BAKED_PROPERTIES) {
      const value = style.getPropertyValue(property);
      if (value && value !== 'none') (copy as SVGElement).style.setProperty(property, value);
    }
    copy.removeAttribute('class');
  });
  clone.removeAttribute('class');
  clone.removeAttribute('style');
  clone.removeAttribute('aria-label');
  clone.removeAttribute('role');
  clone.setAttribute('width', String(width));
  clone.setAttribute('height', String(height));
  return new XMLSerializer().serializeToString(clone);
}

/** Rasterise baked SVG markup at the given device pixel ratio; resolves to null when the browser cannot. */
export async function rasterizeMascot(markup: string, width: number, height: number, dpr: number): Promise<MascotSprite | null> {
  if (!canvasSupported() || width <= 0 || height <= 0) return null;
  try {
    const image = new Image();
    image.src = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(markup)}`;
    await image.decode();
    const canvas = makeCanvas(width * dpr, height * dpr);
    const ctx = canvas.getContext('2d');
    if (!ctx) return null;
    ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
    return { width, height, canvas };
  } catch {
    return null;
  }
}

/** Draw the crisp vector raster. */
function drawCrisp(ctx: CanvasRenderingContext2D, at: Placement, alpha = 1): void {
  if (alpha <= 0) return;
  ctx.globalAlpha = alpha;
  ctx.drawImage(at.sprite.canvas, at.x, at.y, at.width, at.height);
  ctx.globalAlpha = 1;
}

/** Draw Denoise in onto a cleared canvas: eight steps of coarse blocks resolving to fine pixels, then a snap to the crisp raster. */
export function drawDenoiseIn(ctx: CanvasRenderingContext2D, at: Placement, steps: DenoiseStep[], ms: number): void {
  const index = Math.max(0, Math.floor(ms / DENOISE_STEP_MS));
  const step = steps[Math.min(index, steps.length - 1)]!;
  const snap = index < steps.length ? 0 : clamp((ms - steps.length * DENOISE_STEP_MS) / DENOISE_SNAP_MS);
  const cw = at.width / step.n;
  const ch = at.height / step.n;
  ctx.globalAlpha = 1 - snap;
  for (const cell of step.cells) {
    ctx.fillStyle = cell.fill;
    ctx.fillRect(at.x + cell.gx * cw, at.y + cell.gy * ch, cw + 0.5, ch + 0.5);
  }
  ctx.globalAlpha = 1;
  drawCrisp(ctx, at, snap);
}

/**
 * When a stream cell leaves, as a share of the flight (before the per-cell spread). Cells that face the
 * destination leave first: `dx`, `dy` is the direction of travel, and a cell's rank is how far along
 * that direction it sits within the grid.
 */
function streamDelay(gx: number, gy: number, dx: number, dy: number, spread: number): number {
  const span = Math.abs(dx) + Math.abs(dy);
  const along = span === 0 ? 0.5 : ((((gx + 0.5) / STREAM_CELLS - 0.5) * dx + ((gy + 0.5) / STREAM_CELLS - 0.5) * dy) / span) + 0.5;
  return 0.06 + 0.36 * (1 - clamp(along)) + 0.1 * spread;
}

/**
 * Draw one frame of the pixel stream onto a cleared canvas: he breaks into a 16 x 16 grid whose cells
 * fly along an arc, noisier in mid-flight, and reassemble at the new spot and size. `p` runs from 0 to 1.
 */
export function drawStream(ctx: CanvasRenderingContext2D, from: Placement, to: Placement, cells: StreamCell[], p: number, hop: number): void {
  const n = STREAM_CELLS;
  const brk = clamp(p / 0.08);
  const land = clamp((p - 0.88) / 0.12);
  drawCrisp(ctx, from, 1 - brk);
  const fromW = from.width / n;
  const fromH = from.height / n;
  const toW = to.width / n;
  const toH = to.height / n;
  const dx = to.x + to.width / 2 - (from.x + from.width / 2);
  const dy = to.y + to.height / 2 - (from.y + from.height / 2);
  for (const cell of cells) {
    const u = clamp((p - streamDelay(cell.gx, cell.gy, dx, dy, cell.h3)) / 0.42);
    const e = sine(u);
    const px = lerp(from.x + cell.gx * fromW, to.x + cell.gx * toW, e);
    const py = lerp(from.y + cell.gy * fromH, to.y + cell.gy * toH, e) - Math.sin(Math.PI * e) * hop * (0.6 + 0.7 * cell.h1);
    const side = 1 - 0.15 * Math.sin(Math.PI * u);
    const w = lerp(fromW, toW, e) * side - lerp(0.6, 0, land);
    const h = lerp(fromH, toH, e) * side - lerp(0.6, 0, land);
    ctx.fillStyle = rgba(mix(cell.col, cell.noise, 0.65 * Math.sin(Math.PI * u)), brk * (1 - land * 0.3));
    ctx.fillRect(px, py, w, h);
  }
  drawCrisp(ctx, to, land);
}

/** A rectangle in page (viewport) coordinates. */
export interface Bounds {
  x: number;
  y: number;
  width: number;
  height: number;
}

/** Grow a rectangle by a margin on each side. */
export function padBounds(rect: Bounds, left: number, top = left, right = left, bottom = top): Bounds {
  return { x: rect.x - left, y: rect.y - top, width: rect.width + left + right, height: rect.height + top + bottom };
}

/** The smallest rectangle that holds both. */
export function unionBounds(a: Bounds, b: Bounds): Bounds {
  const x = Math.min(a.x, b.x);
  const y = Math.min(a.y, b.y);
  return { x, y, width: Math.max(a.x + a.width, b.x + b.width) - x, height: Math.max(a.y + a.height, b.y + b.height) - y };
}

export interface Overlay {
  ctx: CanvasRenderingContext2D;
  /** Move and resize the canvas to cover these bounds (cut to the viewport); drawing keeps using page coordinates. */
  fit(bounds: Bounds): void;
  clear(): void;
  remove(): void;
}

/**
 * Create a pointer-transparent canvas fixed above the page, sized to the area an effect covers rather than the
 * whole viewport. It sits on the mascot's effect layer, or on `layer` when his spot is in a dialog or sheet.
 */
export function createOverlay(bounds: Bounds, dpr: number, layer?: number): Overlay | null {
  const canvas = makeCanvas(1, 1);
  const ctx = canvas.getContext('2d');
  if (!ctx) return null;
  canvas.style.cssText = `position: fixed; margin: 0; z-index: ${layer ?? 'var(--z-mascot-effect)'}; pointer-events: none;`;
  let current: Bounds | null = null;
  const overlay: Overlay = {
    ctx,
    fit(next) {
      const x = Math.max(0, Math.floor(next.x));
      const y = Math.max(0, Math.floor(next.y));
      const width = Math.max(1, Math.min(window.innerWidth, Math.ceil(next.x + next.width)) - x);
      const height = Math.max(1, Math.min(window.innerHeight, Math.ceil(next.y + next.height)) - y);
      if (current && current.x === x && current.y === y && current.width === width && current.height === height) return;
      current = { x, y, width, height };
      canvas.width = Math.ceil(width * dpr);
      canvas.height = Math.ceil(height * dpr);
      canvas.style.left = `${x}px`;
      canvas.style.top = `${y}px`;
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
      ctx.setTransform(dpr, 0, 0, dpr, -x * dpr, -y * dpr);
    },
    clear() {
      if (current) ctx.clearRect(current.x, current.y, current.width, current.height);
    },
    remove() {
      canvas.remove();
    },
  };
  overlay.fit(bounds);
  document.body.append(canvas);
  return overlay;
}
