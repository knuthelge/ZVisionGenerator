/**
 * Pure helpers for the mascot's artwork: latent-noise colours, the little canvas he paints, and the
 * pixel confetti and Zs. Everything is deterministic, so the same inputs always draw the same picture.
 */

/** Blob's body silhouette in the mascot's SVG units (viewBox -14 -18 228 218): a paint dollop with a flicked tip. */
export const BODY_PATH =
  'M100,184 C50,184 26,166 28,132 C30,96 56,64 94,55 C97,42 108,32 124,32 C117,39 115,46 118,56 C152,68 174,98 172,134 C170,166 150,184 100,184 Z';

export type RGB = readonly [number, number, number];

export const clamp = (x: number, a = 0, b = 1): number => Math.min(b, Math.max(a, x));
export const lerp = (a: number, b: number, t: number): number => a + (b - a) * t;
export const mix = (c1: RGB, c2: RGB, t: number): RGB => [lerp(c1[0], c2[0], t), lerp(c1[1], c2[1], t), lerp(c1[2], c2[2], t)];

/** Format a colour as a CSS `rgb()` value. */
const rgb = (c: RGB): string => `rgb(${Math.round(c[0])},${Math.round(c[1])},${Math.round(c[2])})`;

/** Hash three integers to a stable number in [0, 1). */
export function hash(a: number, b: number, c: number): number {
  let h = Math.imul(a | 0, 374761393) ^ Math.imul(b | 0, 668265263) ^ Math.imul(c | 0, 1274126177);
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  h ^= h >>> 16;
  return (h >>> 0) / 4294967296;
}

function gauss(a: number, b: number, c: number): number {
  const u = Math.max(1e-6, hash(a, b, c));
  const v = hash(a + 17, b + 31, c + 7);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/** Latent-noise colour: palette-biased Gaussian RGB, so the static reads teal and coral rather than TV snow. */
export function noiseRGB(a: number, b: number, c: number): RGB {
  const pick = hash(a + 3, b + 5, c + 11);
  const base: RGB = pick < 0.55 ? [45, 190, 172] : pick < 0.78 ? [235, 130, 112] : pick < 0.9 ? [230, 180, 60] : [70, 76, 86];
  return [0, 1, 2].map((i) => clamp(base[i]! + gauss(a + i * 101, b, c) * 62, 0, 255)) as unknown as RGB;
}

/** Calmer noise for the small in-mood pictures: 30% toward graphite so it never turns neon. */
const softNoise = (a: number, b: number, c: number): RGB => mix(noiseRGB(a, b, c), [96, 104, 110], 0.3);

/** The tile's side length in cells. */
export const TILE_CELLS = 12;
/** How many sampler steps the tile shows between pure noise and the finished picture. */
export const TILE_STEP_COUNT = 8;

/** Block size and noise level of each step, coarse to fine. */
const TILE_STEPS: readonly { block: number; sigma: number }[] = [
  { block: 6, sigma: 0.95 },
  { block: 4, sigma: 0.82 },
  { block: 4, sigma: 0.66 },
  { block: 3, sigma: 0.5 },
  { block: 2, sigma: 0.36 },
  { block: 2, sigma: 0.22 },
  { block: 1, sigma: 0.12 },
  { block: 1, sigma: 0.04 },
];

/** A tiny picture for the tile: dusk sky, amber sun, teal hills. u and v run over [0, 1]. */
function tilePixel(u: number, v: number): RGB {
  const sky = mix([19, 78, 74], [251, 143, 124], clamp((v - 0.05) / 0.7));
  const dx = u - 0.66;
  const dy = v - 0.4;
  if (dx * dx + dy * dy < 0.028) return [251, 191, 36];
  if (v > 0.82 + 0.05 * Math.sin(u * 11 + 2)) return [17, 130, 117];
  if (v > 0.7 + 0.07 * Math.sin(u * 7 + 0.6)) return [45, 212, 191];
  return sky;
}

export interface TileCell {
  x: number;
  y: number;
  fill: string;
}

/** Lay out the tile's cells at a sampler step (0 = pure noise to 7 = nearly done), or the finished picture for `null`. */
export function tileCells(step: number | null): TileCell[] {
  const n = TILE_CELLS;
  const level = step === null ? { block: 1, sigma: 0 } : TILE_STEPS[Math.min(TILE_STEPS.length - 1, Math.max(0, Math.floor(step)))]!;
  const cells: TileCell[] = [];
  for (let by = 0; by < n; by += level.block) {
    for (let bx = 0; bx < n; bx += level.block) {
      const base = tilePixel((bx + level.block / 2) / n, (by + level.block / 2) / n);
      for (let y = by; y < Math.min(n, by + level.block); y += 1) {
        for (let x = bx; x < Math.min(n, bx + level.block); x += 1) {
          cells.push({ x, y, fill: rgb(level.sigma === 0 ? base : mix(base, softNoise(x, y, step ?? 0), level.sigma)) });
        }
      }
    }
  }
  return cells;
}

/**
 * Map a job's progress (0 to 1) to the tile's step. The tile finishes its last step at `fraction` of
 * the job, where the mascot moves on to adding the finishing touches.
 */
export function tileStep(progress: number, fraction: number): number {
  return Math.min(TILE_STEP_COUNT - 1, Math.max(0, Math.floor((progress / fraction) * TILE_STEP_COUNT)));
}

/** The thought bubble's picture: 6 x 6 cells in which an idea has half formed out of noise. */
const BUBBLE_CELLS = 6;

export function bubbleCells(): TileCell[] {
  const n = BUBBLE_CELLS;
  const cells: TileCell[] = [];
  for (let y = 0; y < n; y += 1) {
    for (let x = 0; x < n; x += 1) {
      cells.push({ x, y, fill: rgb(mix(tilePixel((x + 0.5) / n, (y + 0.5) / n), softNoise(x, y, 303), 0.5)) });
    }
  }
  return cells;
}

const CONFETTI_PALETTE = ['#fbbf24', '#fb8f7c', '#2dd4bf', '#ccfbf1', '#60a5fa', '#f9a8d4'] as const;

export interface ConfettiPiece {
  x: number;
  size: number;
  /** The second square of a two-pixel piece, or null for a single pixel. */
  pair: string | null;
  fill: string;
  /** Seconds into the loop the piece starts, as a negative delay. */
  delay: number;
  /** Sideways drift over its fall, in SVG units. */
  sway: number;
  /** Where the piece rests in the reduced-motion still frame. */
  restY: number;
}

/** Square palette confetti, scattered once: some single pixels, some pairs. */
export function confettiPieces(count = 16): ConfettiPiece[] {
  return Array.from({ length: count }, (_, i) => ({
    x: -8 + hash(i, 1, 2) * 216,
    size: [5, 7, 9][i % 3]!,
    pair: i % 4 === 0 ? CONFETTI_PALETTE[(i + 2) % 6] : null,
    fill: CONFETTI_PALETTE[i % 6]!,
    delay: -hash(i, 3, 4) * 1.6,
    sway: Math.round((hash(i, 5, 6) - 0.5) * 24),
    restY: Math.round(-10 + hash(i, 7, 8) * 110),
  }));
}

const Z_GLYPH = ['11111', '00010', '00100', '01000', '11111'] as const;

export interface ZGlyph {
  cell: number;
  /** Seconds into the loop the Z starts. */
  delay: number;
  /** The top-left corner of each lit cell. */
  pixels: { x: number; y: number }[];
}

/** Three 5 x 5 pixel Zs of growing size. */
export function zGlyphs(): ZGlyph[] {
  return (
    [
      [146, 50, 3.6, 0],
      [162, 30, 4.6, 1.2],
      [182, 6, 5.8, 2.4],
    ] as const
  ).map(([ox, oy, cell, delay]) => {
    const pixels: ZGlyph['pixels'] = [];
    Z_GLYPH.forEach((row, gy) => {
      [...row].forEach((bit, gx) => {
        if (bit === '1') pixels.push({ x: Number((ox + gx * cell).toFixed(1)), y: Number((oy + gy * cell).toFixed(1)) });
      });
    });
    return { cell, delay, pixels };
  });
}
