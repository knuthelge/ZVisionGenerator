/**
 * How a MascotSpot appears and travels. A spot that pops in with no previous spot is sampled in like an
 * image (Denoise in). A spot that follows another within the hand-off window receives him as a stream of
 * pixels (Latent stream), flown along an arc above the page. A spot nobody takes over simply disappears.
 * Each effect is a canvas overlay that exists only for the transition, rasterised once from the mascot's
 * SVG; at rest the mascot costs nothing.
 */
import { backOut } from 'svelte/easing';
import type { TransitionConfig } from 'svelte/transition';
import { prefersReducedMotion } from './Mascot.svelte';
import {
  bakeMascotMarkup,
  canvasSupported,
  createOverlay,
  DENOISE_IN_MS,
  denoiseSteps,
  drawDenoiseIn,
  drawStream,
  padBounds,
  rasterizeMascot,
  STREAM_RISE,
  streamCells,
  unionBounds,
  type MascotSprite,
  type Overlay,
} from './mascotCanvas';

interface Departure {
  rect: DOMRect;
  at: number;
  /** The mascot as he looked when he left, to be rasterised only if a spot takes him over. */
  markup: string | null;
}

// Where the mascot last left from, keyed so separate mascots never swap places.
const departures = new Map<string, Departure>();
const HANDOFF_WINDOW_MS = 100;
const DPR_CAP = 2;
/** Room around the drawing for the sampler's edge cells, in px. */
const EDGE_MARGIN = 4;

function motionAllowed(): boolean {
  if (typeof Element === 'undefined' || typeof Element.prototype.animate !== 'function') return false;
  return !prefersReducedMotion();
}

function isFresh(entry: { at: number } | undefined): boolean {
  return entry !== undefined && performance.now() - entry.at <= HANDOFF_WINDOW_MS;
}

function pixelRatio(): number {
  return Math.min(DPR_CAP, window.devicePixelRatio || 1);
}

/** Measure the mascot's drawing, which can be smaller than the spot that wraps it. */
function measure(spot: Element): DOMRect {
  return (spot.querySelector('svg') ?? spot).getBoundingClientRect();
}

/**
 * Put the overlay over the page, or just over the dialog, sheet or viewer his spot sits in, so the
 * layers above that cover the effect as they cover the spot.
 */
function layerFor(spot: Element): number | undefined {
  // The effect layer is over the page and under the viewer, sheets, dialogs and toasts (`--z-mascot-effect`).
  const base = Number.parseInt(getComputedStyle(document.documentElement).getPropertyValue('--z-mascot-effect'), 10);
  if (!Number.isFinite(base)) return undefined;
  let layer = base;
  for (let el: Element | null = spot.parentElement; el; el = el.parentElement) {
    const z = Number.parseInt(getComputedStyle(el).zIndex, 10);
    if (z >= layer) layer = z + 1;
  }
  return layer > base ? layer : undefined;
}

/** Rasterise the mascot's baked markup, or resolve to null when this browser cannot. */
function spriteOf(markup: string, rect: { width: number; height: number }): Promise<MascotSprite | null> {
  return rasterizeMascot(markup, rect.width, rect.height, pixelRatio());
}

function bakeSpot(spot: Element): string | null {
  const svg = spot.querySelector('svg');
  if (!svg || !canvasSupported()) return null;
  const { width, height } = svg.getBoundingClientRect();
  return width === 0 || height === 0 ? null : bakeMascotMarkup(svg, width, height);
}

/**
 * Record where every mascot currently sits. Call this before a DOM update that
 * swaps spots: once the new spot is inserted the old one can be pushed aside by
 * layout, so measuring during the outro would start the flight from the wrong place.
 */
export function rememberMascotSpots(): void {
  document.querySelectorAll<HTMLElement>('[data-mascot-spot]').forEach((spot) => {
    const key = spot.dataset.mascotSpot;
    if (key) departures.set(key, { rect: measure(spot), at: performance.now(), markup: null });
  });
}

/** Run a draw callback every frame until it reports it is done. */
function animate(draw: (ms: number) => boolean): void {
  const start = performance.now();
  const frame = (now: number): void => {
    if (draw(now - start)) requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
}

/** Vanish; the next spot takes over from where this one was last remembered. */
function depart(node: Element, key: string): TransitionConfig {
  const previous = departures.get(key);
  const fresh = isFresh(previous);
  departures.set(key, {
    rect: fresh ? previous!.rect : measure(node),
    at: fresh ? previous!.at : performance.now(),
    markup: motionAllowed() ? bakeSpot(node) : null,
  });
  return { duration: 0 };
}

/** Show a spot again, once an effect has drawn his arrival over it. */
function reveal(spot: HTMLElement, overlay?: Overlay): void {
  spot.style.removeProperty('visibility');
  overlay?.remove();
}

/**
 * Sample the mascot in over his spot: the sampler's steps resolve noise to pixels, then it snaps to the
 * vector. The spot is hidden until the overlay hands over to it, and is tracked every frame so a sheet
 * or dialog still sliding in is met where it ends up.
 */
function playDenoiseIn(spot: HTMLElement, sprite: MascotSprite): void {
  const overlay = createOverlay(padBounds(measure(spot), EDGE_MARGIN), pixelRatio(), layerFor(spot));
  if (!overlay) {
    reveal(spot);
    return;
  }
  const steps = denoiseSteps(sprite);
  animate((ms) => {
    const rect = measure(spot);
    if (ms >= DENOISE_IN_MS || !spot.isConnected || rect.width === 0) {
      reveal(spot, overlay);
      return false;
    }
    overlay.fit(padBounds(rect, EDGE_MARGIN));
    overlay.clear();
    drawDenoiseIn(overlay.ctx, { sprite, x: rect.left, y: rect.top, width: rect.width, height: rect.height }, steps, ms);
    return true;
  });
}

/**
 * Stream the mascot from the previous spot to this one along an arc above the page, so the flight is
 * never clipped by the containers either spot sits in. The landing spot is re-measured every frame.
 */
function playStream(spot: HTMLElement, from: { rect: DOMRect; sprite: MascotSprite }, sprite: MascotSprite, duration: number): void {
  const overlay = createOverlay(padBounds(unionBounds(from.rect, measure(spot)), EDGE_MARGIN), pixelRatio(), layerFor(spot));
  if (!overlay) {
    reveal(spot);
    return;
  }
  const cells = streamCells(sprite);
  const origin = { sprite: from.sprite, x: from.rect.left, y: from.rect.top, width: from.rect.width, height: from.rect.height };
  animate((ms) => {
    const rect = measure(spot);
    const p = Math.min(1, ms / duration);
    if (p >= 1 || !spot.isConnected || rect.width === 0) {
      reveal(spot, overlay);
      return false;
    }
    const hop = Math.min(80, 24 + Math.hypot(origin.x - rect.left, origin.y - rect.top) * 0.15);
    // Cover both spots and the arc between them, re-fitted if the landing spot moves.
    overlay.fit(padBounds(unionBounds(from.rect, rect), EDGE_MARGIN, EDGE_MARGIN + hop * STREAM_RISE, EDGE_MARGIN, EDGE_MARGIN));
    overlay.clear();
    drawStream(overlay.ctx, origin, { sprite, x: rect.left, y: rect.top, width: rect.width, height: rect.height }, cells, p, hop);
    return true;
  });
}

const popIn = (): TransitionConfig => ({
  duration: 320,
  easing: backOut,
  css: (t) => `opacity: ${Math.min(1, t * 2)}; transform: scale(${t});`,
});

/** Stream in from the previous spot, or be sampled in when there is no previous spot. */
function arrive(node: Element, key: string): () => TransitionConfig {
  // Deferred so a same-tick departure is recorded before this runs.
  return () => {
    if (typeof Element === 'undefined' || typeof Element.prototype.animate !== 'function') return { duration: 0 };
    const departure = departures.get(key);
    departures.delete(key);
    // The spot he is leaving, when this arrival is the hand-off for it.
    const from = isFresh(departure) && departure!.markup !== null ? departure! : null;
    // No travelling or sampling at all: he is simply there.
    if (prefersReducedMotion()) return { duration: 0 };
    const spot = node as HTMLElement;
    const markup = bakeSpot(spot);
    if (markup === null) return from ? { duration: 0 } : popIn();
    const arriving = spriteOf(markup, measure(spot));
    const leaving = from ? spriteOf(from.markup!, from.rect) : Promise.resolve(null);
    spot.style.visibility = 'hidden';
    void Promise.all([leaving, arriving]).then(([before, after]) => {
      if (!after || !spot.isConnected) {
        reveal(spot);
      } else if (from && before) {
        const to = measure(spot);
        const duration = Math.min(1000, 650 + Math.hypot(from.rect.left - to.left, from.rect.top - to.top) * 0.5);
        playStream(spot, { rect: from.rect, sprite: before }, after, duration);
      } else {
        playDenoiseIn(spot, after);
      }
    });
    return { duration: 0 };
  };
}

export { arrive, depart };
