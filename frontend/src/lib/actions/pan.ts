/** The direction a drag runs in. */
export type PanAxis = 'x' | 'y';

/** A finger position at a moment, for {@link releaseVelocity}. */
export interface PanSample {
  x: number;
  y: number;
  at: number;
}

/** How far back the release speed looks: the end of a flick, not the whole drag. */
const VELOCITY_WINDOW_MS = 80;

/** Return the axis a first movement runs along; a tie goes to y, where the page's own scrolling lives. */
export function lockAxis(dx: number, dy: number): PanAxis {
  return Math.abs(dx) > Math.abs(dy) ? 'x' : 'y';
}

/**
 * Return the finger's speed along an axis when it lifted, in px per ms, from the samples of its last moments.
 *
 * Only the last {@link VELOCITY_WINDOW_MS} count, so a slow start and a fast finish reads as the flick it ends with.
 */
export function releaseVelocity(samples: readonly PanSample[], axis: PanAxis): number {
  const last = samples[samples.length - 1];
  if (!last) return 0;
  let first = last;
  for (let i = samples.length - 2; i >= 0 && last.at - samples[i].at <= VELOCITY_WINDOW_MS; i--) first = samples[i];
  const ms = last.at - first.at;
  if (ms <= 0) return 0;
  return (axis === 'x' ? last.x - first.x : last.y - first.y) / ms;
}

/** Return how far a surface moves for a pull past its end: less and less, never more than `limit`. */
export function rubberBand(offset: number, limit = 64): number {
  const sign = Math.sign(offset);
  const distance = Math.abs(offset);
  return sign * (limit * distance) / (limit + distance);
}

/**
 * Whether something between `target` and `root` can still scroll in the direction a drag moves the content.
 *
 * `toward` is the sign of the finger's movement: a finger moving down (+1) scrolls content toward its top, so it is
 * the content's to take while anything is scrolled down.
 */
export function canScrollToward(target: Element, root: Element, axis: PanAxis, toward: 1 | -1): boolean {
  for (let el: Element | null = target; el && el !== root.parentElement; el = el.parentElement) {
    if (!(el instanceof HTMLElement)) continue;
    const scroll = axis === 'y' ? el.scrollTop : el.scrollLeft;
    const size = axis === 'y' ? el.scrollHeight - el.clientHeight : el.scrollWidth - el.clientWidth;
    if (size <= 1) continue;
    if (toward > 0 ? scroll > 0 : scroll < size - 1) return true;
  }
  return false;
}

/** The first movement of a drag, offered to {@link PanOptions.claim}. */
export interface PanStart {
  axis: PanAxis;
  dx: number;
  dy: number;
  target: Element;
}

/** Where a drag is, along the axis it was claimed on. */
export interface PanMove {
  axis: PanAxis;
  /** Distance from where the finger went down, in px. */
  offset: number;
}

/** A drag that ended: where it got to and how fast the finger was moving when it lifted. */
export interface PanEnd extends PanMove {
  /** px per ms along the axis, from the last moments of the drag. */
  velocity: number;
}

export interface PanOptions {
  /** The axes this surface moves along; a first movement along another axis is left to the page. */
  axes: readonly PanAxis[];
  /** Whether this drag is the surface's: return false to leave it to the content (a scroller, a control). */
  claim?: (start: PanStart) => boolean;
  onstart?: (start: PanStart) => void;
  onmove: (move: PanMove) => void;
  onend: (end: PanEnd) => void;
  /** A claimed drag was taken away (a second finger, the browser cancelling the touch). */
  oncancel?: () => void;
  /** Off, the surface does not follow a finger. */
  enabled?: boolean;
}

/** Controls that take a finger's drag themselves: a slider, a text box. */
export const OWN_DRAG = 'input, textarea, select, video, [data-own-drag]';

/**
 * Let a surface follow one finger along an axis.
 *
 * The gesture is decided on the first movement, and a claimed drag cancels that movement and every one after it.
 * Deciding later loses on iOS and Chrome: once the browser has started scrolling on an uncancelled move, every later
 * move is uncancellable and the page, the content or pull-to-refresh moves along with the surface. A first movement
 * along another axis, or one `claim` declines, is left to the browser for the whole touch.
 */
export function pan(node: HTMLElement, initial: PanOptions): { update: (next: PanOptions) => void; destroy: () => void } {
  let options = initial;
  let origin: { x: number; y: number; target: Element } | null = null;
  let axis: PanAxis | null = null;
  let samples: PanSample[] = [];

  function reset(): void {
    origin = null;
    axis = null;
    samples = [];
  }

  function onTouchStart(event: TouchEvent): void {
    if (axis) {
      // A second finger: the drag was for one.
      options.oncancel?.();
      reset();
      return;
    }
    reset();
    if (options.enabled === false || event.touches.length !== 1) return;
    const touch = event.touches[0];
    origin = { x: touch.clientX, y: touch.clientY, target: event.target as Element };
  }

  function onTouchMove(event: TouchEvent): void {
    if (!origin || event.touches.length !== 1) return;
    const touch = event.touches[0];
    const dx = touch.clientX - origin.x;
    const dy = touch.clientY - origin.y;
    if (!axis) {
      if (dx === 0 && dy === 0) return;
      const locked = lockAxis(dx, dy);
      const start: PanStart = { axis: locked, dx, dy, target: origin.target };
      if (!options.axes.includes(locked) || (options.claim && !options.claim(start))) {
        origin = null;
        return;
      }
      axis = locked;
      options.onstart?.(start);
    }
    if (event.cancelable) event.preventDefault();
    samples.push({ x: touch.clientX, y: touch.clientY, at: event.timeStamp || Date.now() });
    if (samples.length > 12) samples.shift();
    options.onmove({ axis, offset: axis === 'x' ? dx : dy });
  }

  function onTouchEnd(event: TouchEvent): void {
    if (!origin || !axis) { reset(); return; }
    const touch = event.changedTouches[0];
    const offset = touch ? (axis === 'x' ? touch.clientX - origin.x : touch.clientY - origin.y) : 0;
    if (event.type === 'touchcancel') options.oncancel?.();
    else options.onend({ axis, offset, velocity: releaseVelocity(samples, axis) });
    reset();
  }

  node.addEventListener('touchstart', onTouchStart, { passive: true });
  node.addEventListener('touchmove', onTouchMove, { passive: false });
  node.addEventListener('touchend', onTouchEnd);
  node.addEventListener('touchcancel', onTouchEnd);

  return {
    update(next: PanOptions): void {
      options = next;
    },
    destroy(): void {
      node.removeEventListener('touchstart', onTouchStart);
      node.removeEventListener('touchmove', onTouchMove);
      node.removeEventListener('touchend', onTouchEnd);
      node.removeEventListener('touchcancel', onTouchEnd);
    },
  };
}
