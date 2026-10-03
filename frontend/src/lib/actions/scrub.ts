/** Numeric bounds and granularity shared by scrubbing and nudging. */
export interface NumberSpec {
  step: number;
  min?: number;
  max?: number;
}

export interface ScrubOptions extends NumberSpec {
  /** Current value; `null` scrubs from `min` (or 0). */
  value: number | null;
  onchange: (value: number) => void;
  disabled?: boolean;
}

const DRAG_THRESHOLD_PX = 3;

function decimals(step: number): number {
  const text = String(step);
  return text.includes('.') ? text.split('.')[1].length : 0;
}

/** Snap a value to the spec's step precision and clamp it to its bounds. */
export function clampToSpec(value: number, spec: NumberSpec): number {
  const rounded = Number(value.toFixed(decimals(spec.step)));
  const lower = spec.min ?? -Infinity;
  const upper = spec.max ?? Infinity;
  return Math.min(upper, Math.max(lower, rounded));
}

/** Round a value to the nearest multiple of the step, then clamp it to the bounds. */
export function snapToSpec(value: number, spec: NumberSpec): number {
  return clampToSpec(Math.round(value / spec.step) * spec.step, spec);
}

/** Return the value after dragging `dx` pixels from `start`. */
export function scrubValue(start: number, dx: number, spec: NumberSpec, pxPerStep = 4): number {
  return clampToSpec(start + Math.round(dx / pxPerStep) * spec.step, spec);
}

/** Return the value after one arrow-key nudge; `big` moves ten steps. */
export function nudgeValue(value: number | null, direction: 1 | -1, spec: NumberSpec, big = false): number {
  const start = value ?? spec.min ?? 0;
  return clampToSpec(start + direction * spec.step * (big ? 10 : 1), spec);
}

/**
 * Make an element (usually a row label) change a number by dragging left or right.
 *
 * A press without movement is left alone, so a `<label for>` still focuses its input.
 */
export function scrub(node: HTMLElement, initial: ScrubOptions): { update: (next: ScrubOptions) => void; destroy: () => void } {
  let options = initial;
  let startX = 0;
  let startValue = 0;
  let pointerId: number | null = null;
  let dragging = false;

  function onPointerDown(event: PointerEvent): void {
    if (options.disabled || event.button !== 0) return;
    pointerId = event.pointerId;
    startX = event.clientX;
    startValue = options.value ?? options.min ?? 0;
    dragging = false;
    node.setPointerCapture?.(event.pointerId);
  }

  function onPointerMove(event: PointerEvent): void {
    if (event.pointerId !== pointerId) return;
    const dx = event.clientX - startX;
    if (!dragging && Math.abs(dx) < DRAG_THRESHOLD_PX) return;
    dragging = true;
    event.preventDefault();
    const next = scrubValue(startValue, dx, options);
    if (next !== options.value) options.onchange(next);
  }

  function onPointerUp(event: PointerEvent): void {
    if (event.pointerId !== pointerId) return;
    node.releasePointerCapture?.(event.pointerId);
    pointerId = null;
  }

  function onClick(event: MouseEvent): void {
    // A finished drag must not also focus the input or toggle a control.
    if (!dragging) return;
    event.preventDefault();
    dragging = false;
  }

  node.addEventListener('pointerdown', onPointerDown);
  node.addEventListener('pointermove', onPointerMove);
  node.addEventListener('pointerup', onPointerUp);
  node.addEventListener('pointercancel', onPointerUp);
  node.addEventListener('click', onClick, true);

  return {
    update(next: ScrubOptions): void { options = next; },
    destroy(): void {
      node.removeEventListener('pointerdown', onPointerDown);
      node.removeEventListener('pointermove', onPointerMove);
      node.removeEventListener('pointerup', onPointerUp);
      node.removeEventListener('pointercancel', onPointerUp);
      node.removeEventListener('click', onClick, true);
    },
  };
}
