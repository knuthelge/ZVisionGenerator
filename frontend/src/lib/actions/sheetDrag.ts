/** What a finger did on a sheet when it lifted, for {@link sheetDragOutcome}. */
export interface SheetDragEnd {
  /** Distance dragged down, in px (negative is up). */
  dy: number;
  /** How long the drag took, in ms. */
  ms: number;
  /** The sheet's height, in px. */
  height: number;
}

/** A drag this far down closes the sheet, as does a quick flick. */
const CLOSE_FRACTION = 0.25;
const CLOSE_MIN_PX = 80;
const FLICK_PX_PER_MS = 0.5;
const FLICK_MIN_PX = 24;
/** Movement before the drag decides whether it is a pull on the sheet or a scroll or slide inside it. */
const DECIDE_PX = 8;
const SETTLE_MS = 180;

/** Return whether a finger lifted after this drag closes the sheet or lets it spring back. */
export function sheetDragOutcome({ dy, ms, height }: SheetDragEnd): 'close' | 'stay' {
  if (dy >= Math.max(CLOSE_MIN_PX, height * CLOSE_FRACTION)) return 'close';
  if (dy >= FLICK_MIN_PX && dy / Math.max(ms, 1) >= FLICK_PX_PER_MS) return 'close';
  return 'stay';
}

/** Controls that take a finger's drag themselves: a slider, a text box. */
const OWN_DRAG = 'input, textarea, select, [data-own-drag]';

/** Whether something between the touch and the sheet is scrolled down, so a pull down scrolls it first. */
function scrolledInside(target: Element, sheet: HTMLElement): boolean {
  for (let el: Element | null = target; el && el !== sheet.parentElement; el = el.parentElement) {
    if (el instanceof HTMLElement && el.scrollTop > 0 && el.scrollHeight > el.clientHeight) return true;
  }
  return false;
}

export interface SheetDragOptions {
  /** Called when the sheet was pulled down far or fast enough. */
  onclose: () => void;
  /** Off, the sheet does not follow a finger (e.g. when it is shown as a desktop panel). */
  enabled?: boolean;
}

/**
 * Let a finger pull a bottom sheet down to close it, as iOS sheets do.
 *
 * The pull starts anywhere on the sheet once its content is scrolled to the top, and not on a control that drags
 * itself (a slider, a text box). The sheet follows the finger; lifted far or fast enough it slides away and closes,
 * otherwise it springs back.
 */
export function sheetDrag(node: HTMLElement, initial: SheetDragOptions): { update: (next: SheetDragOptions) => void; destroy: () => void } {
  let options = initial;
  let start: { x: number; y: number; at: number } | null = null;
  let dragging = false;
  let dy = 0;

  function setOffset(px: number, animate: boolean): void {
    node.style.transition = animate ? `translate ${SETTLE_MS}ms ease-out` : 'none';
    node.style.translate = px === 0 ? '' : `0 ${px}px`;
  }

  function onTouchStart(event: TouchEvent): void {
    start = null;
    dragging = false;
    if (options.enabled === false || event.touches.length !== 1) return;
    const target = event.target as Element;
    if (target.closest(OWN_DRAG) || scrolledInside(target, node)) return;
    const touch = event.touches[0];
    start = { x: touch.clientX, y: touch.clientY, at: Date.now() };
  }

  function onTouchMove(event: TouchEvent): void {
    if (!start) return;
    const touch = event.touches[0];
    const moveY = touch.clientY - start.y;
    const moveX = touch.clientX - start.x;
    if (!dragging) {
      if (Math.abs(moveY) < DECIDE_PX && Math.abs(moveX) < DECIDE_PX) return;
      // Down and mostly vertical pulls the sheet; anything else belongs to the content.
      if (moveY <= 0 || Math.abs(moveX) > moveY) { start = null; return; }
      dragging = true;
    }
    event.preventDefault();
    dy = Math.max(0, moveY);
    setOffset(dy, false);
  }

  function onTouchEnd(): void {
    const began = start;
    start = null;
    if (!dragging || !began) return;
    dragging = false;
    const outcome = sheetDragOutcome({ dy, ms: Date.now() - began.at, height: node.offsetHeight });
    if (outcome === 'stay') {
      setOffset(0, true);
      return;
    }
    setOffset(node.offsetHeight, true);
    setTimeout(() => {
      options.onclose();
      // A sheet kept in the page while closed must come back in place.
      node.style.transition = '';
      node.style.translate = '';
    }, SETTLE_MS);
  }

  node.addEventListener('touchstart', onTouchStart, { passive: true });
  node.addEventListener('touchmove', onTouchMove, { passive: false });
  node.addEventListener('touchend', onTouchEnd);
  node.addEventListener('touchcancel', onTouchEnd);

  return {
    update(next: SheetDragOptions): void {
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
