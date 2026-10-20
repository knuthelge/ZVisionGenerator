import { canScrollToward, OWN_DRAG, pan, rubberBand, type PanEnd, type PanStart } from './pan';

/** How a pull on a sheet ended, for {@link sheetDragOutcome}. */
export interface SheetDragEnd {
  /** Distance dragged down, in px (negative is up). */
  offset: number;
  /** Speed down when the finger lifted, in px per ms. */
  velocity: number;
  /** The sheet's height, in px. */
  height: number;
}

/** A pull this far down closes the sheet, as does a flick. */
const CLOSE_FRACTION = 0.25;
const CLOSE_MIN_PX = 80;
const FLICK_PX_PER_MS = 0.5;
const FLICK_MIN_PX = 16;
/** How long the sheet takes to spring back or slide away after the finger lifts. */
export const SHEET_SETTLE_MS = 220;
const SETTLE_EASE = 'cubic-bezier(0.32, 0.72, 0, 1)';

/** Return whether a finger lifted after this pull closes the sheet or lets it spring back. */
export function sheetDragOutcome({ offset, velocity, height }: SheetDragEnd): 'close' | 'stay' {
  if (offset >= Math.max(CLOSE_MIN_PX, height * CLOSE_FRACTION) && velocity > -FLICK_PX_PER_MS) return 'close';
  if (offset >= FLICK_MIN_PX && velocity >= FLICK_PX_PER_MS) return 'close';
  return 'stay';
}

/** Mark the parts of a sheet a finger always pulls (the grab bar, the head) with this attribute. */
export const SHEET_HANDLE = '[data-sheet-handle]';

/** Whether a first movement is a pull on the sheet rather than a scroll or a slide inside it. */
export function claimsSheetPull({ axis, dy, target }: PanStart, sheet: Element): boolean {
  if (axis !== 'y') return false;
  if (target.closest(SHEET_HANDLE)) return true;
  if (dy <= 0 || target.closest(OWN_DRAG)) return false;
  return !canScrollToward(target, sheet, 'y', 1);
}

export interface SheetDragOptions {
  /** Called once the sheet has slid away after a pull far or fast enough. */
  onclose: () => void;
  /** Off, the sheet does not follow a finger (e.g. when it is shown as a desktop panel). */
  enabled?: boolean;
}

/**
 * Let a finger pull a bottom sheet down to close it, as iOS sheets do.
 *
 * The handle (the grab bar and the head, marked {@link SHEET_HANDLE}) always pulls; the rest of the sheet pulls once
 * its content is scrolled to the top, and not on a control that drags itself. The sheet follows the finger, resists
 * a pull up, and on release either slides away and closes or springs back.
 */
export function sheetDrag(node: HTMLElement, initial: SheetDragOptions): { update: (next: SheetDragOptions) => void; destroy: () => void } {
  let options = initial;

  function setOffset(px: number, animate: boolean): void {
    node.style.transition = animate ? `translate ${SHEET_SETTLE_MS}ms ${SETTLE_EASE}` : 'none';
    node.style.translate = `0 ${px}px`;
  }

  function clear(): void {
    node.style.transition = '';
    node.style.translate = '';
  }

  function onend({ offset, velocity }: PanEnd): void {
    if (sheetDragOutcome({ offset, velocity, height: node.offsetHeight }) === 'stay') {
      setOffset(0, true);
      setTimeout(clear, SHEET_SETTLE_MS);
      return;
    }
    setOffset(node.offsetHeight, true);
    setTimeout(() => {
      options.onclose();
      // A sheet kept in the page while closed must come back in place.
      clear();
    }, SHEET_SETTLE_MS);
  }

  const panAction = pan(node, panOptions());

  function panOptions() {
    return {
      axes: ['y'] as const,
      enabled: options.enabled,
      claim: (start: PanStart) => claimsSheetPull(start, node),
      onmove: ({ offset }: { offset: number }) => setOffset(offset >= 0 ? offset : rubberBand(offset, 24), false),
      onend,
      oncancel: () => { setOffset(0, true); setTimeout(clear, SHEET_SETTLE_MS); },
    };
  }

  return {
    update(next: SheetDragOptions): void {
      options = next;
      panAction.update(panOptions());
    },
    destroy(): void {
      panAction.destroy();
    },
  };
}
