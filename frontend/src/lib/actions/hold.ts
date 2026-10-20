/** How long a press must be held, in ms. */
export const HOLD_MS = 480;
/** How far a finger may move before the press counts as a scroll or a drag. */
const HOLD_SLOP_PX = 10;

export interface HoldOptions {
  /** Called once the press has been held long enough. */
  onhold: () => void;
  /** Presses starting on these are not holds (a button inside the held surface, a video's controls). */
  ignore?: string;
  enabled?: boolean;
}

/**
 * Fire a callback when a finger presses and holds, as a long press opens a menu on a phone.
 *
 * Only a finger or a pen holds; a mouse clicks as usual. While the press is held the node carries `data-holding`
 * (with `--hold-ms`), so it can show the hold filling up; moving, lifting early or the browser taking the touch for a
 * scroll cancels it. A hold that fired swallows the click that follows it and the browser's own long-press menu.
 */
export function hold(node: HTMLElement, initial: HoldOptions): { update: (next: HoldOptions) => void; destroy: () => void } {
  let options = initial;
  let timer: ReturnType<typeof setTimeout> | undefined;
  let start: { x: number; y: number } | null = null;
  let fired = false;
  let touching = false;

  function cancel(): void {
    clearTimeout(timer);
    start = null;
    delete node.dataset.holding;
  }

  function onPointerDown(event: PointerEvent): void {
    fired = false;
    touching = event.pointerType === 'touch' || event.pointerType === 'pen';
    if (!touching || options.enabled === false || event.button !== 0 || (node as HTMLButtonElement).disabled) return;
    if (options.ignore && (event.target as Element).closest(options.ignore)) return;
    start = { x: event.clientX, y: event.clientY };
    node.style.setProperty('--hold-ms', `${HOLD_MS}ms`);
    node.dataset.holding = '';
    timer = setTimeout(() => {
      cancel();
      fired = true;
      navigator.vibrate?.(10);
      options.onhold();
    }, HOLD_MS);
  }

  function onPointerMove(event: PointerEvent): void {
    if (start && Math.hypot(event.clientX - start.x, event.clientY - start.y) > HOLD_SLOP_PX) cancel();
  }

  function onClick(event: MouseEvent): void {
    if (!fired) return;
    fired = false;
    event.preventDefault();
    event.stopImmediatePropagation();
  }

  // A finger's long press would also open the browser's menu (copy, save image); the hold replaces it.
  function onContextMenu(event: Event): void {
    if (touching && options.enabled !== false) event.preventDefault();
  }

  node.addEventListener('pointerdown', onPointerDown);
  node.addEventListener('pointermove', onPointerMove);
  node.addEventListener('pointerup', cancel);
  node.addEventListener('pointercancel', cancel);
  node.addEventListener('pointerleave', cancel);
  node.addEventListener('click', onClick, true);
  node.addEventListener('contextmenu', onContextMenu);

  return {
    update(next: HoldOptions): void {
      options = next;
    },
    destroy(): void {
      cancel();
      node.removeEventListener('pointerdown', onPointerDown);
      node.removeEventListener('pointermove', onPointerMove);
      node.removeEventListener('pointerup', cancel);
      node.removeEventListener('pointercancel', cancel);
      node.removeEventListener('pointerleave', cancel);
      node.removeEventListener('click', onClick, true);
      node.removeEventListener('contextmenu', onContextMenu);
    },
  };
}
