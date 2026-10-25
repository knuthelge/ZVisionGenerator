/** How long a press must be held, in ms. */
export const HOLD_MS = 480;
/** How far a finger may move before the press counts as a scroll or a drag. */
const HOLD_SLOP_PX = 10;
/** How long after the finger lifts the click it makes is still the hold's, and swallowed: the browser's own click
 * follows the lift at once, while a person's next tap takes longer than this. */
const RELEASE_MS = 120;

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
 * scroll cancels it, and so does a second finger, as a pinch starts. A hold that fired swallows the click that
 * follows it and the browser's own long-press menu.
 */
export function hold(node: HTMLElement, initial: HoldOptions): { update: (next: HoldOptions) => void; destroy: () => void } {
  let options = initial;
  let timer: ReturnType<typeof setTimeout> | undefined;
  let start: { x: number; y: number } | null = null;
  /** The finger that went down first; only it holds, moves or lifts the hold. */
  let pointerId: number | null = null;
  let fired = false;
  let touching = false;

  function stopPending(): void {
    clearTimeout(timer);
    start = null;
    delete node.dataset.holding;
    document.removeEventListener('pointerdown', onOtherDown, true);
  }

  // A second finger anywhere on the page makes a pinch, even off the node, where the node never hears it.
  function onOtherDown(event: PointerEvent): void {
    if (event.pointerId !== pointerId) stopPending();
  }

  function end(event: PointerEvent): void {
    if (event.pointerId !== pointerId) return;
    pointerId = null;
    stopPending();
    // The finger lifting after a hold still clicks whatever is under it by then, such as an item of the menu the
    // hold opened: that click is swallowed wherever it lands, then clicks are ordinary again.
    if (fired) {
      releaseTimer = setTimeout(() => {
        // No click came (iOS sends none after a long press): clicks on the node are ordinary again.
        fired = false;
        clearRelease();
      }, RELEASE_MS);
    }
  }

  let releaseTimer: ReturnType<typeof setTimeout> | undefined;
  function clearRelease(): void {
    clearTimeout(releaseTimer);
    document.removeEventListener('click', swallowRelease, true);
  }

  function swallowRelease(event: MouseEvent): void {
    event.preventDefault();
    event.stopImmediatePropagation();
    // That was the hold's click; the next one on the node is an ordinary click again.
    fired = false;
    clearRelease();
  }

  function onPointerDown(event: PointerEvent): void {
    // A second finger makes a pinch, not a hold.
    if (!event.isPrimary) { stopPending(); return; }
    // A first finger starts a new gesture: whatever the last one left (a lift that never arrived) is over.
    stopPending();
    clearRelease();
    pointerId = null;
    fired = false;
    touching = event.pointerType === 'touch' || event.pointerType === 'pen';
    if (!touching || options.enabled === false || event.button !== 0 || (node as HTMLButtonElement).disabled) return;
    if (options.ignore && (event.target as Element).closest(options.ignore)) return;
    pointerId = event.pointerId;
    start = { x: event.clientX, y: event.clientY };
    node.style.setProperty('--hold-ms', `${HOLD_MS}ms`);
    node.dataset.holding = '';
    document.addEventListener('pointerdown', onOtherDown, true);
    timer = setTimeout(() => {
      stopPending();
      fired = true;
      document.addEventListener('click', swallowRelease, true);
      navigator.vibrate?.(10);
      options.onhold();
    }, HOLD_MS);
  }

  function onPointerMove(event: PointerEvent): void {
    if (event.pointerId !== pointerId || !start) return;
    if (Math.hypot(event.clientX - start.x, event.clientY - start.y) > HOLD_SLOP_PX) stopPending();
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
  node.addEventListener('pointerup', end);
  node.addEventListener('pointercancel', end);
  node.addEventListener('pointerleave', end);
  node.addEventListener('click', onClick, true);
  node.addEventListener('contextmenu', onContextMenu);

  return {
    update(next: HoldOptions): void {
      options = next;
    },
    destroy(): void {
      stopPending();
      clearRelease();
      node.removeEventListener('pointerdown', onPointerDown);
      node.removeEventListener('pointermove', onPointerMove);
      node.removeEventListener('pointerup', end);
      node.removeEventListener('pointercancel', end);
      node.removeEventListener('pointerleave', end);
      node.removeEventListener('click', onClick, true);
      node.removeEventListener('contextmenu', onContextMenu);
    },
  };
}
