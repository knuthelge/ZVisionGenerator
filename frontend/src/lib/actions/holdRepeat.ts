/** Delay before a held press starts repeating, and the interval between repeats. */
export const HOLD_DELAY_MS = 420;
export const HOLD_INTERVAL_MS = 80;
/** How far a finger may move on the button before the press counts as a scroll or a drag. */
export const HOLD_SLOP_PX = 8;

/**
 * Fire a callback on press, then repeatedly while the press is held (a stepper's − and +).
 *
 * The press is a pointer event, so it works for a finger and a mouse alike; releasing, leaving or
 * cancelling the pointer stops the repeat. A mouse steps on press. A finger steps on release, or when
 * the hold starts repeating, so a scroll or a sheet pull that starts on the button changes nothing.
 */
export function holdRepeat(node: HTMLElement, initial: () => void): { update: (next: () => void) => void; destroy: () => void } {
  let fire = initial;
  let delay: ReturnType<typeof setTimeout> | null = null;
  let repeat: ReturnType<typeof setInterval> | null = null;
  // A finger's press that has not stepped yet, and where it landed.
  let pending: { x: number; y: number } | null = null;

  function stop(): void {
    if (delay !== null) clearTimeout(delay);
    if (repeat !== null) clearInterval(repeat);
    delay = null;
    repeat = null;
    pending = null;
  }

  function onPointerDown(event: PointerEvent): void {
    if (event.button !== 0 || (node as HTMLButtonElement).disabled) return;
    // Keep focus where it is: the field beside the button, not the button.
    event.preventDefault();
    stop();
    const touch = event.pointerType === 'touch' || event.pointerType === 'pen';
    if (touch) pending = { x: event.clientX, y: event.clientY };
    else fire();
    delay = setTimeout(() => {
      if (pending) fire();
      pending = null;
      repeat = setInterval(() => fire(), HOLD_INTERVAL_MS);
    }, HOLD_DELAY_MS);
  }

  function onPointerMove(event: PointerEvent): void {
    if (pending && Math.hypot(event.clientX - pending.x, event.clientY - pending.y) > HOLD_SLOP_PX) stop();
  }

  function onPointerUp(): void {
    if (pending) fire();
    stop();
  }

  // A long press must not open the context menu on a phone.
  function onContextMenu(event: Event): void {
    event.preventDefault();
  }

  node.addEventListener('pointerdown', onPointerDown);
  node.addEventListener('pointermove', onPointerMove);
  node.addEventListener('pointerup', onPointerUp);
  node.addEventListener('pointercancel', stop);
  node.addEventListener('pointerleave', stop);
  node.addEventListener('contextmenu', onContextMenu);

  return {
    update(next) { fire = next; },
    destroy() {
      stop();
      node.removeEventListener('pointerdown', onPointerDown);
      node.removeEventListener('pointermove', onPointerMove);
      node.removeEventListener('pointerup', onPointerUp);
      node.removeEventListener('pointercancel', stop);
      node.removeEventListener('pointerleave', stop);
      node.removeEventListener('contextmenu', onContextMenu);
    },
  };
}
