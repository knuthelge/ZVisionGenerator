/** Delay before a held press starts repeating, and the interval between repeats. */
export const HOLD_DELAY_MS = 420;
export const HOLD_INTERVAL_MS = 80;

/**
 * Fire a callback on press, then repeatedly while the press is held (a stepper's − and +).
 *
 * The press is a pointer event, so it works for a finger and a mouse alike; releasing, leaving or
 * cancelling the pointer stops the repeat.
 */
export function holdRepeat(node: HTMLElement, initial: () => void): { update: (next: () => void) => void; destroy: () => void } {
  let fire = initial;
  let delay: ReturnType<typeof setTimeout> | null = null;
  let repeat: ReturnType<typeof setInterval> | null = null;

  function stop(): void {
    if (delay !== null) clearTimeout(delay);
    if (repeat !== null) clearInterval(repeat);
    delay = null;
    repeat = null;
  }

  function onPointerDown(event: PointerEvent): void {
    if (event.button !== 0 || (node as HTMLButtonElement).disabled) return;
    // Keep focus where it is: the field beside the button, not the button.
    event.preventDefault();
    stop();
    fire();
    delay = setTimeout(() => {
      repeat = setInterval(() => fire(), HOLD_INTERVAL_MS);
    }, HOLD_DELAY_MS);
  }

  // A long press must not open the context menu on a phone.
  function onContextMenu(event: Event): void {
    event.preventDefault();
  }

  node.addEventListener('pointerdown', onPointerDown);
  node.addEventListener('pointerup', stop);
  node.addEventListener('pointercancel', stop);
  node.addEventListener('pointerleave', stop);
  node.addEventListener('contextmenu', onContextMenu);

  return {
    update(next) { fire = next; },
    destroy() {
      stop();
      node.removeEventListener('pointerdown', onPointerDown);
      node.removeEventListener('pointerup', stop);
      node.removeEventListener('pointercancel', stop);
      node.removeEventListener('pointerleave', stop);
      node.removeEventListener('contextmenu', onContextMenu);
    },
  };
}
