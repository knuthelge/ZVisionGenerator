import { pushEscapeLayer } from './escapeStack';

/** Where a popover's edge lines up with its anchor. */
export type PopoverAlign = 'start' | 'end';

export interface PopoverOptions {
  anchor: HTMLElement | null;
  align?: PopoverAlign;
  /** Called when the popover should close: outside click, Escape, or the page scrolling or resizing. */
  onclose: () => void;
}

interface Box {
  top: number;
  bottom: number;
  left: number;
  right: number;
  width: number;
  height: number;
}

const GAP_PX = 6;
const MARGIN_PX = 8;

/**
 * Place a popover below its anchor, or above when it would run off the bottom of the viewport,
 * aligned to the anchor's start or end edge and kept inside the viewport horizontally.
 */
export function placePopover(anchor: Box, popover: Box, viewport: { width: number; height: number }, align: PopoverAlign): { top: number; left: number } {
  const below = anchor.bottom + GAP_PX;
  const top = below + popover.height > viewport.height - MARGIN_PX
    ? Math.max(MARGIN_PX, anchor.top - GAP_PX - popover.height)
    : below;
  const preferred = align === 'end' ? anchor.right - popover.width : anchor.left;
  const left = Math.min(Math.max(MARGIN_PX, preferred), viewport.width - popover.width - MARGIN_PX);
  return { top, left };
}

/**
 * Show an element as a popover next to its anchor.
 *
 * Moves the element to `<body>` so scrolling or overflow on its ancestors cannot clip it, fixes it
 * beside the anchor, and closes it on an outside click, Escape (returning focus to the anchor), or
 * when the page scrolls or resizes, since a fixed popover cannot follow its anchor.
 */
export function popover(node: HTMLElement, initial: PopoverOptions): { update: (next: PopoverOptions) => void; destroy: () => void } {
  let options = initial;
  document.body.appendChild(node);
  node.style.position = 'fixed';

  function place(): void {
    if (!options.anchor) return;
    const { top, left } = placePopover(
      options.anchor.getBoundingClientRect(),
      node.getBoundingClientRect(),
      { width: window.innerWidth, height: window.innerHeight },
      options.align ?? 'start'
    );
    node.style.top = `${top}px`;
    node.style.left = `${left}px`;
  }

  function onPointerDown(event: PointerEvent): void {
    const target = event.target as Node;
    // A popover opened from inside this one (a menu in a panel) lives in <body> too; using it is not an outside click.
    const nested = (target instanceof Element ? target : target.parentElement)?.closest('[data-popover-nested]');
    if (nested && nested !== node) return;
    if (!node.contains(target) && !options.anchor?.contains(target)) options.onclose();
  }

  // Escape closes the popover when it is the innermost overlay, so a dialog behind it stays open.
  function onEscape(): void {
    options.onclose();
    options.anchor?.focus();
  }

  function onViewportChange(event: Event): void {
    if (event.type === 'scroll' && node.contains(event.target as Node)) return;
    options.onclose();
  }

  place();
  document.addEventListener('pointerdown', onPointerDown, true);
  const releaseEscape = pushEscapeLayer(onEscape);
  window.addEventListener('resize', onViewportChange);
  window.addEventListener('scroll', onViewportChange, true);

  return {
    update(next: PopoverOptions): void {
      options = next;
      place();
    },
    destroy(): void {
      document.removeEventListener('pointerdown', onPointerDown, true);
      releaseEscape();
      window.removeEventListener('resize', onViewportChange);
      window.removeEventListener('scroll', onViewportChange, true);
      node.remove();
    },
  };
}
