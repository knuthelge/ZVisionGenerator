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
    if (!node.contains(target) && !options.anchor?.contains(target)) options.onclose();
  }

  // Capture phase, so an Escape meant for the popover does not also close a dialog behind it.
  function onKeydown(event: KeyboardEvent): void {
    if (event.key !== 'Escape') return;
    event.preventDefault();
    event.stopPropagation();
    options.onclose();
    options.anchor?.focus();
  }

  function onViewportChange(event: Event): void {
    if (event.type === 'scroll' && node.contains(event.target as Node)) return;
    options.onclose();
  }

  place();
  document.addEventListener('pointerdown', onPointerDown, true);
  document.addEventListener('keydown', onKeydown, true);
  window.addEventListener('resize', onViewportChange);
  window.addEventListener('scroll', onViewportChange, true);

  return {
    update(next: PopoverOptions): void {
      options = next;
      place();
    },
    destroy(): void {
      document.removeEventListener('pointerdown', onPointerDown, true);
      document.removeEventListener('keydown', onKeydown, true);
      window.removeEventListener('resize', onViewportChange);
      window.removeEventListener('scroll', onViewportChange, true);
      node.remove();
    },
  };
}
