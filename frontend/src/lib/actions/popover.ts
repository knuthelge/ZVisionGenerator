import { pushEscapeLayer } from './escapeStack';

/** Where a popover's edge lines up with its anchor. */
export type PopoverAlign = 'start' | 'end';

/** Which side of its anchor a popover prefers: it opens on the other side when the preferred one has no room. */
export type PopoverSide = 'below' | 'above';

export interface PopoverOptions {
  anchor: HTMLElement | null;
  align?: PopoverAlign;
  /** The side to open on when it fits; below by default. */
  side?: PopoverSide;
  /** Called when the popover should close: outside click, Escape, or the page scrolling or resizing. */
  onclose: () => void;
  /** Where Escape puts focus; the anchor when absent. */
  returnFocus?: HTMLElement | null;
  /**
   * Stay open when the page scrolls or resizes and move with the anchor instead. For a popover opened by a tap, where
   * the phone's keyboard sliding away (it resizes and scrolls the page) would otherwise close it as it opens.
   */
  followViewport?: boolean;
  /**
   * Size the popover to the room it has. It is as tall as its content, opens on the side of its anchor with more room when its
   * own side has too little, and is capped at the room that side offers, so a body that scrolls is the caller's to arrange.
   */
  room?: boolean;
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

/** The popover's left edge: lined up with its anchor's start or end edge, and kept inside the viewport. */
function leftFor(anchor: Box, width: number, viewportWidth: number, align: PopoverAlign): number {
  const wanted = align === 'end' ? anchor.right - width : anchor.left;
  return Math.min(Math.max(MARGIN_PX, wanted), viewportWidth - width - MARGIN_PX);
}

/**
 * Place a popover below its anchor, or above when it would run off the bottom of the viewport (or the other way round with `side: 'above'`),
 * aligned to the anchor's start or end edge and kept inside the viewport horizontally.
 */
export function placePopover(anchor: Box, popover: Box, viewport: { width: number; height: number }, align: PopoverAlign, side: PopoverSide = 'below'): { top: number; left: number } {
  const below = anchor.bottom + GAP_PX;
  const above = anchor.top - GAP_PX - popover.height;
  const top = side === 'above' && above >= MARGIN_PX
    ? above
    : below + popover.height > viewport.height - MARGIN_PX
      ? Math.max(MARGIN_PX, above)
      : below;
  return { top, left: leftFor(anchor, popover.width, viewport.width, align) };
}

/**
 * Place a popover that is as tall as its content: on the preferred side of its anchor when it fits there, else on the side with
 * more room. The height is capped at the room on that side, so it never runs past the viewport's edge.
 */
export function placePopoverInRoom(
  anchor: Box,
  popover: { width: number; height: number },
  viewport: { width: number; height: number },
  align: PopoverAlign,
  side: PopoverSide = 'below'
): { top: number; left: number; maxHeight: number } {
  const roomBelow = Math.max(0, viewport.height - anchor.bottom - GAP_PX - MARGIN_PX);
  const roomAbove = Math.max(0, anchor.top - GAP_PX - MARGIN_PX);
  const rooms = { below: roomBelow, above: roomAbove };
  const other = side === 'above' ? 'below' : 'above';
  const chosen = popover.height > rooms[side] && rooms[other] > rooms[side] ? other : side;
  const height = Math.min(popover.height, rooms[chosen]);
  const top = chosen === 'below' ? anchor.bottom + GAP_PX : anchor.top - GAP_PX - height;
  return { top, left: leftFor(anchor, popover.width, viewport.width, align), maxHeight: height };
}

/**
 * Show an element as a popover next to its anchor.
 *
 * Moves the element to `<body>` so scrolling or overflow on its ancestors cannot clip it, fixes it
 * beside the anchor, re-places it when its size changes, and closes it on an outside click, Escape (returning focus to the anchor, or to `returnFocus`), or
 * when the page scrolls or resizes, since a fixed popover cannot follow its anchor.
 */
export function popover(node: HTMLElement, initial: PopoverOptions): { update: (next: PopoverOptions) => void; destroy: () => void } {
  let options = initial;
  document.body.appendChild(node);
  node.style.position = 'fixed';
  // Marks the node for a dialog's focus trap, which lets focus rest in a popover layered above it.
  node.dataset.popover = '';

  function place(): void {
    if (!options.anchor) return;
    if (options.room) {
      // Measured at its natural height, then capped: a height set before would hide how tall the content wants to be.
      node.style.maxHeight = '';
      const box = node.getBoundingClientRect();
      const fit = placePopoverInRoom(options.anchor.getBoundingClientRect(), box, { width: window.innerWidth, height: window.innerHeight }, options.align ?? 'start', options.side);
      node.style.maxHeight = `${fit.maxHeight}px`;
      node.style.top = `${fit.top}px`;
      node.style.left = `${fit.left}px`;
      return;
    }
    const { top, left } = placePopover(
      options.anchor.getBoundingClientRect(),
      node.getBoundingClientRect(),
      { width: window.innerWidth, height: window.innerHeight },
      options.align ?? 'start',
      options.side
    );
    node.style.top = `${top}px`;
    node.style.left = `${left}px`;
  }

  /** Whether *target* is inside a popover opened from this one (a menu in a panel), which lives in <body> too. */
  function inNested(target: Node): boolean {
    const nested = (target instanceof Element ? target : target.parentElement)?.closest('[data-popover-nested]');
    return Boolean(nested) && nested !== node;
  }

  function onPointerDown(event: PointerEvent): void {
    const target = event.target as Node;
    if (inNested(target)) return;
    if (!node.contains(target) && !options.anchor?.contains(target)) options.onclose();
  }

  // Escape closes the popover when it is the innermost overlay, so a dialog behind it stays open.
  function onEscape(): void {
    options.onclose();
    (options.returnFocus ?? options.anchor)?.focus();
  }

  function onViewportChange(event: Event): void {
    // Scrolling inside this popover, or inside a menu opened from it, moves nothing it is anchored to.
    if (event.type === 'scroll' && (node.contains(event.target as Node) || inNested(event.target as Node))) return;
    if (options.followViewport) {
      place();
      return;
    }
    options.onclose();
  }

  // A popover that grows or shrinks after it opens (a group expanding in a menu) needs a new spot: it may now run
  // off the bottom of the viewport. Placement sets top and left (and, in room mode, a max-height that settles at the same size), so it cannot retrigger this on its own.
  const sizeWatch = typeof ResizeObserver === 'undefined' ? null : new ResizeObserver(() => place());
  sizeWatch?.observe(node);
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
      sizeWatch?.disconnect();
      document.removeEventListener('pointerdown', onPointerDown, true);
      releaseEscape();
      window.removeEventListener('resize', onViewportChange);
      window.removeEventListener('scroll', onViewportChange, true);
      delete node.dataset.popover;
      node.remove();
    },
  };
}
