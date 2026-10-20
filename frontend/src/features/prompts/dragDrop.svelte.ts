import type { PromptBuilder } from './builder.svelte';
import { dropPosition } from './document';

export type DragKind = 'set' | 'entry' | 'snippet';

/** Where a drop would land: before/after an item, at the end of a set, or onto an entry (insert a snippet). */
export type DropTarget =
  | { kind: 'set'; id: string; position: 'before' | 'after' }
  | { kind: 'entry'; setId: string; id: string; position: 'before' | 'after' }
  | { kind: 'set-end'; setId: string }
  | { kind: 'snippet'; id: string; position: 'before' | 'after' }
  | { kind: 'insert'; entryId: string };

/**
 * Return where a drag would land over `el` at height `clientY`, or null when nothing there takes it.
 *
 * Drop zones are marked in the markup: an entry card with `data-dnd-entry` (its id) and `data-dnd-set`, a set with
 * `data-dnd-set` (and `data-dnd-empty` when it shows no entries) and a `.set-head`, a snippet row with
 * `data-dnd-snippet`. An entry drops before or after any entry (in any set) or onto a set's header to go to its end; a
 * set before or after another set; a snippet before or after another snippet, or onto an entry to add a `$reference`.
 */
export function dropTargetAt(drag: { kind: DragKind; id: string }, el: Element | null, clientY: number): DropTarget | null {
  if (!el) return null;
  const entryEl = el.closest<HTMLElement>('[data-dnd-entry]');
  if (entryEl) {
    const id = entryEl.dataset.dndEntry!;
    if (drag.kind === 'entry' && drag.id !== id) {
      return { kind: 'entry', setId: entryEl.dataset.dndSet!, id, position: dropPosition(entryEl.getBoundingClientRect(), clientY) };
    }
    if (drag.kind === 'snippet') return { kind: 'insert', entryId: id };
  }
  const setEl = el.closest<HTMLElement>('[data-dnd-set]:not([data-dnd-entry])');
  if (setEl) {
    const setId = setEl.dataset.dndSet!;
    if (drag.kind === 'set' && drag.id !== setId) return { kind: 'set', id: setId, position: dropPosition(setEl.getBoundingClientRect(), clientY) };
    if (drag.kind === 'entry' && (el.closest('.set-head') !== null || setEl.dataset.dndEmpty === 'true')) return { kind: 'set-end', setId };
  }
  const snippetEl = el.closest<HTMLElement>('[data-dnd-snippet]');
  if (snippetEl && drag.kind === 'snippet' && drag.id !== snippetEl.dataset.dndSnippet) {
    return { kind: 'snippet', id: snippetEl.dataset.dndSnippet!, position: dropPosition(snippetEl.getBoundingClientRect(), clientY) };
  }
  return null;
}

/**
 * Drag-to-arrange state for the prompt builder.
 *
 * A drag starts on a grip (or, with a mouse, a snippet's row), follows the pointer over the drop zones marked in the
 * markup (see {@link dropTargetAt}) and applies the move when it is let go. {@link reorderDrag} drives it from pointer
 * events, so a finger, a pen and a mouse arrange the same way.
 */
export class DragDrop {
  dragging = $state<{ kind: DragKind; id: string } | null>(null);
  target = $state<DropTarget | null>(null);
  readonly #builder: PromptBuilder;

  constructor(builder: PromptBuilder) {
    this.#builder = builder;
  }

  start(kind: DragKind, id: string): void {
    this.#builder.endTyping();
    this.dragging = { kind, id };
    this.target = null;
  }

  end(): void {
    this.dragging = null;
    this.target = null;
  }

  /** The pointer is over `el` at height `clientY`. */
  over(el: Element | null, clientY: number): void {
    if (!this.dragging) return;
    const next = dropTargetAt(this.dragging, el, clientY);
    if (JSON.stringify(next) !== JSON.stringify(this.target)) this.target = next;
  }

  /** Let go: apply the move to where the drag would land, if anywhere. */
  drop(): void {
    const drag = this.dragging;
    const target = this.target;
    this.end();
    if (!drag || !target) return;
    const builder = this.#builder;
    if (drag.kind === 'set' && target.kind === 'set') builder.moveSet(drag.id, target.id, target.position);
    else if (drag.kind === 'entry' && target.kind === 'entry') builder.moveEntry(drag.id, target.setId, target.id, target.position);
    else if (drag.kind === 'entry' && target.kind === 'set-end') builder.moveEntry(drag.id, target.setId, null, 'after');
    else if (drag.kind === 'snippet' && target.kind === 'snippet') builder.moveSnippet(drag.id, target.id, target.position);
    else if (drag.kind === 'snippet' && target.kind === 'insert') {
      const name = builder.doc.snippets.find((item) => item.id === drag.id)?.name;
      if (name) builder.insertSnippetRef(target.entryId, name);
    }
  }

  /** The drop indicator for an item: 'before', 'after', 'into' or null. */
  markFor(kind: 'set' | 'entry' | 'snippet', id: string): 'before' | 'after' | 'into' | null {
    const target = this.target;
    if (!target) return null;
    if (kind === 'set' && target.kind === 'set-end' && target.setId === id) return 'into';
    if (kind === 'entry' && target.kind === 'insert' && target.entryId === id) return 'into';
    if (target.kind === kind && 'id' in target && target.id === id && 'position' in target) return target.position;
    return null;
  }
}

export interface ReorderOptions {
  dnd: DragDrop;
  kind: DragKind;
  id: string;
  /** A finger or a pen starts only on this part (the grip), so a swipe elsewhere on the row still scrolls. */
  handle?: string;
  enabled?: boolean;
}

/** A mouse must move this far before a press becomes a drag, so a click stays a click. */
const MOUSE_SLOP_PX = 4;
/** Within this distance of a scroller's edge, a drag scrolls it. */
const EDGE_PX = 48;
const EDGE_SPEED_PX = 12;

/** The nearest ancestor that scrolls vertically. */
function scrollParent(node: Element): HTMLElement | null {
  for (let el = node.parentElement; el; el = el.parentElement) {
    const overflow = getComputedStyle(el).overflowY;
    if ((overflow === 'auto' || overflow === 'scroll') && el.scrollHeight > el.clientHeight) return el;
  }
  return null;
}

/**
 * Drag an item of the prompt builder to a new place with a mouse, a finger or a pen.
 *
 * A finger starts on the handle at once (it is `touch-action: none`, so the page does not scroll); a mouse starts
 * after a few pixels, so a click on the row still works. While dragging, the drop target follows the pointer and the
 * list scrolls near its edges; letting go applies the move and swallows the click that would follow.
 */
export function reorderDrag(node: HTMLElement, initial: ReorderOptions): { update: (next: ReorderOptions) => void; destroy: () => void } {
  let options = initial;
  let pending: { x: number; y: number; pointerId: number } | null = null;
  let dragging = false;
  let scroller: HTMLElement | null = null;
  let lastY = 0;
  let edgeFrame = 0;
  // The press that ended a drag is not a click on the row (a snippet's row opens it).
  let swallowClick = false;

  function begin(pointerId: number): void {
    pending = null;
    dragging = true;
    node.setPointerCapture?.(pointerId);
    scroller = scrollParent(node);
    options.dnd.start(options.kind, options.id);
  }

  function stop(): void {
    pending = null;
    dragging = false;
    cancelAnimationFrame(edgeFrame);
    edgeFrame = 0;
  }

  // Near a scroller's edge the list keeps scrolling while the pointer rests there.
  function edgeScroll(): void {
    edgeFrame = 0;
    if (!dragging || !scroller) return;
    const box = scroller.getBoundingClientRect();
    const step = lastY < box.top + EDGE_PX ? -EDGE_SPEED_PX : lastY > box.bottom - EDGE_PX ? EDGE_SPEED_PX : 0;
    if (step === 0) return;
    scroller.scrollTop += step;
    edgeFrame = requestAnimationFrame(edgeScroll);
  }

  function onPointerDown(event: PointerEvent): void {
    swallowClick = false;
    if (options.enabled === false || event.button !== 0) return;
    const finger = event.pointerType !== 'mouse';
    if (finger && options.handle && !(event.target as Element).closest(options.handle)) return;
    if ((event.target as Element).closest('input, textarea, select, [contenteditable="true"]')) return;
    if (finger) {
      event.preventDefault();
      begin(event.pointerId);
    } else {
      pending = { x: event.clientX, y: event.clientY, pointerId: event.pointerId };
    }
  }

  function onPointerMove(event: PointerEvent): void {
    if (pending && Math.hypot(event.clientX - pending.x, event.clientY - pending.y) > MOUSE_SLOP_PX) begin(pending.pointerId);
    if (!dragging) return;
    event.preventDefault();
    lastY = event.clientY;
    options.dnd.over(document.elementFromPoint(event.clientX, event.clientY), event.clientY);
    if (!edgeFrame) edgeFrame = requestAnimationFrame(edgeScroll);
  }

  function onPointerUp(): void {
    const dropped = dragging;
    stop();
    if (!dropped) return;
    swallowClick = true;
    options.dnd.drop();
  }

  function onClick(event: MouseEvent): void {
    if (!swallowClick) return;
    swallowClick = false;
    event.preventDefault();
    event.stopPropagation();
  }

  function onPointerCancel(): void {
    if (dragging) options.dnd.end();
    stop();
  }

  node.addEventListener('pointerdown', onPointerDown);
  node.addEventListener('pointermove', onPointerMove);
  node.addEventListener('pointerup', onPointerUp);
  node.addEventListener('pointercancel', onPointerCancel);
  node.addEventListener('click', onClick, true);

  return {
    update(next: ReorderOptions): void {
      options = next;
    },
    destroy(): void {
      stop();
      node.removeEventListener('pointerdown', onPointerDown);
      node.removeEventListener('pointermove', onPointerMove);
      node.removeEventListener('pointerup', onPointerUp);
      node.removeEventListener('pointercancel', onPointerCancel);
      node.removeEventListener('click', onClick, true);
    },
  };
}
