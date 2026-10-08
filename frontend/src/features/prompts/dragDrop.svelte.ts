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
 * Drag-and-drop arranging for the prompt builder.
 *
 * Sets and entries are dragged by their grips, snippets by their row. Entries drop before or after any entry
 * (in any set) or onto a set header to go to its end; a snippet dropped onto an entry adds a `$reference`.
 */
export class DragDrop {
  dragging = $state<{ kind: DragKind; id: string } | null>(null);
  target = $state<DropTarget | null>(null);
  readonly #builder: PromptBuilder;

  constructor(builder: PromptBuilder) {
    this.#builder = builder;
  }

  start(event: DragEvent, kind: DragKind, id: string, ghost: HTMLElement | null): void {
    this.#builder.endTyping();
    this.dragging = { kind, id };
    if (event.dataTransfer) {
      event.dataTransfer.effectAllowed = 'move';
      event.dataTransfer.setData('text/plain', `${kind}:${id}`);
      if (ghost) event.dataTransfer.setDragImage(ghost, 24, 16);
    }
  }

  end(): void {
    this.dragging = null;
    this.target = null;
  }

  /** Dragging over a set (its header or body). */
  overSet(event: DragEvent, setId: string, element: HTMLElement, onHeader: boolean, isEmpty: boolean): void {
    const drag = this.dragging;
    if (drag?.kind === 'set' && drag.id !== setId) {
      this.#accept(event, { kind: 'set', id: setId, position: dropPosition(element.getBoundingClientRect(), event.clientY) });
    } else if (drag?.kind === 'entry' && (onHeader || isEmpty)) {
      this.#accept(event, { kind: 'set-end', setId });
    }
  }

  /** Dragging over an entry card. */
  overEntry(event: DragEvent, setId: string, entryId: string, element: HTMLElement): void {
    const drag = this.dragging;
    if (drag?.kind === 'entry' && drag.id !== entryId) {
      event.stopPropagation();
      this.#accept(event, { kind: 'entry', setId, id: entryId, position: dropPosition(element.getBoundingClientRect(), event.clientY) });
    } else if (drag?.kind === 'snippet') {
      event.stopPropagation();
      this.#accept(event, { kind: 'insert', entryId });
    }
  }

  /** Dragging over a snippet row. */
  overSnippet(event: DragEvent, snippetId: string, element: HTMLElement): void {
    const drag = this.dragging;
    if (drag?.kind === 'snippet' && drag.id !== snippetId) {
      this.#accept(event, { kind: 'snippet', id: snippetId, position: dropPosition(element.getBoundingClientRect(), event.clientY) });
    }
  }

  drop(event: DragEvent): void {
    const drag = this.dragging;
    const target = this.target;
    this.end();
    if (!drag || !target) return;
    event.preventDefault();
    event.stopPropagation();
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

  #accept(event: DragEvent, target: DropTarget): void {
    event.preventDefault();
    if (event.dataTransfer) event.dataTransfer.dropEffect = 'move';
    if (JSON.stringify(target) !== JSON.stringify(this.target)) this.target = target;
  }
}
