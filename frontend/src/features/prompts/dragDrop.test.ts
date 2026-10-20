import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { PromptBuilder } from './builder.svelte';
import { DragDrop, dropTargetAt, reorderDrag } from './dragDrop.svelte';

/** Two sets, the second empty, an entry in the first, and two snippets, each 100px tall from its top. */
function page(): HTMLElement {
  const root = document.createElement('div');
  root.innerHTML = `
    <section data-dnd-set="s1" data-dnd-empty="false">
      <header class="set-head"><span class="set-title">One</span></header>
      <div data-dnd-entry="e1" data-dnd-set="s1"><span class="e1-text">Entry 1</span></div>
      <div data-dnd-entry="e2" data-dnd-set="s1"><span class="e2-text">Entry 2</span></div>
    </section>
    <section data-dnd-set="s2" data-dnd-empty="true"><header class="set-head"></header><p class="s2-body">Empty</p></section>
    <ul><li data-dnd-snippet="n1"><span class="n1-text">a</span></li><li data-dnd-snippet="n2"><span class="n2-text">b</span></li></ul>`;
  for (const el of root.querySelectorAll<HTMLElement>('[data-dnd-entry], [data-dnd-set], [data-dnd-snippet]')) {
    el.getBoundingClientRect = () => ({ top: 0, height: 100, bottom: 100, left: 0, right: 100, width: 100, x: 0, y: 0, toJSON: () => ({}) });
  }
  document.body.append(root);
  return root;
}

const at = (root: HTMLElement, selector: string) => root.querySelector(selector);

describe('dropTargetAt', () => {
  let root: HTMLElement;
  beforeEach(() => { root = page(); });
  afterEach(() => root.remove());

  it('drops an entry before or after another entry by which half the pointer is over', () => {
    expect(dropTargetAt({ kind: 'entry', id: 'e1' }, at(root, '.e2-text'), 20)).toEqual({ kind: 'entry', setId: 's1', id: 'e2', position: 'before' });
    expect(dropTargetAt({ kind: 'entry', id: 'e1' }, at(root, '.e2-text'), 80)).toEqual({ kind: 'entry', setId: 's1', id: 'e2', position: 'after' });
  });

  it('drops an entry at the end of a set from its header or an empty set', () => {
    expect(dropTargetAt({ kind: 'entry', id: 'e1' }, at(root, '.set-title'), 10)).toEqual({ kind: 'set-end', setId: 's1' });
    expect(dropTargetAt({ kind: 'entry', id: 'e1' }, at(root, '.s2-body'), 10)).toEqual({ kind: 'set-end', setId: 's2' });
  });

  it('drops a set around another set, also over its entries', () => {
    expect(dropTargetAt({ kind: 'set', id: 's2' }, at(root, '.e1-text'), 90)).toEqual({ kind: 'set', id: 's1', position: 'after' });
  });

  it('inserts a snippet into an entry, and moves it among the snippets', () => {
    expect(dropTargetAt({ kind: 'snippet', id: 'n1' }, at(root, '.e1-text'), 50)).toEqual({ kind: 'insert', entryId: 'e1' });
    expect(dropTargetAt({ kind: 'snippet', id: 'n1' }, at(root, '.n2-text'), 10)).toEqual({ kind: 'snippet', id: 'n2', position: 'before' });
  });

  it('has nowhere to land over itself or outside every zone', () => {
    expect(dropTargetAt({ kind: 'entry', id: 'e1' }, at(root, '.e1-text'), 10)).toBeNull();
    expect(dropTargetAt({ kind: 'snippet', id: 'n1' }, document.body, 10)).toBeNull();
    expect(dropTargetAt({ kind: 'set', id: 's1' }, null, 10)).toBeNull();
  });
});

function fakeBuilder() {
  return { endTyping: vi.fn(), moveSet: vi.fn(), moveEntry: vi.fn(), moveSnippet: vi.fn(), insertSnippetRef: vi.fn(), doc: { snippets: [{ id: 'n1', name: 'style' }] } };
}

describe('DragDrop', () => {
  let root: HTMLElement;
  beforeEach(() => { root = page(); });
  afterEach(() => root.remove());

  it('moves an entry to where it was let go', () => {
    const builder = fakeBuilder();
    const dnd = new DragDrop(builder as unknown as PromptBuilder);
    dnd.start('entry', 'e1');
    dnd.over(at(root, '.e2-text'), 80);
    expect(dnd.markFor('entry', 'e2')).toBe('after');
    dnd.drop();
    expect(builder.moveEntry).toHaveBeenCalledWith('e1', 's1', 'e2', 'after');
    expect(dnd.dragging).toBeNull();
  });

  it('adds a snippet reference when a snippet is let go on an entry', () => {
    const builder = fakeBuilder();
    const dnd = new DragDrop(builder as unknown as PromptBuilder);
    dnd.start('snippet', 'n1');
    dnd.over(at(root, '.e1-text'), 50);
    dnd.drop();
    expect(builder.insertSnippetRef).toHaveBeenCalledWith('e1', 'style');
  });

  it('changes nothing when let go where nothing takes it', () => {
    const builder = fakeBuilder();
    const dnd = new DragDrop(builder as unknown as PromptBuilder);
    dnd.start('entry', 'e1');
    dnd.over(at(root, '.e2-text'), 80);
    dnd.over(document.body, 10);
    dnd.drop();
    expect(builder.moveEntry).not.toHaveBeenCalled();
  });
});

/** Dispatch a pointer event; jsdom has no PointerEvent, so the pointer fields are set by hand. */
function pointer(node: Element, type: string, pointerType: string, x: number, y: number): void {
  const event = new MouseEvent(type, { bubbles: true, cancelable: true, clientX: x, clientY: y, button: 0 });
  Object.defineProperty(event, 'pointerType', { value: pointerType });
  Object.defineProperty(event, 'pointerId', { value: 1 });
  node.dispatchEvent(event);
}

describe('reorderDrag', () => {
  let root: HTMLElement;
  beforeEach(() => {
    root = page();
    document.elementFromPoint = () => root.querySelector('.e2-text');
  });
  afterEach(() => root.remove());

  it('lets a finger drag from the grip at once, and swallows the click that follows the drop', () => {
    const builder = fakeBuilder();
    const dnd = new DragDrop(builder as unknown as PromptBuilder);
    const grip = document.createElement('span');
    root.querySelector('[data-dnd-entry="e1"]')!.append(grip);
    const action = reorderDrag(grip, { dnd, kind: 'entry', id: 'e1' });
    const onclick = vi.fn();
    root.addEventListener('click', onclick);

    pointer(grip, 'pointerdown', 'touch', 0, 10);
    expect(dnd.dragging).toEqual({ kind: 'entry', id: 'e1' });
    pointer(grip, 'pointermove', 'touch', 0, 80);
    pointer(grip, 'pointerup', 'touch', 0, 80);
    grip.click();
    expect(builder.moveEntry).toHaveBeenCalledWith('e1', 's1', 'e2', 'after');
    expect(onclick).not.toHaveBeenCalled();
    action.destroy();
  });

  it('keeps a mouse click a click, and starts a mouse drag once it moves', () => {
    const builder = fakeBuilder();
    const dnd = new DragDrop(builder as unknown as PromptBuilder);
    const row = root.querySelector('[data-dnd-entry="e1"]') as HTMLElement;
    const action = reorderDrag(row, { dnd, kind: 'entry', id: 'e1' });

    pointer(row, 'pointerdown', 'mouse', 0, 10);
    pointer(row, 'pointerup', 'mouse', 0, 10);
    expect(builder.moveEntry).not.toHaveBeenCalled();

    pointer(row, 'pointerdown', 'mouse', 0, 10);
    pointer(row, 'pointermove', 'mouse', 0, 30);
    expect(dnd.dragging).not.toBeNull();
    pointer(row, 'pointerup', 'mouse', 0, 30);
    expect(builder.moveEntry).toHaveBeenCalled();
    action.destroy();
  });

  it('leaves a finger elsewhere on the row to scroll the page', () => {
    const dnd = new DragDrop(fakeBuilder() as unknown as PromptBuilder);
    const row = root.querySelector('[data-dnd-snippet="n1"]') as HTMLElement;
    const action = reorderDrag(row, { dnd, kind: 'snippet', id: 'n1', handle: '.grip' });
    pointer(row.querySelector('.n1-text')!, 'pointerdown', 'touch', 0, 10);
    expect(dnd.dragging).toBeNull();
    action.destroy();
  });
});
