import type {
  DocumentEntry,
  DocumentProblem,
  EnhanceAxis,
  EnhanceSettings,
  EntryEnhance,
  PromptDocument,
  PromptDocumentLoad,
  PromptDocumentPreview,
  PromptDocumentSaved,
  PromptValue,
} from '$lib/types';
import {
  copyEntry,
  copySet,
  findEntry,
  moveItem,
  newEntry,
  newId,
  newSet,
  newSnippet,
  renameRefsInValue,
  setNameProblem,
  snippetNameProblem,
  structuredUses,
  textAsFields,
  uniqueName,
  valueAsText,
} from './document';
import { UndoHistory } from './history';

export type PreviewFn = (document: PromptDocument, rollEntryId: string | null) => Promise<PromptDocumentPreview>;
export type DropPosition = 'before' | 'after';
/** Which value of an entry a text field edits. */
export type EntryValueKey = 'prompt' | 'negative';

const PREVIEW_DELAY_MS = 250;

function snapshot(document: PromptDocument): string {
  return JSON.stringify($state.snapshot(document));
}

/**
 * Editing state of one prompt file in the builder.
 *
 * Every structural change goes through {@link edit}, which records one undo step and marks the document
 * changed; typing in a text field records one step per editing session ({@link beginTyping}, then {@link type}),
 * taken at the first keystroke so a session without typing leaves no step.
 * Previews are fetched from the server a moment after the last change.
 */
export class PromptBuilder {
  path = $state<string | null>(null);
  revision = $state('');
  /** The file text the document was loaded from; a forced save applies the edits onto it. */
  baseText = $state('');
  doc = $state<PromptDocument>({ snippets: [], sets: [] });
  dirty = $state(false);
  selectedId = $state<string | null>(null);
  /** The text field being edited (`<entry id>:prompt`, `<entry id>:negative`, `<entry id>:field:<n>`, `snippet:<id>`). */
  editingKey = $state<string | null>(null);
  preview = $state<PromptDocumentPreview | null>(null);
  rolled = $state<{ entryId: string; base: string; prompt: string } | null>(null);
  axes = $state<EnhanceAxis[]>([]);
  defaults = $state<EnhanceSettings>({ style: 'keep', mood: 'keep', details: [], length: 'same', motion: [] });

  readonly #history = new UndoHistory<string>(100);
  readonly #previewFn: PreviewFn;
  readonly #onchange: () => void;
  #previewTimer: ReturnType<typeof setTimeout> | undefined;
  #previewRequest = 0;
  /** A typing session has started but nothing is typed yet; the first keystroke records its undo step. */
  #typingPending = false;

  constructor(previewFn: PreviewFn, onchange: () => void = () => undefined) {
    this.#previewFn = previewFn;
    this.#onchange = onchange;
  }

  get problems(): DocumentProblem[] {
    return this.preview?.problems ?? [];
  }

  get errors(): DocumentProblem[] {
    return this.problems.filter((problem) => problem.severity === 'error');
  }

  problemsFor(id: string): DocumentProblem[] {
    return this.problems.filter((problem) => problem.target === id);
  }

  /** Show a loaded file; `restored` replaces its document with unsaved edits kept from an earlier visit. */
  load(payload: PromptDocumentLoad & { document: PromptDocument }, restored: PromptDocument | null = null): void {
    this.path = payload.path;
    this.revision = payload.revision;
    this.baseText = payload.raw_text;
    this.axes = payload.enhance_matrix.axes;
    this.defaults = payload.enhance_matrix.defaults;
    this.doc = restored ?? payload.document;
    this.dirty = restored !== null;
    this.selectedId = this.doc.sets[0]?.entries[0]?.id ?? null;
    this.editingKey = null;
    this.#typingPending = false;
    this.rolled = null;
    this.preview = null;
    this.#history.clear();
    this.refreshPreview(0);
  }

  /** Show no document: the file at `path` can't be built and is being repaired. */
  close(path: string | null): void {
    this.dispose();
    this.path = path;
    this.revision = '';
    this.baseText = '';
    this.doc = { snippets: [], sets: [] };
    this.dirty = false;
    this.selectedId = null;
    this.editingKey = null;
    this.#typingPending = false;
    this.rolled = null;
    this.preview = null;
    this.#history.clear();
  }

  /** The document as JSON, as it would be sent to the server now. */
  snapshot(): string {
    return snapshot(this.doc);
  }

  /**
   * Take a save's result; `sent` is the {@link snapshot} that was saved.
   *
   * When nothing changed since, the saved file's document replaces this one. Edits made while the save ran are
   * kept instead: items that were saved move to the saved file's positional ids, and items that were not (added
   * meanwhile, or brought back by undo) get fresh ids, so no old position is reused. The document stays unsaved.
   * Either way undo restarts, since older snapshots use the old ids.
   */
  applySaved(saved: PromptDocumentSaved, sent: string): void {
    const moved = new Map<string, string>();
    const remap = (id: string): string => {
      if (!moved.has(id)) moved.set(id, saved.ids[id] ?? newId());
      return moved.get(id)!;
    };
    this.revision = saved.revision;
    this.baseText = saved.raw_text;
    if (snapshot(this.doc) === sent) {
      this.doc = saved.document;
      this.dirty = false;
      this.editingKey = null;
    } else {
      for (const snippet of this.doc.snippets) snippet.id = remap(snippet.id);
      for (const set of this.doc.sets) {
        set.id = remap(set.id);
        for (const entry of set.entries) entry.id = remap(entry.id);
      }
      this.editingKey = this.editingKey ? remapEditingKey(this.editingKey, remap) : null;
      this.dirty = true;
    }
    this.selectedId = this.selectedId ? (moved.get(this.selectedId) ?? saved.ids[this.selectedId] ?? null) : null;
    this.#typingPending = this.editingKey !== null;
    this.rolled = null;
    this.#history.clear();
    this.refreshPreview(0);
  }

  /** Record an undo step, then apply `change`. */
  edit(change: (document: PromptDocument) => void): void {
    this.#history.push(snapshot(this.doc));
    change(this.doc);
    // Typing after this change is its own undo step.
    this.#typingPending = this.editingKey !== null;
    this.#changed();
  }

  /** Start a text-editing session: the whole session is one undo step. */
  beginTyping(key: string): void {
    if (this.editingKey === key) return;
    this.editingKey = key;
    this.#typingPending = true;
  }

  endTyping(): void {
    this.editingKey = null;
    this.#typingPending = false;
  }

  /** Apply a keystroke-sized change inside a typing session (no undo step of its own). */
  type(change: (document: PromptDocument) => void): void {
    if (this.#typingPending) {
      this.#history.push(snapshot(this.doc));
      this.#typingPending = false;
    }
    change(this.doc);
    this.#changed();
  }

  undo(): boolean {
    const previous = this.#history.pop();
    if (previous === undefined) return false;
    this.doc = JSON.parse(previous) as PromptDocument;
    this.editingKey = null;
    this.#typingPending = false;
    if (this.selectedId && !findEntry(this.doc, this.selectedId)) this.selectedId = null;
    this.#changed();
    return true;
  }

  select(entryId: string | null): void {
    this.selectedId = entryId;
  }

  /** Fetch previews and problems for the current document, `delay` ms from now. */
  refreshPreview(delay = PREVIEW_DELAY_MS): void {
    clearTimeout(this.#previewTimer);
    this.#previewTimer = setTimeout(() => void this.#fetchPreview(null), delay);
  }

  /** Pick one option of every choice in an entry's prompt, as a run would. */
  async roll(entryId: string): Promise<void> {
    await this.#fetchPreview(entryId);
  }

  clearRoll(): void {
    this.rolled = null;
  }

  dispose(): void {
    clearTimeout(this.#previewTimer);
    this.#previewRequest += 1;
  }

  // ─── Sets ─────────────────────────────────────────────────────────────

  addSet(): string {
    const set = newSet(uniqueName('new_set', this.doc.sets.map((item) => item.name)));
    this.edit((document) => { document.sets.push(set); });
    return set.id;
  }

  /** Rename a set; returns why the name was refused, or null. */
  renameSet(setId: string, name: string): string | null {
    const set = this.doc.sets.find((item) => item.id === setId);
    if (!set || set.name === name) return null;
    const problem = setNameProblem(name, this.doc.sets.filter((item) => item !== set).map((item) => item.name));
    if (problem) return problem;
    this.edit(() => { set.name = name; });
    return null;
  }

  duplicateSet(setId: string): void {
    const index = this.doc.sets.findIndex((item) => item.id === setId);
    if (index < 0) return;
    const set = this.doc.sets[index];
    const copy = copySet($state.snapshot(set), uniqueName(`${set.name}_copy`, this.doc.sets.map((item) => item.name)));
    this.edit((document) => { document.sets.splice(index + 1, 0, copy); });
  }

  deleteSet(setId: string): void {
    this.edit((document) => {
      document.sets = document.sets.filter((item) => item.id !== setId);
    });
    if (this.selectedId && !findEntry(this.doc, this.selectedId)) this.selectedId = null;
  }

  moveSet(setId: string, refId: string, position: DropPosition): void {
    const set = this.doc.sets.find((item) => item.id === setId);
    const ref = this.doc.sets.find((item) => item.id === refId) ?? null;
    if (!set || set === ref) return;
    this.edit((document) => moveItem(document.sets, set, document.sets, ref, position));
  }

  setAllActive(setId: string, active: boolean): void {
    const set = this.doc.sets.find((item) => item.id === setId);
    if (!set) return;
    this.edit(() => set.entries.forEach((entry) => { entry.active = active; }));
  }

  // ─── Entries ──────────────────────────────────────────────────────────

  /** Add an empty entry at the end of a set and start editing its prompt. */
  addEntry(setId: string): string | null {
    const set = this.doc.sets.find((item) => item.id === setId);
    if (!set) return null;
    const entry = newEntry();
    this.edit(() => { set.entries.push(entry); });
    this.selectedId = entry.id;
    // Start typing the new prompt; adding it was the undo step.
    this.editingKey = `${entry.id}:prompt`;
    return entry.id;
  }

  duplicateEntry(entryId: string): void {
    const found = findEntry(this.doc, entryId);
    if (!found) return;
    const copy = copyEntry($state.snapshot(found.entry));
    this.edit(() => { found.set.entries.splice(found.index + 1, 0, copy); });
    this.selectedId = copy.id;
  }

  deleteEntry(entryId: string): void {
    const found = findEntry(this.doc, entryId);
    if (!found) return;
    this.edit(() => { found.set.entries.splice(found.index, 1); });
    if (this.selectedId === entryId) this.selectedId = null;
  }

  /** Move an entry before/after another entry, or to the end of a set when `refId` is null. */
  moveEntry(entryId: string, toSetId: string, refId: string | null, position: DropPosition): void {
    const found = findEntry(this.doc, entryId);
    const target = this.doc.sets.find((item) => item.id === toSetId);
    if (!found || !target || entryId === refId) return;
    const ref = refId === null ? null : (target.entries.find((item) => item.id === refId) ?? null);
    this.edit(() => moveItem(found.set.entries, found.entry, target.entries, ref, position));
    this.selectedId = entryId;
  }

  toggleActive(entryId: string): void {
    const entry = findEntry(this.doc, entryId)?.entry;
    if (entry) this.edit(() => { entry.active = !entry.active; });
  }

  setEnhance(entryId: string, enhance: EntryEnhance): void {
    const entry = findEntry(this.doc, entryId)?.entry;
    if (entry) this.edit(() => { entry.enhance = enhance; });
  }

  /** Show or remove an entry's negative prompt. */
  setNegative(entryId: string, shown: boolean): void {
    const entry = findEntry(this.doc, entryId)?.entry;
    if (!entry) return;
    this.edit(() => { entry.negative = shown ? { kind: 'text', text: '' } : null; });
  }

  /** Switch a value between text and fields (structured values become text). */
  setValueMode(entryId: string, key: EntryValueKey, mode: 'text' | 'fields'): void {
    const entry = findEntry(this.doc, entryId)?.entry;
    const value = entry?.[key];
    if (!entry || !value || value.kind === mode) return;
    const text = valueAsText(value);
    const next: PromptValue = mode === 'text' ? { kind: 'text', text } : { kind: 'fields', fields: textAsFields(text) };
    this.edit(() => { entry[key] = next; });
  }

  addField(entryId: string, key: EntryValueKey): void {
    const value = findEntry(this.doc, entryId)?.entry[key];
    if (value?.kind !== 'fields') return;
    this.edit(() => { value.fields.push({ key: '', value: '' }); });
  }

  removeField(entryId: string, key: EntryValueKey, index: number): void {
    const value = findEntry(this.doc, entryId)?.entry[key];
    if (value?.kind !== 'fields') return;
    this.edit(() => { value.fields.splice(index, 1); });
  }

  renameField(entryId: string, key: EntryValueKey, index: number, name: string): void {
    const value = findEntry(this.doc, entryId)?.entry[key];
    if (value?.kind !== 'fields' || value.fields[index]?.key === name) return;
    this.edit(() => { value.fields[index].key = name; });
  }

  // ─── Snippets ─────────────────────────────────────────────────────────

  addSnippet(): string {
    const snippet = newSnippet(uniqueName('snippet', this.doc.snippets.map((item) => item.name)));
    this.edit((document) => { document.snippets.push(snippet); });
    return snippet.id;
  }

  /** Rename a snippet and every `$reference` to it; returns why it was refused, or null. */
  renameSnippet(snippetId: string, name: string): string | null {
    const snippet = this.doc.snippets.find((item) => item.id === snippetId);
    if (!snippet || snippet.name === name) return null;
    const problem = snippetNameProblem(name, this.doc.snippets.filter((item) => item !== snippet).map((item) => item.name));
    if (problem) return problem;
    if (structuredUses(this.doc, snippet.name)) return `$${snippet.name} is used in a value the builder can't edit, so its references can't be renamed here.`;
    const from = snippet.name;
    this.edit((document) => {
      snippet.name = name;
      for (const item of document.snippets) item.value = renameRefsInValue(item.value, from, name);
      for (const set of document.sets) {
        for (const entry of set.entries) {
          entry.prompt = renameRefsInValue(entry.prompt, from, name);
          if (entry.negative) entry.negative = renameRefsInValue(entry.negative, from, name);
        }
      }
    });
    return null;
  }

  deleteSnippet(snippetId: string): void {
    this.edit((document) => {
      document.snippets = document.snippets.filter((item) => item.id !== snippetId);
    });
  }

  moveSnippet(snippetId: string, refId: string, position: DropPosition): void {
    const snippet = this.doc.snippets.find((item) => item.id === snippetId);
    const ref = this.doc.snippets.find((item) => item.id === refId) ?? null;
    if (!snippet || snippet === ref) return;
    this.edit((document) => moveItem(document.snippets, snippet, document.snippets, ref, position));
  }

  /** Add a `$name` reference to the end of an entry's prompt (a new, unnamed field in fields mode). */
  insertSnippetRef(entryId: string, name: string): void {
    const entry = findEntry(this.doc, entryId)?.entry;
    if (!entry) return;
    const prompt = entry.prompt;
    this.edit(() => {
      if (prompt.kind === 'text') prompt.text = `${prompt.text.trimEnd()} $${name}`.trim();
      else if (prompt.kind === 'fields') prompt.fields.push({ key: '', value: `$${name}` });
    });
    this.selectedId = entryId;
  }

  // ─── Internals ────────────────────────────────────────────────────────

  #changed(): void {
    this.dirty = true;
    this.refreshPreview();
    this.#onchange();
  }

  async #fetchPreview(rollEntryId: string | null): Promise<void> {
    const request = ++this.#previewRequest;
    const document = $state.snapshot(this.doc) as PromptDocument;
    try {
      const preview = await this.#previewFn(document, rollEntryId);
      if (request !== this.#previewRequest) return;
      this.preview = preview;
      if (preview.rolled) {
        this.rolled = { entryId: preview.rolled.entry_id, base: preview.entries[preview.rolled.entry_id]?.prompt ?? '', prompt: preview.rolled.prompt };
      }
    } catch {
      // A failed preview keeps the last one; the next change asks again.
    }
  }
}

/** An editing key (`<id>:prompt`, `<id>:negative:<n>`, `snippet:<id>`) with its item id remapped. */
function remapEditingKey(key: string, remap: (id: string) => string): string {
  if (key.startsWith('snippet:')) return `snippet:${remap(key.slice('snippet:'.length))}`;
  const colon = key.indexOf(':');
  return colon < 0 ? remap(key) : remap(key.slice(0, colon)) + key.slice(colon);
}

/** The entry being previewed, with its set and position. */
export function selectedEntry(builder: PromptBuilder): { entry: DocumentEntry; setName: string; index: number } | null {
  if (!builder.selectedId) return null;
  const found = findEntry(builder.doc, builder.selectedId);
  return found ? { entry: found.entry, setName: found.set.name, index: found.index } : null;
}
