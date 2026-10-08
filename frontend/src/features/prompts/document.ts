import type { DocumentEntry, DocumentSet, DocumentSnippet, EnhanceAxis, EnhanceSettings, EntryEnhance, PromptDocument, PromptField, PromptValue } from '$lib/types';

/** Matches a `$name` snippet reference or an innermost `{a|b|c}` choice (same patterns as `utils/prompt_compose.py`). */
const TOKEN_RE = /\$([A-Za-z_][A-Za-z0-9_]*)|\{([^{}]+)\}/g;
const REF_RE = /\$([A-Za-z_][A-Za-z0-9_]*)/g;
export const SNIPPET_NAME_RE = /^[A-Za-z_][A-Za-z0-9_]*$/;
export const SNIPPETS_KEY = 'snippets';
/** Joins flattened parts, like `SEPARATOR` in `utils/prompt_compose.py`. */
const SEPARATOR = '. ';

export type TextPart = { kind: 'text'; text: string } | { kind: 'snippet'; name: string };
export type PromptToken = TextPart | { kind: 'choice'; start: number; end: number; options: TextPart[][] };

/** Split prompt text into plain text, `$snippet` references, and `{a|b}` choices (with their position in the text). */
export function tokenizePrompt(text: string): PromptToken[] {
  const tokens: PromptToken[] = [];
  let last = 0;
  for (const match of text.matchAll(TOKEN_RE)) {
    const index = match.index ?? 0;
    if (index > last) tokens.push({ kind: 'text', text: text.slice(last, index) });
    if (match[1] !== undefined) {
      tokens.push({ kind: 'snippet', name: match[1] });
    } else {
      tokens.push({ kind: 'choice', start: index, end: index + match[0].length, options: match[2].split('|').map(splitRefs) });
    }
    last = index + match[0].length;
  }
  if (last < text.length) tokens.push({ kind: 'text', text: text.slice(last) });
  return tokens;
}

function splitRefs(text: string): TextPart[] {
  const parts: TextPart[] = [];
  let last = 0;
  for (const match of text.matchAll(REF_RE)) {
    const index = match.index ?? 0;
    if (index > last) parts.push({ kind: 'text', text: text.slice(last, index) });
    parts.push({ kind: 'snippet', name: match[1] });
    last = index + match[0].length;
  }
  if (last < text.length) parts.push({ kind: 'text', text: text.slice(last) });
  return parts;
}

/** The options of the `{...}` choice at `start..end` in `text`. */
export function choiceOptions(text: string, start: number, end: number): string[] {
  return text.slice(start + 1, end - 1).split('|');
}

/** Replace the choice at `start..end`: several options become `{a|b}`, one option becomes plain text. */
export function replaceChoice(text: string, start: number, end: number, options: string[]): string {
  const replacement = options.length > 1 ? `{${options.join('|')}}` : (options[0] ?? '');
  return text.slice(0, start) + replacement + text.slice(end);
}

/** The innermost `{…}` choice whose braces surround `caret`, or null. */
export function choiceAt(text: string, caret: number): { start: number; end: number } | null {
  for (const match of text.matchAll(/\{[^{}]+\}/g)) {
    const start = match.index ?? 0;
    const end = start + match[0].length;
    if (caret > start && caret < end) return { start, end };
  }
  return null;
}

/** Whether selected text can become a choice: not blank, and no braces of its own. */
export function canMakeChoice(selected: string): boolean {
  return selected.trim() !== '' && !/[{}]/.test(selected);
}

/** The partial `$name` being typed just before `caret`, or null. */
export function snippetQueryAt(text: string, caret: number): string | null {
  const match = text.slice(0, caret).match(/\$([A-Za-z0-9_]*)$/);
  return match ? match[1] : null;
}

/** Flatten a prompt value the way `flatten_value` in `utils/prompt_compose.py` does (without resolving snippets). */
export function flattenValue(value: unknown): string {
  if (value === null || value === undefined) return '';
  if (typeof value === 'string') return value.trim();
  if (typeof value === 'boolean') return String(value);
  if (typeof value === 'number') return String(value);
  if (Array.isArray(value)) return value.map(flattenValue).filter(Boolean).join(SEPARATOR);
  if (typeof value === 'object') {
    return Object.entries(value as Record<string, unknown>)
      .map(([key, item]) => {
        const flat = flattenValue(item);
        return flat ? `${key}: ${flat}` : key;
      })
      .join(SEPARATOR);
  }
  return String(value);
}

/** The value as the prompt loader sees it. */
export function valueData(value: PromptValue): unknown {
  if (value.kind === 'text') return value.text;
  if (value.kind === 'fields') return Object.fromEntries(value.fields.map((field) => [field.key, field.value]));
  return value.data;
}

/** Text that produces the same prompt as `value` (structure flattened). */
export function valueAsText(value: PromptValue): string {
  return value.kind === 'text' ? value.text : flattenValue(valueData(value));
}

/** Fields for a text value: one unnamed field holding the text, for the user to name. */
export function textAsFields(text: string): PromptField[] {
  return [{ key: '', value: text.trim() }];
}

/** `base`, or `base_2`, `base_3`, … — the first that is not taken. */
export function uniqueName(base: string, taken: Iterable<string>): string {
  const used = new Set(taken);
  if (!used.has(base)) return base;
  let counter = 2;
  while (used.has(`${base}_${counter}`)) counter += 1;
  return `${base}_${counter}`;
}

/** Why `name` can't name a set, or null. */
export function setNameProblem(name: string, otherNames: string[]): string | null {
  if (!name.trim()) return 'A set needs a name.';
  if (name === SNIPPETS_KEY) return `"${SNIPPETS_KEY}" is reserved for snippets.`;
  if (otherNames.includes(name)) return `There is already a set called ${name}.`;
  return null;
}

/** Why `name` can't name a snippet, or null. */
export function snippetNameProblem(name: string, otherNames: string[]): string | null {
  if (!SNIPPET_NAME_RE.test(name)) return 'Snippet names start with a letter or _ and use only letters, digits and _.';
  if (otherNames.includes(name)) return `There is already a snippet called $${name}.`;
  return null;
}

/** Rename every `$from` reference to `$to` in a text. */
export function renameRefs(text: string, from: string, to: string): string {
  return text.replace(new RegExp(`\\$${from}(?![A-Za-z0-9_])`, 'g'), `$${to}`);
}

/** Rename references in an editable value; structured values are returned unchanged. */
export function renameRefsInValue(value: PromptValue, from: string, to: string): PromptValue {
  if (value.kind === 'text') return { kind: 'text', text: renameRefs(value.text, from, to) };
  if (value.kind === 'fields') return { kind: 'fields', fields: value.fields.map((field) => ({ key: field.key, value: renameRefs(field.value, from, to) })) };
  return value;
}

/** Whether a structured (read-only) value anywhere in the document references `$name`. */
export function structuredUses(document: PromptDocument, name: string): boolean {
  const pattern = new RegExp(`\\$${name}(?![A-Za-z0-9_])`);
  const values: (PromptValue | null)[] = [
    ...document.snippets.map((snippet) => snippet.value),
    ...document.sets.flatMap((set) => set.entries.flatMap((entry) => [entry.prompt, entry.negative])),
  ];
  return values.some((value) => value?.kind === 'structured' && pattern.test(JSON.stringify(value.data)));
}

/** Move `item` out of `from` and into `to`, before or after `ref` (or at the end when `ref` is null). */
export function moveItem<T>(from: T[], item: T, to: T[], ref: T | null, position: 'before' | 'after'): void {
  const fromIndex = from.indexOf(item);
  if (fromIndex < 0) return;
  from.splice(fromIndex, 1);
  const refIndex = ref === null ? -1 : to.indexOf(ref);
  const index = refIndex < 0 ? to.length : refIndex + (position === 'after' ? 1 : 0);
  to.splice(index, 0, item);
}

/** Drop before or after an element, by which half of it the pointer is over. */
export function dropPosition(rect: { top: number; height: number }, clientY: number): 'before' | 'after' {
  return clientY < rect.top + rect.height / 2 ? 'before' : 'after';
}

let idCounter = 0;
/** A fresh id for an item that is not in the file yet. */
export function newId(): string {
  idCounter += 1;
  return `new-${Date.now().toString(36)}-${idCounter}`;
}

export function newEntry(): DocumentEntry {
  return { id: newId(), prompt: { kind: 'text', text: '' }, negative: null, active: true, enhance: null };
}

export function newSet(name: string): DocumentSet {
  return { id: newId(), name, entries: [] };
}

export function newSnippet(name: string): DocumentSnippet {
  return { id: newId(), name, value: { kind: 'text', text: '' } };
}

/** A deep copy with fresh ids, for Duplicate. */
export function copyEntry(entry: DocumentEntry): DocumentEntry {
  return { ...structuredClone(entry), id: newId() };
}

export function copySet(set: DocumentSet, name: string): DocumentSet {
  return { id: newId(), name, entries: set.entries.map(copyEntry) };
}

/** The prompt id (`set:index`) a run uses for an entry; inactive entries have none. */
export function optionId(document: PromptDocument, entryId: string): string | null {
  for (const set of document.sets) {
    const index = set.entries.findIndex((entry) => entry.id === entryId);
    if (index >= 0) return set.entries[index].active ? `${set.name}:${index}` : null;
  }
  return null;
}

/** Find an entry with its set and position. */
export function findEntry(document: PromptDocument, entryId: string): { set: DocumentSet; entry: DocumentEntry; index: number } | null {
  for (const set of document.sets) {
    const index = set.entries.findIndex((entry) => entry.id === entryId);
    if (index >= 0) return { set, entry: set.entries[index], index };
  }
  return null;
}

export function findSnippet(document: PromptDocument, snippetId: string): DocumentSnippet | null {
  return document.snippets.find((snippet) => snippet.id === snippetId) ?? null;
}

/** Keep the selected prompt ids that still exist after a save, under their new ids. */
export function remapOptionIds(ids: string[], map: Record<string, string>): string[] {
  return ids.filter((id) => id in map).map((id) => map[id]);
}

/** Full settings for the enhance picker: an entry's mapping over the defaults, ignoring unknown values. */
export function enhanceSettingsFor(enhance: EntryEnhance, axes: EnhanceAxis[], defaults: EnhanceSettings): EnhanceSettings {
  const settings: EnhanceSettings = { ...defaults, details: [...defaults.details], motion: [...defaults.motion] };
  if (enhance === null || enhance === true) return settings;
  for (const axis of axes) {
    const raw = enhance[axis.key];
    const slugs = axis.options.map((option) => option.slug);
    if (axis.multi && Array.isArray(raw)) (settings[axis.key] as string[]) = raw.filter((slug) => slugs.includes(slug));
    else if (!axis.multi && typeof raw === 'string' && slugs.includes(raw)) (settings[axis.key] as string) = raw;
  }
  return settings;
}

/** Whether `settings` ask for no change at all (the server refuses that). */
export function isNoOpEnhance(settings: EnhanceSettings): boolean {
  return settings.style === 'keep' && settings.mood === 'keep' && settings.details.length === 0 && settings.length === 'same' && settings.motion.length === 0;
}

/** Every chosen enhance option in words, e.g. "Cinematic · Mysterious · +2 details". */
export function enhanceSummary(enhance: EntryEnhance, axes: EnhanceAxis[]): string {
  if (enhance === null) return '';
  if (enhance === true) return 'Default options';
  return enhanceParts(enhance, axes).join(' · ') || 'Custom options';
}

/** A label short enough for a prompt row: the first two choices, then how many more, e.g. "Cinematic · Epic +2". */
export function enhanceShortLabel(enhance: EntryEnhance, axes: EnhanceAxis[]): string {
  if (enhance === null || enhance === true) return enhanceSummary(enhance, axes);
  const parts = enhanceParts(enhance, axes);
  if (parts.length === 0) return 'Custom options';
  return parts.length > 2 ? `${parts[0]} · ${parts[1]} +${parts.length - 2}` : parts.join(' · ');
}

function enhanceParts(enhance: Exclude<EntryEnhance, null | true>, axes: EnhanceAxis[]): string[] {
  const label = (key: string, slug: unknown): string =>
    axes.find((axis) => axis.key === key)?.options.find((option) => option.slug === slug)?.label ?? String(slug);
  const parts: string[] = [];
  if (enhance.style && enhance.style !== 'keep') parts.push(label('style', enhance.style));
  if (enhance.mood && enhance.mood !== 'keep') parts.push(label('mood', enhance.mood));
  if (enhance.length && enhance.length !== 'same') parts.push(label('length', enhance.length));
  const details = Array.isArray(enhance.details) ? enhance.details.length : 0;
  if (details) parts.push(`+${details} detail${details > 1 ? 's' : ''}`);
  const motion = Array.isArray(enhance.motion) ? enhance.motion.length : 0;
  if (motion) parts.push(`${motion} motion`);
  return parts;
}
