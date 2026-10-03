import type { PageId } from '$lib/types';

type KeyInput = Pick<KeyboardEvent, 'key' | 'metaKey' | 'ctrlKey' | 'altKey'>;

// Inputs that take no typed text, so page shortcuts still apply while they have focus.
const NON_TEXT_INPUTS = new Set(['checkbox', 'radio', 'button', 'submit', 'reset']);

/** Return whether a key event comes from a field the user is typing in. */
export function isTyping(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  if (target instanceof HTMLInputElement) return !NON_TEXT_INPUTS.has(target.type);
  return target.isContentEditable || ['TEXTAREA', 'SELECT'].includes(target.tagName);
}

/** Return whether a key press carries no ⌘, Ctrl or Alt modifier; Shift is allowed (it makes `?`). */
export function isPlainKey(event: KeyInput): boolean {
  return !event.metaKey && !event.ctrlKey && !event.altKey;
}

/** Return whether the platform's command modifier (⌘ on macOS, Ctrl elsewhere) is held. */
export function isCommandKey(event: KeyInput): boolean {
  return event.metaKey || event.ctrlKey;
}

/** Return whether a modal dialog (asset viewer, Modal) currently owns the keyboard. */
export function hasOpenModal(root: ParentNode = document): boolean {
  return root.querySelector('[aria-modal="true"]') !== null;
}

/** Return whether a page-level single-key shortcut may run for this event. */
export function acceptsPageShortcut(event: KeyboardEvent): boolean {
  return !event.defaultPrevented && isPlainKey(event) && !isTyping(event.target) && !hasOpenModal();
}

/** Pages reached by pressing `G` then the page's key. */
export const GO_TO_KEYS: Readonly<Record<string, PageId>> = {
  w: 'workspace',
  g: 'gallery',
  m: 'models',
  c: 'config',
};

/** How long `G` waits for the second key, in milliseconds. */
export const CHORD_TIMEOUT_MS = 1500;

export interface GoChordStep {
  /** When `G` started waiting for its second key, or null when nothing is pending. */
  waitingSince: number | null;
  /** The page the completed chord opens. */
  page: PageId | null;
}

/** Advance the `G` → page chord by one key press made at `now`. */
export function stepGoChord(waitingSince: number | null, key: string, now: number): GoChordStep {
  const lower = key.toLowerCase();
  if (waitingSince !== null && now - waitingSince <= CHORD_TIMEOUT_MS) {
    return { waitingSince: null, page: GO_TO_KEYS[lower] ?? null };
  }
  return { waitingSince: lower === 'g' ? now : null, page: null };
}

export interface ShortcutEntry {
  /** Alternatives; each is one key or combination, e.g. `['Del', '⌫']` or `['⌘/Ctrl+E']`. */
  keys: readonly string[];
  label: string;
}

export interface ShortcutGroup {
  title: string;
  entries: readonly ShortcutEntry[];
}

/** Every shortcut outside the asset viewer, grouped for the `?` list. */
export const APP_SHORTCUTS: readonly ShortcutGroup[] = [
  {
    title: 'Anywhere',
    entries: [
      { keys: ['?'], label: 'Show keyboard shortcuts' },
      { keys: ['G → W'], label: 'Go to Workspace' },
      { keys: ['G → G'], label: 'Go to Gallery' },
      { keys: ['G → M'], label: 'Go to Models' },
      { keys: ['G → C'], label: 'Go to Config' },
    ],
  },
  {
    title: 'Workspace',
    entries: [
      { keys: ['⌘/Ctrl+↵'], label: 'Generate' },
      { keys: ['⌘/Ctrl+⇧+↵'], label: 'Generate with a new seed' },
      { keys: ['⌘/Ctrl+E'], label: 'Enhance the prompt' },
      { keys: ['Alt+1'], label: 'Focus the prompt' },
      { keys: ['Alt+2'], label: 'Focus the settings' },
      { keys: ['Esc'], label: 'Stop the running job' },
    ],
  },
  {
    title: 'Gallery',
    entries: [
      { keys: ['←', '↑', '↓', '→'], label: 'Move between assets' },
      { keys: ['↵'], label: 'Open the viewer' },
      { keys: ['Space', 'X'], label: 'Select or deselect' },
      { keys: ['⌘/Ctrl+A'], label: 'Select all loaded assets' },
      { keys: ['Del', '⌫'], label: 'Delete the selection or focused asset' },
      { keys: ['Esc'], label: 'Clear the selection' },
    ],
  },
];
