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
  p: 'prompts',
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

export interface ChordStep<T> {
  /** When the start key began waiting for the second key, or null when nothing is pending. */
  waitingSince: number | null;
  /** The value the completed chord selects. */
  value: T | null;
  /** Whether this key press belongs to the chord (started or completed it), so it must not run another action. */
  consumed: boolean;
}

/** Advance a two-key chord (`startKey` then one of `values`' keys, within CHORD_TIMEOUT_MS) by one key press at `now`. */
export function stepChord<T>(startKey: string, values: Readonly<Record<string, T>>, waitingSince: number | null, key: string, now: number): ChordStep<T> {
  const lower = key.toLowerCase();
  if (waitingSince !== null && now - waitingSince <= CHORD_TIMEOUT_MS) {
    const value = values[lower] ?? null;
    return { waitingSince: null, value, consumed: value !== null };
  }
  const starts = lower === startKey;
  return { waitingSince: starts ? now : null, value: null, consumed: starts };
}

/** Advance the `G` → page chord by one key press made at `now`. */
export function stepGoChord(waitingSince: number | null, key: string, now: number): GoChordStep {
  const step = stepChord('g', GO_TO_KEYS, waitingSince, key, now);
  return { waitingSince: step.waitingSince, page: step.value };
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
      { keys: ['G → P'], label: 'Go to Prompts' },
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
      { keys: ['Alt+3'], label: 'Focus the history' },
      { keys: ['Esc'], label: 'Stop the running job' },
      { keys: ['P'], label: 'Pause or resume the running job' },
      { keys: ['N'], label: 'Skip to the next image' },
      { keys: ['R'], label: 'Repeat the current image' },
    ],
  },
  {
    title: 'Prompts',
    entries: [
      { keys: ['⌘/Ctrl+S'], label: 'Save the prompt file' },
      { keys: ['⌘/Ctrl+Z'], label: 'Undo (outside text fields)' },
      { keys: ['↵'], label: 'Edit the focused prompt text or choice' },
      { keys: ['$'], label: 'Insert a snippet (while editing)' },
      { keys: ['⌥/Alt+↵'], label: 'Edit or start a random choice at the caret (while editing)' },
      { keys: ['Esc', '⌘/Ctrl+↵'], label: 'Finish editing' },
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
