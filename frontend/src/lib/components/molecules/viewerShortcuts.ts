import { isPlainKey, type ShortcutEntry } from '$lib/keyboard';

export type ViewerAction =
  | 'close' | 'prev' | 'next' | 'first' | 'last' | 'details' | 'reuse' | 'reference'
  | 'download' | 'copy' | 'delete' | 'delete-now' | 'help';

/** Shortcuts shown in the viewer's help panel, in display order. */
export const VIEWER_SHORTCUTS: readonly ShortcutEntry[] = [
  { keys: ['←', '→'], label: 'Previous / next asset' },
  { keys: ['Home', 'End'], label: 'First / last asset' },
  { keys: ['R'], label: 'Reuse settings' },
  { keys: ['E'], label: 'Use as reference' },
  { keys: ['D'], label: 'Download' },
  { keys: ['C'], label: 'Copy the prompt' },
  { keys: ['Del', '⌫'], label: 'Delete' },
  { keys: ['⇧+Del'], label: 'Delete without asking' },
  { keys: ['I'], label: 'Toggle details' },
  { keys: ['?'], label: 'Show shortcuts' },
  { keys: ['Esc'], label: 'Close' },
];

const KEY_ACTIONS: Record<string, ViewerAction> = {
  Escape: 'close',
  ArrowLeft: 'prev',
  ArrowRight: 'next',
  Home: 'first',
  End: 'last',
  i: 'details',
  r: 'reuse',
  e: 'reference',
  d: 'download',
  c: 'copy',
  // Mac keyboards send Backspace from the key labelled "delete".
  Delete: 'delete',
  Backspace: 'delete',
  '?': 'help',
};

/** Map a key press to a viewer action, or null when the viewer should ignore it. */
export function viewerActionFor(event: Pick<KeyboardEvent, 'key' | 'shiftKey' | 'metaKey' | 'ctrlKey' | 'altKey'>): ViewerAction | null {
  if (!isPlainKey(event)) return null;
  const key = event.key.length === 1 ? event.key.toLowerCase() : event.key;
  const action = KEY_ACTIONS[key] ?? null;
  return action === 'delete' && event.shiftKey ? 'delete-now' : action;
}
