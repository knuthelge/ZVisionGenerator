import { isPlainKey, stepChord, type ChordStep, type ShortcutEntry } from '$lib/keyboard';
import type { UpscaleFactor } from '$lib/types';

export type ViewerAction =
  | 'close' | 'prev' | 'next' | 'first' | 'last' | 'details' | 'reuse' | 'reference'
  | 'download' | 'copy' | 'delete' | 'delete-now' | 'help';

type KeyInput = Pick<KeyboardEvent, 'key' | 'shiftKey' | 'metaKey' | 'ctrlKey' | 'altKey'>;

/** Shortcuts shown in the viewer's help panel, in display order. */
export const VIEWER_SHORTCUTS: readonly ShortcutEntry[] = [
  { keys: ['←', '→'], label: 'Previous / next asset' },
  { keys: ['Home', 'End'], label: 'First / last asset' },
  { keys: ['R'], label: 'Reuse settings' },
  { keys: ['E'], label: 'Use as reference' },
  { keys: ['X → 2'], label: 'Upscale 2×' },
  { keys: ['X → 4'], label: 'Upscale 4×' },
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

const UPSCALE_KEYS: Readonly<Record<string, UpscaleFactor>> = { '2': 2, '4': 4 };

/** Map a key press to a viewer action, or null when the viewer should ignore it. */
export function viewerActionFor(event: KeyInput): ViewerAction | null {
  if (!isPlainKey(event)) return null;
  const key = event.key.length === 1 ? event.key.toLowerCase() : event.key;
  const action = KEY_ACTIONS[key] ?? null;
  return action === 'delete' && event.shiftKey ? 'delete-now' : action;
}

/** Advance the `X` → `2`/`4` upscale chord by one key press made at `now`; Shift+X and modified keys never start it. */
export function stepUpscaleChord(waitingSince: number | null, event: KeyInput, now: number): ChordStep<UpscaleFactor> {
  if (!isPlainKey(event) || (waitingSince === null && event.shiftKey)) return { waitingSince: null, value: null, consumed: false };
  return stepChord('x', UPSCALE_KEYS, waitingSince, event.key, now);
}
