import { isPlainKey } from '$lib/keyboard';

/** What a key press asks of the player. */
export type PlayerCommand =
  | { type: 'toggle' }
  /** Move *frames* frames (negative goes back) from the shown frame. */
  | { type: 'step'; frames: number }
  /** Move *seconds* seconds (negative goes back) from the shown position. */
  | { type: 'seek'; seconds: number }
  /** Change the volume by *delta* (0 to 1 is the whole range). */
  | { type: 'volume'; delta: number }
  | { type: 'start' }
  | { type: 'end' }
  /** Go to *fraction* of the clip, 0 to 1. */
  | { type: 'jump'; fraction: number };

/** Where in the player the key was pressed: a button, the volume slider, or anywhere else (the timeline). */
export type KeyTarget = 'button' | 'volume' | 'timeline';

/** The state a key's meaning depends on. */
export interface KeyContext {
  kind: 'video' | 'audio';
  paused: boolean;
  target: KeyTarget;
  /** Whether the player has a volume control; without one the arrow keys up and down are the page's. */
  volume: boolean;
}

type KeyInput = Pick<KeyboardEvent, 'key' | 'shiftKey' | 'metaKey' | 'ctrlKey' | 'altKey'>;

/** How far a volume key moves the volume. */
export const VOLUME_STEP = 0.1;
/** How far an arrow seeks a playing video, or a take. */
export const SEEK_SECONDS = 1;
/** The same with Shift: seconds, or frames while a video is paused. */
export const SEEK_BIG = 5;

/** Return the command a key press means inside the player, or `null` for a key the player leaves to the page. */
export function playerKeyCommand(event: KeyInput, context: KeyContext): PlayerCommand | null {
  if (!isPlainKey(event)) return null;
  switch (event.key) {
    case ' ':
      // A focused button activates itself.
      return context.target === 'button' ? null : { type: 'toggle' };
    case 'ArrowLeft':
    case 'ArrowRight': {
      const direction = event.key === 'ArrowLeft' ? -1 : 1;
      if (context.target === 'volume') return { type: 'volume', delta: direction * VOLUME_STEP };
      const amount = event.shiftKey ? SEEK_BIG : SEEK_SECONDS;
      if (context.kind === 'video' && context.paused) return { type: 'step', frames: direction * amount };
      return { type: 'seek', seconds: direction * amount };
    }
    case 'ArrowUp':
      return context.volume ? { type: 'volume', delta: VOLUME_STEP } : null;
    case 'ArrowDown':
      return context.volume ? { type: 'volume', delta: -VOLUME_STEP } : null;
    case 'Home':
      return { type: 'start' };
    case 'End':
      return { type: 'end' };
    default:
      // The digits 0 to 9 go to 0% to 90%, as on the common video sites.
      return /^[0-9]$/.test(event.key) ? { type: 'jump', fraction: Number(event.key) / 10 } : null;
  }
}

const NAVIGATION_KEYS = new Set(['Tab', 'ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Home', 'End', 'PageUp', 'PageDown']);

/** Return whether a key press moves through the page by keyboard (Tab, the arrows, Home, End, Page Up and Down), which is what brings focus rings in. Space and Enter act on a control; they are not navigation. */
export function navigatesByKey(event: KeyInput): boolean {
  return isPlainKey(event) && NAVIGATION_KEYS.has(event.key);
}
