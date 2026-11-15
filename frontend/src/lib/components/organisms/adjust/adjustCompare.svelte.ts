/** What holds Compare on: a toggle stays until the next tap or edit, a hold lasts while its source is pressed. */
export type CompareHold = 'button' | 'picture' | 'key';

/** Keys whose release ends a keyboard hold, so a modifier let go before the letter cannot leave the original showing. */
export const RELEASE_KEYS: ReadonlySet<string> = new Set(['Shift', 'Control', 'Alt', 'AltGraph', 'Meta']);

/** The state of Compare: whether the original shows, from a toggle or a hold. */
export interface CompareState {
  /** Whether the original is on the picture, by a toggle or a hold. */
  readonly showing: boolean;
  /** Whether a tap left Compare on. */
  readonly toggled: boolean;
  /** What holds it, or null. */
  readonly held: CompareHold | null;
  /** Flip the toggle (a tap, or Enter or Space on the button). */
  toggle(): void;
  /** Turn the toggle off: any edit does. */
  clearToggle(): void;
  /** Start a hold from a source. A hold already running keeps its source. */
  hold(source: CompareHold): void;
  /** End the hold, but only when `source` is the one holding. */
  release(source: CompareHold): void;
  /** End any hold (window blur, the page hidden). */
  releaseAll(): void;
}

/** Create the state of Compare, with its reactive fields for a component to read. */
export function createCompare(): CompareState {
  let toggled = $state(false);
  let held = $state<CompareHold | null>(null);
  return {
    get showing(): boolean {
      return toggled || held !== null;
    },
    get toggled(): boolean {
      return toggled;
    },
    get held(): CompareHold | null {
      return held;
    },
    toggle(): void {
      toggled = !toggled;
    },
    clearToggle(): void {
      toggled = false;
    },
    hold(source): void {
      held ??= source;
    },
    release(source): void {
      if (held === source) held = null;
    },
    releaseAll(): void {
      held = null;
    },
  };
}
