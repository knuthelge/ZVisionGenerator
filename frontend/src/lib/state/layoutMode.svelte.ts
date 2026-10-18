/** Which composition of the Workspace fits the screen, and whether the pointer is a finger. */
export type LayoutMode = 'compact' | 'tall' | 'wide';

/** Phones, and any screen too short for a sidebar (a phone held sideways). */
export const COMPACT_QUERY = '(max-width: 639px), (max-height: 499px)';
/** Portrait tablets: the stage over Compose and Settings side by side. */
export const TALL_QUERY = '(min-width: 640px) and (max-width: 1023px)';
/** A finger instead of a mouse: taller targets and steppers in place of label scrubbing. */
export const COARSE_QUERY = '(pointer: coarse)';

/** The mode for the two width queries; compact wins over tall (a short wide window is compact). */
export function layoutModeFor(matches: { compact: boolean; tall: boolean }): LayoutMode {
  if (matches.compact) return 'compact';
  if (matches.tall) return 'tall';
  return 'wide';
}

let _compact = $state(false);
let _tall = $state(false);
let _coarse = $state(false);

function watch(query: string, apply: (matches: boolean) => void): void {
  if (typeof matchMedia !== 'function') return;
  const list = matchMedia(query);
  apply(list.matches);
  list.addEventListener('change', (event) => apply(event.matches));
}

watch(COMPACT_QUERY, (matches) => { _compact = matches; });
watch(TALL_QUERY, (matches) => { _tall = matches; });
watch(COARSE_QUERY, (matches) => { _coarse = matches; });

export const layout = {
  get mode(): LayoutMode { return layoutModeFor({ compact: _compact, tall: _tall }); },
  get compact(): boolean { return this.mode === 'compact'; },
  /** The pointer is coarse (touch): controls grow and numbers get steppers, at every width. */
  get coarse(): boolean { return _coarse; },
};
