import type { DraftState, LyricsVersion } from '$lib/types';

/**
 * The versions of a song's lyrics the helper made. `lyrics` always holds the shown version (what Generate sends); `lyricsVersions` keeps
 * the whole text of every version and `lyricsCurrent` says which one is shown. The user's own lyrics are saved as `mine` when the first
 * result arrives, and stay first; typing into a version makes it `mine` too.
 *
 * Anything else may set `lyrics` (a Reuse, a recent prompt, the Edit as text box on its own): the versions then no longer describe it,
 * so they are ignored until a result starts a new set from what is shown.
 */

/** How many versions are kept, the user's own included. The oldest result goes first. */
export const LYRICS_VERSION_LIMIT = 5;

type Versioned = Pick<DraftState, 'lyrics' | 'lyricsVersions' | 'lyricsCurrent'>;

/** The fields a change of versions sets in the draft. */
export type VersionPatch = Pick<DraftState, 'lyricsVersions' | 'lyricsCurrent'>;

/** Whether the versions describe the lyrics shown: the current version is the text in the box. */
export function inStep(state: Versioned): boolean {
  return state.lyricsVersions[state.lyricsCurrent]?.text === state.lyrics;
}

/**
 * The versions after a result: the shown lyrics are kept as `mine` first when the set is new, the result follows as the newest and is
 * shown. Past `LYRICS_VERSION_LIMIT`, the oldest result is dropped; the user's own stay.
 */
export function pushVersion(state: Versioned, text: string): VersionPatch {
  const kept = inStep(state) ? state.lyricsVersions : state.lyrics.trim() === '' ? [] : [{ text: state.lyrics, mine: true }];
  const added = [...kept, { text, mine: false }];
  const pinned = added[0]?.mine ? 1 : 0;
  const versions = added.length > LYRICS_VERSION_LIMIT ? [...added.slice(0, pinned), ...added.slice(pinned + added.length - LYRICS_VERSION_LIMIT)] : added;
  return { lyricsVersions: versions, lyricsCurrent: versions.length - 1 };
}

/** Where the shown version sits among the versions, as a 1-based `index` of `count`; null with fewer than two or when they are out of step. */
export function versionStepper(state: Versioned): { index: number; count: number } | null {
  if (state.lyricsVersions.length < 2 || !inStep(state)) return null;
  return { index: state.lyricsCurrent + 1, count: state.lyricsVersions.length };
}

/** The fields to set to show version *index* (1-based); an index outside the versions changes nothing. */
export function showVersion(state: Versioned, index: number): (VersionPatch & Pick<DraftState, 'lyrics'>) | null {
  const version: LyricsVersion | undefined = state.lyricsVersions[index - 1];
  return version === undefined ? null : { lyrics: version.text, lyricsVersions: state.lyricsVersions, lyricsCurrent: index - 1 };
}

/** Whether a helper wrote the version shown, which the Lyrics heading marks with a ✦: the versions are in step and the shown one is not the user's. */
export function writtenByHelper(state: Versioned): boolean {
  return inStep(state) && state.lyricsVersions[state.lyricsCurrent]?.mine === false;
}

/**
 * The fields to set when *text* is typed into the lyrics: the shown version follows what is typed, so stepping away and back finds the
 * edit, and it becomes the user's own (its ✦ goes). Versions out of step are left alone.
 */
export function editLyrics(state: Versioned, text: string): Pick<DraftState, 'lyrics'> & Partial<VersionPatch> {
  if (!inStep(state)) return { lyrics: text };
  return { lyrics: text, lyricsVersions: state.lyricsVersions.map((version, at) => (at === state.lyricsCurrent ? { text, mine: true } : version)) };
}
