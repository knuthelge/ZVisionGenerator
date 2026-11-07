/**
 * The versions of an Enhanced text, as pure rules shared by every box that keeps them: the Image and Video prompt pair and each Audio box.
 *
 * A store keeps its versions in its own shape; these functions only need to know what value an entry holds (a string for an audio box,
 * a `{ prompt, scene }` pair for a video) and when two values are the same text.
 */

/** How many versions of an Enhanced text are kept. */
export const VERSION_LIMIT = 5;

/** What the stepper shows: `index` null is "– of N" with ‹ bringing back version `back`. */
export interface StepperState {
  index: number | null;
  count: number;
  back?: number;
}

/** How to read a store's versions: the value an entry holds, and whether two values are the same text. */
export interface VersionReader<V, E> {
  valueOf: (entry: E) => V;
  same: (a: V, b: V) => boolean;
}

/**
 * Add *entry* as the newest version, keeping the earlier ones (whatever text they were made from) up to `VERSION_LIMIT`. An earlier version with
 * the same text as *entry* moves to the newest place instead of being copied, so each text is one version and the stepper never loops.
 */
export function pushVersion<V, E>(history: readonly E[], entry: E, reader: VersionReader<V, E>): E[] {
  const value = reader.valueOf(entry);
  return [...history.filter((earlier) => !reader.same(reader.valueOf(earlier), value)), entry].slice(-VERSION_LIMIT);
}

/** The position (from 0) of the version *shown* is; with identical versions (typing can make two the same) the last match counts. -1 when it is not one of them. */
export function shownPosition<V, E>(history: readonly E[], shown: V, reader: VersionReader<V, E>): number {
  return history.map((entry) => reader.same(reader.valueOf(entry), shown)).lastIndexOf(true);
}

/**
 * What the version stepper shows for *count* versions, or null for none (it is hidden then).
 *
 * With Enhanced text (*used*): the shown version's 1-based `index` of `count`, once there are two versions; none when the text is not one of them
 * (typed by hand, *position* -1). Without Enhanced text: "– of N", only after Scratch emptied it, with `back` the 1-based version that was scratched.
 */
export function versionStepper(count: number, used: boolean, position: number, scratched: number | null): StepperState | null {
  if (count === 0) return null;
  if (!used) return scratched !== null && scratched < count ? { index: null, count, back: scratched + 1 } : null;
  return count < 2 || position < 0 ? null : { index: position + 1, count };
}

/** The position Scratch remembers for the text shown, so ‹ brings it back; null when no Enhanced text is shown or it is not one of the versions. */
export function scratchedPosition<V, E>(history: readonly E[], shown: V, used: boolean, reader: VersionReader<V, E>): number | null {
  const position = used ? shownPosition(history, shown, reader) : -1;
  return position < 0 ? null : position;
}

/** *history* with the shown version (at *position*, or none when -1) changed by *edit*: what typing in a shown version does to its entry. */
export function editedHistory<E>(history: E[], position: number, edit: (entry: E) => E): E[] {
  return position < 0 ? history : history.map((entry, at) => (at === position ? edit(entry) : entry));
}

/** *source* without the record of how it was written; null stays null. */
export function withoutRecord<S extends { record?: unknown }>(source: S | null): S | null {
  if (source === null) return null;
  const { record: _record, ...rest } = source;
  return rest as S;
}
