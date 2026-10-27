import type { DraftState, EnhancedRewrite, EnhancedSource } from '$lib/types';
import { isEnhancedStale } from './promptEnhance';

/** How many rewrites of one text the draft keeps. */
const REWRITE_HISTORY_LIMIT = 5;

/** Whether two rewrites were made from the same text, in the same mode and, for an extension, for the same source and length; the record is ignored. */
function sameSource(a: EnhancedSource, b: EnhancedSource): boolean {
  return (
    a.prompt === b.prompt &&
    a.scene === b.scene &&
    a.mode === b.mode &&
    (a.sourceVideoPath ?? null) === (b.sourceVideoPath ?? null) &&
    (a.extendFrames ?? null) === (b.extendFrames ?? null)
  );
}

/** Add *rewrite* as the newest entry, dropping entries made from another source text and keeping the last `REWRITE_HISTORY_LIMIT`. */
export function pushRewrite(history: readonly EnhancedRewrite[], rewrite: EnhancedRewrite): EnhancedRewrite[] {
  return [...history.filter((entry) => sameSource(entry.from, rewrite.from)), rewrite].slice(-REWRITE_HISTORY_LIMIT);
}

/**
 * Where the shown Enhanced pair sits in the history, as a 1-based `index` of `count` entries.
 *
 * Null when there are fewer than two entries, auto-enhance is on, the Enhanced text is out of date, or the
 * shown pair is not one of the entries (it was edited, cleared or came from somewhere else). With identical
 * entries the last match counts.
 */
export function rewriteStepper(state: DraftState): { index: number; count: number } | null {
  const history = state.enhancedHistory;
  if (history.length < 2 || state.enhanceAuto || isEnhancedStale(state)) return null;
  const position = history.map((entry) => entry.prompt === state.enhancedPrompt && entry.scene === state.enhancedScene).lastIndexOf(true);
  return position < 0 ? null : { index: position + 1, count: history.length };
}
