import type { DraftState, EnhancedRewrite, EnhancedSource } from '$lib/types';
import { editedHistory, pushVersion, scratchedPosition, shownPosition, versionStepper, withoutRecord, type StepperState, type VersionReader } from './versions';

/** Image and Video versions hold a prompt and a scene: two are the same text when both parts are equal. */
const READER: VersionReader<{ prompt: string; scene: string }, EnhancedRewrite> = {
  valueOf: ({ prompt, scene }) => ({ prompt, scene }),
  same: (a, b) => a.prompt === b.prompt && a.scene === b.scene,
};

/** The pair the draft's Enhanced text shows. */
function shownPair(state: Pick<EnhancedState, 'enhancedPrompt' | 'enhancedScene'>): { prompt: string; scene: string } {
  return { prompt: state.enhancedPrompt, scene: state.enhancedScene };
}

/** The draft fields the Enhanced text and its versions live in. */
type EnhancedState = Pick<DraftState, 'enhancedPrompt' | 'enhancedScene' | 'enhancedFrom' | 'enhancedHistory' | 'enhancedScratched'>;

/** The fields to set to show *entry*, or nothing shown: the versions stay, and none is scratched. */
function showing(prompt: string, scene: string, from: EnhancedSource | null): Pick<EnhancedState, 'enhancedPrompt' | 'enhancedScene' | 'enhancedFrom' | 'enhancedScratched'> {
  return { enhancedPrompt: prompt, enhancedScene: scene, enhancedFrom: from, enhancedScratched: null };
}

/** Whether the draft has Enhanced text in either part. */
function hasEnhanced(state: Pick<EnhancedState, 'enhancedPrompt' | 'enhancedScene'>): boolean {
  return state.enhancedPrompt.trim() !== '' || state.enhancedScene.trim() !== '';
}

/** Add *rewrite* as the newest version, keeping the earlier ones (whatever text they were made from) up to `VERSION_LIMIT`. */
export function pushRewrite(history: readonly EnhancedRewrite[], rewrite: EnhancedRewrite): EnhancedRewrite[] {
  return pushVersion(history, rewrite, READER);
}

/** The fields an Enhance result sets: the result is shown and is the newest version. */
export function rewriteResultPatch(state: Pick<EnhancedState, 'enhancedHistory'>, rewrite: EnhancedRewrite): Partial<DraftState> {
  return { ...showing(rewrite.prompt, rewrite.scene, rewrite.from), enhancedHistory: pushRewrite(state.enhancedHistory, rewrite) };
}

/** What the stepper shows for the draft, or null for none (it is hidden then); see `versionStepper`. Hidden while *running* and when auto-enhance is on. */
export function rewriteStepper(state: DraftState, running = false): StepperState | null {
  if (running || state.enhanceAuto) return null;
  return versionStepper(state.enhancedHistory.length, hasEnhanced(state), shownPosition(state.enhancedHistory, shownPair(state), READER), state.enhancedScratched);
}

/** The fields to set to show version *index* (1-based); an index outside the versions changes nothing. */
export function stepRewritePatch(state: Pick<EnhancedState, 'enhancedHistory'>, index: number): Partial<DraftState> {
  const entry = state.enhancedHistory[index - 1];
  return entry === undefined ? {} : showing(entry.prompt, entry.scene, entry.from);
}

/** The fields to set to empty the Enhanced text, so Generate sends your text; the versions stay and ‹ brings back the one that was emptied. */
export function scratchRewritePatch(state: EnhancedState): Partial<DraftState> {
  return { enhancedPrompt: '', enhancedScene: '', enhancedFrom: null, enhancedScratched: scratchedPosition(state.enhancedHistory, shownPair(state), hasEnhanced(state), READER) };
}

/** The fields to set to drop every version and the Enhanced text: what a recent-prompt pick does. */
export function clearRewritesPatch(): Partial<DraftState> {
  return { enhancedPrompt: '', enhancedScene: '', enhancedFrom: null, enhancedHistory: [], enhancedScratched: null };
}

/**
 * The fields to set when the Enhanced text becomes *prompt* and *scene* by hand (typing, a trigger word put in). Text typed by hand is no longer what
 * the enhancer wrote, so the record is dropped. Typing in a shown version replaces that version's text, found by its text before the edit, and drops its
 * record there too; emptying both parts is a Scratch.
 */
export function editEnhancedPatch(state: EnhancedState, edit: { prompt: string; scene: string }): Partial<DraftState> {
  if (edit.prompt.trim() === '' && edit.scene.trim() === '') return scratchRewritePatch(state);
  const position = hasEnhanced(state) ? shownPosition(state.enhancedHistory, shownPair(state), READER) : -1;
  const history = editedHistory(state.enhancedHistory, position, (entry) => ({ prompt: edit.prompt, scene: edit.scene, from: withoutRecord(entry.from) as EnhancedSource }));
  return { ...showing(edit.prompt, edit.scene, withoutRecord(state.enhancedFrom)), enhancedHistory: history };
}
