import type { DraftState, PromptEntryText } from '$lib/types';

/** Why nothing can be saved right now. */
export type SaveBlock = 'json' | 'enhancing' | 'empty';

/** The prompt entry to save, or the reason there is none. */
export type SaveSelection = { entry: PromptEntryText } | { block: SaveBlock };

/** What the Compose pane is showing. */
export interface SaveView {
  enhancedView: boolean;
  enhancing: boolean;
  jsonMode: boolean;
  pairMode: boolean;
  showNegative: boolean;
}

type SaveState = Pick<DraftState, 'prompt' | 'scene' | 'negativePrompt' | 'enhancedPrompt' | 'enhancedScene' | 'enhanceAuto'>;

/** Pick the prompt text in view as a prompt-file entry; the Enhanced view falls back to your text for a part with no enhanced text. */
export function promptToSave(state: SaveState, view: SaveView): SaveSelection {
  if (view.jsonMode) return { block: 'json' };
  const enhancedPrompt = state.enhancedPrompt.trim();
  const enhancedScene = state.enhancedScene.trim();
  let prompt = state.prompt.trim();
  let scene = view.pairMode ? state.scene.trim() : '';
  if (view.enhancedView) {
    if (view.enhancing) return { block: 'enhancing' };
    if (state.enhanceAuto) return { block: 'empty' };
    if (!enhancedPrompt && (!view.pairMode || !enhancedScene)) return { block: 'empty' };
    prompt = enhancedPrompt || prompt;
    if (view.pairMode) scene = enhancedScene || scene;
  }
  if (!prompt) return { block: 'empty' };
  const negative = view.showNegative ? state.negativePrompt.trim() : '';
  const entry: PromptEntryText = { prompt };
  if (scene) entry.scene = scene;
  if (negative) entry.negative = negative;
  return { entry };
}

/** The unique, non-blank paths in the order given. */
export function saveTargets(...candidates: (string | null | undefined)[]): string[] {
  const seen = new Set<string>();
  for (const candidate of candidates) {
    const path = candidate?.trim();
    if (path) seen.add(path);
  }
  return [...seen];
}

/** The folder of a file path, with its trailing separator, or null for no path. */
export function folderOf(path: string | null): string | null {
  if (!path) return null;
  const cut = Math.max(path.lastIndexOf('/'), path.lastIndexOf('\\'));
  return cut < 0 ? null : path.slice(0, cut + 1);
}
