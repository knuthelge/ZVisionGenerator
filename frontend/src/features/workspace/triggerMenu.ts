import { enhancedPartUsed, submittedCaption, submittedPair } from '$lib/state/promptEnhance';
import { loraLabel, type LoraChip } from '$lib/state/loras';
import { addTrigger, containsTrigger, splitTriggers } from '$lib/state/triggers';
import type { DraftState, LoraInfo } from '$lib/types';
import type { WorkspaceCapabilities } from './capabilities';

/** How the prompt boxes are laid out: which box a trigger goes into, and whether it can be written at all. */
export interface TriggerModes {
  /** A video: the Scene box, with the prompt as the Action. */
  pair: boolean;
  /** The prompt is a JSON caption. */
  json: boolean;
  /** The prompts come from a file. */
  file: boolean;
}

export function triggerModes(caps: Pick<WorkspaceCapabilities, 'showScene' | 'showPromptInline' | 'supportsJsonPrompt'>, state: Pick<DraftState, 'promptSource' | 'jsonPromptEnabled'>): TriggerModes {
  const file = state.promptSource === 'file';
  return {
    pair: caps.showScene && !file && caps.showPromptInline,
    json: caps.showPromptInline && caps.supportsJsonPrompt && state.jsonPromptEnabled,
    file,
  };
}

export interface TriggerWord {
  word: string;
  /** The text Generate sends already holds the word. */
  inPrompt: boolean;
}

/** The trigger words of one LoRA that is in use. */
export interface TriggerGroup {
  /** The LoRA's file name. */
  name: string;
  /** What the menu calls the LoRA. */
  title: string;
  words: TriggerWord[];
}

export interface TriggerMenu {
  groups: TriggerGroup[];
  /** The words (counted once however many LoRAs name them) that the text Generate sends does not hold yet. */
  missing: number;
}

export interface TriggerInput {
  state: DraftState;
  modes: TriggerModes;
  chips: readonly LoraChip[];
  installed: readonly LoraInfo[];
  /** Triggers of downloading LoRAs, by LoRA name. */
  pending: Readonly<Record<string, string>>;
}

/** The trigger of a chip's LoRA: the installed file's, else what its download named. */
export function chipTrigger(name: string, installed: readonly LoraInfo[], pending: Readonly<Record<string, string>>): string | null {
  return installed.find((lora) => lora.name === name)?.trigger || pending[name] || null;
}

/** The draft field a part's text lives in: its Enhanced text when Generate sends that, else the text as written. */
export function partKey(state: DraftState, part: 'scene' | 'prompt'): 'scene' | 'enhancedScene' | 'prompt' | 'enhancedPrompt' {
  if (part === 'scene') return enhancedPartUsed(state, 'scene') ? 'enhancedScene' : 'scene';
  return enhancedPartUsed(state, 'action') ? 'enhancedPrompt' : 'prompt';
}

/** The text Generate sends for the prompt boxes: the Enhanced text where that is used, with the Scene before the Action. */
function sentText(state: DraftState, modes: TriggerModes): string {
  const caption = modes.json ? state.jsonPrompt : submittedCaption(state);
  if (caption !== null) return caption;
  const pair = submittedPair(state);
  return modes.pair ? `${pair.scene}\n${pair.prompt}` : pair.prompt;
}

/**
 * The Triggers menu of the prompt box: the words of the LoRAs in use, grouped by LoRA, each marked when the text
 * Generate sends already holds it (whole word, any case). Null when the menu is not offered: no LoRA in use has a
 * trigger, or the prompt is a JSON caption or comes from a file (nothing is written into those).
 */
export function triggerMenu({ state, modes, chips, installed, pending }: TriggerInput): TriggerMenu | null {
  if (modes.json || modes.file || submittedCaption(state) !== null) return null;
  const text = sentText(state, modes);
  const groups: TriggerGroup[] = [];
  const missing = new Set<string>();
  for (const chip of chips) {
    const words = splitTriggers(chipTrigger(chip.name, installed, pending));
    if (words.length === 0) continue;
    const title = installed.find((lora) => lora.name === chip.name)?.title ?? loraLabel(chip.name);
    groups.push({ name: chip.name, title, words: words.map((word) => ({ word, inPrompt: containsTrigger(text, word) })) });
    for (const { word, inPrompt } of groups[groups.length - 1].words) if (!inPrompt) missing.add(word.toLowerCase());
  }
  return groups.length === 0 ? null : { groups, missing: missing.size };
}

/** The draft change that puts *word* at the start of the scene (video) or prompt, in the Enhanced text when Generate sends that. */
export function insertWordPatch(state: DraftState, modes: TriggerModes, word: string): Partial<DraftState> {
  const key = partKey(state, modes.pair ? 'scene' : 'prompt');
  return { [key]: addTrigger(state[key], word) };
}
