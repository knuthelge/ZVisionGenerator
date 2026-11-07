import { orderedActions } from './scriptHelper';
import { autoSoundSeconds } from './soundSettings';
import type {
  AudioEnhanceAxis,
  AudioEnhanceContext,
  AudioEnhanceMatrix,
  AudioEnhanceRecord,
  AudioEnhanceSettings,
  AudioEnhanceTask,
  AudioRewriteTask,
  AudioSettingsVocabulary,
  AudioWorkflow,
  DraftState,
  PromptEnhancerContract,
  ScriptAction,
  ScriptSettings,
} from '$lib/types';

/** The kind each task writes for, and so the axes its popover offers. */
const TASK_KIND = { sound: 'sound', style: 'music' } as const satisfies Record<AudioEnhanceTask, 'sound' | 'music'>;

/** The task that rewrites the text of a kind's box; null for a kind that has no audio enhancer (Speech's text is read as written). */
export function enhanceTaskOf(kind: AudioWorkflow): AudioEnhanceTask | null {
  return kind === 'sound' ? 'sound' : kind === 'music' ? 'style' : null;
}

/** The task that writes the text of a kind's box: the caption task, or for Speech the script helper. */
export function rewriteTaskOf(kind: AudioWorkflow): AudioRewriteTask {
  return kind === 'speech' ? 'script' : kind === 'sound' ? 'sound' : 'style';
}

/**
 * The axes a task's popover offers, in the matrix's order. A take that does not sing (*singing* false) has no Singer: the axes that need a
 * singing voice are left out.
 */
export function axesOf(matrix: AudioEnhanceMatrix, task: AudioEnhanceTask, singing = true): AudioEnhanceAxis[] {
  return matrix.axes.filter((axis) => axis.kind === TASK_KIND[task] && (singing || !axis.vocal));
}

/** Whether *value* is a list of strings. */
function isList(value: unknown): value is string[] {
  return Array.isArray(value);
}

/** The options of *task* as the popover shows them: the server's defaults, then the user's last choices, with any option the server no longer offers left out. */
export function effectiveAudioSettings(state: Pick<DraftState, 'enhanceSettingsAudio'>, task: AudioEnhanceTask, matrix: AudioEnhanceMatrix): AudioEnhanceSettings {
  const stored = state.enhanceSettingsAudio[task] ?? {};
  const merged = { ...matrix.defaults, ...stored } as unknown as Record<string, string | string[]>;
  for (const axis of axesOf(matrix, task)) {
    const offered = new Set(axis.options.map((option) => option.slug));
    const value = merged[axis.key];
    if (isList(value)) merged[axis.key] = value.filter((slug) => offered.has(slug));
    else if (typeof value !== 'string' || !offered.has(value)) merged[axis.key] = axis.default[0] ?? 'keep';
  }
  return merged as unknown as AudioEnhanceSettings;
}

/** Whether every option of *task* is at its default and the direction is blank. */
export function isDefaultAudioSettings(settings: AudioEnhanceSettings, task: AudioEnhanceTask, matrix: AudioEnhanceMatrix, singing = true): boolean {
  return Object.keys(audioSettingsPayload(settings, task, matrix, singing)).length === 0;
}

/**
 * The options as sent to the server and recorded with the take: only what differs from the default on the task's own axes (without Singer
 * when the take does not sing), and the direction when it is not blank.
 */
export function audioSettingsPayload(settings: AudioEnhanceSettings, task: AudioEnhanceTask, matrix: AudioEnhanceMatrix, singing = true): Partial<AudioEnhanceSettings> {
  const payload: Record<string, string | string[]> = {};
  const values = settings as unknown as Record<string, string | string[]>;
  const defaults = matrix.defaults as unknown as Record<string, string | string[]>;
  for (const axis of axesOf(matrix, task, singing)) {
    const value = values[axis.key];
    const fallback = defaults[axis.key];
    const same = isList(value) && isList(fallback) ? value.length === fallback.length && value.every((slug) => fallback.includes(slug)) : value === fallback;
    if (!same) payload[axis.key] = value;
  }
  if (settings.note.trim() !== '') payload.note = settings.note;
  return payload as Partial<AudioEnhanceSettings>;
}

/** The part of the Audio tab's view the context reads: which controls the open row shows, and what its Lyrics make. */
export interface ContextView {
  shown: { showLyrics: boolean; showTempo: boolean; showKey: boolean };
  music: { kind: 'song' | 'instrumental' } | null;
}

/** How much of the lyrics a style rewrite is told; the server keeps the same. */
export const LYRICS_CONTEXT_CHARS = 600;

/**
 * Whether the open Music take has a singing voice: its model takes lyrics and they hold words. A model that takes none (Stable Audio) and
 * lyrics that are empty or only `[section]` tags make an instrumental, which has no Singer.
 */
export function musicSings(state: Pick<DraftState, 'lyrics'>, view: Pick<ContextView, 'shown' | 'music'>): boolean {
  if (!view.shown.showLyrics) return false;
  return view.music === null ? state.lyrics.trim() !== '' : view.music.kind === 'song';
}

/** Whether the take sings, for the task: Sound has no voice to hide, so only a style depends on the lyrics. */
export function taskSings(state: Pick<DraftState, 'lyrics'>, task: AudioEnhanceTask, view: Pick<ContextView, 'shown' | 'music'>): boolean {
  return task !== 'style' || musicSings(state, view);
}

/**
 * What the enhancer is told about the take besides the text. A Sound's type and the length it will have (the length set, else what Auto
 * stands for at that type), so the rewrite describes only what fits. A Music style: whether the song is instrumental, the start of the lyrics
 * of a model that sings them, and the tempo and key when set. The length of a song is never sent.
 */
export function audioEnhanceContext(
  state: Pick<DraftState, 'soundType' | 'audioSeconds' | 'lyrics' | 'audioTempo' | 'audioKey'>,
  task: AudioEnhanceTask,
  vocabulary: AudioSettingsVocabulary | undefined,
  view: ContextView,
): AudioEnhanceContext {
  if (task === 'sound') {
    const seconds = state.audioSeconds > 0 ? state.audioSeconds : vocabulary ? autoSoundSeconds(state.soundType, vocabulary) : 0;
    return { sound_type: state.soundType, ...(seconds > 0 ? { seconds } : {}) };
  }
  const sings = musicSings(state, view);
  return {
    instrumental: !sings,
    ...(sings ? { lyrics: state.lyrics.slice(0, LYRICS_CONTEXT_CHARS) } : {}),
    ...(view.shown.showTempo && state.audioTempo !== null ? { tempo: state.audioTempo } : {}),
    ...(view.shown.showKey && state.audioKey ? { key: state.audioKey } : {}),
  };
}

/** The record Generate sends for a rewrite the enhancer *model* wrote with *settings*. */
export function audioEnhanceRecord(task: AudioEnhanceTask, model: string, settings: Partial<AudioEnhanceSettings>): AudioEnhanceRecord {
  return { settings, model, format: 'text', task };
}

/** Read the settings of a script record: the actions in run order, and with a fit the seconds and the pace; null when they name no action or a fit has no time. */
function parseScriptSettings(value: unknown): ScriptSettings | null {
  const raw = (typeof value === 'object' && value !== null ? value : {}) as Partial<ScriptSettings>;
  const actions = Array.isArray(raw.actions) ? orderedActions(raw.actions.filter((action): action is ScriptAction => action === 'fit' || action === 'delivery')) : [];
  if (actions.length === 0) return null;
  if (!actions.includes('fit')) return { actions };
  if (typeof raw.seconds !== 'number') return null;
  return { actions, seconds: raw.seconds, ...(typeof raw.pace === 'string' ? { pace: raw.pace } : {}) };
}

/**
 * Read the record a saved take carries (a Reuse's `enhance` param, a job's `enhance_record`): the record in the form Generate sends it,
 * or null when it is broken or names no model or task.
 */
export function parseAudioRecord(value: string | undefined): AudioEnhanceRecord | null {
  if (!value) return null;
  try {
    const parsed = JSON.parse(value) as Partial<AudioEnhanceRecord> | null;
    if (typeof parsed !== 'object' || parsed === null || typeof parsed.model !== 'string' || parsed.model === '') return null;
    if (parsed.task !== 'sound' && parsed.task !== 'style' && parsed.task !== 'script') return null;
    const tier = parsed.tier === 'fast' || parsed.tier === 'smart' || parsed.tier === 'custom' ? parsed.tier : undefined;
    const settings = parsed.task === 'script' ? parseScriptSettings(parsed.settings) : (parsed.settings ?? {});
    if (settings === null) return null;
    return { settings, model: parsed.model, ...(tier === undefined ? {} : { tier }), format: 'text', task: parsed.task };
  } catch {
    return null;
  }
}

/** What a task says when there is no text to rewrite. */
const EMPTY_TEXT_HINT: Record<AudioEnhanceTask, string> = { sound: 'Describe the sound first.', style: 'Write the style first.' };

/**
 * Why Enhance cannot run now, or null. The enhancer needs the memory a take uses, so it waits for the take to finish (and, a moment, for
 * the take to be submitted). With *text* given, an empty box is a reason too.
 */
export function enhanceBlock(contract: PromptEnhancerContract, task: AudioEnhanceTask, state: { jobsActive: boolean; busy: boolean; text?: string }): string | null {
  if (contract.model === null) return contract.error ?? 'No prompt enhancer model is configured.';
  if (state.jobsActive) return 'Available when the take is done.';
  if (state.busy) return 'Available once the take is submitted.';
  return state.text !== undefined && state.text.trim() === '' ? EMPTY_TEXT_HINT[task] : null;
}
