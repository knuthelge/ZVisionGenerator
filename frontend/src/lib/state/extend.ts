import type { DraftState, ExtendSourceInfo, WorkspaceContext } from '$lib/types';
import { submittedPair } from './promptEnhance';

/** Form fields that choose the size or length of a non-extend clip; an extension takes both from its source. */
const EXTEND_IGNORED_FIELDS = ['ratio', 'size', 'width', 'height', 'frames', 'image_path', 'image_file'] as const;

/** Collapse every whitespace run to one space and trim the ends, as the server does before comparing prompts. */
export function normalizePromptText(text: string): string {
  return text.split(/\s+/).filter(Boolean).join(' ');
}

/**
 * Whether the Prompt still equals the prompt recorded in a legacy source.
 *
 * A legacy source recorded one prompt and no scene, so pre-filling it into the Prompt would repeat its action
 * and dialogue. True right after that prefill; false for any source with a recorded scene or no recorded prompt.
 */
export function isOriginalLegacyPrompt(state: Pick<DraftState, 'prompt'>, info: ExtendSourceInfo | null): boolean {
  if (info === null || !info.legacy || info.prompt === null) return false;
  const original = normalizePromptText(info.prompt);
  return original !== '' && normalizePromptText(state.prompt) === original;
}

/**
 * The Scene and Prompt a source pre-fills.
 *
 * A recorded scene fills Scene and leaves Prompt empty (what happens next); a legacy source fills Prompt with its
 * old prompt and leaves Scene empty; a source with no recorded text leaves both empty.
 */
export function extendPrefillFor(info: ExtendSourceInfo): { scene: string; prompt: string } {
  if (info.scene) return { scene: info.scene, prompt: '' };
  if (info.legacy && info.prompt) return { scene: '', prompt: info.prompt };
  return { scene: '', prompt: '' };
}

/** Whether a source can be extended: it was read and reports no problems. */
export function extendSourceUsable(info: ExtendSourceInfo | null): boolean {
  return info !== null && info.problems.length === 0 && info.fps !== null && info.frame_count !== null;
}

/**
 * Set the Extend-specific fields of a Workspace submission and settle its `scene` field.
 *
 * An extension always sends `scene`, even blank: absent means "the source's recorded scene" and blank means "no scene".
 * It sends the source and the new frames, and none of the fields that choose a size or length. Other workflows
 * leave a blank scene out of the form, which is today's form data.
 */
export function finalizeSceneFields(form: FormData, state: DraftState): void {
  if (state.workflow !== 'extend') {
    if (String(form.get('scene') ?? '').trim() === '') form.delete('scene');
    return;
  }
  for (const field of EXTEND_IGNORED_FIELDS) form.delete(field);
  for (const field of ['video_upscale_enabled', 'video_upscale_factor', 'upscale']) form.delete(field);
  if (state.extendSourcePath) form.set('source_video', state.extendSourcePath);
  if (state.extendFrames !== null) form.set('extend_frames', String(state.extendFrames));
  if (!form.has('scene')) form.set('scene', submittedPair(state).scene);
}

/** Where the inspection of the chosen source stands. */
export type ExtendSourceStatus = 'idle' | 'checking' | 'ready' | 'not_found' | 'failed';

/** Generate's reason when no usable source is chosen: nothing chosen, or no video at the path. */
export const CHOOSE_SOURCE_NOTE = 'Choose a video to extend.';
/** Generate's reason when the source was read and cannot be extended. */
export const REJECTED_SOURCE_NOTE = "This video can't be extended. See Source video.";

/** Whether the chosen video model can extend a clip, with the name the note calls it by. */
export interface ExtendModelSupport {
  supported: boolean;
  label: string;
}

/** Return what the workspace knows about the model's ability to extend, or null when it knows nothing (the model then is not blocked). */
export function extendModelSupport(ctx: WorkspaceContext | null, model: string): ExtendModelSupport | null {
  const defaults = ctx?.video_model_defaults?.[model];
  if (!ctx || !defaults) return null;
  return { supported: defaults.supports_extend, label: ctx.video_models.find((entry) => entry.id === model)?.label ?? model };
}

/** Whether the chosen source lets an extension start, and the reason shown when it does not. */
export interface ExtendGuard {
  /** `ok`: usable (or not an extension); `unsupported`: the model cannot extend; `none`: nothing usable chosen; `checking`: waiting for the server; `rejected`: read but not extendable. */
  kind: 'ok' | 'unsupported' | 'none' | 'checking' | 'rejected';
  blocked: boolean;
  /** The reason to show; null while checking (a note would only flicker) and when not blocked. */
  note: string | null;
}

/** Decide whether Generate may run for an extension, from the model, the chosen path, what the server said about it and the inspection status. */
export function extendGuard(
  state: Pick<DraftState, 'workflow' | 'extendSourcePath'>,
  info: ExtendSourceInfo | null,
  status: ExtendSourceStatus,
  model: ExtendModelSupport | null = null,
): ExtendGuard {
  if (state.workflow !== 'extend') return { kind: 'ok', blocked: false, note: null };
  if (model && !model.supported) return { kind: 'unsupported', blocked: true, note: `${model.label} can't extend a clip. Pick another video model.` };
  if (!state.extendSourcePath || status === 'not_found') return { kind: 'none', blocked: true, note: CHOOSE_SOURCE_NOTE };
  if (status === 'failed') return { kind: 'rejected', blocked: true, note: REJECTED_SOURCE_NOTE };
  if (info === null) return { kind: 'checking', blocked: true, note: null };
  if (!extendSourceUsable(info)) return { kind: 'rejected', blocked: true, note: REJECTED_SOURCE_NOTE };
  return { kind: 'ok', blocked: false, note: null };
}

/** The Enhance hint for an extension that cannot start, or null when there is nothing to tell. */
export function extendEnhanceHint(guard: ExtendGuard): string | null {
  if (guard.kind === 'none') return 'Choose a video to extend first.';
  if (guard.kind === 'rejected') return 'Choose a video that can be extended first.';
  if (guard.kind === 'unsupported') return 'Choose a video model that can extend first.';
  return null;
}

/** One fact about the source, shown in the Source video section. */
export interface SourceFact {
  key: string;
  text: string;
  title?: string;
}

/** Shown on the Rotated fact. */
export const ROTATED_TITLE = 'Shown upright. The original is re-encoded once when joined.';

/** Round to two places for display (6.0417 is 6.04); whole numbers stay whole. */
function shortNumber(value: number): string {
  return String(Number(value.toFixed(2)));
}

/** Seconds as shown in the facts and the total: always two places (6.00, 6.04). */
function twoPlaceSeconds(seconds: number): string {
  return seconds.toFixed(2);
}

/** The facts line of a source: displayed size, frame rate, length, frames, then rotation, audio and whether text was recorded. */
export function extendSourceFacts(info: ExtendSourceInfo): SourceFact[] {
  const facts: SourceFact[] = [];
  if (info.width !== null && info.height !== null) facts.push({ key: 'size', text: `${info.width} × ${info.height}` });
  if (info.fps !== null) facts.push({ key: 'fps', text: `${shortNumber(info.fps)} fps` });
  if (info.duration_seconds !== null) facts.push({ key: 'duration', text: `${twoPlaceSeconds(info.duration_seconds)} s` });
  if (info.frame_count !== null) facts.push({ key: 'frames', text: `${info.frame_count} frames` });
  if (info.rotation) facts.push({ key: 'rotated', text: 'Rotated', title: ROTATED_TITLE });
  if (info.has_audio !== null) facts.push({ key: 'audio', text: info.has_audio ? 'Audio' : 'No audio' });
  if (!info.scene && !info.prompt) facts.push({ key: 'no-prompt', text: 'No prompt recorded' });
  return facts;
}
