import type { DraftState, EnhanceContext, EnhanceSettings, PromptEnhancerContract, Workflow, WorkflowMode } from '$lib/types';

/** Media mode of a canonical workflow. */
export function workflowMode(workflow: Workflow): WorkflowMode {
  return workflow === 'txt2vid' || workflow === 'img2vid' ? 'video' : 'image';
}

const FALLBACK_SETTINGS: EnhanceSettings = { style: 'keep', mood: 'keep', details: ['lighting', 'composition'], length: 'same', motion: ['action'], sound: [], note: '' };

/** The user's last-used options, with axes they never set (e.g. saved before an axis existed) taken from the backend matrix defaults. */
export function effectiveEnhanceSettings(state: DraftState, contract: PromptEnhancerContract | undefined): EnhanceSettings {
  const defaults = { ...FALLBACK_SETTINGS, ...contract?.matrix.defaults };
  return { ...defaults, details: [...defaults.details], motion: [...defaults.motion], sound: [...defaults.sound], ...state.enhanceSettings };
}

/** Whether the job's video has an audio track, so the Sound axis applies. */
export function hasAudio(state: DraftState): boolean {
  return workflowMode(state.workflow) === 'video' && state.audio;
}

/** Whether *settings* would ask for no change at all (keep style and mood, no details, same length, no direction, no video motion or sound). */
export function isNoOpSettings(settings: EnhanceSettings, mode: WorkflowMode, audio = false): boolean {
  const hasMotion = mode === 'video' && settings.motion.length > 0;
  const hasSound = mode === 'video' && audio && settings.sound.length > 0;
  return settings.style === 'keep' && settings.mood === 'keep' && settings.details.length === 0 && settings.length === 'same' && !settings.note.trim() && !hasMotion && !hasSound;
}

/** Settings as sent to the backend; `motion` only applies to video, `sound` only to video with audio, and a blank `note` is left out. */
export function settingsPayload(settings: EnhanceSettings, mode: WorkflowMode, audio = false): Partial<EnhanceSettings> {
  const { motion, sound, note, ...rest } = settings;
  const base: Partial<EnhanceSettings> = note.trim() ? { ...rest, note } : rest;
  if (mode !== 'video') return base;
  return audio ? { ...base, motion, sound } : { ...base, motion };
}

/** The job facts an on-demand enhancement needs: frame shape, clip length, audio and reference image. */
export function enhanceContext(state: DraftState): EnhanceContext {
  const video = workflowMode(state.workflow) === 'video';
  const custom = state.dimensionMode === 'custom';
  return {
    ratio: state.ratio || null,
    width: custom ? state.width : null,
    height: custom ? state.height : null,
    frames: video ? state.frameCount : null,
    fps: video ? state.fps : null,
    audio: hasAudio(state),
    reference_image: state.workflow === 'img2img' || state.workflow === 'img2vid',
  };
}

/** Whether the Enhanced box replaces the inline prompt on submit. */
export function enhancedOverrideActive(state: DraftState): boolean {
  return state.promptSource === 'inline' && !state.jsonPromptEnabled && !state.enhanceAuto && state.enhancedPrompt.trim() !== '';
}

/** The `prompt` form value to submit: the Enhanced text when it applies, otherwise the prompt box. */
export function submittedPrompt(state: DraftState): string {
  return enhancedOverrideActive(state) ? state.enhancedPrompt : state.prompt;
}

/** Whether the Enhanced text was produced from a different prompt or workflow mode than the current one. */
export function isEnhancedStale(state: DraftState): boolean {
  const source = state.enhancedFrom;
  if (source === null || state.enhancedPrompt.trim() === '') return false;
  return source.prompt !== state.prompt || source.mode !== workflowMode(state.workflow);
}

/** Status line for an enhancement phase. */
export function enhancePhaseMessage(phase: 'downloading' | 'loading' | 'generating' | 'generating_cpu', downloadSize: string | null): string {
  if (phase === 'downloading') return `Downloading enhancer model${downloadSize ? ` (≈${downloadSize}, first use only)` : ' (first use only)'}…`;
  if (phase === 'loading') return 'Loading enhancer…';
  if (phase === 'generating_cpu') return 'Enhancing on the CPU (no GPU found); this can take a few minutes…';
  return 'Enhancing…';
}

/** Short note for a clamped length request. */
export function clampedNote(length: string): string {
  return length === 'shorter' ? 'Too short to shorten; kept about the same length.' : 'Already at maximum length; kept about the same length.';
}
