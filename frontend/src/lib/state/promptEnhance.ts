import type { DraftState, EnhanceContext, EnhancedSource, EnhanceSettings, PromptEnhancerContract, Workflow, WorkflowMode } from '$lib/types';

/** Media mode of a canonical workflow. */
export function workflowMode(workflow: Workflow): WorkflowMode {
  return workflow === 'txt2vid' || workflow === 'img2vid' || workflow === 'extend' ? 'video' : 'image';
}

const FALLBACK_SETTINGS: EnhanceSettings = { style: 'keep', mood: 'keep', details: ['lighting', 'composition'], length: 'same', motion: ['action'], sound: [], note: '' };

/** The user's last-used options, with axes they never set (e.g. saved before an axis existed) taken from the backend matrix defaults. */
export function effectiveEnhanceSettings(state: DraftState, contract: PromptEnhancerContract | undefined): EnhanceSettings {
  const defaults = { ...FALLBACK_SETTINGS, ...contract?.matrix.defaults };
  return { ...defaults, details: [...defaults.details], motion: [...defaults.motion], sound: [...defaults.sound], ...state.enhanceSettings };
}

/**
 * Whether the job's video has an audio track, so the Sound axis applies; *supported* is false where the video backend writes no audio.
 *
 * @param sourceHasAudio Extend only: whether the source has audio; a silent source yields a silent clip. Null when unknown.
 */
export function hasAudio(state: DraftState, supported = true, sourceHasAudio: boolean | null = null): boolean {
  if (state.workflow === 'extend' && sourceHasAudio === false) return false;
  return supported && workflowMode(state.workflow) === 'video' && state.audio;
}

/**
 * Whether *settings* would ask for no change at all (keep style and mood, no details, same length, no direction, no video motion or sound).
 *
 * @param split The job is a video whose scene is blank and whose prompt is not, so the rewrite splits it.
 */
export function isNoOpSettings(settings: EnhanceSettings, mode: WorkflowMode, audio = false, split = false): boolean {
  // A video prompt written without a scene is split into scene and action, which is always a change.
  if (mode === 'video' && split) return false;
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

/**
 * The job facts an on-demand enhancement needs: frame shape, clip length, audio and reference image.
 *
 * An extension names its source and new frames instead; the server reads the source's size, frame rate and recorded text.
 *
 * @param sourceHasAudio Extend only: whether the source has audio, when known.
 */
export function enhanceContext(state: DraftState, audioSupported = true, sourceHasAudio: boolean | null = null): EnhanceContext {
  if (state.workflow === 'extend') {
    return {
      ratio: null,
      width: null,
      height: null,
      frames: null,
      fps: null,
      audio: hasAudio(state, audioSupported, sourceHasAudio),
      reference_image: false,
      source_video: state.extendSourcePath,
      extend_frames: state.extendFrames,
    };
  }
  const video = workflowMode(state.workflow) === 'video';
  const custom = state.dimensionMode === 'custom';
  return {
    ratio: state.ratio || null,
    width: custom ? state.width : null,
    height: custom ? state.height : null,
    frames: video ? state.frameCount : null,
    fps: video ? state.fps : null,
    audio: hasAudio(state, audioSupported),
    reference_image: state.workflow === 'img2img' || state.workflow === 'img2vid',
  };
}

/**
 * Whether an enhancement of *state* always changes the text, so no setting makes it a no-op.
 *
 * True for a video prompt with no scene but with a prompt (split into scene and action), and for an extension,
 * which always writes a new action.
 */
export function splitsPrompt(state: DraftState): boolean {
  if (state.workflow === 'extend') return true;
  return workflowMode(state.workflow) === 'video' && state.scene.trim() === '' && state.prompt.trim() !== '';
}

/**
 * Whether auto-enhance always changes the text of *state*'s job, so no setting makes it a no-op.
 *
 * Auto-enhance rewrites each prompt at submit time (also in file mode, where the prompt box is blank), so unlike
 * `splitsPrompt` it needs no prompt: a blank default scene is enough for a video, and an extension always writes a new action.
 */
export function autoSplitsPrompt(state: DraftState): boolean {
  if (state.workflow === 'extend') return true;
  return workflowMode(state.workflow) === 'video' && state.scene.trim() === '';
}

/** Whether there is text to run: a scene or a prompt for video, a prompt for images. An extension may run with neither (it continues naturally). */
export function canSubmitText(state: DraftState): boolean {
  if (state.workflow === 'extend') return true;
  if (state.prompt.trim() !== '') return true;
  return workflowMode(state.workflow) === 'video' && state.scene.trim() !== '';
}

/** What an enhancement started now would be produced from, kept with its result to tell when it goes out of date. */
export function enhancedSourceFor(state: DraftState): EnhancedSource {
  const mode = workflowMode(state.workflow);
  const source: EnhancedSource = { prompt: state.prompt, scene: mode === 'video' ? state.scene : '', mode };
  if (state.workflow === 'extend') {
    source.sourceVideoPath = state.extendSourcePath;
    source.extendFrames = state.extendFrames;
  }
  return source;
}

/** Whether the Enhanced text replaces the inline text on submit; for video, text in either Enhanced box counts. */
export function enhancedOverrideActive(state: DraftState): boolean {
  if (state.promptSource !== 'inline' || state.jsonPromptEnabled || state.enhanceAuto) return false;
  if (state.enhancedPrompt.trim() !== '') return true;
  return workflowMode(state.workflow) === 'video' && state.enhancedScene.trim() !== '';
}

/** The scene and prompt to submit: the Enhanced pair when it applies, otherwise the boxes. Images never carry a scene. */
export function submittedPair(state: DraftState): { scene: string; prompt: string } {
  const video = workflowMode(state.workflow) === 'video';
  if (enhancedOverrideActive(state)) return { scene: video ? state.enhancedScene : '', prompt: state.enhancedPrompt };
  return { scene: video ? state.scene : '', prompt: state.prompt };
}

/** The `prompt` form value to submit: the Enhanced text when it applies, otherwise the prompt box. */
export function submittedPrompt(state: DraftState): string {
  return submittedPair(state).prompt;
}

/** Whether the Enhanced text was produced from a different scene, prompt, workflow mode or, for an extension, source or length than the current one. */
export function isEnhancedStale(state: DraftState): boolean {
  const source = state.enhancedFrom;
  if (source === null) return false;
  const mode = workflowMode(state.workflow);
  if (state.enhancedPrompt.trim() === '' && (mode !== 'video' || state.enhancedScene.trim() === '')) return false;
  if (source.prompt !== state.prompt || source.mode !== mode || (mode === 'video' && (source.scene ?? '') !== state.scene)) return true;
  // Outside an extension the source and length mean nothing, so an Enhanced pair made for one is out of date once the workflow is another.
  const extending = state.workflow === 'extend';
  return (source.sourceVideoPath ?? null) !== (extending ? state.extendSourcePath : null) || (source.extendFrames ?? null) !== (extending ? state.extendFrames : null);
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
