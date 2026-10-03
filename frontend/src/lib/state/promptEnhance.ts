import type { DraftState, EnhanceSettings, PromptEnhancerContract, Workflow, WorkflowMode } from '$lib/types';

/** Media mode of a canonical workflow. */
export function workflowMode(workflow: Workflow): WorkflowMode {
  return workflow === 'txt2vid' || workflow === 'img2vid' ? 'video' : 'image';
}

/** The user's last-used options, falling back to the backend matrix defaults. */
export function effectiveEnhanceSettings(state: DraftState, contract: PromptEnhancerContract | undefined): EnhanceSettings {
  const defaults = contract?.matrix.defaults ?? { style: 'keep', details: ['lighting', 'composition'], length: 'same', motion: ['action'] };
  return state.enhanceSettings ?? { ...defaults, details: [...defaults.details], motion: [...defaults.motion] };
}

/** Whether *settings* would ask for no change at all (keep style, no details, same length, no motion). */
export function isNoOpSettings(settings: EnhanceSettings, mode: WorkflowMode): boolean {
  const hasMotion = mode === 'video' && settings.motion.length > 0;
  return settings.style === 'keep' && settings.details.length === 0 && settings.length === 'same' && !hasMotion;
}

/** Settings as sent to the backend; `motion` only applies to video. */
export function settingsPayload(settings: EnhanceSettings, mode: WorkflowMode): Partial<EnhanceSettings> {
  const { motion, ...rest } = settings;
  return mode === 'video' ? { ...rest, motion } : rest;
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
