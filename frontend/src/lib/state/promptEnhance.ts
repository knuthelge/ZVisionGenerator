import type { DraftState, EnhanceRecord, EnhanceContext, EnhancedSource, EnhanceFormat, EnhanceParts, EnhancerTier, EnhanceSettings, PromptEnhancerContract, PromptFileOption, Workflow, WorkflowMode } from '$lib/types';

/** The enhancer a run uses: Smart when chosen and available here, otherwise Fast. */
export function effectiveTier(state: Pick<DraftState, 'enhanceModel'>, contract: PromptEnhancerContract | undefined | null): EnhancerTier {
  return state.enhanceModel === 'smart' && contract?.smart && !contract.smart.unavailable ? 'smart' : 'fast';
}

/** What a run writes: a JSON caption only for an image model that takes JSON captions. */
export function effectiveFormat(state: Pick<DraftState, 'enhanceFormat' | 'workflow'>, jsonCapable: boolean): EnhanceFormat {
  return state.enhanceFormat === 'json' && jsonCapable && workflowMode(state.workflow) === 'image' ? 'json' : 'text';
}

/** Whether *text* is a JSON object, i.e. a structured caption rather than a prompt. */
export function isJsonCaption(text: string): boolean {
  if (!text.trim().startsWith('{')) return false;
  try {
    const parsed: unknown = JSON.parse(text);
    return typeof parsed === 'object' && parsed !== null && !Array.isArray(parsed);
  } catch {
    return false;
  }
}

/** A JSON caption laid out for reading and editing; other text is returned unchanged. */
export function prettyCaption(text: string): string {
  return isJsonCaption(text) ? JSON.stringify(JSON.parse(text), null, 2) : text;
}

/** A prompt-file entry's `enhance:` spec as a run in *mode* applies it, or null when that run does not enhance the entry. */
export function entryEnhanceSpec(option: Pick<PromptFileOption, 'enhance' | 'enhance_image' | 'enhance_caption'>, mode: WorkflowMode, caption: boolean): string | null {
  if (mode === 'video') return option.enhance ?? null;
  return (caption ? option.enhance_caption : option.enhance_image) ?? null;
}

/** Media mode of a canonical workflow. */
export function workflowMode(workflow: Workflow): WorkflowMode {
  return workflow === 'txt2vid' || workflow === 'img2vid' || workflow === 'extend' ? 'video' : 'image';
}

/** The enhance options when nothing is picked: Keep everywhere, Same length (the server's defaults). */
export const DEFAULT_ENHANCE_SETTINGS: EnhanceSettings = { style: 'keep', mood: 'keep', theme: 'keep', details: [], length: 'same', camera: 'keep', pace: 'keep', motion: [], sound: [], note: '' };

/**
 * What Rewrite starts as: an extension keeps its scene and writes the next action, and so does Image to Video, whose scene
 * describes the starting image; Text to Video writes both. A blank Scene still means both (see `runParts`).
 */
export function defaultParts(workflow: Workflow): EnhanceParts {
  return workflow === 'extend' || workflow === 'img2vid' ? 'action' : 'both';
}

/**
 * Whether *settings* equal *defaults* (the server's, else nothing picked and Same length) on every option and the direction;
 * the remembered Rewrite choices do not count.
 */
export function isDefaultEnhanceSettings(settings: EnhanceSettings, defaults: Partial<EnhanceSettings> = DEFAULT_ENHANCE_SETTINGS): boolean {
  const base = { ...DEFAULT_ENHANCE_SETTINGS, ...defaults };
  return (Object.keys(DEFAULT_ENHANCE_SETTINGS) as (keyof EnhanceSettings)[]).every((key) => {
    const value = settings[key];
    const fallback = base[key];
    if (Array.isArray(value) || Array.isArray(fallback)) {
      const [a, b] = [value ?? [], fallback ?? []] as string[][];
      return a.length === b.length && a.every((item) => b.includes(item));
    }
    return key === 'note' ? String(value ?? '').trim() === String(fallback ?? '').trim() : value === fallback;
  });
}

/** Restore enhance options from their JSON (a queued job's or an asset's); a Rewrite choice is remembered for *workflow*. */
export function restoredEnhanceSettings(value: string | undefined, workflow: Workflow | undefined): EnhanceSettings | null {
  if (!value) return null;
  try {
    const parsed = JSON.parse(value) as Partial<EnhanceSettings>;
    if (typeof parsed !== 'object' || parsed === null) return null;
    const { parts: _parts, partsByWorkflow: _remembered, ...known } = parsed;
    const settings: EnhanceSettings = { ...DEFAULT_ENHANCE_SETTINGS, ...known };
    const parts: EnhanceParts | undefined = parsed.parts === 'both' || parsed.parts === 'action' ? parsed.parts : undefined;
    return parts !== undefined && workflow !== undefined ? { ...settings, partsByWorkflow: { [workflow]: parts } } : settings;
  } catch {
    return null;
  }
}

/**
 * What Reuse restores from an asset's `enhance` record (JSON): the options for *workflow*'s mode, the enhancer (Fast or
 * Smart; a custom model leaves the choice alone), the format, and the record kept with the Enhanced text.
 */
export function reusedEnhance(value: string | undefined, workflow: Workflow): { settings: EnhanceSettings; tier: EnhancerTier | null; format: EnhanceFormat; record: EnhanceRecord } | null {
  if (!value) return null;
  try {
    const parsed = JSON.parse(value) as { settings?: Partial<EnhanceSettings>; model?: unknown; tier?: unknown; format?: unknown };
    if (typeof parsed?.model !== 'string' || !parsed.model) return null;
    const settings = restoredEnhanceSettings(JSON.stringify(parsed.settings ?? {}), workflow);
    if (!settings) return null;
    const format: EnhanceFormat = parsed.format === 'json' ? 'json' : 'text';
    const tier = parsed.tier === 'fast' || parsed.tier === 'smart' ? parsed.tier : null;
    return { settings, tier, format, record: { settings: parsed.settings ?? {}, model: parsed.model, format } };
  } catch {
    return null;
  }
}

/** The draft field that holds the enhance options of *workflow*'s mode: images and video remember theirs apart. */
export function enhanceSettingsKey(workflow: Workflow): 'enhanceSettings' | 'enhanceSettingsVideo' {
  return workflowMode(workflow) === 'video' ? 'enhanceSettingsVideo' : 'enhanceSettings';
}

/** The Rewrite choice remembered for the current workflow, or that workflow's default. */
export function rememberedParts(state: Pick<DraftState, 'workflow' | 'enhanceSettingsVideo'>): EnhanceParts {
  return state.enhanceSettingsVideo?.partsByWorkflow?.[state.workflow] ?? defaultParts(state.workflow);
}

/** Settings with the Rewrite choice for *workflow* remembered in `partsByWorkflow`; the other fields and workflows are kept. */
export function withRememberedParts(settings: EnhanceSettings, workflow: Workflow, parts: EnhanceParts): EnhanceSettings {
  return { ...settings, partsByWorkflow: { ...settings.partsByWorkflow, [workflow]: parts } };
}

/** Whether Rewrite = Action only has to wait for a scene: an inline video run with a blank Scene splits the text into both parts. */
export function actionOnlyBlocked(state: DraftState): boolean {
  return workflowMode(state.workflow) === 'video' && state.promptSource === 'inline' && state.scene.trim() === '';
}

/**
 * The parts a run rewrites: the remembered choice, except that a blank Scene always means both.
 *
 * The remembered choice itself is not changed. Prompt-file entries bring their own scenes, so a file run never falls back here.
 */
export function runParts(state: DraftState, remembered: EnhanceParts = rememberedParts(state)): EnhanceParts {
  return remembered === 'action' && actionOnlyBlocked(state) ? 'both' : remembered;
}

/** *settings* with every option value the server no longer offers left out (a single choice falls back to its default). */
function offeredOnly(settings: EnhanceSettings, contract: PromptEnhancerContract | undefined): EnhanceSettings {
  if (!contract) return settings;
  const next: EnhanceSettings = { ...settings };
  for (const axis of contract.matrix.axes) {
    const offered = new Set(axis.options.map((option) => option.slug));
    const record = next as unknown as Record<string, string | string[]>;
    const value = record[axis.key];
    if (Array.isArray(value)) record[axis.key] = value.filter((slug) => offered.has(slug));
    else if (typeof value === 'string' && !offered.has(value)) record[axis.key] = axis.default[0] ?? 'keep';
  }
  return next;
}

/** The user's last-used options, with axes they never set (e.g. saved before an axis existed) taken from the backend matrix defaults. */
export function effectiveEnhanceSettings(state: DraftState, contract: PromptEnhancerContract | undefined): EnhanceSettings {
  const defaults = { ...DEFAULT_ENHANCE_SETTINGS, ...contract?.matrix.defaults };
  const settings = offeredOnly({ ...defaults, details: [...defaults.details], motion: [...defaults.motion], sound: [...defaults.sound], ...state[enhanceSettingsKey(state.workflow)] }, contract);
  delete settings.parts;
  return workflowMode(state.workflow) === 'video' ? { ...settings, parts: rememberedParts(state) } : settings;
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
 * Settings as sent to the backend; `camera`, `pace`, `motion` and `parts` only apply to video, `sound` only to video with audio,
 * and a blank `note` is left out.
 *
 * The per-workflow Rewrite memory stays in the browser.
 */
export function settingsPayload(settings: EnhanceSettings, mode: WorkflowMode, audio = false): Partial<EnhanceSettings> {
  const { camera, pace, motion, sound, note, parts, partsByWorkflow: _remembered, ...rest } = settings;
  const base: Partial<EnhanceSettings> = note.trim() ? { ...rest, note } : rest;
  if (mode !== 'video') return base;
  const video = parts === undefined ? { ...base, camera, pace, motion } : { ...base, parts, camera, pace, motion };
  return audio ? { ...video, sound } : video;
}

/**
 * The job facts an on-demand enhancement needs: frame shape, clip length, audio, reference image, references and last frame.
 *
 * An extension names its source, new frames and video model instead; the server reads the source's size, frame rate and recorded text,
 * and the model's family sets the step the new frames are a multiple of.
 *
 * @param sourceHasAudio Extend only: whether the source has audio, when known.
 * @param endImage Whether the clip also ends on a last-frame image (chosen, or uploaded and pending).
 * @param references How many reference rows the draft has, uploads included (an uploaded row has an empty path in the draft).
 */
export function enhanceContext(
  state: DraftState,
  audioSupported = true,
  sourceHasAudio: boolean | null = null,
  endImage = false,
  references = 0,
): EnhanceContext {
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
      model: state.model || null,
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
    references,
    end_image: video && endImage,
  };
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

/** Whether Generate sends the Enhanced text of one part: its Enhanced box has text, and the text is inline with no JSON caption and no enhancing at job time. */
export function enhancedPartUsed(state: DraftState, part: 'scene' | 'action'): boolean {
  if (state.promptSource !== 'inline' || state.jsonPromptEnabled || state.enhanceAuto) return false;
  if (part === 'action') return state.enhancedPrompt.trim() !== '';
  return workflowMode(state.workflow) === 'video' && state.enhancedScene.trim() !== '';
}

/** Whether Generate sends Enhanced text for at least one part. */
export function anyEnhancedUsed(state: DraftState): boolean {
  return enhancedPartUsed(state, 'scene') || enhancedPartUsed(state, 'action');
}

/** The scene and prompt to submit: each part is its Enhanced text when that has text, otherwise your text. Images never carry a scene. */
export function submittedPair(state: DraftState): { scene: string; prompt: string } {
  const video = workflowMode(state.workflow) === 'video';
  return {
    scene: !video ? '' : enhancedPartUsed(state, 'scene') ? state.enhancedScene : state.scene,
    prompt: enhancedPartUsed(state, 'action') ? state.enhancedPrompt : state.prompt,
  };
}

/** The Enhanced JSON caption Generate sends as `json_prompt` in place of the prompt, or null when the Enhanced text is not one. */
export function submittedCaption(state: DraftState): string | null {
  if (workflowMode(state.workflow) !== 'image' || !enhancedPartUsed(state, 'action')) return null;
  return isJsonCaption(state.enhancedPrompt) ? state.enhancedPrompt : null;
}

/** The `prompt` form value to submit. */
export function submittedPrompt(state: DraftState): string {
  return submittedPair(state).prompt;
}

/** Whether the Enhanced text was produced from a different scene, prompt, workflow mode or, for an extension, source or length than the current one. */
export function isEnhancedStale(state: DraftState): boolean {
  const source = state.enhancedFrom;
  if (source === null) return false;
  const mode = workflowMode(state.workflow);
  if (state.enhancedPrompt.trim() === '' && (mode !== 'video' || state.enhancedScene.trim() === '')) return false;
  if (source.prompt !== state.prompt || source.mode !== mode || (mode === 'video' && source.scene !== state.scene)) return true;
  // Outside an extension the source and length mean nothing, so an Enhanced pair made for one is out of date once the workflow is another.
  const extending = state.workflow === 'extend';
  return (source.sourceVideoPath ?? null) !== (extending ? state.extendSourcePath : null) || (source.extendFrames ?? null) !== (extending ? state.extendFrames : null);
}

/** Status line for an enhancement phase; *caption* runs write a JSON caption. */
export function enhancePhaseMessage(phase: 'downloading' | 'loading' | 'generating' | 'generating_cpu', downloadSize: string | null, caption = false): string {
  if (phase === 'downloading') return `Downloading enhancer model${downloadSize ? ` (≈${downloadSize}, first use only)` : ' (first use only)'}…`;
  if (phase === 'loading') return 'Loading enhancer…';
  if (phase === 'generating_cpu') return 'Enhancing on the CPU (no GPU found); this can take a few minutes…';
  return caption ? 'Writing a JSON caption…' : 'Enhancing…';
}

/** Short note for a clamped length request. */
export function clampedNote(length: string): string {
  return length === 'shorter' ? 'Too short to shorten; kept about the same length.' : 'Already at maximum length; kept about the same length.';
}
