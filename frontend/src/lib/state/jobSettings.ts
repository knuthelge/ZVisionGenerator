import { enhanceSettingsKey, prettyCaption, restoredEnhanceSettings, reusedEnhance, workflowMode } from './promptEnhance';
import type { DraftState, EnhancedSource, EnhanceParts, EnhanceSettings, JobSettings, PromptSource, Workflow } from '$lib/types';
import { EXTEND_FRAME_STEP, snapExtendFrames } from './videoDuration';

/** Fields the workspace prefill path applies on top of the model's defaults (the same keys as a reuse URL). */
const PREFILL_KEYS = ['workflow', 'model', 'steps', 'guidance', 'seed', 'ratio', 'size', 'width', 'height', 'frames', 'lora', 'image_path', 'end_image_path', 'source_video', 'extend_frames'] as const;

export interface JobSettingsPrefill {
  /** Applied like a reuse URL: workflow and model first, so their defaults load before the values. */
  params: Record<string, string>;
  /** Every other submitted setting, applied to the draft afterwards. */
  patch: Partial<DraftState>;
}

/** Whether a job was submitted from the form, so its settings can be loaded back (an upscale was not). */
export function hasJobSettings(settings: JobSettings | null | undefined): boolean {
  return Object.keys(settings ?? {}).length > 0;
}

/** Return every value submitted for a field. */
export function values(settings: JobSettings, key: string): string[] {
  const value = settings[key];
  if (value === undefined) return [];
  return Array.isArray(value) ? value : [value];
}

/** Return the first value submitted for a field. */
export function first(settings: JobSettings, key: string): string | undefined {
  return values(settings, key)[0];
}

/** A checkbox sent with a hidden "false" fallback is on when any other value came with it. */
export function checked(settings: JobSettings, key: string): boolean {
  return values(settings, key).some((value) => value !== 'false');
}

/** Parse a submitted number; blank, `null`, and non-numeric values become null. */
export function numberOrNull(value: string | undefined): number | null {
  if (value === undefined || value.trim() === '' || value === 'null') return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

/** The `record` of the Enhanced text from the job's `enhance_record` field, or nothing when it has none (or a broken one). */
function enhanceRecord(value: string | undefined, workflow: Workflow): Pick<EnhancedSource, 'record'> {
  const reused = reusedEnhance(value, workflow);
  return reused ? { record: reused.record } : {};
}

/**
 * Turn the form fields a job was submitted with back into workspace state, so the form can be refilled from it.
 *
 * @param rememberedParts The Rewrite choice the draft remembers for each video workflow; the job's own workflow takes the job's.
 * @param extendStep The step between the lengths the job's model can extend by.
 */
export function jobSettingsPrefill(settings: JobSettings, rememberedParts?: EnhanceSettings['partsByWorkflow'], extendStep: number = EXTEND_FRAME_STEP): JobSettingsPrefill {
  const params: Record<string, string> = {};
  for (const key of PREFILL_KEYS) {
    const value = first(settings, key);
    // A blank LoRA list is kept: the job ran without LoRAs, so loading it clears them.
    if (value !== undefined && (value !== '' || key === 'lora')) params[key] = value;
  }

  const promptSource = (first(settings, 'prompt_source') ?? 'inline') as PromptSource;
  const isVideo = first(settings, 'mode') === 'video';
  const workflow = (first(settings, 'workflow') as Workflow | undefined) ?? (isVideo ? 'txt2vid' : 'txt2img');
  const promptInput = first(settings, 'prompt_input');
  // An image run of an Enhanced JSON caption sent it as `json_prompt` beside the user's own prompt; a caption typed in the { } box has no own prompt.
  const ranCaption = !isVideo && settings.json_prompt !== undefined && promptInput !== undefined;
  const jsonPrompt = ranCaption ? undefined : first(settings, 'json_prompt');
  // A run that used Enhanced text also recorded the user's own text; it becomes the boxes, and each part that differs fills its Enhanced box.
  const ranEnhanced = settings.scene_input !== undefined || promptInput !== undefined;
  const ranScene = first(settings, 'scene') ?? '';
  const ranPrompt = ranCaption ? prettyCaption(first(settings, 'json_prompt') ?? '') : first(settings, 'prompt') ?? '';
  const ownScene = ranEnhanced ? first(settings, 'scene_input') ?? '' : ranScene;
  const ownPrompt = ranEnhanced ? promptInput ?? '' : ranPrompt;
  // A part that ran as written is not Enhanced text: only a part that differs from its recorded input fills its Enhanced box.
  const enhancedPrompt = ranEnhanced && ranPrompt !== ownPrompt ? ranPrompt : '';
  const enhancedScene = ranEnhanced && ranScene !== ownScene ? ranScene : '';
  const hasEnhanced = enhancedPrompt.trim() !== '' || enhancedScene.trim() !== '';
  const patch: Partial<DraftState> = {
    promptSource,
    // Without recorded inputs the submitted text is what ran (an Enhanced override included), so it becomes the text.
    prompt: ownPrompt,
    enhancedPrompt,
    enhancedScene,
    enhancedFrom: hasEnhanced ? { prompt: ownPrompt, scene: ownScene, mode: workflowMode(workflow), ...enhanceRecord(first(settings, 'enhance_record'), workflow) } : null,
    jsonPromptEnabled: jsonPrompt !== undefined,
    negativePrompt: first(settings, 'negative_prompt') ?? '',
    seed: numberOrNull(first(settings, 'seed')),
    dimensionMode: settings.width !== undefined ? 'custom' : 'ratio',
    enhanceAuto: settings.enhance_auto !== undefined,
    enhanceModel: first(settings, 'enhance_model') === 'smart' ? 'smart' : 'fast',
    enhanceFormat: first(settings, 'enhance_format') === 'json' ? 'json' : 'text',
  };
  if (isVideo) patch.scene = ownScene;
  // An extension's boxes are the job's own text, blank scene included; the source's recorded text must not replace them.
  if (isVideo && workflow === 'extend') {
    const source = first(settings, 'source_video') ?? null;
    const frames = numberOrNull(first(settings, 'extend_frames'));
    patch.extendPrefill = { path: source ?? '', scene: '', prompt: '' };
    // The Enhanced pair was made for this source and length, so it is current (as loadFromUrl records for a Reuse).
    if (patch.enhancedFrom) patch.enhancedFrom = { ...patch.enhancedFrom, sourceVideoPath: source, extendFrames: frames === null ? null : snapExtendFrames(frames, extendStep) };
  }
  if (jsonPrompt !== undefined) patch.jsonPrompt = jsonPrompt;
  if (promptSource === 'file') {
    patch.promptFilePath = first(settings, 'prompts_file') ?? null;
    patch.promptFileOptionIds = values(settings, 'prompt_option_id');
  }
  const runs = numberOrNull(first(settings, 'runs'));
  if (runs !== null) patch.runs = runs;
  const strength = numberOrNull(first(settings, 'image_strength'));
  if (strength !== null) patch.referenceImageStrength = strength;
  if (settings.scheduler !== undefined) patch.scheduler = first(settings, 'scheduler') || null;
  if (settings.first_sigma !== undefined) patch.firstSigma = numberOrNull(first(settings, 'first_sigma'));
  if (settings.quantize !== undefined) patch.quantize = numberOrNull(first(settings, 'quantize'));
  const enhance = restoredEnhanceSettings(first(settings, 'enhance_settings'), first(settings, 'workflow') as Workflow | undefined);
  if (enhance) {
    const key = enhanceSettingsKey(workflow);
    patch[key] = key === 'enhanceSettingsVideo' && rememberedParts ? { ...enhance, partsByWorkflow: { ...rememberedParts, ...enhance.partsByWorkflow } } : enhance;
  }

  for (const [name, enabledKey, amountKey] of [
    ['sharpen', 'postprocessSharpenEnabled', 'postprocessSharpenAmount'],
    ['contrast', 'postprocessContrastEnabled', 'postprocessContrastAmount'],
    ['saturation', 'postprocessSaturationEnabled', 'postprocessSaturationAmount'],
  ] as const) {
    if (settings[`${name}_enabled`] === undefined) continue;
    patch[enabledKey] = first(settings, `${name}_enabled`) === 'true';
    const amount = numberOrNull(first(settings, `${name}_amount`));
    if (amount !== null) patch[amountKey] = amount;
    // Sharpen on with no amount means it followed the config (auto); restore that instead of keeping a typed amount.
    else if (amountKey === 'postprocessSharpenAmount' && patch[enabledKey]) patch.postprocessSharpenAmount = null;
  }

  if (isVideo) {
    patch.audio = checked(settings, 'audio');
    patch.lowMemory = checked(settings, 'low_memory');
    patch.videoUpscaleEnabled = settings.video_upscale_factor !== undefined;
    const factor = numberOrNull(first(settings, 'video_upscale_factor'));
    if (factor !== null) patch.videoUpscaleFactor = factor;
    patch.videoUpscaleReview = patch.videoUpscaleEnabled && checked(settings, 'review');
  } else {
    patch.upscaleEnabled = settings.upscale !== undefined;
    const factor = numberOrNull(first(settings, 'upscale'));
    if (factor !== null) patch.upscaleFactor = factor;
    patch.upscaleReview = patch.upscaleEnabled && checked(settings, 'review');
    if (patch.upscaleEnabled) {
      patch.upscaleDenoise = numberOrNull(first(settings, 'upscale_denoise'));
      patch.upscaleSteps = numberOrNull(first(settings, 'upscale_steps'));
      patch.upscaleGuidance = numberOrNull(first(settings, 'upscale_guidance'));
      patch.upscaleSharpen = first(settings, 'upscale_sharpen') !== 'false';
    }
  }
  return { params, patch };
}
