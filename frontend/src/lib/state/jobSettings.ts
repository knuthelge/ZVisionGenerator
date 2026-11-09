import { enhanceSettingsKey, prettyCaption, restoredEnhanceSettings, reusedEnhance, workflowMode } from './promptEnhance';
import type { DraftState, EnhancedSource, EnhanceParts, EnhanceSettings, JobSettings, PrefillParams, PromptSource, Workflow } from '$lib/types';
import { EXTEND_FRAME_STEP, snapExtendFrames } from './videoDuration';

/** Fields the workspace prefill path applies on top of the model's defaults (the same keys as a reuse URL). */
const PREFILL_KEYS = ['workflow', 'model', 'steps', 'guidance', 'seed', 'ratio', 'size', 'width', 'height', 'frames', 'lora', 'image_path', 'end_image_path', 'audio_track_path', 'audio_track_start', 'source_video', 'extend_frames'] as const;

export interface JobSettingsPrefill {
  /** Applied like a reuse URL: workflow and model first, so their defaults load before the values. */
  params: PrefillParams;
  /** Every other submitted setting, applied to the draft afterwards. */
  patch: Partial<DraftState>;
}

/** The fields of an audio job's settings the Audio tab loads back, as a Reuse link names them (the speaker's consent is never recorded, so it is never among them). `prompt_input` and `enhance_record` are read apart: see `audioJobPrefill`. */
const AUDIO_PREFILL_KEYS = [
  ...['workflow', 'model', 'prompt', 'lyrics', 'seconds', 'tempo', 'key', 'speaker'],
  ...['emotion', 'manner', 'pace', 'pitch', 'expressiveness', 'sound_type', 'saved_voice', 'voice_path', 'voice_text', 'seed', 'runs'],
] as const;

/** Whether an audio job recorded how its text was written: its `enhance_record` field holds a non-empty JSON object. Not whether `prompt_input` is truthy, as the user's own text may be empty. */
function hasEnhanceRecord(value: string | undefined): value is string {
  if (value === undefined) return false;
  try {
    const record: unknown = JSON.parse(value);
    return typeof record === 'object' && record !== null && Object.keys(record).length > 0;
  } catch {
    return false;
  }
}

/**
 * The prefill params of an audio job, named as a Reuse link names them. A job that used Enhanced text (it sent an
 * `enhance_record`) ran the Enhanced text as `prompt`: the user's own text, which may be empty, is the `prompt` param,
 * the text that ran is `enhanced_prompt` and the record is `enhance`. Any other job's `prompt` is its text.
 */
function audioJobPrefill(settings: JobSettings): PrefillParams {
  const params: PrefillParams = { prompt: '' };
  for (const key of AUDIO_PREFILL_KEYS) {
    const value = first(settings, key);
    if (value !== undefined) params[key] = value;
  }
  const record = first(settings, 'enhance_record');
  if (hasEnhanceRecord(record)) {
    params.enhanced_prompt = params.prompt as string;
    params.prompt = first(settings, 'prompt_input') ?? '';
    params.enhance = record;
  }
  return params;
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
 * Turn the finishing settings a job (or a Reuse URL) names back into draft state: strength, Sharpen, Contrast, Saturation,
 * Transparent and the creative Upscale. A setting the job did not name is left out, except the Upscale, which reads a missing
 * `upscale` as off. A video job has none of the image-only settings, so only the shared post-processing applies.
 */
export function outputSettingsPatch(settings: JobSettings, isVideo: boolean): Partial<DraftState> {
  const patch: Partial<DraftState> = {};
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
  if (isVideo) return patch;

  const strength = numberOrNull(first(settings, 'image_strength'));
  if (strength !== null) patch.referenceImageStrength = strength;
  // An image job that did not name the field ran without it, like any other unchecked box.
  patch.postprocessTransparent = checked(settings, 'transparent');
  patch.upscaleEnabled = settings.upscale !== undefined;
  const factor = numberOrNull(first(settings, 'upscale'));
  if (factor !== null) patch.upscaleFactor = factor;
  if (patch.upscaleEnabled) {
    patch.upscaleDenoise = numberOrNull(first(settings, 'upscale_denoise'));
    patch.upscaleSteps = numberOrNull(first(settings, 'upscale_steps'));
    patch.upscaleGuidance = numberOrNull(first(settings, 'upscale_guidance'));
    patch.upscaleSharpen = first(settings, 'upscale_sharpen') !== 'false';
  }
  return patch;
}

/**
 * Turn the form fields a job was submitted with back into workspace state, so the form can be refilled from it.
 *
 * An audio job fills the Audio tab's kind, model, text, lyrics, length, tempo, key, voice, delivery, sound type, seed and batch.
 *
 * @param rememberedParts The Rewrite choice the draft remembers for each video workflow; the job's own workflow takes the job's.
 * @param extendStep The step between the lengths the job's model can extend by.
 */
export function jobSettingsPrefill(settings: JobSettings, rememberedParts?: EnhanceSettings['partsByWorkflow'], extendStep: number = EXTEND_FRAME_STEP): JobSettingsPrefill {
  // An audio job fills the Audio tab's boxes like a Reuse does; its tick stays unticked, as nothing records it.
  if (first(settings, 'mode') === 'audio') {
    return { params: audioJobPrefill(settings), patch: {} };
  }
  const params: PrefillParams = {};
  for (const key of PREFILL_KEYS) {
    const value = first(settings, key);
    // A blank LoRA list is kept: the job ran without LoRAs, so loading it clears them.
    if (value !== undefined && (value !== '' || key === 'lora')) params[key] = value;
  }
  // The references the job ran with, in prompt order; none named clears them, as the job had none.
  if (first(settings, 'mode') !== 'video') params.reference_path = values(settings, 'reference_path');

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
  if (settings.scheduler !== undefined) patch.scheduler = first(settings, 'scheduler') || null;
  if (settings.first_sigma !== undefined) patch.firstSigma = numberOrNull(first(settings, 'first_sigma'));
  if (settings.quantize !== undefined) patch.quantize = numberOrNull(first(settings, 'quantize'));
  const enhance = restoredEnhanceSettings(first(settings, 'enhance_settings'), first(settings, 'workflow') as Workflow | undefined);
  if (enhance) {
    const key = enhanceSettingsKey(workflow);
    patch[key] = key === 'enhanceSettingsVideo' && rememberedParts ? { ...enhance, partsByWorkflow: { ...rememberedParts, ...enhance.partsByWorkflow } } : enhance;
  }

  Object.assign(patch, outputSettingsPatch(settings, isVideo));

  if (isVideo) {
    // A job that followed an audio track sent no Audio switch (the track is its sound); the switch keeps its value.
    if (settings.audio !== undefined || settings.audio_track_path === undefined) patch.audio = checked(settings, 'audio');
    patch.lowMemory = checked(settings, 'low_memory');
    patch.videoUpscaleEnabled = settings.video_upscale_factor !== undefined;
    const factor = numberOrNull(first(settings, 'video_upscale_factor'));
    if (factor !== null) patch.videoUpscaleFactor = factor;
    patch.videoUpscaleReview = patch.videoUpscaleEnabled && checked(settings, 'review');
  } else {
    patch.upscaleReview = patch.upscaleEnabled === true && checked(settings, 'review');
  }
  return { params, patch };
}
