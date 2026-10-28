// Pure helpers for settings presets: the key vocabulary, reading a preset off the draft, comparing the two,
// and turning a preset into the draft patch an apply makes.
// The vocabulary mirrors the server's preset keys: Web form field names, `null` meaning "not set".

import { formatLoraString, keepInstalledLoras, parseLoraString } from './loras';
import { workflowMode } from './promptEnhance';
import type { DraftState, PresetMode, PresetSettings, PresetValue, SettingsPreset, WorkspaceContext } from '$lib/types';

const IMAGE_KEYS = [
  'quantize', 'lora', 'ratio', 'size', 'width', 'height', 'runs', 'steps', 'guidance', 'image_strength', 'scheduler', 'first_sigma',
  'sharpen_enabled', 'sharpen_amount', 'contrast_enabled', 'contrast_amount', 'saturation_enabled', 'saturation_amount',
  'upscale', 'upscale_denoise', 'upscale_steps', 'upscale_guidance', 'upscale_sharpen', 'review',
] as const;
const VIDEO_KEYS = ['lora', 'ratio', 'size', 'width', 'height', 'runs', 'steps', 'frames', 'audio', 'low_memory', 'video_upscale_factor', 'review'] as const;

/** The settings every preset of a mode holds, in file order. */
export const PRESET_KEYS: Record<PresetMode, readonly string[]> = { image: IMAGE_KEYS, video: VIDEO_KEYS };

const SIZE_KEYS = ['ratio', 'size', 'width', 'height'] as const;

type DraftKey = keyof DraftState;

// The draft keys each preset key owns; applying a preset writes exactly these (plus the model and the LoRA memory).
const COMMON_DRAFT_KEYS: Record<string, DraftKey[]> = {
  lora: ['loraString'], ratio: ['ratio'], size: ['size', 'dimensionMode'], width: ['width'], height: ['height'], runs: ['runs'], steps: ['steps'],
};
const IMAGE_DRAFT_KEYS: Record<string, DraftKey[]> = {
  ...COMMON_DRAFT_KEYS,
  quantize: ['quantize'], guidance: ['guidance'], image_strength: ['referenceImageStrength'], scheduler: ['scheduler'], first_sigma: ['firstSigma'],
  sharpen_enabled: ['postprocessSharpenEnabled'], sharpen_amount: ['postprocessSharpenAmount'],
  contrast_enabled: ['postprocessContrastEnabled'], contrast_amount: ['postprocessContrastAmount'],
  saturation_enabled: ['postprocessSaturationEnabled'], saturation_amount: ['postprocessSaturationAmount'],
  upscale: ['upscaleEnabled', 'upscaleFactor'], upscale_denoise: ['upscaleDenoise'], upscale_steps: ['upscaleSteps'],
  upscale_guidance: ['upscaleGuidance'], upscale_sharpen: ['upscaleSharpen'], review: ['upscaleReview'],
};
const VIDEO_DRAFT_KEYS: Record<string, DraftKey[]> = {
  ...COMMON_DRAFT_KEYS,
  frames: ['frameCount'], audio: ['audio'], low_memory: ['lowMemory'],
  video_upscale_factor: ['videoUpscaleEnabled', 'videoUpscaleFactor'], review: ['videoUpscaleReview'],
};
const DRAFT_KEYS: Record<PresetMode, Record<string, DraftKey[]>> = { image: IMAGE_DRAFT_KEYS, video: VIDEO_DRAFT_KEYS };

function _mode(state: Pick<DraftState, 'workflow'>): PresetMode {
  return workflowMode(state.workflow);
}

/** Return the draft keys applying a preset of the state's mode may write, apart from `appliedPreset`. */
export function presetDraftKeys(state: Pick<DraftState, 'workflow'>): Set<DraftKey> {
  const owned = PRESET_KEYS[_mode(state)].flatMap((key) => DRAFT_KEYS[_mode(state)][key] ?? []);
  // A preset may change the model, and with it the LoRAs remembered per family and the model remembered for the mode.
  return new Set<DraftKey>([...owned, 'model', 'loraString', 'lorasByFamily', 'lastImageModel', 'lastVideoModel']);
}

/** Return *value* while its feature is on, else null: a setting of an off feature is not saved. */
function _whenOn<T extends number | boolean | null>(enabled: boolean, value: T): T | null {
  return enabled ? value : null;
}

/** Read the model and every setting of the state's mode off the draft, as a preset would hold them. */
export function presetSettingsFromDraft(state: DraftState, ctx: WorkspaceContext): { model: string; settings: PresetSettings } {
  const mode = _mode(state);
  const custom = state.dimensionMode === 'custom';
  const raw: PresetSettings = {
    lora: formatLoraString(parseLoraString(state.loraString, ctx.loras)),
    ratio: state.ratio,
    size: custom ? 'custom' : state.size,
    width: custom ? state.width : null,
    height: custom ? state.height : null,
    runs: state.runs,
    steps: state.steps,
  };
  if (mode === 'video') {
    Object.assign(raw, {
      frames: state.frameCount,
      audio: state.audio,
      low_memory: state.lowMemory,
      video_upscale_factor: state.videoUpscaleEnabled ? state.videoUpscaleFactor : null,
      review: _whenOn(state.videoUpscaleEnabled, state.videoUpscaleReview),
    });
  } else {
    Object.assign(raw, {
      quantize: state.quantize,
      guidance: state.guidance,
      image_strength: state.referenceImageStrength,
      scheduler: state.scheduler,
      first_sigma: state.firstSigma,
      sharpen_enabled: state.postprocessSharpenEnabled,
      sharpen_amount: _whenOn(state.postprocessSharpenEnabled, state.postprocessSharpenAmount),
      contrast_enabled: state.postprocessContrastEnabled,
      contrast_amount: _whenOn(state.postprocessContrastEnabled, state.postprocessContrastAmount),
      saturation_enabled: state.postprocessSaturationEnabled,
      saturation_amount: _whenOn(state.postprocessSaturationEnabled, state.postprocessSaturationAmount),
      upscale: state.upscaleEnabled ? state.upscaleFactor : null,
      upscale_denoise: _whenOn(state.upscaleEnabled, state.upscaleDenoise),
      upscale_steps: _whenOn(state.upscaleEnabled, state.upscaleSteps),
      upscale_guidance: _whenOn(state.upscaleEnabled, state.upscaleGuidance),
      upscale_sharpen: _whenOn(state.upscaleEnabled, state.upscaleSharpen),
      review: _whenOn(state.upscaleEnabled, state.upscaleReview),
    });
  }
  return { model: state.model, settings: Object.fromEntries(PRESET_KEYS[mode].map((key) => [key, raw[key] ?? null])) };
}

/** A value as compared: an unset text or an absent key is null. */
function _plain(value: PresetValue | undefined): PresetValue {
  return value === undefined || value === '' ? null : value;
}

/** The size keys as one unit: a custom size is its width and height, any other the ratio and preset. */
function _sizeUnit(settings: PresetSettings): string {
  return settings.size === 'custom' ? `custom:${_plain(settings.width)}x${_plain(settings.height)}` : `${_plain(settings.ratio)}/${_plain(settings.size)}`;
}

/** Return the settings in comparable form: LoRAs normalized, the size as one unit (`size`), absent keys null. */
function _comparable(settings: PresetSettings, mode: PresetMode, ctx: WorkspaceContext): Record<string, PresetValue> {
  const result: Record<string, PresetValue> = {};
  for (const key of PRESET_KEYS[mode]) {
    if (key === 'lora') result.lora = formatLoraString(parseLoraString(String(settings.lora ?? ''), ctx.loras));
    else if (key === 'size') result.size = _sizeUnit(settings);
    else if (key === 'review') result.review = settings.review === true;
    else if (!(SIZE_KEYS as readonly string[]).includes(key)) result[key] = _plain(settings[key]);
  }
  return result;
}

/** Tell whether settings read off the form (`current`, in *mode*) differ from a preset: its model and every setting, with the size keys as one unit and the LoRA lists normalized. */
export function presetDiffersFrom(preset: SettingsPreset, mode: PresetMode, current: { model: string; settings: PresetSettings }, ctx: WorkspaceContext): boolean {
  if (preset.mode !== mode) return true;
  if (preset.model !== null && preset.model !== current.model) return true;
  const wanted = _comparable(preset.settings, mode, ctx);
  const actual = _comparable(current.settings, mode, ctx);
  return Object.keys(wanted).some((key) => wanted[key] !== actual[key]);
}

/** Tell whether the form differs from a preset. */
export function presetDiffers(preset: SettingsPreset, state: DraftState, ctx: WorkspaceContext): boolean {
  return presetDiffersFrom(preset, _mode(state), presetSettingsFromDraft(state, ctx), ctx);
}

/** Return the applied preset from a list of presets, or null when none is applied, it is gone, or it belongs to the other mode. */
export function appliedPreset(presets: readonly SettingsPreset[], state: Pick<DraftState, 'workflow' | 'appliedPreset'>): SettingsPreset | null {
  const mode = _mode(state);
  const slug = state.appliedPreset ?? null;
  return slug === null ? null : presets.find((preset) => preset.mode === mode && preset.slug === slug) ?? null;
}

/** What applying a preset sets: the model to hydrate, the reuse-URL params, and the draft patch. */
export interface PresetPrefill {
  model: string;
  params: Record<string, string>;
  patch: Partial<DraftState>;
}

/** Add each non-null value of *entries* to a patch; a null value leaves the model default in place. */
function _assign(patch: Partial<DraftState>, entries: [DraftKey, PresetValue][]): void {
  for (const [key, value] of entries) {
    if (value !== null && value !== undefined) Object.assign(patch, { [key]: value });
  }
}

/**
 * Turn a preset into what applying it sets on top of the chosen model's defaults.
 *
 * The model is the preset's when the mode's model list has it, else the current one stays. LoRAs that are not
 * installed are dropped; a preset without LoRAs or quantization sends `''`, so the LoRAs remembered for the family
 * do not come back.
 */
export function presetPrefill(preset: SettingsPreset, state: DraftState, ctx: WorkspaceContext): PresetPrefill {
  const s = preset.settings;
  const models = preset.mode === 'image' ? ctx.image_models : ctx.video_models;
  const model = preset.model !== null && models.some((candidate) => candidate.id === preset.model) ? preset.model : state.model;

  const isSet = (key: string): boolean => s[key] !== null && s[key] !== undefined;
  const params: Record<string, string> = { lora: keepInstalledLoras(String(s.lora ?? ''), ctx.loras) };
  for (const key of ['ratio', 'size', 'width', 'height', 'steps', 'guidance', 'frames']) {
    if (isSet(key)) params[key] = String(s[key]);
  }
  const patch: Partial<DraftState> = {};
  _assign(patch, [['runs', s.runs]]);

  if (preset.mode === 'image') {
    params.quantize = isSet('quantize') ? String(s.quantize) : '';
    // A null scheduler or first sigma is a choice (the model's own), so it is written, unlike a missing toggle.
    patch.scheduler = s.scheduler as string | null;
    patch.firstSigma = s.first_sigma as number | null;
    // A null sharpen amount means "follow the config", which is a choice too.
    patch.postprocessSharpenAmount = s.sharpen_amount as number | null;
    _assign(patch, [
      ['referenceImageStrength', s.image_strength], ['postprocessSharpenEnabled', s.sharpen_enabled],
      ['postprocessContrastEnabled', s.contrast_enabled], ['postprocessContrastAmount', s.contrast_amount],
      ['postprocessSaturationEnabled', s.saturation_enabled], ['postprocessSaturationAmount', s.saturation_amount],
      ['upscaleFactor', s.upscale], ['upscaleSharpen', s.upscale_sharpen],
    ]);
    // With upscale on, a null denoise, step count or guidance means "auto", which is a choice too; with it off they stay at the model's.
    if (isSet('upscale')) {
      patch.upscaleDenoise = s.upscale_denoise as number | null;
      patch.upscaleSteps = s.upscale_steps as number | null;
      patch.upscaleGuidance = s.upscale_guidance as number | null;
    }
    patch.upscaleEnabled = isSet('upscale');
    patch.upscaleReview = isSet('upscale') && s.review === true;
  } else {
    _assign(patch, [['audio', s.audio], ['lowMemory', s.low_memory], ['videoUpscaleFactor', s.video_upscale_factor]]);
    patch.videoUpscaleEnabled = isSet('video_upscale_factor');
    patch.videoUpscaleReview = isSet('video_upscale_factor') && s.review === true;
  }
  return { model, params, patch };
}
