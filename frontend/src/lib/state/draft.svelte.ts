import { AUDIO_FIELD_DEFAULTS, audioModelById, audioPrefill, availableKinds, hydrateAudio } from './audioDraft';
import { extendPrefillFor } from './extend';
import { downloadStore } from './downloads.svelte';
import { heldUploads } from './heldUploads.svelte';
import { isPrefilledTrack, trackOf, trackPatch, trackSet, trackStartPatch, type TrackTab } from './audioTrack';
import { imageSupport, lastSlotPatch, startSlotPatch, type SlotValue, type ImageTab } from './imageSlots';
import { keepInstalledLoras, modelLoraFamily, withoutLora } from './loras';
import { outputSettingsPatch } from './jobSettings';
import { presetDraftKeys, presetPrefill } from './presets';
import { enhanceSettingsKey, isJsonCaption, prettyCaption, reusedEnhance, workflowMode } from './promptEnhance';
import { alignFrames, defaultExtendFrames, EXTEND_FRAME_STEP, snapExtendFrames } from './videoDuration';
import { isAudioWorkflow, navTab, videoKindOf, workflowFor, workspaceTab, type MediaTab } from './workspaceTab';
import { voiceConsent } from './voiceConsent.svelte';
import { pickScopes, RESET_SCOPE, REUSE_SCOPE, type Scope } from './resetScope';
import { firstValues, listOf } from '$lib/prefillParams';
import type { NavTab, VideoKind, TabText, LoraInfo, DraftState, ExtendSourceInfo, FrameRule, PrefillParams, SettingsPreset, WorkspacePrefill, WorkspaceContext, AudioWorkflow, Workflow, WorkspaceWorkflow, ImageModelDefaults, VideoModelDefaults } from '$lib/types';

const STORAGE_KEY = 'ziv-workspace-draft-v1';
const SCHEMA_VERSION = 3;
/** Highest sharpen amount the backend accepts (`MAX_SHARPEN_AMOUNT` in utils/config.py). */
export const MAX_SHARPEN_AMOUNT = 1.5;

function _canonicalWorkflow(raw: string, ctx: WorkspaceContext): WorkspaceWorkflow | null {
  const workflowValue = raw.trim();
  return ctx.workflow_contract.values.find((workflow) => workflow === workflowValue) ?? null;
}

const DEFAULT_DRAFT: DraftState = {
  workflow: 'txt2img',
  promptSource: 'inline',
  prompt: '',
  scene: '',
  jsonPromptEnabled: false,
  jsonPrompt: '',
  firstSigma: null,
  negativePrompt: '',
  promptFilePath: null,
  promptFileOptionIds: [],
  promptFileLoaded: null,
  model: '',
  ratio: '',
  dimensionMode: 'ratio',
  size: '',
  steps: 0,
  guidance: 0,
  width: 0,
  height: 0,
  runs: 1,
  seed: null,
  loraString: '',
  lorasByFamily: {},
  lastImageModel: '',
  lastVideoModel: '',
  lastImageQuantize: null,
  settingsModel: '',
  referenceImagePath: null,
  referenceImageUpload: null,
  firstFramePath: null,
  firstFrameUpload: null,
  referenceImagePaths: [],
  missingReferencePaths: [],
  endImagePath: null,
  endImageUpload: null,
  audioTrackPath: null,
  audioTrackUpload: null,
  audioTrackStart: 0,
  extendAudioTrackPath: null,
  extendAudioTrackUpload: null,
  extendAudioTrackStart: 0,
  referenceImageStrength: 0,
  frameCount: 0,
  fps: 24,
  audio: false,
  lowMemory: false,
  upscaleEnabled: false,
  upscaleFactor: 2,
  quantize: null,
  historyCollapsed: false,
  sidebarCollapsed: false,
  lastGeneratedAt: null,
  version: SCHEMA_VERSION,
  scheduler: null,
  postprocessSharpenEnabled: true,
  postprocessSharpenAmount: null,
  postprocessContrastEnabled: false,
  postprocessContrastAmount: 1.0,
  postprocessSaturationEnabled: false,
  postprocessSaturationAmount: 1.0,
  postprocessTransparent: false,
  upscaleDenoise: null,
  upscaleSteps: null,
  upscaleGuidance: null,
  upscaleSharpen: true,
  upscaleReview: false,
  videoUpscaleEnabled: false,
  videoUpscaleFactor: 2,
  videoUpscaleReview: false,
  enhancedPrompt: '',
  enhancedScene: '',
  enhancedFrom: null,
  enhancedHistory: [],
  savedText: {},
  enhanceSettings: null,
  enhanceSettingsVideo: null,
  enhanceAuto: false,
  enhanceModel: 'fast',
  enhanceFormat: 'text',
  extendSourcePath: null,
  extendFrames: null,
  missingSourceVideo: false,
  extendPrefill: null,
  appliedPreset: null,
  videoKind: 'new',
  audioKind: 'music',
  audioModels: { music: '', speech: '', sound: '' },
  ...AUDIO_FIELD_DEFAULTS,
};

const _URL_PARAM_CONTROL_IDS: Partial<Record<string, string>> = {
  prompt: 'prompt_inline',
  scene: 'prompt_scene',
  enhanced_scene: 'prompt_scene',
  enhanced_prompt: 'prompt_inline',
  model: 'model',
  steps: 'steps',
  guidance: 'guidance',
  seed: 'seed',
  ratio: 'ratio',
  size: 'size',
  width: 'custom_dimensions',
  height: 'custom_dimensions',
  frames: 'frame_count',
  lora: 'loras',
  quantize: 'quantize',
  source_video: 'source_video',
  extend_frames: 'extend_length',
  image_path: 'reference_image_path',
  end_image_path: 'end_image_path',
  audio_track_path: 'audio_track_path',
  reference_path: 'references',
  negative_prompt: 'negative_prompt',
  scheduler: 'scheduler',
  image_strength: 'image_strength',
  transparent: 'postprocess_transparent',
  sharpen_enabled: 'postprocess_sharpen',
  sharpen_amount: 'postprocess_sharpen',
  contrast_enabled: 'postprocess_contrast',
  contrast_amount: 'postprocess_contrast',
  saturation_enabled: 'postprocess_saturation',
  saturation_amount: 'postprocess_saturation',
  upscale: 'image_upscale_enabled',
  upscale_denoise: 'image_upscale_denoise',
  upscale_steps: 'image_upscale_steps',
  upscale_guidance: 'image_upscale_guidance',
  upscale_sharpen: 'image_upscale_sharpen',
};

/** The Reuse URL param each finishing-setting draft key comes from, so a hidden control's value is skipped. */
const _OUTPUT_KEY_PARAMS: Partial<Record<keyof DraftState, string>> = {
  referenceImageStrength: 'image_strength',
  postprocessTransparent: 'transparent',
  postprocessSharpenEnabled: 'sharpen_enabled',
  postprocessSharpenAmount: 'sharpen_amount',
  postprocessContrastEnabled: 'contrast_enabled',
  postprocessContrastAmount: 'contrast_amount',
  postprocessSaturationEnabled: 'saturation_enabled',
  postprocessSaturationAmount: 'saturation_amount',
  upscaleEnabled: 'upscale',
  upscaleFactor: 'upscale',
  upscaleDenoise: 'upscale_denoise',
  upscaleSteps: 'upscale_steps',
  upscaleGuidance: 'upscale_guidance',
  upscaleSharpen: 'upscale_sharpen',
};

const _CLEAR_FIELD_STATE_UPDATES: Partial<Record<string, Partial<DraftState>>> = {
  negative_prompt: { negativePrompt: '' },
  guidance: { guidance: DEFAULT_DRAFT.guidance },
  quantize: { quantize: null },
  frames: { frameCount: DEFAULT_DRAFT.frameCount },
  audio: { audio: DEFAULT_DRAFT.audio },
  low_memory: { lowMemory: DEFAULT_DRAFT.lowMemory },
  upscale: { upscaleEnabled: false, upscaleFactor: DEFAULT_DRAFT.upscaleFactor },
  sharpen_enabled: { postprocessSharpenEnabled: false },
  sharpen_amount: { postprocessSharpenAmount: null },
  contrast_enabled: { postprocessContrastEnabled: false },
  contrast_amount: { postprocessContrastAmount: 1.0 },
  saturation_enabled: { postprocessSaturationEnabled: false },
  saturation_amount: { postprocessSaturationAmount: 1.0 },
  upscale_denoise: { upscaleDenoise: null },
  upscale_steps: { upscaleSteps: null },
  upscale_guidance: { upscaleGuidance: null },
  upscale_sharpen: { upscaleSharpen: true },
  upscale_save_pre: {},
  transparent: { postprocessTransparent: false },
};

/**
 * Return the state with the workflow its tab and images call for. The tab stays; whether it starts from text or from an
 * image follows from the tab's start image and whether the model takes one.
 */
function _withDerivedWorkflow(ctx: WorkspaceContext | null, state: DraftState): DraftState {
  // The Audio tab's workflow is its kind, whatever the images are.
  if (isAudioWorkflow(state.workflow)) return state;
  const tab = workspaceTab(state.workflow);
  const workflow = workflowFor(tab, state, imageSupport(ctx, state.model, tab === 'video' ? 'video' : 'image').start);
  return workflow === state.workflow ? state : { ...state, workflow };
}

/** The draft fields that hold a tab's own text: what is typed and the Enhanced text made from it. */
const TEXT_KEYS = [
  'prompt', 'scene', 'negativePrompt', 'jsonPromptEnabled', 'jsonPrompt', 'enhancedPrompt', 'enhancedScene', 'enhancedFrom', 'enhancedHistory',
] as const satisfies readonly (keyof TabText)[];

const EMPTY_TEXT: TabText = {
  prompt: '', scene: '', negativePrompt: '', jsonPromptEnabled: false, jsonPrompt: '', enhancedPrompt: '', enhancedScene: '', enhancedFrom: null, enhancedHistory: [],
};

function _textOf(state: DraftState): TabText {
  return Object.fromEntries(TEXT_KEYS.map((key) => [key, state[key]])) as unknown as TabText;
}

/**
 * Return the state with the text of the mode a workflow belongs to. Image and Video each keep their own: moving to the other
 * mode saves the open text under the mode being left and brings back what the entered one saved (nothing, for a mode not
 * visited yet). Video's two kinds share one text, so moving between them changes nothing.
 */
function _withTextOfMode(state: DraftState, workflow: WorkspaceWorkflow): DraftState {
  const from = workflowMode(state.workflow);
  const to = workflowMode(workflow);
  if (from === to) return state;
  const { [to]: entering, ...rest } = state.savedText;
  return { ...state, ...(entering ?? EMPTY_TEXT), savedText: { ...rest, [from]: _textOf(state) } };
}

/** The draft fields that hold an image path: one per slot, and the reference rows. */
const SLOT_KEYS = ['referenceImagePath', 'firstFramePath', 'endImagePath', 'referenceImagePaths'] as const satisfies readonly (keyof DraftState)[];

/** The paths the state's image slots hold (the start image, the frames and the reference rows); a slot can hold the empty path of an upload. */
export function slotPaths(state: Pick<DraftState, (typeof SLOT_KEYS)[number]>, keys: readonly (typeof SLOT_KEYS)[number][] = SLOT_KEYS): string[] {
  return keys.flatMap((key) => {
    const value = state[key];
    return Array.isArray(value) ? value : value ? [value] : [];
  });
}

/** The state's missing-file flags without the one for a replaced path, unless another slot still holds that path. */
function _withoutFlag(state: DraftState, replaced: string | null): string[] {
  if (!replaced || slotPaths(state).includes(replaced)) return state.missingReferencePaths;
  return state.missingReferencePaths.filter((path) => path !== replaced);
}

/** The missing-file flags of the paths that slots a prefill leaves alone still hold. */
function _flagsOfUntouchedSlots(state: DraftState, prefill: Partial<DraftState>): string[] {
  const kept = new Set(slotPaths(state, SLOT_KEYS.filter((key) => !(key in prefill))));
  return state.missingReferencePaths.filter((path) => kept.has(path));
}

function _applyImageDefaults(state: DraftState, defaults: ImageModelDefaults): DraftState {
  const pp = defaults.postprocess;
  const uu = defaults.upscale;
  return {
    ...state,
    ratio: defaults.ratio,
    dimensionMode: 'ratio',
    size: defaults.size,
    steps: defaults.steps,
    guidance: defaults.guidance,
    width: defaults.width,
    height: defaults.height,
    referenceImageStrength: defaults.image_strength,
    scheduler: defaults.scheduler,
    postprocessSharpenEnabled: pp.sharpen !== false,
    postprocessSharpenAmount: typeof pp.sharpen === 'number' ? pp.sharpen : null,
    postprocessContrastEnabled: pp.contrast !== false,
    postprocessContrastAmount: typeof pp.contrast === 'number' ? pp.contrast : 1.0,
    postprocessSaturationEnabled: pp.saturation !== false,
    postprocessSaturationAmount: typeof pp.saturation === 'number' ? pp.saturation : 1.0,
    upscaleEnabled: uu.enabled,
    upscaleFactor: uu.factor ?? 2,
    upscaleDenoise: uu.denoise,
    upscaleSteps: uu.steps,
    upscaleGuidance: uu.guidance,
    upscaleSharpen: uu.sharpen,
  };
}

function _applyVideoDefaults(state: DraftState, defaults: VideoModelDefaults): DraftState {
  return {
    ...state,
    ratio: defaults.ratio,
    dimensionMode: 'ratio',
    size: defaults.size,
    steps: defaults.steps,
    width: defaults.width,
    height: defaults.height,
    // A configured default may be off the model's grid; align it so Reset matches what the server generates.
    frameCount: alignFrames(defaults.frame_count, _frameRuleOf(defaults)),
    audio: defaults.audio,
    lowMemory: defaults.low_memory,
    videoUpscaleEnabled: defaults.upscale.enabled,
    videoUpscaleFactor: defaults.upscale.factor ?? 2,
  };
}

function _visibleControlsForWorkflow(ctx: WorkspaceContext, workflow: WorkspaceWorkflow): Set<string> {
  return new Set(ctx.workflow_contract.definitions[workflow]?.visible_controls ?? []);
}

function _applyClearFields(state: DraftState, clearFields: string[]): DraftState {
  let nextState = state;
  for (const clearField of clearFields) {
    const updates = _CLEAR_FIELD_STATE_UPDATES[clearField];
    if (updates) {
      nextState = { ...nextState, ...updates };
    }
  }
  return nextState;
}

/** Draft keys the Settings pane edits; each can be compared with and reset to its model default. */
export const SETTING_KEYS = [
  'ratio', 'dimensionMode', 'size', 'width', 'height', 'runs', 'frameCount', 'steps', 'guidance', 'referenceImageStrength',
  'seed', 'scheduler', 'firstSigma',
  'postprocessSharpenEnabled', 'postprocessSharpenAmount', 'postprocessContrastEnabled', 'postprocessContrastAmount',
  'postprocessSaturationEnabled', 'postprocessSaturationAmount', 'postprocessTransparent',
  'upscaleEnabled', 'upscaleFactor', 'upscaleDenoise', 'upscaleSteps', 'upscaleGuidance', 'upscaleSharpen', 'upscaleReview',
  'audio', 'lowMemory', 'videoUpscaleEnabled', 'videoUpscaleFactor', 'videoUpscaleReview', 'audioSeconds',
] as const satisfies readonly (keyof DraftState)[];

export type SettingKey = (typeof SETTING_KEYS)[number];
export type SettingDefaults = Pick<DraftState, SettingKey>;

function _pickSettings(state: DraftState): SettingDefaults {
  return Object.fromEntries(SETTING_KEYS.map((key) => [key, state[key]])) as SettingDefaults;
}

function _isVideoWorkflow(workflow: WorkspaceWorkflow): boolean {
  return workflowMode(workflow) === 'video';
}

/**
 * Remember the state's model as the last one used in its own mode (image or video), whichever workflow is open, and as the one the
 * settings belong to. An image model also remembers its quantization, which the video workflows clear.
 */
function _rememberModel(ctx: WorkspaceContext, state: DraftState): DraftState {
  if (ctx.video_models.some((model) => model.id === state.model)) return { ...state, lastVideoModel: state.model, settingsModel: state.model };
  if (ctx.image_models.some((model) => model.id === state.model)) return { ...state, lastImageModel: state.model, lastImageQuantize: state.quantize, settingsModel: state.model };
  return state;
}

/** Return the backend defaults for a model in a workflow's mode, or null when none are known. */
function _modelDefaultsFor(ctx: WorkspaceContext, workflow: WorkspaceWorkflow, model: string): ImageModelDefaults | VideoModelDefaults | null {
  const isVideoMode = _isVideoWorkflow(workflow);
  const defaultsMap = isVideoMode ? ctx.video_model_defaults : ctx.image_model_defaults;
  const fallback = isVideoMode ? ctx.video_defaults : ctx.defaults;
  const currentModel = isVideoMode ? ctx.current_video_model : ctx.current_image_model;
  return (defaultsMap?.[model] ?? (model === currentModel ? fallback : null)) as ImageModelDefaults | VideoModelDefaults | null;
}

/** Return the key a model's LoRAs are remembered under: `family:size` when the model has a size, else its family, else the model itself. */
function _loraFamilyKey(ctx: WorkspaceContext, model: string): string {
  const found = modelLoraFamily(ctx, model);
  if (!found?.family) return `model:${model}`;
  return found.sizeVariant ? `${found.family}:${found.sizeVariant}` : found.family;
}

/**
 * Carry LoRAs across a model change: a new family gets back the LoRAs last chosen for it (or none), and the old
 * family's are remembered. Within one family the LoRAs stay. Remembered LoRAs that were since removed are dropped.
 *
 * The Audio tab has no LoRAs: entering it remembers the family it comes from and holds none, and leaving it brings back
 * the LoRAs of the family it goes to, so a visit to Audio never changes what Image and Video remember.
 */
function _swapFamilyLoras(ctx: WorkspaceContext, state: DraftState, fromModel: string, toModel: string): Partial<DraftState> {
  if (!fromModel || !toModel) return {};
  const fromAudio = audioModelById(ctx, fromModel) !== null;
  const toAudio = audioModelById(ctx, toModel) !== null;
  if (fromAudio && toAudio) return {};
  const fromKey = _loraFamilyKey(ctx, fromModel);
  const toKey = _loraFamilyKey(ctx, toModel);
  if (fromKey === toKey) return {};
  const lorasByFamily = fromAudio ? state.lorasByFamily : { ...state.lorasByFamily, [fromKey]: state.loraString };
  return {
    lorasByFamily,
    loraString: toAudio ? '' : keepInstalledLoras(state.lorasByFamily[toKey] ?? '', ctx.loras, downloadStore.loraNames()),
  };
}

/** The frame counts a video model accepts: its family's rule, else any count. */
function _frameRuleOf(defaults: VideoModelDefaults | undefined): FrameRule {
  return defaults?.frame_rule ?? { alignment: 1, offset: 1, round: 'nearest' };
}

/** Return a video model's defaults, falling back to the default video model's. */
function _videoDefaultsOf(ctx: WorkspaceContext, model: string): VideoModelDefaults | undefined {
  return ctx.video_model_defaults?.[model] ?? ctx.video_defaults;
}

/**
 * Return a video model's frame rate and frame rule, falling back to the default video model's.
 *
 * @param sourceFps An extension's frame rate is its source's: when given, it replaces the model's.
 */
export function videoTimingFor(ctx: WorkspaceContext, model: string, sourceFps: number | null = null): { fps: number; frameRule: FrameRule } {
  const defaults = _videoDefaultsOf(ctx, model);
  return { fps: sourceFps ?? defaults?.fps ?? 24, frameRule: _frameRuleOf(defaults) };
}

/** Return the step between the lengths a video model can extend by (8 when it does not say). */
export function extendStepFor(ctx: WorkspaceContext, model: string): number {
  return _videoDefaultsOf(ctx, model)?.extend_frame_step ?? EXTEND_FRAME_STEP;
}

/** Return the only step counts a video model offers, or null when it takes any count. */
export function stepOptionsFor(ctx: WorkspaceContext, model: string): number[] | null {
  const options = _videoDefaultsOf(ctx, model)?.step_options;
  return options && options.length > 0 ? options : null;
}

/** Move a step count to the nearest offered one (the higher on a tie), as the server does. */
export function snapSteps(steps: number, options: number[]): number {
  return options.reduce((best, option) => {
    const distance = Math.abs(option - steps);
    const bestDistance = Math.abs(best - steps);
    return distance < bestDistance || (distance === bestDistance && option > best) ? option : best;
  });
}

/** Apply a workflow's cleared fields and then the model's defaults to a state. */
function _withModelDefaults(ctx: WorkspaceContext, state: DraftState): DraftState {
  const clearFields = ctx.workflow_contract.definitions[state.workflow]?.clear_fields ?? [];
  const cleared = _applyClearFields(state, clearFields);
  const modelDefaults = _modelDefaultsFor(ctx, state.workflow, state.model);
  if (!modelDefaults) return cleared;
  return _isVideoWorkflow(state.workflow)
    ? _applyVideoDefaults(cleared, modelDefaults as VideoModelDefaults)
    : _applyImageDefaults(cleared, modelDefaults as ImageModelDefaults);
}

/** Return a fresh draft for a workflow and model with that model's defaults applied. */
function _defaultState(ctx: WorkspaceContext, workflow: WorkspaceWorkflow, model: string): DraftState {
  return _withModelDefaults(ctx, { ...DEFAULT_DRAFT, workflow, model });
}

/**
 * Return the default value of every setting for the state's workflow and model.
 *
 * Reads `defaultsFor`, so it resolves exactly as a Reset does, without touching the draft.
 */
export function settingDefaultsFor(ctx: WorkspaceContext, state: Pick<DraftState, 'workflow' | 'model'>): SettingDefaults {
  return _pickSettings(defaultsFor(ctx, { ...DEFAULT_DRAFT, ...state }, null));
}

/** Return the size presets offered for a ratio; image presets beyond the model's dimension limit are left out. */
export function offeredSizes(ctx: WorkspaceContext, workflow: WorkspaceWorkflow, model: string, ratio: string): string[] {
  if (_isVideoWorkflow(workflow)) return ctx.video_size_options[ratio] ?? [];
  const options = ctx.image_size_options[ratio] ?? [];
  const max = (_modelDefaultsFor(ctx, workflow, model) as ImageModelDefaults | null)?.dimension_max ?? null;
  if (max === null) return options;
  const dims = ctx.image_size_dimensions[ratio] ?? {};
  return options.filter((size) => !dims[size] || (dims[size][0] <= max && dims[size][1] <= max));
}

/** Whether a stored size can still be submitted: a custom size, or a preset the model offers. */
function _presetIsValid(ctx: WorkspaceContext, state: DraftState): boolean {
  // A custom size stands on its own width and height; its ratio/size labels are not submitted.
  if (state.dimensionMode === 'custom') return state.width > 0 && state.height > 0;
  return offeredSizes(ctx, state.workflow, state.model, state.ratio).includes(state.size);
}

const _SIZE_KEYS = ['ratio', 'size', 'dimensionMode', 'width', 'height'] as const;

/**
 * Return the settings of `current` that differ from `baseline` (compared with `Object.is`).
 *
 * The size settings (ratio, size, dimension mode, width, height) are kept or dropped as a whole,
 * so a changed width never pairs with another model's height.
 */
export function changedSettings(current: SettingDefaults, baseline: SettingDefaults): Partial<SettingDefaults> {
  const changed = Object.fromEntries(SETTING_KEYS.filter((key) => !Object.is(current[key], baseline[key])).map((key) => [key, current[key]])) as Partial<SettingDefaults>;
  if (_SIZE_KEYS.some((key) => key in changed)) Object.assign(changed, Object.fromEntries(_SIZE_KEYS.map((key) => [key, current[key]])));
  return changed;
}

/** The pixel limits of a model: its smallest side, its largest side (null when unbounded), and the grid sides snap to. */
export interface DimensionLimits { min: number; max: number | null; step: number }

/** Snap a side length to the nearest multiple of the step, within the smallest and largest multiples the limits allow. */
export function snapDimension(value: number, limits: DimensionLimits): number {
  const { min, max, step } = limits;
  const lowest = Math.ceil(min / step) * step;
  const highest = max === null ? Infinity : Math.floor(max / step) * step;
  return Math.min(highest, Math.max(lowest, Math.round(value / step) * step));
}

/** Scale a custom size into a model's limits keeping its aspect ratio, then snap each side to the step. */
export function fitDimensions(width: number, height: number, limits: DimensionLimits): { width: number; height: number } {
  const larger = Math.max(width, height);
  const smaller = Math.min(width, height);
  let scale = 1;
  if (limits.max !== null && larger > limits.max) scale = limits.max / larger;
  else if (smaller < limits.min) scale = limits.min / smaller;
  return { width: snapDimension(width * scale, limits), height: snapDimension(height * scale, limits) };
}

/** Return the preset of a ratio with the most pixels; presets without known pixel sizes lose to ones with, and the last listed wins a tie. */
function _largestPreset(ctx: WorkspaceContext, workflow: WorkspaceWorkflow, ratio: string, presets: string[]): string {
  const dims = (_isVideoWorkflow(workflow) ? ctx.video_size_dimensions : ctx.image_size_dimensions)?.[ratio] ?? {};
  const area = (size: string): number => (dims[size] ? dims[size][0] * dims[size][1] : 0);
  return presets.reduce((best, size) => (area(size) >= area(best) ? size : best));
}

/** Fit the size of a state to its model: a preset the model does not offer becomes its largest one, a custom size is scaled into the model's limits. */
function _fitSizeToModel(ctx: WorkspaceContext, state: DraftState, defaults: ImageModelDefaults | VideoModelDefaults): DraftState {
  if (state.dimensionMode === 'custom') {
    if (_isVideoWorkflow(state.workflow) || state.width <= 0 || state.height <= 0) return state;
    const image = defaults as ImageModelDefaults;
    return { ...state, ...fitDimensions(state.width, state.height, { min: image.dimension_min, max: image.dimension_max, step: image.dimension_step }) };
  }
  const offered = offeredSizes(ctx, state.workflow, state.model, state.ratio);
  if (offered.includes(state.size)) return state;
  if (offered.length > 0) return { ...state, size: _largestPreset(ctx, state.workflow, state.ratio, offered) };
  return { ...state, ..._pickSizeSettings(defaults) };
}

function _pickSizeSettings(defaults: ImageModelDefaults | VideoModelDefaults): Pick<DraftState, (typeof _SIZE_KEYS)[number]> {
  return { ratio: defaults.ratio, dimensionMode: 'ratio', size: defaults.size, width: defaults.width, height: defaults.height };
}

/** Return the settings of a draft that differ from its own model's (gated) defaults; none when those defaults are unknown. */
function _changedSinceOldDefaults(ctx: WorkspaceContext, state: DraftState): Partial<SettingDefaults> {
  if (!_modelDefaultsFor(ctx, state.workflow, state.model)) return {};
  const baseline = _applyCapabilityGates(ctx, _defaultState(ctx, state.workflow, state.model));
  return changedSettings(_pickSettings(state), _pickSettings(baseline));
}

/** Fit the kept settings of a state to its model: the size, and the steps of a video model with a step cap. */
function _fitToModel(ctx: WorkspaceContext, state: DraftState): DraftState {
  const defaults = _modelDefaultsFor(ctx, state.workflow, state.model);
  if (!defaults) return state;
  const fitted = _fitSizeToModel(ctx, state, defaults);
  const maxSteps = _isVideoWorkflow(state.workflow) ? (defaults as VideoModelDefaults).max_steps : null;
  return typeof maxSteps === 'number' ? { ...fitted, steps: Math.min(fitted.steps, maxSteps) } : fitted;
}

/**
 * Clear what the state's model cannot use, and align the frame count to its grid.
 *
 * @param sourceFps An extension runs at its source's frame rate, which the draft's store of it holds.
 */
function _applyCapabilityGates(ctx: WorkspaceContext, state: DraftState, sourceFps: number | null = _sourceFps(state.workflow)): DraftState {
  const { workflow, model } = state;
  // Nothing of an image or a video applies to a kind of audio.
  if (isAudioWorkflow(workflow)) return state;
  const isVideoMode = _isVideoWorkflow(workflow);
  const modelDefaults = _modelDefaultsFor(ctx, workflow, model);
  let nextState = state;

  // An Enhanced JSON caption only means something to a model that takes JSON captions.
  const imageTakesJson = !isVideoMode && (modelDefaults as ImageModelDefaults | null)?.supports_json_prompt === true;
  if (!imageTakesJson && isJsonCaption(nextState.enhancedPrompt)) nextState = { ...nextState, enhancedPrompt: '', enhancedFrom: null };
  if (!imageTakesJson && nextState.enhancedHistory.some((entry) => isJsonCaption(entry.prompt))) {
    nextState = { ...nextState, enhancedHistory: nextState.enhancedHistory.filter((entry) => !isJsonCaption(entry.prompt)) };
  }
  if (isVideoMode) {
    nextState = { ...nextState, negativePrompt: '', quantize: null, jsonPromptEnabled: false, jsonPrompt: '', firstSigma: null };
    // A kept or restored count may predate snapping; show the count the server will generate.
    const timing = videoTimingFor(ctx, model, sourceFps);
    nextState.fps = timing.fps;
    // An extension has no clip length of its own: its new frames (`extendFrames`) stay multiples of the model's step, not its frame rule.
    if (workflow !== 'extend') nextState.frameCount = alignFrames(nextState.frameCount, timing.frameRule);
    // Settings of controls the model does not have are cleared, and the steps move to one it offers.
    const video = modelDefaults as VideoModelDefaults | null;
    const hidden = new Set(video?.hidden_controls ?? []);
    if (hidden.has('video_upscale_enabled')) nextState.videoUpscaleEnabled = false;
    if (hidden.has('video_upscale_review')) nextState.videoUpscaleReview = false;
    if (hidden.has('low_memory') && video) nextState.lowMemory = video.low_memory;
    if (video?.step_options?.length) nextState.steps = snapSteps(nextState.steps, video.step_options);
    if (workflow === 'extend' && nextState.extendFrames !== null) nextState.extendFrames = snapExtendFrames(nextState.extendFrames, extendStepFor(ctx, model));
  } else {
    const imageDefaults = modelDefaults as ImageModelDefaults | null;
    nextState = {
      ...nextState,
      negativePrompt: imageDefaults?.supports_negative_prompt ? nextState.negativePrompt : '',
      quantize: imageDefaults?.supports_quantize ? nextState.quantize : null,
      jsonPromptEnabled: imageDefaults?.supports_json_prompt ? nextState.jsonPromptEnabled : false,
      jsonPrompt: imageDefaults?.supports_json_prompt ? nextState.jsonPrompt : '',
      firstSigma: imageDefaults?.supports_first_sigma ? nextState.firstSigma : null,
      scheduler: imageDefaults?.supports_scheduler === false ? null : nextState.scheduler,
    };
  }
  return nextState;
}

/** The workflow each media tab falls to without a start image. */
const _TEXT_WORKFLOW: Record<MediaTab, Workflow> = { image: 'txt2img', video: 'txt2vid', extend: 'extend' };

/**
 * Return the defaults of every field for the tab the state has open, resolved in three layers: the static defaults, then the
 * model's (with the workflow's cleared fields and the capability gates), then the values derived from what the draft holds
 * (the Extend length from its source's frame rate, the Extend track the source brought, a sound model's default length).
 * Only the fields of the open tab's scopes are meaningful; a Reset picks them with `RESET_SCOPE`.
 *
 * @param extendSource What the server reported about the chosen extend source, or null.
 */
export function defaultsFor(ctx: WorkspaceContext, state: DraftState, extendSource: ExtendSourceInfo | null): DraftState {
  const tab = workspaceTab(state.workflow);
  if (tab === 'audio') {
    const { workflow, model, audioKind, audioModels } = state;
    return hydrateAudio(ctx, { ...DEFAULT_DRAFT, workflow, model, audioKind, audioModels });
  }
  const sourceFps = extendSource?.fps ?? null;
  const modelled = _withModelDefaults(ctx, { ...DEFAULT_DRAFT, promptSource: ctx.default_prompt_source, workflow: _TEXT_WORKFLOW[tab], model: state.model });
  const defaults = _applyCapabilityGates(ctx, modelled, tab === 'extend' ? sourceFps : null);
  if (tab !== 'extend') return defaults;
  // The source stays across a Reset, so the track it filled in comes back with it, from where the source's sound ended.
  const sourceTrack = state.extendPrefill?.path === state.extendSourcePath ? (state.extendPrefill?.audioTrack ?? null) : null;
  return {
    ...defaults,
    extendFrames: sourceFps ? defaultExtendFrames(sourceFps, extendStepFor(ctx, state.model)) : state.extendFrames,
    ...(sourceTrack !== null ? trackPatch('extend', { path: sourceTrack.path }, sourceTrack.start) : {}),
  };
}

function loadFromStorage(): DraftState {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return { ...DEFAULT_DRAFT };
    const parsed = JSON.parse(raw) as Partial<DraftState>;
    if (parsed.version !== SCHEMA_VERSION) return { ...DEFAULT_DRAFT };
    return { ...DEFAULT_DRAFT, ...parsed };
  } catch {
    return { ...DEFAULT_DRAFT };
  }
}

/** The uploads held for the tabs a transition resets are forgotten with their fields; a voice takes its consent tick with it. */
function _clearHeldUploads(scopes: readonly Scope[]): void {
  if (scopes.includes('slotsImage')) heldUploads.clearTab('image');
  if (scopes.includes('slotsVideo')) heldUploads.clearTab('video');
  if (scopes.includes('slotsExtend')) heldUploads.clearTab('extend');
  if (scopes.includes('voice')) {
    heldUploads.clearTab('audio');
    voiceConsent.reset();
  }
}

let _draft = $state<DraftState>(loadFromStorage());
let _authorityReady = $state(false);
// What the server last reported about the extend source; fetched again on load, so it is not saved with the draft.
let _extendSource = $state<ExtendSourceInfo | null>(null);

/** The frame rate an extension runs at: its source's, once known. */
function _sourceFps(workflow: WorkspaceWorkflow): number | null {
  return workflow === 'extend' ? _extendSource?.fps ?? null : null;
}

export const draft = {
  get state(): DraftState { return _draft; },
  get authorityReady(): boolean { return _authorityReady; },
  /** What the server reported about the chosen extend source; null before it answers or without a source. */
  get extendSource(): ExtendSourceInfo | null { return _extendSource; },

  loadDraft(): void {
    _draft = loadFromStorage();
    _authorityReady = false;
    _extendSource = null;
  },

  /**
   * Apply URL query/hash params to the draft using canonical workflow values only.
   * Only fields present in params are updated; absent fields are left unchanged. A param whose control the workflow
   * does not show is skipped, unless `ungated` is set.
   *
   * With `exact` (a Reuse, a Load settings; needs a context, and not a Use as), the settings are the take's exactly: the
   * scopes of `REUSE_SCOPE` for the tab first take their defaults, so what the params omit is the default, not what
   * the form held, and the size is then fitted to the model. A Use as, a link and a preset apply only what they name.
   *
   * The images go to the slots of the tab the params open (`image_path` is the start image in Image and the first frame in
   * Video). With `replaceSlots`, every slot of that tab is set and one the params do not name is emptied; a Use as
   * (`use_as`) sets only its slot. With a context, the workflow is derived from the images at the end.
   */
  loadFromUrl(input: PrefillParams, ctx: WorkspaceContext | null = null, options: { ungated?: boolean; replaceSlots?: boolean; exact?: boolean } = {}): void {
    // A name that repeats (reference_path) arrives as a list; every other parameter reads as its first value.
    const params = firstValues(input);
    const lists = Object.fromEntries(Object.entries(input).map(([key, value]) => [key, listOf(value)]));
    const prefill: WorkspacePrefill = {};
    let workflow = _draft.workflow;
    const exact = ctx !== null && options.exact === true && params.use_as === undefined;
    if (params.workflow && ctx) {
      const canonical = _canonicalWorkflow(params.workflow, ctx);
      if (canonical) {
        prefill.workflow = canonical;
        workflow = canonical;
      }
    }

    // The Audio tab has its own fields: its kind opens, and a Use as → Voice or a Reuse sets only what it names.
    if (isAudioWorkflow(workflow)) {
      const { patch, voice } = audioPrefill(params, ctx, workflow, options.ungated === true);
      // The tick follows the voice: a prefill that replaces the voice leaves it for the person to give, and one that does not
      // (a song or a sound) leaves the voice and its tick alone.
      if (voice) voiceConsent.reset(voice.exempt);
      const opened: DraftState = { ..._withTextOfMode(_draft, workflow), ...(exact ? pickScopes(DEFAULT_DRAFT, REUSE_SCOPE.audio) : {}), ...prefill, ...patch, workflow, audioKind: workflow };
      const hydrated = ctx ? hydrateAudio(ctx, opened) : opened;
      _draft = ctx ? { ...hydrated, ..._swapFamilyLoras(ctx, _draft, _draft.model, hydrated.model) } : hydrated;
      return;
    }

    const visibleControls = ctx ? _visibleControlsForWorkflow(ctx, workflow) : null;
    const canApply = (paramKey: string): boolean => {
      const controlId = _URL_PARAM_CONTROL_IDS[paramKey];
      return options.ungated === true || !controlId || visibleControls === null || visibleControls.has(controlId);
    };

    // An extension names its source: the URL then describes the whole job, so its text replaces whatever the boxes held.
    const extending = workflow === 'extend' && canApply('source_video') && Boolean(params.source_video || params.missing_source_video);
    // Reusing text fills both boxes of the pair (a missing part is blank), and the Enhanced pair when the asset kept one.
    // A structured caption reuses into the JSON box; the text boxes are cleared so nothing older stands beside it.
    if (params.json_prompt && canApply('prompt') && workflowMode(workflow) === 'image') {
      Object.assign(prefill, { jsonPrompt: prettyCaption(params.json_prompt), jsonPromptEnabled: true, prompt: '', enhancedPrompt: '', enhancedFrom: null });
    }
    const reusesText = extending || ['prompt', 'scene', 'enhanced_prompt', 'enhanced_scene'].some((key) => params[key]);
    if (extending) {
      // A source that moved still names its old path; the Workspace asks for it to be chosen again.
      prefill.extendSourcePath = params.source_video || null;
      prefill.missingSourceVideo = params.missing_source_video === 'true';
      if (prefill.extendSourcePath !== _draft.extendSourcePath) _extendSource = null;
    }
    if (params.extend_frames && canApply('extend_frames')) {
      const frames = Number(params.extend_frames);
      // A multiple of the model's step, not of its frame rule: the new frames are not a clip length.
      if (Number.isFinite(frames)) prefill.extendFrames = snapExtendFrames(frames, ctx ? extendStepFor(ctx, prefill.model ?? _draft.model) : EXTEND_FRAME_STEP);
    }
    if (reusesText && canApply('prompt')) {
      prefill.prompt = params.prompt ?? '';
      prefill.enhancedPrompt = canApply('enhanced_prompt') ? params.enhanced_prompt ?? '' : '';
      const video = workflowMode(workflow) === 'video';
      // An image never carries a scene; a stale Enhanced scene would silently win once the workflow turns to video.
      if (video && canApply('scene')) prefill.scene = params.scene ?? '';
      prefill.enhancedScene = video && canApply('enhanced_scene') ? params.enhanced_scene ?? '' : '';
      // The reused Enhanced pair came from the reused inputs, so it is not out of date; with none, a stale Enhanced box would silently win.
      const enhanced = `${prefill.enhancedPrompt}${prefill.enhancedScene}`.trim() !== '';
      prefill.enhancedFrom = enhanced ? { prompt: prefill.prompt, scene: prefill.scene ?? '', mode: workflowMode(workflow) } : null;
      // How the reused rewrite was made: its options, enhancer and format come back, and Generate records them again.
      const reused = enhanced ? reusedEnhance(params.enhance, workflow) : null;
      if (reused && prefill.enhancedFrom) {
        prefill.enhancedFrom.record = reused.record;
        const key = enhanceSettingsKey(workflow);
        const remembered = key === 'enhanceSettingsVideo' ? _draft.enhanceSettingsVideo?.partsByWorkflow : undefined;
        prefill[key] = remembered || reused.settings.partsByWorkflow ? { ...reused.settings, partsByWorkflow: { ...remembered, ...reused.settings.partsByWorkflow } } : reused.settings;
        if (reused.tier) prefill.enhanceModel = reused.tier;
        if (workflowMode(workflow) === 'image') prefill.enhanceFormat = reused.format;
      }
      if (enhanced && prefill.enhancedFrom && workflow === 'extend') {
        prefill.enhancedFrom.sourceVideoPath = prefill.extendSourcePath !== undefined ? prefill.extendSourcePath : _draft.extendSourcePath;
        prefill.enhancedFrom.extendFrames = prefill.extendFrames !== undefined ? prefill.extendFrames : _draft.extendFrames;
      }
      // Text the URL carried is settled for this source: a later inspection of it must not replace it.
      // That holds for blank text too (a Reuse that recorded none keeps both boxes blank), but only when the URL named the
      // boxes at all: an Extend action names none, and the first inspection then fills them from the source's recorded text.
      if (extending && prefill.extendSourcePath) {
        const named = ['scene', 'prompt', 'enhanced_scene', 'enhanced_prompt'].some((key) => key in params);
        prefill.extendPrefill = named ? { path: prefill.extendSourcePath, scene: prefill.scene ?? '', prompt: prefill.prompt } : null;
      }
    }
    if (params.model && canApply('model')) prefill.model = params.model;
    if (params.steps && canApply('steps')) prefill.steps = Number(params.steps);
    if (params.guidance && canApply('guidance')) prefill.guidance = Number(params.guidance);
    if (params.seed && canApply('seed')) prefill.seed = Number(params.seed);
    if (params.ratio && canApply('ratio')) prefill.ratio = params.ratio;
    if (params.size && canApply('size')) prefill.size = params.size;
    if (params.width && canApply('width')) prefill.width = Number(params.width);
    if (params.height && canApply('height')) prefill.height = Number(params.height);
    if (params.frames && canApply('frames')) {
      // Show the count the server will generate, not one it would re-align.
      const rule = ctx ? videoTimingFor(ctx, prefill.model ?? _draft.model).frameRule : { alignment: 1, offset: 1, round: 'nearest' as const };
      prefill.frameCount = alignFrames(Number(params.frames), rule);
    }
    // A named but blank list clears the LoRAs: a reused job runs with exactly its source's.
    if (params.lora !== undefined && canApply('lora')) prefill.loraString = params.lora;
    // A named but blank level clears the quantization, like the LoRAs: a reused image has exactly its source's level.
    // A level the model does not offer is dropped rather than left for the server to reject.
    if (params.quantize !== undefined && ctx && canApply('quantize')) {
      const level = Number(params.quantize);
      const supported = (_modelDefaultsFor(ctx, workflow, prefill.model ?? _draft.model) as ImageModelDefaults | null)?.supports_quantize === true;
      prefill.quantize = supported && params.quantize !== '' && ctx.quantize_options.includes(level) ? level : null;
    }
    // The images belong to the tab the prefill opens. A Use as (`use_as`) sets only its slot; a full prefill replaces every
    // slot of the tab, so what it does not name is empty (a reused text-made asset has no start image).
    const slotTab = workspaceTab(workflow);
    const replacing = options.replaceSlots === true && params.use_as === undefined;
    if (slotTab !== 'extend') {
      if (params.image_path && canApply('image_path')) Object.assign(prefill, startSlotPatch(slotTab, { path: params.image_path }));
      else if (replacing) Object.assign(prefill, startSlotPatch(slotTab, null));
    }
    if (slotTab === 'video') {
      if (params.end_image_path && canApply('end_image_path')) Object.assign(prefill, lastSlotPatch({ path: params.end_image_path }));
      else if (replacing) Object.assign(prefill, lastSlotPatch(null));
    }
    // The audio track belongs to the same tab (Video or Extend) and follows the images' rule: a full prefill replaces it,
    // so a job made without a track clears it. A Use as leaves it alone.
    if (slotTab !== 'image') {
      if (params.audio_track_path && canApply('audio_track_path')) {
        const start = Number(params.audio_track_start);
        Object.assign(prefill, trackPatch(slotTab, { path: params.audio_track_path }, Number.isFinite(start) && start > 0 ? start : 0));
      } else if (replacing) Object.assign(prefill, trackPatch(slotTab, null));
    }
    // A named list, even an empty one, replaces the references: a reused image has exactly its source's. The list is kept
    // whole, whatever the model takes; the Workspace shows and sends the first few.
    if (lists.reference_path !== undefined && canApply('reference_path')) prefill.referenceImagePaths = lists.reference_path.filter((path) => path !== '');
    else if (replacing && slotTab === 'image') prefill.referenceImagePaths = [];
    // Flags from the slots the prefill replaced are dropped, those of the other slots stay; the Workspace checks the new paths.
    if (SLOT_KEYS.some((key) => key in prefill)) prefill.missingReferencePaths = _flagsOfUntouchedSlots(_draft, prefill);
    // An image Reuse names the finishing settings the asset was made with; Transparent is named on its own too (a viewer-upscaled
    // asset sends only it), and without Sharpen the rest are left alone, or Upscale would read the missing `upscale` as off.
    if (workflowMode(workflow) === 'image') {
      const finishing: Partial<DraftState> = params.sharpen_enabled !== undefined ? outputSettingsPatch(input, false) : {};
      if (params.transparent !== undefined) finishing.postprocessTransparent = params.transparent !== 'false';
      for (const [key, value] of Object.entries(finishing)) {
        const param = _OUTPUT_KEY_PARAMS[key as keyof DraftState];
        if (param === undefined || canApply(param)) Object.assign(prefill, { [key]: value });
      }
    }
    if (params.negative_prompt && canApply('negative_prompt')) prefill.negativePrompt = params.negative_prompt;
    if (params.scheduler && canApply('scheduler')) prefill.scheduler = params.scheduler;
    // A preset the current model does not offer keeps the current preset size instead; an exact Reuse is fitted to the model below.
    if (ctx && !exact && prefill.size !== undefined && prefill.size !== 'custom'
      && !offeredSizes(ctx, workflow, prefill.model ?? _draft.model, prefill.ratio ?? _draft.ratio).includes(prefill.size)) {
      delete prefill.size;
      delete prefill.ratio;
    }
    // A preset that has since changed size keeps the asset's own width/height, so the reused job matches it.
    if (ctx && prefill.size !== undefined && prefill.size !== 'custom' && prefill.width !== undefined && prefill.height !== undefined) {
      const table = workflowMode(workflow) === 'video' ? ctx.video_size_dimensions ?? {} : ctx.image_size_dimensions;
      const preset = table[prefill.ratio ?? _draft.ratio]?.[prefill.size];
      if (preset && (preset[0] !== prefill.width || preset[1] !== prefill.height)) prefill.size = 'custom';
    }
    // Reused assets carry both their preset and their pixel size; a "custom" size means width/height rule.
    if (prefill.ratio !== undefined || prefill.size !== undefined) prefill.dimensionMode = prefill.size === 'custom' ? 'custom' : 'ratio';
    else if (prefill.width !== undefined || prefill.height !== undefined) prefill.dimensionMode = 'custom';
    // A model named here without a re-hydration still swaps LoRAs by family; LoRAs the URL names win over remembered ones.
    const loras = ctx && prefill.model !== undefined ? _swapFamilyLoras(ctx, _draft, _draft.model, prefill.model) : {};
    // Opening the other mode brings back its own text first, so a prefill sets the opened tab's text only.
    const opened = { ..._withTextOfMode(_draft, workflow), ...loras };
    // An exact prefill puts the take's scopes back to their defaults for its model first; the source's own track is not a default of a take.
    const reset = exact ? pickScopes(defaultsFor(ctx, { ...opened, workflow, model: prefill.model ?? opened.model, extendPrefill: null }, _extendSource), REUSE_SCOPE[workspaceTab(workflow)]) : {};
    _draft = { ...opened, ...reset, ...prefill };
    if (ctx) _draft = _withDerivedWorkflow(ctx, _draft);
    if (exact) _draft = _fitToModel(ctx, _draft);
  },

  /**
   * Hydrate draft from the backend workspace context for the current workflow.
   * Corrects the model if it is invalid for the current workflow mode, and
   * populates ratio, size, steps, guidance, and video-specific defaults from
   * the backend contract.
   *
   * If preferredModel is provided, it is used when it exists in the valid model
   * list for the current workflow; otherwise the context default is used. Without
   * one, the draft's model is kept when valid, else the model last used in this
   * mode (image or video) when it is still offered, else the context default.
   * With `keepSettings`, the settings and prompt source chosen for the same model are kept;
   * otherwise the model's defaults apply. With `keepChangedSettings`, a different model keeps
   * the settings that differ from the old model's defaults (fitted to the new model's sizes and
   * step cap) and takes its own defaults for the rest; the same model keeps everything.
   */
  hydrateFromContext(ctx: WorkspaceContext, preferredModel: string | null = null, options: { keepSettings?: boolean; keepChangedSettings?: boolean } = {}): void {
    // A machine that cannot extend gives the Extend kind back to New.
    if (_draft.workflow === 'extend' && !ctx.workflow_contract.values.includes('extend')) _draft = _withDerivedWorkflow(ctx, { ..._draft, workflow: 'txt2vid', videoKind: 'new' });
    // The Audio tab settles on a kind and one of its models; a machine that runs no audio model gives the tab back to Image.
    if (isAudioWorkflow(_draft.workflow) && availableKinds(ctx).length === 0) _draft = { ..._draft, workflow: 'txt2img' };
    // The model the draft holds belongs to the mode it came from, which the workflow may just have left: remember it first.
    _draft = _rememberModel(ctx, _draft);
    if (isAudioWorkflow(_draft.workflow)) {
      const hydrated = hydrateAudio(ctx, _draft, preferredModel);
      _draft = { ...hydrated, ..._swapFamilyLoras(ctx, _draft, _draft.model, hydrated.model) };
      _authorityReady = true;
      this.saveDraft();
      return;
    }
    const workflow = _draft.workflow;
    const isVideoMode = _isVideoWorkflow(workflow);
    const validModels = isVideoMode ? ctx.video_models : ctx.image_models;

    // Determine model: preferredModel > current draft model (if still valid) > the model last used in this mode
    // (if still valid) > context default. An explicit preferredModel that is not valid goes straight to the default.
    const remembered = isVideoMode ? _draft.lastVideoModel : _draft.lastImageModel;
    const candidates = preferredModel !== null ? [preferredModel] : [_draft.model, remembered];
    const model = candidates.find((id) => id !== '' && validModels.some((m) => m.id === id))
      ?? (isVideoMode ? ctx.current_video_model : ctx.current_image_model)
      ?? validModels[0]?.id
      ?? '';

    // The Audio tab leaves the settings of the model it came from in the draft: they still belong to that model.
    const comingFromAudio = audioModelById(ctx, _draft.model) !== null;
    const sameModel = model === (comingFromAudio ? _draft.settingsModel : _draft.model);
    const keepSettings = options.keepSettings === true && sameModel;
    const keepChanged = options.keepChangedSettings === true;
    const keepsPromptSource = (keepSettings || keepChanged) && ctx.prompt_sources.includes(_draft.promptSource);
    const promptSource = keepsPromptSource ? _draft.promptSource : ctx.default_prompt_source;
    // Coming back to the image model last run from a mode without quantization (video, audio) brings its quantization back.
    const returnsToImage = !isVideoMode && model === _draft.lastImageModel && !ctx.image_models.some((m) => m.id === _draft.model);
    const quantize = returnsToImage ? _draft.lastImageQuantize : _draft.quantize;
    let nextState = _withModelDefaults(ctx, { ..._draft, workflow, model, promptSource, quantize, ..._swapFamilyLoras(ctx, _draft, _draft.model, model) });
    if (keepSettings || (keepChanged && sameModel)) {
      // Same model: the user's settings win over its defaults. A preset that no longer exists falls back to the default size.
      const saved: Partial<DraftState> = _pickSettings(_draft);
      if (!_presetIsValid(ctx, _draft)) {
        for (const key of _SIZE_KEYS) delete saved[key];
      }
      nextState = { ...nextState, ...saved };
    } else if (keepChanged) {
      nextState = _fitToModel(ctx, { ...nextState, ..._changedSinceOldDefaults(ctx, _draft) });
    }
    // Settings of a model of the other mode (or none) were not set by the applied preset's: it no longer describes them.
    if (comingFromAudio && !sameModel) nextState = { ...nextState, appliedPreset: null };
    nextState = _rememberModel(ctx, _applyCapabilityGates(ctx, nextState));

    // The workflow follows from the tab and its images, whatever the model: one it cannot start from an image runs from text.
    _draft = _withDerivedWorkflow(ctx, nextState);
    _authorityReady = true;
    this.saveDraft();
  },

  /**
   * Open a tab: the workflow becomes the one its images call for (the Audio tab's is its kind, the Video tab's the kind it was
   * left on). When the tab is of the other mode (image or video) the
   * applied preset is forgotten and the text swaps to what that mode kept; the Audio tab in between keeps both settings and preset. The model and defaults follow when the Workspace calls `onTabChange` with its context.
   * Without a context the model's capabilities are unknown, so the tab's start image is assumed to be usable.
   */
  setTab(tab: NavTab): void {
    if (tab === navTab(_draft.workflow)) return;
    // Leaving Video notes the kind it was on, so coming back opens it again.
    const left = videoKindOf(_draft.workflow);
    const videoKind = left ?? _draft.videoKind;
    // The Audio tab opens on the kind it was left on, the Video tab on its kind.
    const workflow = tab === 'audio' ? _draft.audioKind : tab === 'video' && videoKind === 'extend' ? 'extend' : workflowFor(tab, _draft, true);
    // The Audio tab sits between the modes without taking the settings: only Image to Video (or back) forgets the preset.
    const leavesMode = [workflow, _draft.workflow].every((w) => !isAudioWorkflow(w)) && workflowMode(workflow) !== workflowMode(_draft.workflow);
    _draft = { ..._withTextOfMode(_draft, workflow), workflow, videoKind, ...(leavesMode ? { appliedPreset: null } : {}) };
    this.saveDraft();
  },

  /**
   * Open a kind of the Video tab: New (a text or first-frame clip) or Extend. Only the workflow changes; each kind keeps its own
   * inputs, and the text, the model and the settings are shared. The model and defaults follow when the Workspace calls `onTabChange`.
   *
   * @param ctx Tells whether the model takes a first frame; without it one is assumed.
   */
  setVideoKind(kind: VideoKind, ctx: WorkspaceContext | null): void {
    const workflow = kind === 'extend' ? 'extend' : workflowFor('video', _draft, imageSupport(ctx, _draft.model, 'video').start);
    _draft = { ..._draft, workflow, videoKind: kind };
    this.saveDraft();
  },

  /**
   * Re-hydrate the model and defaults for the tab that was just opened (`setTab`). Switching between tabs of the same mode
   * keeps the model, so it keeps the user's settings too.
   */
  onTabChange(ctx: WorkspaceContext): void {
    this.hydrateFromContext(ctx, null, { keepSettings: true });
  },

  /**
   * Set or clear the start image of a tab (the first frame in Video): the draft holds its path or the upload's file name,
   * the file is held for Generate, the replaced path's missing flag is dropped, and the workflow is derived again. Nothing
   * else changes, so the settings and the model stay.
   *
   * @param ctx Tells whether the model takes a start image; without it one is assumed.
   */
  setStartImage(tab: ImageTab, value: SlotValue, ctx: WorkspaceContext | null): void {
    const replaced = tab === 'image' ? _draft.referenceImagePath : _draft.firstFramePath;
    heldUploads.setStart(tab, value !== null && 'file' in value ? value.file : null);
    const patched = { ..._draft, ...startSlotPatch(tab, value) };
    _draft = _withDerivedWorkflow(ctx, { ...patched, missingReferencePaths: _withoutFlag(patched, replaced) });
    this.saveDraft();
  },

  /** Set or clear the last frame: the draft holds its path or the upload's file name, and the file is held for Generate. */
  setLastFrame(value: SlotValue): void {
    const patched = { ..._draft, ...lastSlotPatch(value) };
    heldUploads.setLastFrame(value !== null && 'file' in value ? value.file : null);
    _draft = { ...patched, missingReferencePaths: _withoutFlag(patched, _draft.endImagePath) };
    this.saveDraft();
  },

  /**
   * Set or clear the audio track of a tab (Video or Extend): the draft holds its path or the upload's file name and
   * the file is held for Generate. A new track starts at 0; clearing puts Start at back to 0.
   */
  setAudioTrack(tab: TrackTab, value: SlotValue): void {
    heldUploads.setTrack(tab, value !== null && 'file' in value ? value.file : null);
    _draft = { ..._draft, ...trackPatch(tab, value) };
    this.saveDraft();
  },

  /** Open a kind of the Audio tab: Music, Speech or Sound, on the model last chosen for it. */
  setAudioKind(kind: AudioWorkflow, ctx: WorkspaceContext | null): void {
    const opened: DraftState = { ..._withTextOfMode(_draft, kind), workflow: kind, audioKind: kind, model: _draft.audioModels[kind] };
    _draft = ctx ? hydrateAudio(ctx, opened) : opened;
    this.saveDraft();
  },

  /**
   * Set or clear the recording the Speech tab's voice follows: the draft holds its path or the upload's file name and the
   * file is held for Generate. A new recording is a new voice, so the speaker's tick is cleared with it; the transcript the
   * person typed stays, whatever the recording does. Choosing one switches the voice to *From a recording*.
   */
  setVoice(value: SlotValue): void {
    heldUploads.setVoice(value !== null && 'file' in value ? value.file : null);
    voiceConsent.reset();
    _draft = {
      ..._draft,
      voicePath: value !== null && 'path' in value ? value.path : null,
      voiceUpload: value !== null && 'file' in value ? value.file.name : null,
      voiceMode: value === null ? _draft.voiceMode : 'recording',
    };
    this.saveDraft();
  },

  /** Set the seconds into the track where the tab's clip starts. */
  setAudioTrackStart(tab: TrackTab, start: number): void {
    _draft = { ..._draft, ...trackStartPatch(tab, Math.max(0, start)) };
    this.saveDraft();
  },

  /**
   * Reset the open tab to its defaults: the fields of the scopes `RESET_SCOPE` lists for the tab take the values `defaultsFor`
   * resolves, and everything else stays. The tab's images and audio track (Image: the start image and references; Video: the
   * first and last frame and the track; Extend: its track, which returns to the source's own) clear with their upload markers
   * and held files, so its workflow falls back to text. The Audio tab puts back its length and voice (the held recording
   * and the tick with it). The batch and the seed are shared by every tab and reset on each. The applied preset clears.
   * Nothing of another tab or of the other mode changes, and the model, LoRAs, quantization, text, prompt file, extend
   * source and layout stay.
   */
  resetSelections(ctx: WorkspaceContext): void {
    const s = _draft;
    const scopes = RESET_SCOPE[workspaceTab(s.workflow)];
    const defaults = defaultsFor(ctx, s, _extendSource);
    _clearHeldUploads(scopes);
    const reset = { ...s, ...pickScopes(defaults, scopes) };
    const kept = new Set(slotPaths(reset));
    _draft = _withDerivedWorkflow(ctx, { ...reset, missingReferencePaths: s.missingReferencePaths.filter((path) => kept.has(path)) });
    this.saveDraft();
  },

  /**
   * Apply a settings preset on top of the chosen model's defaults.
   *
   * Every key of the preset is set, in every workflow. Everything else (prompt, scene, negative prompt, seed,
   * prompt source, enhance options, workflow, reference image and extend source) keeps its value, except what the new
   * model's capability gates clear, as on a manual model switch. A preset's model that is not in the model list, and
   * LoRAs that are not installed, are left out. Never goes through `jobSettingsPrefill`, which would replace the
   * prompt and seed.
   */
  applyPreset(ctx: WorkspaceContext, preset: SettingsPreset): void {
    const before = _draft;
    const touched = presetDraftKeys(before);
    const kept = Object.fromEntries(Object.entries(before).filter(([key]) => !touched.has(key as keyof DraftState) && key !== 'appliedPreset')) as Partial<DraftState>;
    const { model, params, patch } = presetPrefill(preset, before, ctx);
    this.hydrateFromContext(ctx, model);
    // Every preset key lands, whether or not the workflow shows its control.
    this.loadFromUrl(params, ctx, { ungated: true });
    // The kept workflow predates the preset's model, so it is derived again from the tab's images and that model.
    _draft = _withDerivedWorkflow(ctx, { ..._draft, ...patch, ...kept });
    // The restored text and options meet the new model again, as they do when it is picked by hand.
    _draft = _applyCapabilityGates(ctx, _draft);
    // A custom size or step count the preset carried may exceed the kept model's limits, as after a manual model switch.
    _draft = { ..._fitToModel(ctx, _draft), appliedPreset: preset.slug };
    this.saveDraft();
  },

  /** Remember which preset (by slug) the form started from, or none with null. */
  setAppliedPreset(slug: string | null): void {
    _draft = { ..._draft, appliedPreset: slug };
    this.saveDraft();
  },

  saveDraft(): void {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(_draft));
    } catch {
      // storage full or unavailable — ignore
    }
  },

  /** Apply several field updates at once. */
  patch(updates: Partial<DraftState>): void {
    _draft = { ..._draft, ...updates };
    this.saveDraft();
  },

  update<K extends keyof DraftState>(key: K, value: DraftState[K]): void {
    _draft = { ..._draft, [key]: value };
    this.saveDraft();
  },

  /**
   * Take a LoRA out of the current chips and out of what each model family remembers, so no later Generate or family
   * switch can ask for a file that is gone. *installed* lets a chip written as a path be matched by its name.
   */
  removeLora(name: string, installed: LoraInfo[] = []): void {
    const lorasByFamily = Object.fromEntries(Object.entries(_draft.lorasByFamily).map(([key, value]) => [key, withoutLora(value, name, installed)]));
    _draft = { ..._draft, loraString: withoutLora(_draft.loraString, name, installed), lorasByFamily };
    this.saveDraft();
  },

  /**
   * Choose the video to extend, or none with null.
   * Forgets what was known about the previous source and any "source moved" notice.
   */
  setExtendSourcePath(path: string | null): void {
    const changed = path !== _draft.extendSourcePath;
    if (changed) _extendSource = null;
    // A newly chosen source brings its own settings, so the form no longer starts from a preset.
    _draft = { ..._draft, extendSourcePath: path, missingSourceVideo: false, ...(changed && path !== null ? { appliedPreset: null } : {}) };
    this.saveDraft();
  },

  /** Forget the extend source's facts, for example when its file is gone; the chosen path stays. */
  clearExtendSource(): void {
    _extendSource = null;
  },

  /**
   * Take what the server reported about the extend source and pre-fill from it.
   *
   * The first answer for a source sets the frame rate, a missing length (2 s) and, when the source offers them, the model
   * and LoRAs, and fills Scene and Prompt (`extendPrefillFor`): a box is replaced only when it is empty or still holds the
   * previous prefill, unless `replaceText` forces both, and replacing clears the Enhanced pair. Later answers for the same
   * source (a new model or length was inspected) only refresh the facts and never touch the text.
   *
   * @param ctx When given and the source's model differs from the draft's, the model's defaults are loaded.
   */
  applyExtendSource(info: ExtendSourceInfo, ctx: WorkspaceContext | null = null, options: { replaceText?: boolean } = {}): void {
    _extendSource = info;
    const previous = _draft.extendPrefill;
    const sourceKey = _draft.extendSourcePath ?? '';
    const settled = previous !== null && previous.path === sourceKey;
    const updates: Partial<DraftState> = { missingSourceVideo: false };
    if (info.fps) updates.fps = info.fps;
    if (_draft.extendFrames === null && info.extend_frames !== null) updates.extendFrames = info.extend_frames;
    if (!settled || options.replaceText) {
      const fill = extendPrefillFor(info);
      const sceneFree = options.replaceText || _draft.scene === '' || _draft.scene === previous?.scene;
      const promptFree = options.replaceText || _draft.prompt === '' || _draft.prompt === previous?.prompt;
      if (sceneFree) updates.scene = fill.scene;
      if (promptFree) updates.prompt = fill.prompt;
      if (sceneFree || promptFree) {
        updates.enhancedPrompt = '';
        updates.enhancedScene = '';
        updates.enhancedFrom = null;
      }
      // The track follows the same rule as the text: it is replaced only while it is empty or still the previous source's own
      // prefill, so a track the user chose (an upload included) stays. A source without a track empties a prefilled one.
      const track = fill.audioTrack;
      const trackFree = !trackSet(trackOf(_draft, 'extend')) || isPrefilledTrack(_draft, previous?.audioTrack ?? null);
      if (trackFree) {
        Object.assign(updates, trackPatch('extend', track === null ? null : { path: track.path }, track?.start ?? 0));
        heldUploads.setTrack('extend', null);
      }
      updates.extendPrefill = {
        path: sourceKey,
        scene: updates.scene ?? previous?.scene ?? '',
        prompt: updates.prompt ?? previous?.prompt ?? '',
        audioTrack: trackFree ? track : (previous?.audioTrack ?? null),
      };
    }
    _draft = { ..._draft, ...updates };
    if (!settled || options.replaceText) {
      if (info.model && info.model !== _draft.model) {
        if (ctx) this.hydrateFromContext(ctx, info.model);
        else _draft = { ..._draft, model: info.model };
      }
      // After the model change, which swaps in the new family's LoRAs. A source that recorded its settings brings its
      // LoRAs, none included; one without recorded settings leaves them.
      if (info.lora !== null) _draft = { ..._draft, loraString: info.lora };
    }
    this.saveDraft();
  },

  reset(): void {
    _draft = { ...DEFAULT_DRAFT };
    _authorityReady = false;
    _extendSource = null;
    heldUploads.clear();
    voiceConsent.reset();
    localStorage.removeItem(STORAGE_KEY);
  }
};
