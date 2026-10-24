import { extendPrefillFor } from './extend';
import { keepInstalledLoras, modelLoraFamily } from './loras';
import { presetDraftKeys, presetPrefill } from './presets';
import { enhanceSettingsKey, isJsonCaption, prettyCaption, reusedEnhance, workflowMode } from './promptEnhance';
import { alignFrames, defaultExtendFrames, EXTEND_FRAME_STEP, snapExtendFrames } from './videoDuration';
import type { DraftState, ExtendSourceInfo, FrameRule, SettingsPreset, WorkspacePrefill, WorkspaceContext, Workflow, ImageModelDefaults, VideoModelDefaults } from '$lib/types';

const STORAGE_KEY = 'ziv-workspace-draft-v1';
const SCHEMA_VERSION = 3;
// Version 2 drafts always carried a sharpen amount: the old fixed default 0.8 now means "follow the config".
const LEGACY_SHARPEN_DEFAULT = 0.8;
/** Highest sharpen amount the backend accepts (`MAX_SHARPEN_AMOUNT` in utils/config.py). */
export const MAX_SHARPEN_AMOUNT = 1.5;

function _canonicalWorkflow(raw: string, ctx: WorkspaceContext): Workflow | null {
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
  referenceImagePath: null,
  endImagePath: null,
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
  negative_prompt: 'negative_prompt',
  scheduler: 'scheduler',
};

const _CLEAR_FIELD_STATE_UPDATES: Partial<Record<string, Partial<DraftState>>> = {
  negative_prompt: { negativePrompt: '' },
  guidance: { guidance: DEFAULT_DRAFT.guidance },
  image_path: { referenceImagePath: null },
  end_image_path: { endImagePath: null },
  image_strength: { referenceImageStrength: DEFAULT_DRAFT.referenceImageStrength },
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
};

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

function _visibleControlsForWorkflow(ctx: WorkspaceContext, workflow: Workflow): Set<string> {
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
  'postprocessSaturationEnabled', 'postprocessSaturationAmount',
  'upscaleEnabled', 'upscaleFactor', 'upscaleDenoise', 'upscaleSteps', 'upscaleGuidance', 'upscaleSharpen', 'upscaleReview',
  'audio', 'lowMemory', 'videoUpscaleEnabled', 'videoUpscaleFactor', 'videoUpscaleReview',
] as const satisfies readonly (keyof DraftState)[];

export type SettingKey = (typeof SETTING_KEYS)[number];
export type SettingDefaults = Pick<DraftState, SettingKey>;

function _pickSettings(state: DraftState): SettingDefaults {
  return Object.fromEntries(SETTING_KEYS.map((key) => [key, state[key]])) as SettingDefaults;
}

function _isVideoWorkflow(workflow: Workflow): boolean {
  return workflowMode(workflow) === 'video';
}

/** Return the backend defaults for a model in a workflow's mode, or null when none are known. */
function _modelDefaultsFor(ctx: WorkspaceContext, workflow: Workflow, model: string): ImageModelDefaults | VideoModelDefaults | null {
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
 */
function _swapFamilyLoras(ctx: WorkspaceContext, state: DraftState, fromModel: string, toModel: string): Partial<DraftState> {
  if (!fromModel || !toModel) return {};
  const fromKey = _loraFamilyKey(ctx, fromModel);
  const toKey = _loraFamilyKey(ctx, toModel);
  if (fromKey === toKey) return {};
  return {
    lorasByFamily: { ...state.lorasByFamily, [fromKey]: state.loraString },
    loraString: keepInstalledLoras(state.lorasByFamily[toKey] ?? '', ctx.loras),
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
function _defaultState(ctx: WorkspaceContext, workflow: Workflow, model: string): DraftState {
  return _withModelDefaults(ctx, { ...DEFAULT_DRAFT, workflow, model });
}

/**
 * Return the default value of every setting for the state's workflow and model.
 *
 * Uses the same resolution as `resetSelections`, without touching the draft.
 */
export function settingDefaultsFor(ctx: WorkspaceContext, state: Pick<DraftState, 'workflow' | 'model'>): SettingDefaults {
  return _pickSettings(_defaultState(ctx, state.workflow, state.model));
}

/** Return the size presets offered for a ratio; image presets beyond the model's dimension limit are left out. */
export function offeredSizes(ctx: WorkspaceContext, workflow: Workflow, model: string, ratio: string): string[] {
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
function _largestPreset(ctx: WorkspaceContext, workflow: Workflow, ratio: string, presets: string[]): string {
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

/** Clear what the state's model cannot use, and align the frame count to its grid. */
function _applyCapabilityGates(ctx: WorkspaceContext, state: DraftState): DraftState {
  const { workflow, model } = state;
  const isVideoMode = _isVideoWorkflow(workflow);
  const modelDefaults = _modelDefaultsFor(ctx, workflow, model);
  let nextState = state;

  // An Enhanced JSON caption only means something to a model that takes JSON captions.
  const imageTakesJson = !isVideoMode && (modelDefaults as ImageModelDefaults | null)?.supports_json_prompt === true;
  if (!imageTakesJson && isJsonCaption(nextState.enhancedPrompt)) nextState = { ...nextState, enhancedPrompt: '', enhancedFrom: null };
  if (isVideoMode) {
    nextState = { ...nextState, negativePrompt: '', quantize: null, jsonPromptEnabled: false, jsonPrompt: '', firstSigma: null };
    // A kept or restored count may predate snapping; show the count the server will generate.
    const timing = videoTimingFor(ctx, model, _sourceFps(workflow));
    nextState.fps = timing.fps;
    // An extension has no clip length of its own: its new frames (`extendFrames`) stay multiples of the model's step, not its frame rule.
    if (workflow !== 'extend') nextState.frameCount = alignFrames(nextState.frameCount, timing.frameRule);
    // Settings of controls the model does not have are cleared, and the steps move to one it offers.
    const video = modelDefaults as VideoModelDefaults | null;
    const hidden = new Set(video?.hidden_controls ?? []);
    if (hidden.has('video_upscale_enabled')) nextState.videoUpscaleEnabled = false;
    if (hidden.has('video_upscale_review')) nextState.videoUpscaleReview = false;
    if (hidden.has('low_memory') && video) nextState.lowMemory = video.low_memory;
    // The last frame is an Image to Video setting of the model: switching to one without it drops the image.
    if (hidden.has('end_image_path')) nextState.endImagePath = null;
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

/** Upgrade a saved draft from the previous schema; other versions pass through unchanged. */
function _migrateDraft<T extends Partial<DraftState>>(saved: T): T {
  if (saved.version !== 2) return saved;
  const amount = saved.postprocessSharpenAmount;
  // The old control allowed up to 2; the server now rejects amounts above MAX_SHARPEN_AMOUNT.
  const kept = amount === LEGACY_SHARPEN_DEFAULT || amount == null ? null : Math.min(amount, MAX_SHARPEN_AMOUNT);
  return { ...saved, version: SCHEMA_VERSION, postprocessSharpenAmount: kept };
}

function loadFromStorage(): DraftState {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return { ...DEFAULT_DRAFT };
    const parsed = _migrateDraft(JSON.parse(raw) as Partial<DraftState> & { promptFileOptionId?: string | null });
    if (parsed.version !== SCHEMA_VERSION) return { ...DEFAULT_DRAFT };
    const { promptFileOptionId, ...saved } = parsed;
    return {
      ...DEFAULT_DRAFT,
      ...saved,
      // Drafts saved before images and video kept their own enhance options seed both from the one set.
      enhanceSettingsVideo: saved.enhanceSettingsVideo !== undefined ? saved.enhanceSettingsVideo : saved.enhanceSettings ?? null,
      promptFileOptionIds: saved.promptFileOptionIds ?? (promptFileOptionId ? [promptFileOptionId] : []),
    };
  } catch {
    return { ...DEFAULT_DRAFT };
  }
}

let _draft = $state<DraftState>(loadFromStorage());
let _authorityReady = $state(false);
// What the server last reported about the extend source; fetched again on load, so it is not saved with the draft.
let _extendSource = $state<ExtendSourceInfo | null>(null);

/** The frame rate an extension runs at: its source's, once known. */
function _sourceFps(workflow: Workflow): number | null {
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
   */
  loadFromUrl(params: Record<string, string>, ctx: WorkspaceContext | null = null, options: { ungated?: boolean } = {}): void {
    const prefill: WorkspacePrefill = {};
    let workflow = _draft.workflow;
    if (params.workflow && ctx) {
      const canonical = _canonicalWorkflow(params.workflow, ctx);
      if (canonical) {
        prefill.workflow = canonical;
        workflow = canonical;
      }
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
    if (params.image_path && canApply('image_path')) prefill.referenceImagePath = params.image_path;
    if (params.end_image_path && canApply('end_image_path')) prefill.endImagePath = params.end_image_path;
    if (params.negative_prompt && canApply('negative_prompt')) prefill.negativePrompt = params.negative_prompt;
    if (params.scheduler && canApply('scheduler')) prefill.scheduler = params.scheduler;
    // A reused preset the current model does not offer keeps the current preset size instead.
    if (ctx && prefill.size !== undefined && prefill.size !== 'custom'
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
    _draft = { ..._draft, ...loras, ...prefill };
  },

  /**
   * Hydrate draft from the backend workspace context for the current workflow.
   * Corrects the model if it is invalid for the current workflow mode, and
   * populates ratio, size, steps, guidance, and video-specific defaults from
   * the backend contract.
   *
   * If preferredModel is provided, it is used when it exists in the valid model
   * list for the current workflow; otherwise the context default is used.
   * With `keepSettings`, the settings and prompt source chosen for the same model are kept;
   * otherwise the model's defaults apply. With `keepChangedSettings`, a different model keeps
   * the settings that differ from the old model's defaults (fitted to the new model's sizes and
   * step cap) and takes its own defaults for the rest; the same model keeps everything.
   */
  hydrateFromContext(ctx: WorkspaceContext, preferredModel: string | null = null, options: { keepSettings?: boolean; keepChangedSettings?: boolean } = {}): void {
    const workflow = _draft.workflow;
    const isVideoMode = _isVideoWorkflow(workflow);
    const validModels = isVideoMode ? ctx.video_models : ctx.image_models;

    // Determine model: preferredModel > current draft model (if still valid) > context default
    const candidate = preferredModel ?? _draft.model;
    const modelIsValid = candidate !== '' && validModels.some(m => m.id === candidate);
    const model = modelIsValid
      ? candidate
      : ((isVideoMode ? ctx.current_video_model : ctx.current_image_model) ?? validModels[0]?.id ?? '');

    const sameModel = model === _draft.model;
    const keepSettings = options.keepSettings === true && sameModel;
    const keepChanged = options.keepChangedSettings === true;
    const keepsPromptSource = (keepSettings || keepChanged) && ctx.prompt_sources.includes(_draft.promptSource);
    const promptSource = keepsPromptSource ? _draft.promptSource : ctx.default_prompt_source;
    let nextState = _withModelDefaults(ctx, { ..._draft, workflow, model, promptSource, ..._swapFamilyLoras(ctx, _draft, _draft.model, model) });
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
    nextState = _applyCapabilityGates(ctx, nextState);

    _draft = nextState;
    _authorityReady = true;
    this.saveDraft();
  },

  /**
   * Switch workflow and re-hydrate all model/defaults from context for the new
   * workflow mode. Called when the user changes the workflow via the top nav.
   */
  onWorkflowChange(workflow: Workflow, ctx: WorkspaceContext): void {
    _draft = { ..._draft, workflow };
    _authorityReady = false;
    // Switching between workflows of the same mode keeps the model, so it keeps the user's settings too.
    this.hydrateFromContext(ctx, null, { keepSettings: true });
  },

  /**
   * Reset all generation selections to the model's defaults, keeping the
   * workflow, model, LoRAs, quantization, and prompt inputs (including the
   * Enhanced prompt) untouched. Enhancement options and auto-enhance reset. The applied preset clears, since its
   * settings are gone.
   */
  resetSelections(ctx: WorkspaceContext): void {
    const s = _draft;
    _draft = {
      ...DEFAULT_DRAFT,
      workflow: s.workflow,
      model: s.model,
      loraString: s.loraString,
      lorasByFamily: s.lorasByFamily,
      quantize: s.quantize,
      historyCollapsed: s.historyCollapsed,
      sidebarCollapsed: s.sidebarCollapsed,
      lastGeneratedAt: s.lastGeneratedAt,
      // The source is an input, kept like the prompt; its length returns to the 2 s default.
      extendSourcePath: s.extendSourcePath,
      missingSourceVideo: s.missingSourceVideo,
      extendPrefill: s.extendPrefill,
      extendFrames: _extendSource?.fps ? defaultExtendFrames(_extendSource.fps, extendStepFor(ctx, s.model)) : s.extendFrames,
    };
    this.hydrateFromContext(ctx, s.model);
    // Restore prompt inputs after hydration, which re-applies the default prompt source.
    _draft = {
      ..._draft,
      promptSource: s.promptSource,
      prompt: s.prompt,
      scene: s.scene,
      jsonPromptEnabled: s.jsonPromptEnabled,
      jsonPrompt: s.jsonPrompt,
      negativePrompt: s.negativePrompt,
      promptFilePath: s.promptFilePath,
      promptFileOptionIds: s.promptFileOptionIds,
      promptFileLoaded: s.promptFileLoaded,
      enhancedPrompt: s.enhancedPrompt,
      enhancedScene: s.enhancedScene,
      enhancedFrom: s.enhancedFrom,
    };
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
    _draft = { ..._draft, ...patch, ...kept };
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
    // Switching between image and video ends the applied preset's hold; presets belong to one mode.
    const leavesMode = key === 'workflow' && workflowMode(value as Workflow) !== workflowMode(_draft.workflow);
    _draft = { ..._draft, [key]: value, ...(leavesMode ? { appliedPreset: null } : {}) };
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
      updates.extendPrefill = { path: sourceKey, scene: updates.scene ?? previous?.scene ?? '', prompt: updates.prompt ?? previous?.prompt ?? '' };
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
    localStorage.removeItem(STORAGE_KEY);
  }
};
