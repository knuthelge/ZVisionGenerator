import { workflowMode } from './promptEnhance';
import type { DraftState, WorkspacePrefill, WorkspaceContext, Workflow, ImageModelDefaults, VideoModelDefaults } from '$lib/types';

const STORAGE_KEY = 'ziv-workspace-draft-v1';
const SCHEMA_VERSION = 2;

function _canonicalWorkflow(raw: string, ctx: WorkspaceContext): Workflow | null {
  const workflowValue = raw.trim();
  return ctx.workflow_contract.values.find((workflow) => workflow === workflowValue) ?? null;
}

const DEFAULT_DRAFT: DraftState = {
  workflow: 'txt2img',
  promptSource: 'inline',
  prompt: '',
  jsonPromptEnabled: false,
  jsonPrompt: '',
  firstSigma: null,
  negativePrompt: '',
  promptFilePath: null,
  promptFileOptionIds: [],
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
  referenceImagePath: null,
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
  postprocessSharpenAmount: 0.8,
  postprocessContrastEnabled: false,
  postprocessContrastAmount: 1.0,
  postprocessSaturationEnabled: false,
  postprocessSaturationAmount: 1.0,
  upscaleDenoise: null,
  upscaleSteps: null,
  upscaleGuidance: null,
  upscaleSharpen: true,
  videoUpscaleEnabled: false,
  videoUpscaleFactor: 2,
  enhancedPrompt: '',
  enhancedFrom: null,
  enhanceSettings: null,
  enhanceAuto: false,
};

const _URL_PARAM_CONTROL_IDS: Partial<Record<string, string>> = {
  prompt: 'prompt_inline',
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
  image_path: 'reference_image_path',
};

const _CLEAR_FIELD_STATE_UPDATES: Partial<Record<string, Partial<DraftState>>> = {
  negative_prompt: { negativePrompt: '' },
  guidance: { guidance: DEFAULT_DRAFT.guidance },
  image_path: { referenceImagePath: null },
  image_strength: { referenceImageStrength: DEFAULT_DRAFT.referenceImageStrength },
  quantize: { quantize: null },
  frames: { frameCount: DEFAULT_DRAFT.frameCount },
  audio: { audio: DEFAULT_DRAFT.audio },
  low_memory: { lowMemory: DEFAULT_DRAFT.lowMemory },
  upscale: { upscaleEnabled: false, upscaleFactor: DEFAULT_DRAFT.upscaleFactor },
  sharpen_enabled: { postprocessSharpenEnabled: false },
  sharpen_amount: { postprocessSharpenAmount: 0.8 },
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
    postprocessSharpenAmount: typeof pp.sharpen === 'number' ? pp.sharpen : 0.8,
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
    frameCount: defaults.frame_count,
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
  'upscaleEnabled', 'upscaleFactor', 'upscaleDenoise', 'upscaleSteps', 'upscaleGuidance', 'upscaleSharpen',
  'audio', 'lowMemory', 'videoUpscaleEnabled', 'videoUpscaleFactor',
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

/**
 * Return the default value of every setting for the state's workflow and model.
 *
 * Uses the same resolution as `resetSelections`, without touching the draft.
 */
export function settingDefaultsFor(ctx: WorkspaceContext, state: Pick<DraftState, 'workflow' | 'model'>): SettingDefaults {
  return _pickSettings(_withModelDefaults(ctx, { ...DEFAULT_DRAFT, workflow: state.workflow, model: state.model }));
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

function loadFromStorage(): DraftState {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return { ...DEFAULT_DRAFT };
    const parsed = JSON.parse(raw) as Partial<DraftState> & { promptFileOptionId?: string | null };
    if (parsed.version !== SCHEMA_VERSION) return { ...DEFAULT_DRAFT };
    const { promptFileOptionId, ...saved } = parsed;
    return {
      ...DEFAULT_DRAFT,
      ...saved,
      promptFileOptionIds: saved.promptFileOptionIds ?? (promptFileOptionId ? [promptFileOptionId] : []),
    };
  } catch {
    return { ...DEFAULT_DRAFT };
  }
}

let _draft = $state<DraftState>(loadFromStorage());
let _authorityReady = $state(false);

export const draft = {
  get state(): DraftState { return _draft; },
  get authorityReady(): boolean { return _authorityReady; },

  loadDraft(): void {
    _draft = loadFromStorage();
    _authorityReady = false;
  },

  /**
  * Apply URL query/hash params to the draft using canonical workflow values only.
   * Only fields present in params are updated; absent fields are left unchanged.
   */
  loadFromUrl(params: Record<string, string>, ctx: WorkspaceContext | null = null): void {
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
      return !controlId || visibleControls === null || visibleControls.has(controlId);
    };

    if (params.prompt && canApply('prompt')) prefill.prompt = params.prompt;
    if (params.model && canApply('model')) prefill.model = params.model;
    if (params.steps && canApply('steps')) prefill.steps = Number(params.steps);
    if (params.guidance && canApply('guidance')) prefill.guidance = Number(params.guidance);
    if (params.seed && canApply('seed')) prefill.seed = Number(params.seed);
    if (params.ratio && canApply('ratio')) prefill.ratio = params.ratio;
    if (params.size && canApply('size')) prefill.size = params.size;
    if (params.width && canApply('width')) prefill.width = Number(params.width);
    if (params.height && canApply('height')) prefill.height = Number(params.height);
    if (params.frames && canApply('frames')) prefill.frameCount = Number(params.frames);
    if (params.lora && canApply('lora')) prefill.loraString = params.lora;
    if (params.image_path && canApply('image_path')) prefill.referenceImagePath = params.image_path;
    // A reused preset the current model does not offer keeps the current preset size instead.
    if (ctx && prefill.size !== undefined && prefill.size !== 'custom'
      && !offeredSizes(ctx, workflow, prefill.model ?? _draft.model, prefill.ratio ?? _draft.ratio).includes(prefill.size)) {
      delete prefill.size;
      delete prefill.ratio;
    }
    // Reused assets carry both their preset and their pixel size; a "custom" size means width/height rule.
    if (prefill.ratio !== undefined || prefill.size !== undefined) prefill.dimensionMode = prefill.size === 'custom' ? 'custom' : 'ratio';
    else if (prefill.width !== undefined || prefill.height !== undefined) prefill.dimensionMode = 'custom';
    // A reused prompt replaces whatever was enhanced before; a stale Enhanced box would silently win.
    if (prefill.prompt !== undefined) {
      prefill.enhancedPrompt = '';
      prefill.enhancedFrom = null;
    }
    _draft = { ..._draft, ...prefill };
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
   * otherwise the model's defaults apply.
   */
  hydrateFromContext(ctx: WorkspaceContext, preferredModel: string | null = null, options: { keepSettings?: boolean } = {}): void {
    const workflow = _draft.workflow;
    const isVideoMode = _isVideoWorkflow(workflow);
    const validModels = isVideoMode ? ctx.video_models : ctx.image_models;

    // Determine model: preferredModel > current draft model (if still valid) > context default
    const candidate = preferredModel ?? _draft.model;
    const modelIsValid = candidate !== '' && validModels.some(m => m.id === candidate);
    const model = modelIsValid
      ? candidate
      : ((isVideoMode ? ctx.current_video_model : ctx.current_image_model) ?? validModels[0]?.id ?? '');

    const keepSettings = options.keepSettings === true && model === _draft.model;
    const promptSource = keepSettings && ctx.prompt_sources.includes(_draft.promptSource) ? _draft.promptSource : ctx.default_prompt_source;
    let nextState = _withModelDefaults(ctx, { ..._draft, workflow, model, promptSource });
    if (keepSettings) {
      // Same model: the user's settings win over its defaults. A preset that no longer exists falls back to the default size.
      const saved: Partial<DraftState> = _pickSettings(_draft);
      if (!_presetIsValid(ctx, _draft)) {
        for (const key of ['ratio', 'size', 'dimensionMode', 'width', 'height'] as const) delete saved[key];
      }
      nextState = { ...nextState, ...saved };
    }
    const modelDefaults = _modelDefaultsFor(ctx, workflow, model);

    if (isVideoMode) {
      nextState = { ...nextState, negativePrompt: '', quantize: null, jsonPromptEnabled: false, jsonPrompt: '', firstSigma: null };
    } else {
      const imageDefaults = modelDefaults as ImageModelDefaults | null;
      nextState = {
        ...nextState,
        negativePrompt: imageDefaults?.supports_negative_prompt ? nextState.negativePrompt : '',
        quantize: imageDefaults?.supports_quantize ? nextState.quantize : null,
        jsonPromptEnabled: imageDefaults?.supports_json_prompt ? nextState.jsonPromptEnabled : false,
        jsonPrompt: imageDefaults?.supports_json_prompt ? nextState.jsonPrompt : '',
        firstSigma: imageDefaults?.supports_first_sigma ? nextState.firstSigma : null,
      };
    }

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
   * Enhanced prompt) untouched. Enhancement options and auto-enhance reset.
   */
  resetSelections(ctx: WorkspaceContext): void {
    const s = _draft;
    _draft = {
      ...DEFAULT_DRAFT,
      workflow: s.workflow,
      model: s.model,
      loraString: s.loraString,
      quantize: s.quantize,
      historyCollapsed: s.historyCollapsed,
      sidebarCollapsed: s.sidebarCollapsed,
      lastGeneratedAt: s.lastGeneratedAt,
    };
    this.hydrateFromContext(ctx, s.model);
    // Restore prompt inputs after hydration, which re-applies the default prompt source.
    _draft = {
      ..._draft,
      promptSource: s.promptSource,
      prompt: s.prompt,
      jsonPromptEnabled: s.jsonPromptEnabled,
      jsonPrompt: s.jsonPrompt,
      negativePrompt: s.negativePrompt,
      promptFilePath: s.promptFilePath,
      promptFileOptionIds: s.promptFileOptionIds,
      enhancedPrompt: s.enhancedPrompt,
      enhancedFrom: s.enhancedFrom,
    };
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

  reset(): void {
    _draft = { ...DEFAULT_DRAFT };
    _authorityReady = false;
    localStorage.removeItem(STORAGE_KEY);
  }
};
