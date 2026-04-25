import type { DraftState, WorkspacePrefill, WorkspaceContext, Workflow, ImageModelDefaults, VideoModelDefaults } from '$lib/types';

const STORAGE_KEY = 'ziv-workspace-draft-v1';
const SCHEMA_VERSION = 2;

// Mirror the backend canonical workflow aliases so legacy URL params are parsed correctly.
const _WORKFLOW_ALIASES: Record<string, Workflow> = {
  txt2img: 'txt2img', image: 'txt2img', texttoimage: 'txt2img',
  img2img: 'img2img', i2i: 'img2img', imagetoimage: 'img2img',
  txt2vid: 'txt2vid', video: 'txt2vid', texttovideo: 'txt2vid',
  img2vid: 'img2vid', i2v: 'img2vid', imagetovideo: 'img2vid',
};

function _canonicalWorkflow(raw: string): Workflow | null {
  const normalized = raw.trim().toLowerCase().replace(/[^a-z0-9]+/g, '');
  return _WORKFLOW_ALIASES[normalized] ?? null;
}

const DEFAULT_DRAFT: DraftState = {
  workflow: 'txt2img',
  promptSource: 'inline',
  prompt: '',
  negativePrompt: '',
  promptFilePath: null,
  promptFileOptionId: null,
  model: '',
  ratio: '',
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
};

const _URL_PARAM_CONTROL_IDS: Partial<Record<string, string>> = {
  prompt: 'prompt_inline',
  model: 'model',
  steps: 'steps',
  guidance: 'guidance',
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
   * Apply URL query/hash params to the draft. Handles legacy workflow aliases.
   * Only fields present in params are updated; absent fields are left unchanged.
   */
  loadFromUrl(params: Record<string, string>, ctx: WorkspaceContext | null = null): void {
    const prefill: WorkspacePrefill = {};
    let workflow = _draft.workflow;
    if (params.workflow) {
      const canonical = _canonicalWorkflow(params.workflow);
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
    if (params.ratio && canApply('ratio')) prefill.ratio = params.ratio;
    if (params.size && canApply('size')) prefill.size = params.size;
    if (params.width && canApply('width')) prefill.width = Number(params.width);
    if (params.height && canApply('height')) prefill.height = Number(params.height);
    if (params.frames && canApply('frames')) prefill.frameCount = Number(params.frames);
    if (params.lora && canApply('lora')) prefill.loraString = params.lora;
    if (params.image_path && canApply('image_path')) prefill.referenceImagePath = params.image_path;
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
   */
  hydrateFromContext(ctx: WorkspaceContext, preferredModel: string | null = null): void {
    const workflow = _draft.workflow;
    const isVideoMode = workflow === 'txt2vid' || workflow === 'img2vid';
    const validModels = isVideoMode ? ctx.video_models : ctx.image_models;

    // Determine model: preferredModel > current draft model (if still valid) > context default
    const candidate = preferredModel ?? _draft.model;
    const modelIsValid = candidate !== '' && validModels.some(m => m.id === candidate);
    const model = modelIsValid
      ? candidate
      : ((isVideoMode ? ctx.current_video_model : ctx.current_image_model) ?? validModels[0]?.id ?? '');

    const defaultsMap = isVideoMode ? ctx.video_model_defaults : ctx.image_model_defaults;
    const fallback = isVideoMode ? ctx.video_defaults : ctx.defaults;
    const currentModel = isVideoMode ? ctx.current_video_model : ctx.current_image_model;
    const modelDefaults = (defaultsMap?.[model] ?? (model === currentModel ? fallback : null)) as (ImageModelDefaults | VideoModelDefaults | null);
    const clearFields = ctx.workflow_contract.definitions[workflow]?.clear_fields ?? [];
    let nextState = _applyClearFields({
      ..._draft,
      workflow,
      model,
      promptSource: ctx.default_prompt_source,
    }, clearFields);

    if (modelDefaults) {
      nextState = isVideoMode
        ? _applyVideoDefaults(nextState, modelDefaults as VideoModelDefaults)
        : _applyImageDefaults(nextState, modelDefaults as ImageModelDefaults);
    }

    if (isVideoMode) {
      nextState = { ...nextState, negativePrompt: '', quantize: null };
    } else {
      const imageDefaults = modelDefaults as ImageModelDefaults | null;
      nextState = {
        ...nextState,
        negativePrompt: imageDefaults?.supports_negative_prompt ? nextState.negativePrompt : '',
        quantize: imageDefaults?.supports_quantize ? nextState.quantize : null,
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
    this.hydrateFromContext(ctx, null);
  },

  saveDraft(): void {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(_draft));
    } catch {
      // storage full or unavailable — ignore
    }
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
