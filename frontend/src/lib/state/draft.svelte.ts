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
  prompt: '',
  negativePrompt: '',
  model: '',
  ratio: '',
  size: '',
  steps: 20,
  guidance: 7.5,
  width: 1024,
  height: 1024,
  runs: 1,
  seed: null,
  loraString: '',
  referenceImagePath: null,
  referenceImageStrength: 0.7,
  frameCount: 97,
  fps: 24,
  audio: true,
  lowMemory: true,
  upscaleEnabled: false,
  upscaleFactor: 2,
  quantize: null,
  historyCollapsed: false,
  lastGeneratedAt: null,
  version: SCHEMA_VERSION
};

function _applyImageDefaults(state: DraftState, defaults: ImageModelDefaults): DraftState {
  return {
    ...state,
    ratio: defaults.ratio,
    size: defaults.size,
    steps: defaults.steps,
    guidance: defaults.guidance,
    width: defaults.width,
    height: defaults.height,
    referenceImageStrength: defaults.image_strength,
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

let _draft = $state<DraftState>(loadFromStorage());

export const draft = {
  get state(): DraftState { return _draft; },

  loadDraft(): void {
    _draft = loadFromStorage();
  },

  /**
   * Apply URL query/hash params to the draft. Handles legacy workflow aliases.
   * Only fields present in params are updated; absent fields are left unchanged.
   */
  loadFromUrl(params: Record<string, string>): void {
    const prefill: WorkspacePrefill = {};
    if (params.workflow) {
      const canonical = _canonicalWorkflow(params.workflow);
      if (canonical) prefill.workflow = canonical;
    }
    if (params.prompt) prefill.prompt = params.prompt;
    if (params.model) prefill.model = params.model;
    if (params.steps) prefill.steps = Number(params.steps);
    if (params.guidance) prefill.guidance = Number(params.guidance);
    if (params.ratio) prefill.ratio = params.ratio;
    if (params.size) prefill.size = params.size;
    if (params.width) prefill.width = Number(params.width);
    if (params.height) prefill.height = Number(params.height);
    if (params.frames) prefill.frameCount = Number(params.frames);
    if (params.lora) prefill.loraString = params.lora;
    if (params.image_path) prefill.referenceImagePath = params.image_path;
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
    const modelDefaults = (defaultsMap?.[model] ?? fallback) as (ImageModelDefaults | VideoModelDefaults);

    _draft = isVideoMode
      ? _applyVideoDefaults({ ..._draft, model }, modelDefaults as VideoModelDefaults)
      : _applyImageDefaults({ ..._draft, model }, modelDefaults as ImageModelDefaults);
    this.saveDraft();
  },

  /**
   * Switch workflow and re-hydrate all model/defaults from context for the new
   * workflow mode. Called when the user changes the workflow via the top nav.
   */
  onWorkflowChange(workflow: Workflow, ctx: WorkspaceContext): void {
    _draft = { ..._draft, workflow };
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
    localStorage.removeItem(STORAGE_KEY);
  }
};
