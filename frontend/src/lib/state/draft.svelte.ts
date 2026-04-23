import type { DraftState, WorkspacePrefill } from '$lib/types';

const STORAGE_KEY = 'ziv-workspace-draft-v1';
const SCHEMA_VERSION = 1;

const DEFAULT_DRAFT: DraftState = {
  workflow: 'txt2img',
  prompt: '',
  negativePrompt: '',
  model: '',
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
  motionStrength: 0.5,
  quantize: false,
  historyCollapsed: false,
  lastGeneratedAt: null,
  version: SCHEMA_VERSION
};

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

  loadFromUrl(params: Record<string, string>): void {
    const prefill: WorkspacePrefill = {};
    if (params.workflow && ['txt2img', 'img2img', 'txt2vid', 'img2vid'].includes(params.workflow)) {
      prefill.workflow = params.workflow as WorkspacePrefill['workflow'];
    }
    if (params.prompt) prefill.prompt = params.prompt;
    if (params.model) prefill.model = params.model;
    if (params.steps) prefill.steps = Number(params.steps);
    if (params.guidance) prefill.guidance = Number(params.guidance);
    if (params.width) prefill.width = Number(params.width);
    if (params.height) prefill.height = Number(params.height);
    if (params.lora) prefill.loraString = params.lora;
    _draft = { ..._draft, ...prefill };
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
