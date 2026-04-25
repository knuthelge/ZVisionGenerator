// ── Core data types ──────────────────────────────────────────────────────────

export type Workflow = 'txt2img' | 'img2img' | 'txt2vid' | 'img2vid';

export type WorkflowMode = 'image' | 'video';

export type PromptSource = 'inline' | 'file';

export interface ReuseState {
  requested_workflow: Workflow;
  resolved_workflow: Workflow;
  workflow_available: boolean;
  requested_model: string | null;
  resolved_model: string | null;
  model_available: boolean;
  fallback_reasons: string[];
}

export interface GalleryAsset {
  path: string;
  url: string;
  thumbnail_url: string;
  filename: string;
  created_at: string;
  workflow: Workflow;
  prompt: string;
  model: string;
  width?: number;
  height?: number;
  ratio?: string | null;
  size?: string | null;
  frame_count?: number | null;
  image_path?: string | null;
  duration?: number;
  reuse_workspace_url: string; // hash URL: '#/workspace?workflow=...&prompt=...'
  reuse_state?: ReuseState;
  media_type: 'image' | 'video';
}

export interface ModelOption {
  id: string;
  label: string;
  type: 'image' | 'video';
}

export interface LoraInfo {
  name: string;
  path: string;
}

export interface WorkspaceContext {
  image_models: ModelOption[];
  video_models: ModelOption[];
  loras: LoraInfo[];
  history_assets: GalleryAsset[];
  defaults: ImageModelDefaults;
  video_defaults: VideoModelDefaults;
  image_model_defaults: Record<string, ImageModelDefaults>;
  video_model_defaults: Record<string, VideoModelDefaults>;
  current_image_model: string | null;
  current_video_model: string | null;
  config: WebUiConfig;
  output_dir: string;
  quantize_options: number[];
  image_ratios: string[];
  video_ratios: string[];
  image_size_options: Record<string, string[]>;
  video_size_options: Record<string, string[]>;
  scheduler_options: string[];
  prompt_sources: PromptSource[];
  default_prompt_source: PromptSource;
  prompt_file: PromptFileContract;
  workflow_contract: WorkflowContract;
}

export interface PromptFileContract {
  accepted_extensions: string[];
  browse_kind: 'existing_file';
  selection_required: boolean;
}

export interface PromptFileOption {
  id: string;
  set_name: string;
  source_index: number;
  label: string;
  prompt_preview: string;
  negative_preview: string | null;
}

export interface PromptFileInspection {
  path: string;
  options: PromptFileOption[];
}

export interface PromptFileDocument extends PromptFileInspection {
  raw_text: string;
}

export type PathPickerStatus = 'selected' | 'cancelled' | 'unsupported' | 'error';

export interface PathPickerResult {
  status: PathPickerStatus;
  path: string | null;
  message: string | null;
}

export interface UpscaleDefaults {
  enabled: boolean;
  factor: number | null;
  steps: number | null;
}

export interface ImageUpscaleDefaults extends UpscaleDefaults {
  denoise: number | null;
  guidance: number | null;
  sharpen: boolean;
  save_pre: boolean;
}

export interface ImagePostprocessDefaults {
  sharpen: number | boolean;
  contrast: number | boolean;
  saturation: number | boolean;
}

export interface ImageModelDefaults {
  ratio: string;
  size: string;
  steps: number;
  guidance: number;
  width: number;
  height: number;
  scheduler: string | null;
  supports_negative_prompt: boolean;
  supports_quantize: boolean;
  quantize: number | null;
  image_strength: number;
  postprocess: ImagePostprocessDefaults;
  upscale: ImageUpscaleDefaults;
}

export interface VideoModelDefaults {
  ratio: string;
  size: string;
  steps: number;
  width: number;
  height: number;
  frame_count: number;
  audio: boolean;
  low_memory: boolean;
  supports_i2v: boolean;
  supports_quantize: boolean;
  quantize: number | null;
  max_steps: number | null;
  fps: number;
  upscale: UpscaleDefaults;
}

export interface WorkflowContractEntry {
  mode: WorkflowMode;
  model_kind: WorkflowMode;
  visible_controls: string[];
  supports_reference_image: boolean;
  requires_reference_image: boolean;
  clear_fields: string[];
}

export interface WorkflowContract {
  values: Workflow[];
  legacy_aliases: Record<string, Workflow>;
  definitions: Record<Workflow, WorkflowContractEntry>;
  field_precedence: {
    defaults: string[];
    dimensions: string;
  };
}

export interface WebUiConfig {
  visible_sections: string[];
  theme: string;
  gallery_page_size: number;
  output_dir?: string;
  startup_view?: 'workspace' | 'gallery' | 'config';
  default_models?: { image: string | null; video: string | null };
  image_model_options?: string[];
  video_model_options?: string[];
  lora_options?: string[];
  quantize_options?: number[];
  model_cache_dir?: string;
  loras_dir?: string;
  huggingface_token_configured?: boolean;
  huggingface_token_env_var?: string | null;
  image_size_labels?: { value: string; label: string }[];
  default_image_size?: string;
}

// ── Draft / workspace form state ──────────────────────────────────────────────

export interface DraftState {
  workflow: Workflow;
  promptSource: PromptSource;
  prompt: string;
  negativePrompt: string;
  promptFilePath: string | null;
  promptFileOptionId: string | null;
  model: string;
  ratio: string;
  size: string;
  steps: number;
  guidance: number;
  width: number;
  height: number;
  runs: number;
  seed: number | null; // null = random
  loraString: string;
  referenceImagePath: string | null;
  referenceImageStrength: number;
  frameCount: number;
  fps: number;
  audio: boolean;
  lowMemory: boolean;
  upscaleEnabled: boolean;
  upscaleFactor: number;
  quantize: number | null;
  historyCollapsed: boolean;
  lastGeneratedAt: string | null;
  version: number;
  scheduler: string | null;
  postprocessSharpenEnabled: boolean;
  postprocessSharpenAmount: number;
  postprocessContrastEnabled: boolean;
  postprocessContrastAmount: number;
  postprocessSaturationEnabled: boolean;
  postprocessSaturationAmount: number;
  upscaleDenoise: number | null;
  upscaleSteps: number | null;
  upscaleGuidance: number | null;
  upscaleSharpen: boolean;
  videoUpscaleEnabled: boolean;
  videoUpscaleFactor: number;
}

export type WorkspacePrefill = Partial<Omit<DraftState, 'historyCollapsed' | 'lastGeneratedAt' | 'version'>>;

// ── Job / SSE types ────────────────────────────────────────────────────────────

export type JobStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';

export interface JobContext {
  job_id: string;
  workflow: Workflow;
  prompt: string;
  model: string;
  runs: number;
  created_at: string;
  supported_controls?: string[];
}

export interface ActiveJobState extends JobContext {
  status: JobStatus;
  currentStep: number;
  totalSteps: number;
  elapsed: number;
  remaining: number;
  stageName: string;
  stageIndex: number;
  batchLabel: string;
  batchIndex: number;
  paused: boolean;
  message: string;
  outputs: GalleryAsset[];
}

// ── SSE event types ────────────────────────────────────────────────────────────

export interface StepEvent {
  type: 'step_progress';
  job_id: string;
  current_step: number;
  total_steps: number;
  elapsed_secs: number;
  eta_secs?: number;
  workflow_stage_name?: string;
  workflow_stage_index?: number;
  run_index?: number;
  total_runs?: number;
  ran_iterations?: number;
  total_iterations?: number;
}

export interface BatchCompletedEvent {
  type: 'batch_completed';
  job_id: string;
  run_index: number;
  total_runs: number;
  ran_iterations: number;
  output_path: string;
  asset: GalleryAsset;
}

export interface JobCompletedEvent {
  type: 'job_completed';
  job_id: string;
  total_runs: number;
  outputs: GalleryAsset[];
}

export interface JobFailedEvent {
  type: 'job_failed';
  job_id: string;
  error: string;
}

export interface JobCancelledEvent {
  type: 'batch_cancelled';
  job_id: string;
}

export interface ProgressTextEvent {
  type: 'progress_text';
  job_id: string;
  text: string;
}

export interface JobPausedEvent {
  type: 'job_paused';
  job_id: string;
}

export interface JobResumedEvent {
  type: 'job_resumed';
  job_id: string;
}

export type SSEEvent =
  | StepEvent
  | BatchCompletedEvent
  | JobCompletedEvent
  | JobFailedEvent
  | JobCancelledEvent
  | ProgressTextEvent
  | JobPausedEvent
  | JobResumedEvent;

// NOTE: batch_completed is NOT a terminal event (see multi-run guard in sse.ts)
export type SSETerminalEvent = 'job_completed' | 'job_failed' | 'batch_cancelled';

// ── Gallery types ──────────────────────────────────────────────────────────────

export interface GalleryPage {
  assets: GalleryAsset[];
  page: number;
  total_pages: number;
  total_count: number;
}

// ── Config types ───────────────────────────────────────────────────────────────

export interface AppConfig {
  output_dir: string;
  log_level: string;
  ui: WebUiConfig;
  models: Record<string, unknown>;
}

export interface ModelOperationResult {
  tone: 'success' | 'error' | 'info';
  message: string;
  detail: string;
}

// ── Page/navigation types ──────────────────────────────────────────────────────

export type PageId = 'workspace' | 'gallery' | 'config' | 'models';

export interface RouterState {
  page: PageId;
  params: Record<string, string>;
}

// ── Model inventory types ──────────────────────────────────────────────────────

export interface ModelEntry {
  name: string;
  family: string;
  size_label?: string;
  /** Alias used by some backends */
  size?: string;
}

export interface VideoModelEntry {
  name: string;
  family: string;
  supports_i2v: boolean;
  size_label?: string;
}

export interface LoraEntry {
  name: string;
  size_label?: string;
  file_size_mb?: number;
}

export interface ModelInventory {
  models_dir: string;
  loras_dir: string;
  image_models: ModelEntry[];
  video_models: VideoModelEntry[];
  loras: LoraEntry[];
  huggingface_configured: boolean;
  huggingface_token_env_var: string | null;
}
