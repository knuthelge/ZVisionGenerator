// ── Core data types ──────────────────────────────────────────────────────────

export type Workflow = 'txt2img' | 'img2img' | 'txt2vid' | 'img2vid';

export type WorkflowMode = 'image' | 'video';

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
  duration?: number;
  reuse_workspace_url: string; // hash URL: '#/workspace?workflow=...&prompt=...'
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
  defaults: ModelDefaults;
  video_defaults: VideoModelDefaults;
  image_model_defaults: Record<string, ModelDefaults>;
  video_model_defaults: Record<string, ModelDefaults>;
  current_image_model: string;
  current_video_model: string;
  config: WebUiConfig;
  output_dir?: string;
  quantize_options?: number[];
  image_ratios?: string[];
  video_ratios?: string[];
  image_size_options?: Record<string, string[]>;
  video_size_options?: Record<string, string[]>;
}

export interface ModelDefaults {
  steps: number;
  guidance: number;
  width: number;
  height: number;
  quantize?: boolean;
}

export interface VideoModelDefaults {
  steps: number;
  guidance: number;
  width: number;
  height: number;
  frame_count: number;
  fps: number;
  motion_strength?: number;
  quantize?: boolean;
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
  prompt: string;
  negativePrompt: string;
  model: string;
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
  motionStrength: number;
  quantize: boolean;
  historyCollapsed: boolean;
  lastGeneratedAt: string | null;
  version: number;
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
  available_surfaces: string[];
  huggingface_configured: boolean;
  huggingface_token_env_var: string | null;
}
