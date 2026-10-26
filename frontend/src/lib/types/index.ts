// ── Core data types ──────────────────────────────────────────────────────────

export type Workflow = 'txt2img' | 'img2img' | 'txt2vid' | 'img2vid' | 'extend';

/** Workflow of a job: a Workspace workflow, or an upscale of an existing image. */
export type JobWorkflow = Workflow | 'upscale';

export type UpscaleFactor = 2 | 4;

/** How an image upscale runs: the SeedVR2 model (macOS), or the image model's creative pass. */
export type UpscaleMethod = 'seedvr2' | 'creative';

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
  id: string;
  url: string;
  thumbnail_url: string;
  filename: string;
  created_at: string;
  workflow: Workflow;
  prompt: string;
  /** A video's recorded scene (who and where, look, camera); null for images and videos made without one. */
  scene?: string | null;
  model: string;
  width?: number;
  height?: number;
  ratio?: string | null;
  size?: string | null;
  frame_count?: number | null;
  image_path?: string | null;
  /** The image a first-and-last-frame clip ends on; absent or null for other assets. */
  end_image_path?: string | null;
  /** Absolute path of the asset file on the host; lets the asset become a reference image. */
  file_path?: string;
  seed?: number | null;
  steps?: number | null;
  guidance?: number | null;
  lora?: string | null;
  duration?: number;
  reuse_workspace_url: string; // hash URL: '#/workspace?workflow=...&prompt=...'
  has_reusable_config?: boolean;
  reuse_state?: ReuseState;
  media_type: 'image' | 'video';
  /** Recorded settings beyond the reuse fields, shown in the asset viewer. */
  details?: AssetDetails;
  /** How the prompt was enhanced; null when it was not, or the file predates the record. */
  enhance?: AssetEnhance | null;
  /** Upscale options for an image (null for videos and images of unknown size). */
  upscale?: AssetUpscale | null;
  /** Whether a video can be extended; null for images and where the platform cannot extend. */
  extend?: AssetExtend | null;
  /** The video an extension continued, as recorded; null for other assets. */
  source_video?: string | null;
  /** The frames an extension added, as recorded; null for other assets. */
  extend_frames?: number | null;
}

/** How an asset's prompt was enhanced, as recorded in the file; the viewer shows the summary and the user's own text. */
export interface AssetEnhance extends EnhanceRecord {
  /** One line: the enhancer, then the options that are not defaults, e.g. "Fast · Cinematic, Melancholic, Longer". */
  summary: string;
  tier: 'fast' | 'smart' | 'custom';
  /** The user's own prompt (and scene, for video) before the rewrite. */
  prompt_input: string | null;
  scene_input: string | null;
}

/** Whether a video can be extended, with the reason when it cannot. */
export interface AssetExtend {
  available: boolean;
  /** Why the video cannot be extended; null when it can. */
  reason: string | null;
}

/** A problem that stops a video from being extended. */
export interface ExtendProblem {
  code: string;
  message: string;
}

/** How a clip or an extension window fits this machine's memory; null where the server cannot tell. */
export interface ClipMemoryFit {
  status: string;
  required_gb: number;
  budget_gb: number;
  /** `discrete` on a CUDA GPU; absent (unified memory) on Apple Silicon. */
  kind?: MemoryKind;
  /** Discrete GPUs only: the memory the figures are about, whichever is closer to its limit. */
  resource?: 'gpu' | 'system';
  /** MiniMax H3 on a discrete GPU only: the model family, whose advice differs from LTX's. */
  family?: 'h3';
  /** MiniMax H3 only: the estimated generation time, in seconds. */
  expected_seconds?: number;
}

/** What `POST /api/video/memory-fit` reports: the size and length a video job would use, and how that clip fits memory. */
export interface VideoMemoryFitInfo {
  memory_fit: ClipMemoryFit | null;
  width: number;
  height: number;
  frames: number;
}

/**
 * What `POST /api/extend/source` reports about a video chosen for extending.
 *
 * Facts the server could not read are null. `width` and `height` are the displayed size.
 */
export interface ExtendSourceInfo {
  path: string;
  filename: string;
  /** Where the browser can load the video; null when it lies outside the output folder. */
  media_url: string | null;
  width: number | null;
  height: number | null;
  rotation: number | null;
  fps: number | null;
  frame_count: number | null;
  duration_seconds: number | null;
  has_audio: boolean | null;
  /** The scene recorded in the video, or null. */
  scene: string | null;
  /** The prompt recorded in the video, or null. */
  prompt: string | null;
  /** The video recorded only a single prompt and no scene. */
  legacy: boolean;
  /** The model it was made with, when this machine offers it. */
  model: string | null;
  lora: string | null;
  problems: ExtendProblem[];
  /** The new frames, snapped to a multiple of 8 (a 2 s default when none was requested); null when the frame rate is unknown. */
  extend_frames: number | null;
  window_frames: number | null;
  memory_fit: ClipMemoryFit | null;
}

/** How a saved asset was made, as recorded in its embedded settings. */
export interface AssetDetails {
  /** The workflow as recorded, e.g. `upscale`; `workflow` holds the one Reuse settings uses. */
  recorded_workflow?: string | null;
  negative_prompt?: string | null;
  scheduler?: string | null;
  model_family?: string | null;
  image_strength?: number | null;
  generation?: AssetGeneration;
  /** The image an upscale started from; `id` is set while it is still in the gallery. */
  source?: { path: string; width: number | null; height: number | null; id: string | null } | null;
}

export interface AssetGeneration {
  /** Seconds the model spent generating. */
  time?: number;
  quantize?: number;
  upscale?: {
    factor?: number;
    method?: UpscaleMethod;
    model?: string;
    quantize?: number | null;
    softness?: number | null;
    seed?: number | null;
    denoise?: number;
    steps?: number;
    guidance?: number;
    pre_sharpen?: number;
  };
  sharpen?: number;
  contrast?: number;
  saturation?: number;
  audio?: boolean;
  output_format?: string;
}

export interface AssetUpscaleFactor {
  factor: UpscaleFactor;
  width: number;
  height: number;
  allowed: boolean;
  reason: string | null;
  /** Set on a SeedVR2 option that is allowed but needs care: it may not fit in memory, or a video runs one frame per batch. */
  warning?: string | null;
  /** Why there is a warning: `memory`, or `one_frame_batch` for a video upscaled one frame at a time. */
  warning_code?: 'memory' | 'one_frame_batch' | null;
}

/** The SeedVR2 upscale an asset offers. */
export interface AssetSeedvr2Upscale {
  options: AssetUpscaleFactor[];
  /** Label of the model tier that runs, e.g. `SeedVR2 7B q8`. */
  model: string;
  downloaded: boolean;
  /** Size of the first-use download; null once the model is on this machine. */
  download_size_label: string | null;
}

export interface AssetUpscale {
  /** Creative upscale options (the image model's img2img pass). */
  factors: AssetUpscaleFactor[];
  /** SeedVR2 upscale options, or null where SeedVR2 is not offered. */
  seedvr2: AssetSeedvr2Upscale | null;
}

export type MemoryFitStatus = 'fits' | 'tight' | 'too_large';

export interface MemoryFitEstimate {
  status: MemoryFitStatus;
  /** GPU memory needed (unified memory on Apple Silicon). */
  required_gb: number;
  /** Discrete GPUs only: system memory needed for the weights that stream to the GPU. */
  system_gb?: number;
}

/** `unified`: Apple Silicon, one memory pool. `discrete`: a CUDA GPU with its own memory, fed from system memory. */
export type MemoryKind = 'unified' | 'discrete';

/** Estimated memory need per quantize level (`none`, `4`, `8`) against this machine's GPU budget. */
export interface MemoryFit {
  kind?: MemoryKind;
  budget_gb: number;
  /** Discrete GPUs only: this machine's system memory. */
  system_budget_gb?: number;
  by_quantize: Record<string, MemoryFitEstimate>;
  /** LTX MLX video only: the estimate when low-memory mode is off and every component stays loaded. */
  without_low_memory?: MemoryFitEstimate;
  /** Caveat to show with the estimate, e.g. the clip a video estimate is for. */
  note?: string;
}

/** Local availability fields shared by every model listing; `null` means unknown. */
export interface ModelStatusFields {
  downloaded?: boolean | null;
  memory_fit?: MemoryFit | null;
}

export interface ModelOption extends ModelStatusFields {
  id: string;
  label: string;
  type: 'image' | 'video';
  /** One-time download size on this platform, e.g. "35 GB"; image models only, null when unknown. */
  download_size_label?: string | null;
}

/** What the first-run dialog needs: whether it was skipped for good and the model it preselects. */
export interface FirstRunContext {
  skipped: boolean;
  recommended_model: string | null;
}

export interface LoraInfo {
  name: string;
  path: string;
  /** The model family the LoRA was trained for; null or absent when unknown. */
  family?: string | null;
  /** The model size the LoRA fits within its family (for example `4b`); null or absent when it fits any. */
  size_variant?: string | null;
}

export interface WorkspaceContext {
  image_models: ModelOption[];
  video_models: ModelOption[];
  loras: LoraInfo[];
  history_assets: GalleryAsset[];
  active_job: JobSnapshot | null;
  queued_jobs?: JobSnapshot[];
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
  image_size_dimensions: Record<string, Record<string, [number, number]>>;
  /** Resolved [width, height] per video ratio/size preset. */
  video_size_dimensions?: Record<string, Record<string, [number, number]>>;
  scheduler_options: string[];
  prompt_sources: PromptSource[];
  default_prompt_source: PromptSource;
  prompt_file: PromptFileContract;
  workflow_contract: WorkflowContract;
  prompt_enhancer?: PromptEnhancerContract;
  first_run?: FirstRunContext;
}

// ── Prompt enhancer ────────────────────────────────────────────────────────────

export type EnhanceAxisKey = 'style' | 'mood' | 'theme' | 'details' | 'camera' | 'pace' | 'length' | 'motion' | 'sound';

/** Which parts of a video prompt an enhancement writes: the scene and the action, or only the action (the scene is kept as written). */
export type EnhanceParts = 'both' | 'action';

export interface EnhanceSettings {
  style: string;
  mood: string;
  theme: string;
  details: string[];
  length: string;
  /** Video only. */
  camera: string;
  /** Video only. */
  pace: string;
  motion: string[];
  sound: string[];
  /** Free-text direction for the enhancer; empty for none. */
  note: string;
  /** Video only. In the effective settings it is the choice for the current workflow; never stored (see `partsByWorkflow`). */
  parts?: EnhanceParts;
  /** The Rewrite choice remembered for each video workflow; a workflow with none uses its own default. */
  partsByWorkflow?: Partial<Record<Workflow, EnhanceParts>>;
}

/** Facts about the job that shape a rewrite, sent with an on-demand enhancement. */
export interface EnhanceContext {
  ratio: string | null;
  width: number | null;
  height: number | null;
  frames: number | null;
  fps: number | null;
  audio: boolean;
  reference_image: boolean;
  /** The clip also ends on a supplied last-frame image. */
  end_image?: boolean;
  /** Extend only: the video being continued; the server reads its size, frame rate and recorded text. */
  source_video?: string | null;
  /** Extend only: the new frames, a multiple of the video model's extend step. */
  extend_frames?: number | null;
  /** Extend only: the selected video model, whose family sets the extend step; absent or null means the LTX step. */
  model?: string | null;
}

export interface EnhanceAxis {
  key: EnhanceAxisKey;
  label: string;
  multi: boolean;
  video_only: boolean;
  /** Shown only for video with audio on. */
  audio_only: boolean;
  options: EnhanceOptionInfo[];
  default: string[];
}

/** One option of an enhancement axis, as the Web UI lists it. */
export interface EnhanceOptionInfo {
  slug: string;
  label: string;
  /** One line on what the option asks the enhancer for. */
  description: string;
  /** The menu group it is listed under (Style: Photo or Art; Theme: its five groups), or null. */
  group: string | null;
}

export interface PromptEnhancerContract {
  matrix: { axes: EnhanceAxis[]; defaults: EnhanceSettings };
  /** Effective model repo/path; null when none is configured (see `error`). */
  model: string | null;
  revision: string | null;
  downloaded: boolean;
  /** Approximate first-use download size, null once downloaded. */
  download_size_label: string | null;
  default_max_words: number;
  /** Whether this platform's video backend writes an audio track; without one the Sound axis never applies. */
  video_audio?: boolean;
  /** The larger Smart enhancer; null when a custom model is configured (then there is no Fast/Smart choice). */
  smart?: SmartEnhancerContract | null;
  error: string | null;
}

export interface SmartEnhancerContract {
  /** Approximate first-use download size, null once downloaded. */
  download_size_label: string | null;
  /** Why Smart cannot run on this machine, or null when it can. */
  unavailable: string | null;
}

/** Which enhancer model runs: the default Fast one or the larger Smart one. */
export type EnhancerTier = 'fast' | 'smart';

/** What an image enhancement writes: a rewritten prompt or a structured JSON caption. */
export type EnhanceFormat = 'text' | 'json';

/** What an Enhanced prompt (or Enhanced scene and prompt) was produced from; null for hand-typed text (never out of date). */
export interface EnhancedSource {
  prompt: string;
  /** The scene it was produced beside; missing in drafts saved before scenes existed. */
  scene?: string;
  mode: WorkflowMode;
  /** Extend only: the source video it was produced for; a different source makes it out of date. */
  sourceVideoPath?: string | null;
  /** Extend only: the new frames it was produced for. */
  extendFrames?: number | null;
  /** How the rewrite was made; Generate sends it so the asset records it. Missing for typed-in or older Enhanced text. */
  record?: EnhanceRecord;
}

/** The options, model and output format an Enhanced text was written with, as the server records them. */
export interface EnhanceRecord {
  settings: Partial<EnhanceSettings>;
  /** The enhancer that wrote it, ``REPO[@REVISION]``; the server works out Fast, Smart or custom from it. */
  model: string;
  format: EnhanceFormat;
}

/**
 * One line of the enhance stream. Video frames carry the server's split `scene` and `prompt`;
 * image frames carry only `prompt` on `done` (and neither part on `text`).
 */
export type EnhanceFrame =
  | { type: 'status'; phase: 'downloading' | 'loading' | 'generating' | 'generating_cpu' }
  | { type: 'text'; text: string; scene?: string; prompt?: string }
  | { type: 'done'; prompt: string; scene?: string; clamped: boolean; warning?: string; model?: string }
  | { type: 'error'; detail: string };

export interface PromptFileContract {
  accepted_extensions: string[];
  selection_required: boolean;
  trust_boundary: {
    scope: 'server_host_only';
    manual_entry: 'submitted_value_kept_until_backend_validation';
    picker: 'in_app_browser';
    read_write: 'yaml_files_only';
  };
  help: {
    path: string;
    option_required: string;
    option_optional: string;
    empty_options: string;
    stale_selection: string;
    loaded: string;
    ignored_negative_video: string;
    ignored_negative_unsupported: string;
    ignored_scene_image?: string;
  };
}

/** A prompt, negative prompt or snippet value as the prompt builder edits it. */
export type PromptValue =
  | { kind: 'text'; text: string }
  | { kind: 'fields'; fields: PromptField[] }
  /** Lists, nested mappings or other values the builder shows read-only and never rewrites. */
  | { kind: 'structured'; data: unknown };

export interface PromptField {
  key: string;
  value: string;
}

/** An entry's `enhance:` value: off, default options, or the mapping as written. */
export type EntryEnhance = null | true | Partial<Record<EnhanceAxisKey | 'note' | 'parts', string | string[]>>;

export interface DocumentEntry {
  id: string;
  /** The entry's `scene:` (video only), or null when it has none. */
  scene: PromptValue | null;
  prompt: PromptValue;
  negative: PromptValue | null;
  active: boolean;
  enhance: EntryEnhance;
}

export interface DocumentSet {
  id: string;
  name: string;
  entries: DocumentEntry[];
}

export interface DocumentSnippet {
  id: string;
  name: string;
  value: PromptValue;
}

/** A prompt file as the builder edits it. Ids from a load are positional (`n0`, `s0`, `s0.e1`). */
export interface PromptDocument {
  snippets: DocumentSnippet[];
  sets: DocumentSet[];
}

export interface PromptDocumentLoad {
  path: string;
  revision: string;
  raw_text: string;
  /** Missing when the file can't be built; `problem` then says why (repair view). */
  document?: PromptDocument;
  problem?: string;
  enhance_matrix: { axes: EnhanceAxis[]; defaults: EnhanceSettings };
}

export interface PromptDocumentSaved {
  path: string;
  revision: string;
  raw_text: string;
  document: PromptDocument;
  /** Every id sent, mapped to its positional id in the saved file. */
  ids: Record<string, string>;
  /** Active prompt ids before the save mapped to their ids after it. */
  option_id_map: Record<string, string>;
  warnings: string[];
}

/** The text of one prompt-file entry the Workspace saves. */
export interface PromptEntryText {
  prompt: string;
  scene?: string;
  negative?: string;
}

export interface PromptEntryAppended {
  path: string;
  revision: string;
  entry_id: string;
  set_name: string;
}

export interface DocumentProblem {
  target: string;
  severity: 'error' | 'warning';
  message: string;
}

export interface PromptDocumentPreview {
  entries: Record<string, { prompt: string; negative: string | null; scene?: string | null }>;
  problems: DocumentProblem[];
  snippet_uses: Record<string, number>;
  rolled: { entry_id: string; prompt: string } | null;
}

export interface PromptFileOption {
  id: string;
  set_name: string;
  source_index: number;
  label: string;
  prompt_preview: string;
  negative_preview: string | null;
  /** The entry's own scene, or null when it has none. */
  scene_preview?: string | null;
  /** Whether the entry has its own `scene:`. */
  has_scene?: boolean;
  /** Prompt-file `enhance:` settings as an `--enhance` spec (as a video run applies them), or null. */
  enhance?: string | null;
  /** The same settings as a text image run applies them, or null when such a run does not enhance the entry. */
  enhance_image?: string | null;
  /** The same settings as a JSON-caption image run applies them (it enhances every entry with `enhance:`), or null. */
  enhance_caption?: string | null;
}

export interface PromptFileInspection {
  path: string;
  options: PromptFileOption[];
}

/** What a Browse button chooses; the server owns the list (`web/path_browser.py`). */
export type BrowsePurposeName = 'output_directory' | 'prompt_file' | 'prompt_file_folder' | 'checkpoint_file' | 'lora_file' | 'source_video' | 'reference_image';

export interface BrowsePurpose {
  name: BrowsePurposeName;
  kind: 'directory' | 'existing_file';
  label: string;
  accepted_extensions: string[];
}

export interface BrowseEntry {
  name: string;
  path: string;
  kind: 'directory' | 'file';
  size: number | null;
  /** Seconds since the epoch. */
  modified: number | null;
  selectable: boolean;
}

export interface BrowseListing {
  path: string;
  parent: string | null;
  separator: string;
  /** The file the request named, when it named a file rather than a folder. */
  focus: string | null;
  entries: BrowseEntry[];
  /** Files left out because the purpose can't choose them. */
  hidden_count: number;
}

export interface BrowsePlace {
  id: string;
  label: string;
  path: string;
  /** The place this purpose opens at when the field is empty. */
  default: boolean;
}

export interface BrowsePlaces {
  purpose: BrowsePurpose;
  places: BrowsePlace[];
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
  supports_img2img: boolean;
  supports_upscale: boolean;
  supports_json_prompt: boolean;
  supports_first_sigma: boolean;
  supports_scheduler: boolean;
  dimension_min: number;
  dimension_max: number | null;
  dimension_step: number;
  enhance_max_words?: number;
  /** The detected model family; null when it could not be detected. */
  family?: string | null;
  /** The model's size within its family (for example `4b`); null when it has none or is unknown. */
  size_variant?: string | null;
}

/** The frame counts a video family accepts: `alignment * k + offset` for `k >= 1`; `round` picks how another count moves to one. */
export interface FrameRule {
  alignment: number;
  offset: number;
  round: 'nearest' | 'up';
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
  /** The frame counts the model's family accepts. */
  frame_rule: FrameRule;
  /** Workflow controls this model does not have (`low_memory`, `video_upscale_enabled`, ...); they are not shown. */
  hidden_controls?: string[];
  /** The model can extend a clip. */
  supports_extend: boolean;
  /** The only step counts the model offers (ascending); null or absent: any count up to `max_steps`. */
  step_options?: number[] | null;
  /** An extension adds frames in multiples of this. */
  extend_frame_step: number;
  upscale: UpscaleDefaults;
  enhance_max_words?: number;
  /** The detected model family; null when it could not be detected. */
  family?: string | null;
}

export interface WorkflowContractEntry {
  mode: WorkflowMode;
  model_kind: WorkflowMode;
  visible_controls: string[];
  supports_reference_image: boolean;
  requires_reference_image: boolean;
  /** The workflow continues a chosen video (extend). */
  requires_source_video: boolean;
  clear_fields: string[];
}

export interface WorkflowContract {
  values: Workflow[];
  definitions: Record<Workflow, WorkflowContractEntry>;
  field_precedence: {
    defaults: string[];
    dimensions: string;
  };
}

export interface WebUiConfig {
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
  huggingface_token_source?: string | null;
  image_size_labels?: { value: string; label: string }[];
  default_image_size?: string;
}

export type WritableConfigValue = string | number | boolean | null;

export interface WritableConfigField {
  key: string;
  type: 'string' | 'number' | 'boolean';
  clearable: boolean;
  empty_string: 'clear' | 'reject' | 'coerce';
  omitted: 'unchanged';
  null: 'clear' | 'reject';
  value: WritableConfigValue;
  persisted_value: WritableConfigValue;
  persisted_value_shape: string;
  effective_value: WritableConfigValue;
  effective_value_shape: string;
  default_source: string;
  validation_rules: string[];
  owning_consumer: string;
}

export interface WritableConfigSchema {
  version: number;
  semantics: {
    omitted: 'unchanged';
    null: 'clear for clearable fields';
    empty_string: 'normalized by field empty_string behavior before persistence';
  };
  fields: WritableConfigField[];
}

// ── Draft / workspace form state ──────────────────────────────────────────────

export interface DraftState {
  workflow: Workflow;
  promptSource: PromptSource;
  prompt: string;
  /** Video scene (who and where, look, camera); the default scene in prompt-file mode. Image workflows ignore it. */
  scene: string;
  jsonPromptEnabled: boolean;
  jsonPrompt: string;
  firstSigma: number | null;
  negativePrompt: string;
  promptFilePath: string | null;
  promptFileOptionIds: string[];
  /** The prompts file the Workspace last loaded and how many prompts it holds; null until one loads. */
  promptFileLoaded: { path: string; count: number } | null;
  model: string;
  ratio: string;
  /** `ratio`: submit the ratio/size preset. `custom`: submit width/height. */
  dimensionMode: 'ratio' | 'custom';
  size: string;
  steps: number;
  guidance: number;
  width: number;
  height: number;
  runs: number;
  seed: number | null; // null = random
  loraString: string;
  /** LoRAs last chosen for each model family (keyed `model:<id>` when the family is unknown); `loraString` holds the current family's. */
  lorasByFamily: Record<string, string>;
  referenceImagePath: string | null;
  /** Image to Video: the image the clip ends on; null when the model has no last frame or none is chosen. */
  endImagePath: string | null;
  referenceImageStrength: number;
  frameCount: number;
  fps: number;
  audio: boolean;
  lowMemory: boolean;
  upscaleEnabled: boolean;
  upscaleFactor: number;
  quantize: number | null;
  historyCollapsed: boolean;
  /** Compose and Settings are folded into a narrow strip. */
  sidebarCollapsed: boolean;
  lastGeneratedAt: string | null;
  version: number;
  scheduler: string | null;
  postprocessSharpenEnabled: boolean;
  /** Final sharpen amount; null follows the config (`sharpening.normal` / `upscaled`). */
  postprocessSharpenAmount: number | null;
  postprocessContrastEnabled: boolean;
  postprocessContrastAmount: number;
  postprocessSaturationEnabled: boolean;
  postprocessSaturationAmount: number;
  upscaleDenoise: number | null;
  upscaleSteps: number | null;
  upscaleGuidance: number | null;
  upscaleSharpen: boolean;
  /** Pause after the small render, to approve or discard it before the upscale. */
  upscaleReview: boolean;
  videoUpscaleEnabled: boolean;
  videoUpscaleFactor: number;
  /** Pause after the half-size video render, to approve or discard it before the upscale. */
  videoUpscaleReview: boolean;
  /** Text from the Enhance button (or typed); when non-blank it is the inline prompt that gets generated. */
  enhancedPrompt: string;
  /** The Enhanced scene of a video; used together with the Enhanced prompt. */
  enhancedScene: string;
  enhancedFrom: EnhancedSource | null;
  /** Last-used enhancement options for images; null until the user picks any (backend defaults apply). */
  enhanceSettings: EnhanceSettings | null;
  /** Last-used enhancement options for video, with the Rewrite choice of each video workflow; null until any are picked. */
  enhanceSettingsVideo: EnhanceSettings | null;
  /** Enhance each image's prompt on the server at generation time. */
  enhanceAuto: boolean;
  /** The enhancer chosen in the popover; Smart falls back to Fast where it is unavailable. */
  enhanceModel: EnhancerTier;
  /** What image enhancement writes; JSON applies only to models that take JSON captions. */
  enhanceFormat: EnhanceFormat;
  /** Extend: the video to continue; null until one is chosen. Its facts are fetched again on load. */
  extendSourcePath: string | null;
  /** Extend: the new frames, a multiple of 8; null until a source sets the 2 s default. Never aligned to 8k+1. */
  extendFrames: number | null;
  /** Extend: the source could not be found where Reuse recorded it, so it must be chosen again. */
  missingSourceVideo: boolean;
  /** Extend: the source whose Scene and Prompt were settled, and the text its prefill put there (to tell edits from prefill). */
  extendPrefill: { path: string; scene: string; prompt: string } | null;
  /** The slug of the settings preset the form started from; null when none. Cleared on a mode switch, Reset, a loaded source's settings and when the preset is deleted. */
  appliedPreset: string | null;
}

export type WorkspacePrefill = Partial<Omit<DraftState, 'historyCollapsed' | 'sidebarCollapsed' | 'lastGeneratedAt' | 'version'>>;

// ── Settings presets ──────────────────────────────────────────────────────────

export type PresetMode = 'image' | 'video';

/** One preset setting: a form field's value, or null for "not set" (off or automatic). */
export type PresetValue = string | number | boolean | null;

/** Every setting of a mode, by Web form field name; keys the preset does not set are null. */
export type PresetSettings = Record<string, PresetValue>;

/** A saved working setup, as `/api/presets` sends it. */
export interface SettingsPreset {
  slug: string;
  name: string;
  mode: PresetMode;
  model: string | null;
  settings: PresetSettings;
}

/** What creating a preset takes. */
export type SettingsPresetBody = Omit<SettingsPreset, 'slug'>;

/** What updating a preset takes; its mode and slug are in the route. */
export type SettingsPresetUpdate = Omit<SettingsPreset, 'slug' | 'mode'>;

// ── Job / SSE types ────────────────────────────────────────────────────────────

// 'unknown' is client-only: the server restarted and no longer has the job, so its outcome cannot be known.
export type JobStatus = 'queued' | 'pending' | 'running' | 'paused' | 'completed' | 'failed' | 'cancelled' | 'unknown';

export interface JobContext {
  id?: string;
  job_id: string;
  workflow: JobWorkflow;
  /** `video` for a job that upscales an existing video file; absent on other jobs. */
  media_type?: 'image' | 'video';
  prompt: string;
  model: string;
  /** How the Web UI shows `model`: its configured name, else the id. Absent on jobs from an older server. */
  model_label?: string;
  runs: number;
  created_at: string;
  supported_controls?: string[];
  /** Size label, e.g. "2:3 · m · 8 steps". */
  meta?: string;
  /** Size in pixels the job renders at; absent on jobs from an older server. */
  width?: number | null;
  height?: number | null;
  /** Form fields the job was submitted with; the submit response omits them, so the Workspace adds them. */
  settings?: JobSettings;
  /** 1-based place in the queue; null once the job is the active one. */
  queue_position?: number | null;
  /** Short notes about how the job deviates from what was asked, e.g. "original settings unknown". */
  notices?: string[];
}

/** A small render that waits for the user to approve or discard it before its upscale. */
export interface JobReview {
  /** 1-based position of the generation under review within the job. */
  iteration: number;
}

/** Form fields a job was submitted with; a field sent more than once is a list. */
export type JobSettings = Record<string, string | string[]>;

export interface JobSnapshot extends JobContext {
  id: string;
  job_type: string;
  status: JobStatus;
  completed_at?: string | number | null;
  event_count: number;
  last_event?: Record<string, unknown> | null;
  terminal_event?: 'job_completed' | 'job_failed' | 'job_cancelled';
  paused: boolean;
  result_path?: string | null;
  outputs?: GalleryAsset[];
  /** Version of the job's current live preview; 0 when none is available. */
  preview_version?: number;
  started_at?: number | null;
  /** The render waiting for approve or discard before its upscale; null otherwise. */
  review?: JobReview | null;
}

/** The active generation job and the queued ones, oldest first; a job is never in both. */
export interface JobList {
  active_job: JobSnapshot | null;
  queued_jobs: JobSnapshot[];
}

export interface ActiveJobState extends JobContext {
  status: JobStatus;
  currentStep: number;
  totalSteps: number;
  elapsed: number;
  remaining: number;
  stageName: string;
  stageIndex: number;
  promptNumber?: number;
  promptCount?: number;
  batchLabel: string;
  batchIndex: number;
  paused: boolean;
  message: string;
  outputs: GalleryAsset[];
  /** URL of the latest live preview of the current generation; null when none is available. */
  previewUrl: string | null;
  /** The render under review (shown as `previewUrl`); the job waits for approve or discard while this is set. */
  review?: JobReview | null;
  /** Auto-enhanced prompt of the current generation, when the job enhances prompts. */
  enhancedPrompt?: string;
  /** Outcome of auto enhancement for the current generation, planned before the model loaded. */
  enhanceStatus?: EnhanceStatus;
  /** Where the job is in the server's phase order; unset until an event says. */
  phase?: JobPhase;
  /** Server timestamps (epoch seconds) at which each phase started and ended, for the setup durations. */
  phaseMarks?: PhaseMarks;
  /** Rewrites planned before the model loads; 0 when the job does not enhance prompts. */
  enhanceTotal?: number;
  /** Bits of the quantized copy the model load saves first (first use of a model only); unset for a plain load. */
  loadQuant?: number;
  /** Progress of the model download the job is waiting on; cleared once the model or enhancer starts loading. */
  download?: JobDownload | null;
  /** How the job's download ended when the job was cancelled: stopped (owner) or detached (waiting on another download). */
  downloadStop?: 'stopped' | 'detached';
  /** Newest `model_loading` phase: the model's files are downloading, loading into memory, or being saved as a quantized copy. */
  loadPhase?: LoadPhase;
  /** The quantized copy replaces the downloaded files (SeedVR2), so saving it frees disk. */
  loadReplacesDownload?: boolean;
  /** The current item has finished (its generation_finished arrived); the next item has not started yet. */
  itemFinished?: boolean;
  /** Ordered workflow stages of the current item, e.g. ["text_to_image", "upscale"]. */
  stageNames?: string[];
  /** Step counts of the current item's stepped stages that have run, by stage name. */
  stageSteps?: Record<string, number>;
  /** Backend phase of the last step event, e.g. "image_upscale" or "video_upscale_stage_1". */
  stepPhase?: string;
  /** A control the server accepted that has not taken effect yet (pause waits for the current image). */
  controlQueued?: JobControlAction | null;
  /** 'reconnecting' while the event stream is down and the shown progress may be stale. */
  connection?: 'live' | 'reconnecting';
  /** Client time (ms since epoch) of the last event received. */
  lastEventAt?: number;
  /** Server clock minus client clock, in seconds, so live phase durations count from server timestamps. */
  serverOffset?: number;
}

/** Phase of a `model_loading` event. */
export type LoadPhase = 'building' | 'downloading' | 'loading' | 'saving_quant';

/** Server phase order of a job: queued, prompt enhancement, model load, then the generation loop. */
export type JobPhase = 'queue' | 'enhance' | 'load' | 'generate';

/** Epoch-second start and end of the setup phases, as reported by the server. */
export interface PhaseMarks {
  queue?: { start?: number; end?: number };
  enhance?: { start?: number; end?: number };
  load?: { start?: number; end?: number };
  generate?: { start?: number; end?: number };
}

/** A job control the server accepts (`POST /jobs/{id}/controls/{action}`). */
export type JobControlAction = 'pause' | 'resume' | 'next' | 'repeat' | 'approve' | 'discard' | 'quit';

/** Per-iteration auto-enhancement outcome reported on `prompt_started`. */
export type EnhanceStatus = 'off' | 'enhanced' | 'failed' | 'skipped';

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
  preview_version?: number;
}

export interface GenerationFinishedEvent {
  type: 'generation_finished';
  job_id: string;
  job_type?: string;
  timestamp?: number;
  elapsed_secs?: number;
  mode?: WorkflowMode;
  status: 'success' | 'failed' | 'skipped';
  run_index?: number;
  total_runs?: number;
  ran_iterations?: number;
  total_iterations?: number;
  set_name?: string;
  prompt_index?: number;
  total_prompts?: number;
  prompt?: string;
  seed?: number;
  filename?: string;
  output_path?: string;
  generation_time?: number;
  asset?: GalleryAsset;
}

export interface BatchCompletedEvent {
  type: 'batch_completed';
  job_id: string;
  job_type?: string;
  timestamp?: number;
  elapsed_secs?: number;
  mode?: WorkflowMode;
  completed_iterations: number;
  total_iterations: number;
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
  type: 'job_cancelled';
  job_id: string;
  /** `removed` for a job taken off the queue before it started. */
  reason?: string;
  /** How the job's model download ended: `stopped` (the job owned it) or `detached` (it only waited on another download). */
  download?: 'stopped' | 'detached';
}

export interface ProgressTextEvent {
  type: 'progress_text';
  job_id: string;
  text: string;
}

export interface JobPausedEvent {
  type: 'job_paused';
  job_id: string;
  /** Set when the pause is a review before upscaling; the small render is the job preview at `preview_version`. */
  review?: JobReview;
  preview_version?: number;
}

export interface JobResumedEvent {
  type: 'job_resumed';
  job_id: string;
}

/** Byte progress of one part of a model download, as carried by `model_downloading` and the downloads lane. */
export interface DownloadProgressFields {
  repo: string;
  /** 1-based index of the part (repo) being downloaded. */
  part: number;
  parts: number;
  bytes_done: number;
  bytes_total: number | null;
  bytes_per_sec: number | null;
  download_eta_secs: number | null;
  /** The download is another job's or the lane's; this one waits on it. */
  shared: boolean;
}

export interface ModelDownloadingEvent extends DownloadProgressFields {
  type: 'model_downloading';
  mode: 'image' | 'video';
  role: 'model' | 'enhancer';
  model: string;
}

/** The download a job is waiting on, taken from its newest `model_downloading` event. */
export interface JobDownload extends DownloadProgressFields {
  role: 'model' | 'enhancer';
  model: string;
}

export type SSEEvent =
  | StepEvent
  | ModelDownloadingEvent
  | GenerationFinishedEvent
  | BatchCompletedEvent
  | JobCompletedEvent
  | JobFailedEvent
  | JobCancelledEvent
  | ProgressTextEvent
  | JobPausedEvent
  | JobResumedEvent;

// NOTE: batch_completed is NOT a terminal event (see multi-run guard in sse.ts)
export type SSETerminalEvent = 'job_completed' | 'job_failed' | 'job_cancelled';

// ── Gallery types ──────────────────────────────────────────────────────────────

export interface GalleryPage {
  assets: GalleryAsset[];
  page: number;
  total_pages: number;
  total_count: number;
}

// ── Job log types ──────────────────────────────────────────────────────────────

/** How a logged job ended. */
export type JobLogStatus = 'completed' | 'failed' | 'cancelled';

/** The outcome filters the job log offers. */
export type JobLogFilter = 'all' | 'failed' | 'completed';

/** One finished job as the server's job log records it; times are epoch seconds. */
export interface JobLogEntry {
  job_id: string;
  job_type: string;
  workflow: string;
  model: string;
  prompt: string;
  settings: JobSettings;
  status: JobLogStatus;
  created_at: number | null;
  started_at: number | null;
  completed_at: number | null;
  duration_secs: number;
  outputs: string[];
  error: string | null;
  /** How many runs of the job failed; a completed or cancelled job can carry some. */
  failed_runs: number;
}

export interface JobLogPage {
  entries: JobLogEntry[];
  page: number;
  total_pages: number;
  total_count: number;
  page_size: number;
}

// ── Config types ───────────────────────────────────────────────────────────────

export interface AppConfig {
  output_dir: string;
  log_level: string;
  ui: WebUiConfig;
  writable_config: WritableConfigSchema;
  models: Record<string, unknown>;
}

export interface ModelOperationResult {
  tone: 'success' | 'error' | 'info';
  message: string;
  detail: string;
  /** True when the operation continues in the background (an LTX conversion); poll its status. */
  background?: boolean;
}

/** What the converter found in a video checkpoint (LTX-2.3, LTX-2.5 or MiniMax H3) before converting it. */
export interface CheckpointInspection {
  model_type: string;
  /** The version, layout and storage, e.g. "MiniMax H3, ComfyUI single file, pruned (rank 8), INT8 (195)". */
  summary: string;
  advice: string;
  tone: 'success' | 'warning' | 'info';
  /** LTX only. */
  distillation?: 'dev' | 'distilled' | 'unknown';
  recommend_distill?: boolean;
  recommended_quantize?: 4 | 8;
  /** MiniMax H3 only: what the file's name or metadata hints about a step-distilled checkpoint. */
  turbo_hints?: string[];
}

/** The current or last LTX conversion. */
export interface ConversionStatus {
  state: 'idle' | 'running' | 'done' | 'failed';
  name?: string;
  stage?: string;
  done?: number;
  total?: number;
  message?: string | null;
}

// ── Page/navigation types ──────────────────────────────────────────────────────

export type PageId = 'workspace' | 'prompts' | 'gallery' | 'config' | 'models';

export interface RouterState {
  page: PageId;
  params: Record<string, string>;
}

// ── Model inventory types ──────────────────────────────────────────────────────

/**
 * What deleting a model removes: its installed folder, its HuggingFace download (the alias stays), or only its
 * prepared copies (`copy`) while other models still use the download it shares with them.
 */
export interface ModelDeleteInfo {
  kind: 'installed' | 'huggingface' | 'copy';
  repo_id: string | null;
  /** Installed models linking into this HuggingFace download, which deleting it would break. */
  linked_by: string[];
  /** The model's stored quants (e.g. `atlas@q8`), deleted along with it. */
  stored_quants?: string[];
  /** For `copy`: the models still using the shared download, which stays. */
  shared_with?: string[];
  /** For a model built on first use: the source repos deleted with it (those no other built model needs). */
  source_repos?: string[];
}

/** One Models-page download in the downloads lane. */
export type DownloadItem = {
  id: string;
  kind: 'model' | 'enhancer';
  name: string;
  label: string;
  state: 'queued' | 'running' | 'done' | 'failed' | 'cancelled';
  queue_position?: number;
  message: string | null;
} & Partial<DownloadProgressFields>;

/** Response of `GET /api/models/downloads`: queued, running and recently finished items. */
export interface DownloadList {
  items: DownloadItem[];
}

/** What a Models-page download would fetch, from `GET /api/models/download/plan`. */
export interface DownloadPlan {
  kind: 'model' | 'enhancer';
  name: string;
  label: string;
  parts: { repo: string; files: number; bytes: number | null }[];
  total_bytes: number | null;
  /** Free space on the download drive against the estimate; `null` when it cannot be checked. */
  space: { path: string; needed_bytes: number; free_bytes: number; enough: boolean } | null;
}

/** A quantized copy of an installed model, used automatically when that quantize level is selected. */
export interface StoredQuantInfo {
  base_model: string;
  bits: number;
}

export interface ModelEntry extends ModelStatusFields {
  name: string;
  /** How the Web UI shows the model: its configured name, else `name`. */
  label: string;
  family: string;
  size_label?: string;
  /** Where the model comes from: `installed`, `alias`, or `seedvr2` for the plain-upscale model's stored copy. */
  source?: string;
  stored_quant?: StoredQuantInfo | null;
  delete?: ModelDeleteInfo | null;
  /** Disk use of the model's files (with its HuggingFace companions, and stored quants that have no row of their own). */
  disk_size_label?: string | null;
  /** Size of the first-use download, set on the SeedVR2 row. */
  download_size_label?: string | null;
  /** Files are missing, so the Models page offers Download. */
  download_needed: boolean;
}

export interface VideoModelEntry extends ModelStatusFields {
  name: string;
  /** How the Web UI shows the model: its configured name, else `name`. */
  label: string;
  family: string;
  supports_i2v: boolean;
  size_label?: string;
  /** The quantize level a converted LTX checkpoint is stored at (its only level); `null` for other video models. */
  converted_bits?: number | null;
  delete?: ModelDeleteInfo | null;
  /** Disk use of the model's files, as on image rows. */
  disk_size_label?: string | null;
  /** Files are missing, so the Models page offers Download. */
  download_needed: boolean;
}

/** A prompt-enhancer model configured on this platform: the Fast or Smart tier, or a custom `user_model`. */
export interface EnhancerModelEntry {
  tier: 'fast' | 'smart' | 'custom';
  repo: string;
  revision: string | null;
  downloaded: boolean;
  /** Disk use of its Hugging Face download (all revisions, partial downloads included), when any. */
  disk_size_label: string | null;
  /** Size of the first-use download, for the built-in tiers. */
  download_size_label: string | null;
  /** Why Smart cannot run on this machine, or `null`. */
  unavailable: string | null;
  /** Whether a Hugging Face download is on disk to delete (never for a local folder). */
  deletable: boolean;
  /** Files are missing, so the Models page offers Download. */
  download_needed: boolean;
}

/** Hugging Face cache content nothing on the Models page uses. */
export interface OtherDownloads {
  cache_dir: string;
  /** Model repos no alias, enhancer tier or installed model uses; `size_label` is what deleting one frees. */
  repos: { repo_id: string; size_label: string }[];
  /** Files in the cache's shared store that no download links to any more, or `null` when there are none. */
  stranded: { count: number; size_label: string } | null;
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
  enhancer_models?: EnhancerModelEntry[];
  other_downloads?: OtherDownloads;
  loras: LoraEntry[];
  /** Whether this platform can save quantized copies of models (macOS). */
  stored_quants_supported?: boolean;
  huggingface_configured: boolean;
  huggingface_token_source: string | null;
}
