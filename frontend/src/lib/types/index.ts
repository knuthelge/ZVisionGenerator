// ── Core data types ──────────────────────────────────────────────────────────

export type Workflow = 'txt2img' | 'img2img' | 'txt2vid' | 'img2vid' | 'extend';

/** The kinds of audio, in the order the Kind control lists them: also the workflows of the audio mode, one per kind. The server lists a workflow once a model of its kind can run. */
export const AUDIO_KINDS = ['music', 'speech', 'sound'] as const;
export type AudioWorkflow = (typeof AUDIO_KINDS)[number];

/** The types of a sound, as the Sound Type control lists them; `auto` leaves the choice (and the Auto length) to the model. */
export const SOUND_TYPES = ['auto', 'ambience', 'effect', 'one-shot'] as const;
export type SoundType = (typeof SOUND_TYPES)[number];

/** The five settings that steer how a built-in voice says the text, by their ids (the vocabulary is the workspace payload's `audio_settings`). */
export interface SpeechDeliveryValues {
  emotion: string;
  manner: string;
  pace: string;
  pitch: string;
  expressiveness: string;
}

/** One of the 24 keys a song can be asked to be in: its ABC id (`F#m`), its words (`F# minor`) and its mode. */
export interface MusicKeyOption {
  id: string;
  words: string;
  mode: 'major' | 'minor';
}

/** One choice of a delivery setting: its id as the form sends it and its label. */
export interface DeliveryChoice {
  id: string;
  label: string;
}

/** An emotion and the group a picker lists it under; `neutral` is in none. */
export interface EmotionChoice extends DeliveryChoice {
  group: string | null;
}

/** A Vibe: a named starting point that sets all five delivery values. */
export interface VibeOption extends DeliveryChoice {
  delivery: SpeechDeliveryValues;
}

/** A `[mark]` a speech text may hold; `as set` has no field or value. */
export interface SpeechMarkOption {
  id: string;
  field: keyof SpeechDeliveryValues | null;
  value: string | null;
}

/** The vocabulary of the audio settings the Workspace payload serves as `audio_settings` (the part the form reads). */
export interface AudioSettingsVocabulary {
  /** The slowest and fastest tempo of a take, in beats per minute. */
  tempo: { min: number; max: number };
  keys: MusicKeyOption[];
  /** Speech delivery: the emotions (neutral first, then by group), the manners, the paces slowest first, and the Low/Normal/High levels. */
  emotions: EmotionChoice[];
  manners: DeliveryChoice[];
  paces: DeliveryChoice[];
  levels: DeliveryChoice[];
  /** The five values of a voice nobody steered, and the Vibes that fill them. */
  delivery_defaults: SpeechDeliveryValues;
  vibes: VibeOption[];
  /** The marks a speech text holds; they steer the delivery and are not read, so they are not counted. */
  marks: SpeechMarkOption[];
  /** How long speech takes to read: characters a second at a Normal pace, and the factor of every other pace. */
  reading: { chars_per_second: number; pace_factor: Record<string, number> };
  /** The Sound Types as the control lists them (Auto first), each with its label. */
  sound_types: { id: SoundType; label: string }[];
  /** The length in seconds a sound has when its Length is left on Auto, by Sound Type. */
  auto_seconds: Record<SoundType, number>;
}

/** What the Workspace form can run: an image or video workflow, or the kind of the Audio tab. */
export type WorkspaceWorkflow = Workflow | AudioWorkflow;

/** Workflow of a job: a Workspace workflow, an upscale of an existing image, or the kind of an audio job. */
export type JobWorkflow = Workflow | 'upscale' | AudioWorkflow | 'voice_design';

export type UpscaleFactor = 2 | 4;

/** How an image upscale runs: the SeedVR2 model (macOS), or the image model's creative pass. */
export type UpscaleMethod = 'seedvr2' | 'creative' | 'ltx';

export type WorkflowMode = 'image' | 'video' | 'audio';

/** What the user picks in the top bar: an image, a video, or the continuation of a video. The workflow follows from it and the tab's images. */
export type WorkspaceTab = 'image' | 'video' | 'extend' | 'audio';
/** The tabs of the top bar. Extend is a kind of Video, not a tab; `WorkspaceTab` also names its slot. */
export type NavTab = 'image' | 'video' | 'audio';
/** The kinds of the Video tab: a new clip (from text or a first frame) or the extension of a clip. */
export type VideoKind = 'new' | 'extend';

export type PromptSource = 'inline' | 'file';

export interface ReuseState {
  requested_workflow: Workflow | AudioWorkflow;
  resolved_workflow: Workflow | AudioWorkflow;
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
  /** The project folder the file sits in; null for a file at the top of the output folder. */
  project: string | null;
  workflow: Workflow | AudioWorkflow;
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
  /** The audio track a clip followed, with where in it the clip's sound starts and ends (seconds); absent or null for other assets. */
  audio_track_path?: string | null;
  audio_track_start?: number | null;
  audio_track_end?: number | null;
  /** The references an image was made with, in prompt order; empty for assets made without any. */
  reference_images?: string[];
  /** Absolute path of the asset file on the host; lets the asset become a start image or frame. */
  file_path?: string;
  seed?: number | null;
  steps?: number | null;
  guidance?: number | null;
  lora?: string | null;
  /** A take's length in seconds, as read from the file. */
  duration?: number | null;
  /** An audio take: what it is (`music`, `speech`, `sound`); null for a file without this app's settings. */
  kind?: AudioWorkflow | null;
  /** An audio take: 100 values of 0-255, the relative loudness along the take; null or absent while the server has not computed them (the app asks for them), empty when they could not be read. */
  peaks?: number[] | null;
  channels?: number | null;
  sample_rate?: number | null;
  /** An audio take: the label of the model's preset, null when no such preset is configured; `model` stays its id. */
  model_label?: string | null;
  /** An audio file's size in bytes. */
  file_bytes?: number | null;
  /** A speech take made from a recording of a voice (the recording itself is not stored with the take). */
  has_reference_voice?: boolean;
  /** A speech take made in a saved voice that still exists: the voice's name; null once it is deleted or for any other take. */
  voice_name?: string | null;
  /** A song's lyrics, section lines included; null otherwise. */
  lyrics?: string | null;
  /** A song that reached the model's cap and was faded out there. */
  cut?: boolean;
  /** An audio take: what the model noted about it (a setting it could not follow); absent or empty for none. */
  notes?: string[];
  /** A song's tempo in beats per minute and its key id (`Am`); null when the take left them on Auto. */
  tempo?: number | null;
  key?: string | null;
  /** A speech take's delivery as recorded (each null when not recorded), and a sound's type. */
  emotion?: string | null;
  manner?: string | null;
  pace?: string | null;
  pitch?: string | null;
  expressiveness?: string | null;
  sound_type?: SoundType | null;
  reuse_workspace_url: string; // hash URL: '#/workspace?workflow=...&prompt=...'
  has_reusable_config?: boolean;
  reuse_state?: ReuseState;
  /** The image was cut out and has a transparent background; absent or false for opaque assets. */
  transparent?: boolean;
  media_type: 'image' | 'video' | 'audio';
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
  width: number | null;
  height: number | null;
  rotation: number | null;
  fps: number | null;
  frame_count: number | null;
  /** Seconds from the file's start to its first video frame, the offset of the clock a point to extend from is on; null when the file could not be read. */
  video_start: number | null;
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
  /** The audio file the source followed, or null when its sound was generated. */
  audio_track_path: string | null;
  /** Seconds into that file where the source's sound ended (where an extension carries on); null with no track. */
  audio_track_end: number | null;
  problems: ExtendProblem[];
  /** The new frames, snapped to a multiple of 8 (a 2 s default when none was requested); null when the frame rate is unknown. */
  extend_frames: number | null;
  window_frames: number | null;
  memory_fit: ClipMemoryFit | null;
  /** The project folder the source sits in; null when it sits at the top of the output folder or outside it. */
  output_folder: string | null;
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
  /** The subject was cut out onto a transparent background. */
  transparent?: boolean;
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

/** The LTX upscale a video offers on macOS: the clip's latent upsampled 2× and refined in a few steps. */
export interface AssetLtxUpscale {
  options: AssetUpscaleFactor[];
  /** Label of the LTX model that refines the clip, e.g. `LTX-2.3 4-bit`. */
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
  /** LTX upscale options for a video on macOS; absent or null elsewhere. */
  ltx?: AssetLtxUpscale | null;
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

/**
 * The keys every audio model row has. The Workspace lists a model once per kind it makes (a row is a (model, kind) pair, so the
 * same `id` can repeat with another `kind`), and what that pair takes follows from the row: music with lyrics has the Lyrics box,
 * music without lyrics and sound have a length, speech has a voice.
 */
interface AudioModelBase extends ModelStatusFields {
  id: string;
  label: string;
  type: 'audio';
  /** The workflow controls this model does not have as this kind, such as `lyrics` or `length`; the workflow's controls minus these are the ones shown. */
  hidden_controls: string[];
  /** A music row of a model that sings lyrics: the `[section]` list an empty Lyrics box stands for; null for the others. */
  instrumental_sections?: string | null;
  licences: ModelLicence[];
}

/** A music model; `max_seconds` is where it cuts a song. A model that sets the length of its music (no lyrics) also gives the shortest and the default length. */
export interface MusicAudioModel extends AudioModelBase {
  kind: 'music';
  limits: { max_seconds: number; min_seconds?: number; default_seconds?: number };
}

/** One built-in voice of a speech model: its id, its name and the language it speaks natively. */
export interface AudioVoice {
  id: string;
  label: string;
  language: string;
}

/** A speech model: the text it reads in one take, its built-in voices (the first is the default) and the recordings it takes as a voice. */
export interface SpeechAudioModel extends AudioModelBase {
  kind: 'speech';
  limits: { max_text_chars: number; min_voice_seconds: number; max_voice_seconds: number };
  voices: AudioVoice[];
  /** The id of the voice a take reads in when none is chosen; null for a model without built-in voices. */
  default_voice: string | null;
  /** Whether the files that cloning a recording needs are downloaded; null when the model has none to fetch or the state is unknown. */
  voice_download?: { downloaded: boolean } | null;
  /** Whether the model can make a voice from a description (it has VoiceDesign files). */
  can_design: boolean;
}

/** A sound model: the lengths it makes, and the one a new draft starts with. */
export interface SoundAudioModel extends AudioModelBase {
  kind: 'sound';
  limits: { min_seconds: number; max_seconds: number; default_seconds: number };
}

/** One audio model the Workspace offers (`audio_models` of the workspace payload). */
export type AudioModelOption = MusicAudioModel | SpeechAudioModel | SoundAudioModel;

export interface ModelOption extends ModelStatusFields {
  id: string;
  label: string;
  type: 'image' | 'video' | 'audio';
  /** One-time download size on this platform, e.g. "35 GB"; image models only, null when unknown. */
  download_size_label?: string | null;
  /** An LTX video model's version, as the server read it from the model (alias, repo or folder); null for other models. */
  version?: '2.3' | '2.5' | null;
  /** The licences the model comes with (two for LTX with Gemma); empty for an unrestricted model. */
  licences?: ModelLicence[];
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
  /** The id of the curated entry the file is, else null. */
  curated_id?: string | null;
  /** The name shown for the LoRA: the curated entry's, or the repository's for a Hugging Face download; null for a file of your own. */
  title?: string | null;
  /** The trigger words that go with the LoRA, comma-separated, else null. */
  trigger?: string | null;
  /** Where the trigger came from: `user` (typed), `curated`, `card` or `header`. */
  trigger_source?: LoraTriggerSource | null;
  /** Where the app downloaded the file from, else null (a hand-copied file). */
  source?: LoraSource | null;
  /** What the Hub showed when a Hugging Face file (not a curated one) was downloaded, kept for offline; else null. */
  hub?: LoraHub | null;
  /** The file's size in bytes and as shown, or null when it is not listed with one. */
  size_bytes?: number | null;
  size_label?: string | null;
}

/** The facts kept with a Hugging Face download, so the Installed tab describes it without a connection. */
export interface LoraHub {
  name: string;
  author: string | null;
  downloads: number | null;
  licence: string | null;
  licence_label: string | null;
  model_card_url: string;
  thumbnail_url: string | null;
}

export type LoraTriggerSource = 'user' | 'curated' | 'card' | 'header';

export interface LoraSource {
  repo: string | null;
  revision: string | null;
  file: string | null;
  sha256: string | null;
  curated_id: string | null;
}

export interface WorkspaceContext {
  image_models: ModelOption[];
  video_models: ModelOption[];
  /** The audio models that run on this machine, in config order; empty (and the Audio tab hidden) without one. */
  audio_models?: AudioModelOption[];
  /** The vocabulary of the audio settings (tempo range, keys); served with the audio models. */
  audio_settings?: AudioSettingsVocabulary;
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
  /** The licences of the background removal models, which a Transparent job downloads and uses. */
  matte_licences?: ModelLicence[];
}

// ── Prompt enhancer ────────────────────────────────────────────────────────────

export type EnhanceAxisKey = 'style' | 'mood' | 'theme' | 'shot' | 'angle' | 'palette' | 'details' | 'camera' | 'pace' | 'length' | 'motion' | 'sound';

/** Which parts of a video prompt an enhancement writes: the scene and the action, or only the action (the scene is kept as written). */
export type EnhanceParts = 'both' | 'action';

export interface EnhanceSettings {
  style: string;
  mood: string;
  theme: string;
  shot: string;
  angle: string;
  palette: string;
  /** Custom palette's own `#RRGGBB` colours; empty unless `palette` is custom. */
  colors: string[];
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
  /** An audio track is in effect: it is the clip's sound, so none is generated. Sent only then. */
  audio_track?: boolean;
  /** A start image is supplied (img2img's image, or a video's first frame). */
  reference_image: boolean;
  /** How many reference images are supplied; the prompt names them "image 1", "image 2". */
  references?: number;
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
  /** Palette options only: the `#RRGGBB` swatches that show the palette; empty for Keep and Custom. */
  swatches?: string[];
}

/** The keys of the audio option axes: Music's six, then Sound's four. */
export type AudioEnhanceAxisKey = 'genre' | 'music_mood' | 'instruments' | 'without' | 'singer' | 'production' | 'space' | 'distance' | 'over_time' | 'movement';

/** What the audio enhancer writes: a Sound prompt (`sound`) or a Music style (`style`). */
export type AudioEnhanceTask = 'sound' | 'style';

/** The tasks that write a longer text: the lyrics helper (Music), the script helper (Speech) and Describe it's Fill in (both). */
export type AudioWriterTask = 'lyrics' | 'script' | 'fill';

/** The ways the script helper edits a speech text: mark it up with delivery, or fit it to a reading time. */
export type ScriptAction = 'delivery' | 'fit';

/**
 * What a script request asks for, and what the record of a text it wrote keeps: the actions, fit first when both are asked (the server runs them in
 * that order in one request). `seconds` and `pace` belong to `fit` (a record leaves out a Normal pace).
 */
export interface ScriptSettings {
  actions: ScriptAction[];
  seconds?: number;
  pace?: string;
}

/** What can write the text of an audio box: a caption task, or the script helper for Speech's text. */
export type AudioRewriteTask = AudioEnhanceTask | 'script';

/** The fields Fill in can write: Music's Style, Tempo, Key and Lyrics; Speech's Vibe and Text. */
export type FillField = 'style' | 'tempo' | 'key' | 'lyrics' | 'vibe' | 'text';

/** The kinds that have a Describe it box. */
export type DescribeKind = 'music' | 'speech';

/**
 * What the fields Fill in wrote looked like before it wrote them: the draft keys it changed, with their values, so Undo fill puts back exactly
 * what was there. `marked` is the marks the fields had before this fill.
 */
export interface FillSnapshot {
  values: Partial<DraftState>;
  marked: FillField[];
  /** The values those earlier marks were given. */
  filled: Partial<Record<FillField, string>>;
}

/** The Describe it box of one kind: the idea, the fields Fill in wrote (each marked ✦ while it is still as written), what Undo fill puts back, and whether the box is folded to one line (null: folded while the Lyrics or Text have words). */
export interface AudioDescribe {
  text: string;
  marked: FillField[];
  /** The value each marked field was given, so an edit (which makes the field the user's) is told from the fill's own text. */
  filled: Partial<Record<FillField, string>>;
  undo: FillSnapshot | null;
  folded: boolean | null;
}

/** The ways the lyrics helper improves lyrics; `fit` works on the whole song only. */
export type LyricsImprovement = 'tighten' | 'fit' | 'rhymes' | 'simpler';

/** What a lyrics request asks for: improve the words that are there, or write one empty section. */
export interface LyricsSettings {
  action?: 'improve' | 'write';
  improve?: LyricsImprovement[];
  /** Free-text direction; omitted for none. */
  note?: string;
  /** The section to work on, counted over the `[section]` tags from 0; null or omitted is the whole song. */
  section?: number | null;
}

/** One version of the song's lyrics: the whole text, and whether it is the user's own (the first version, kept when a result arrives). */
export interface LyricsVersion {
  text: string;
  mine: boolean;
}

/** A choice on every audio axis, plus the free-text direction. A `sound` task reads Sound's four axes, a `style` task Music's six. */
export interface AudioEnhanceSettings {
  genre: string;
  music_mood: string;
  instruments: string[];
  /** Instruments to leave out; the enhancer writes them as "no guitar". */
  without: string[];
  singer: string;
  production: string;
  space: string;
  distance: string;
  over_time: string;
  movement: string;
  /** Free-text direction for the enhancer; empty for none. */
  note: string;
}

/** One audio option axis, as the server serves it in `prompt_enhancer.audio_matrix`. */
export interface AudioEnhanceAxis {
  key: AudioEnhanceAxisKey;
  label: string;
  multi: boolean;
  /** The kind whose Enhance offers it. */
  kind: 'music' | 'sound';
  /** Applies only when the take has a singing voice (Singer). */
  vocal: boolean;
  options: EnhanceOptionInfo[];
  default: string[];
}

export interface AudioEnhanceMatrix {
  axes: AudioEnhanceAxis[];
  defaults: AudioEnhanceSettings;
}

/** The take's facts that shape an audio rewrite; every key is optional. */
export interface AudioEnhanceContext {
  /** The start of the lyrics (the first 600 characters are used). */
  lyrics?: string;
  /** The start of the speech text (the first 600 characters are used), so Fill in picks a Vibe that suits it. */
  text?: string;
  /** The song's style text, so the lyrics helper writes words that suit it. */
  style?: string;
  /** The song has no singing, so no singer is described. */
  instrumental?: boolean;
  /** Tempo in BPM and key as an ABC id; only a model that writes metadata states them. */
  tempo?: number | string | null;
  key?: string | null;
  /** The length of the take in seconds; a sound's rewrite describes only what fits in it. */
  seconds?: number;
  sound_type?: SoundType;
}

/** How an audio text was written: stored with the take as its `enhance` record. */
export interface AudioEnhanceRecord {
  /** Only the options that are not defaults; a script record holds what the helper did. */
  settings: Partial<AudioEnhanceSettings> | ScriptSettings;
  /** The enhancer that wrote it, ``REPO[@REVISION]``; the server works out Fast, Smart or custom from it. */
  model: string;
  /** Which enhancer it was (Fast, Smart or a custom model); the server works it out, and a saved take carries it. */
  tier?: 'fast' | 'smart' | 'custom';
  format: 'text';
  task: AudioRewriteTask;
}

/** The text boxes of the Audio tab that can hold a rewrite: Music's Style, a Sound's Prompt and Speech's Text. */
export type AudioBox = 'style' | 'sound' | 'text';

/** What an audio rewrite was made from: the own text and the audio model, so a change to either makes it out of date. */
export interface AudioRewriteSource {
  /** The text the rewrite was made from; empty for a take whose own text was empty. */
  input: string;
  /** The audio model the rewrite was written for (its family sets the caption shape). */
  model: string;
  task: AudioRewriteTask;
  /** How the rewrite was written; Generate sends it so the take records it. Missing for typed-in text. */
  record?: AudioEnhanceRecord;
}

/** One rewrite kept in the history of a box. */
export interface AudioRewrite {
  text: string;
  /** Null for text typed in the Enhanced tab (never out of date). */
  from: AudioRewriteSource | null;
}

/**
 * The Enhanced text of one audio box (empty text is none: Generate sends the own text) and its last five versions. `scratched` is the position
 * (from 0) of the version Scratch emptied the box from, which the stepper's ‹ brings back; absent unless the box was scratched.
 */
export interface AudioRewrites extends AudioRewrite {
  history: AudioRewrite[];
  scratched?: number | null;
}

export interface PromptEnhancerContract {
  matrix: { axes: EnhanceAxis[]; defaults: EnhanceSettings };
  /** The audio option axes and their defaults. */
  audio_matrix: AudioEnhanceMatrix;
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
  /** Whether the Smart model is on disk. */
  downloaded: boolean;
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
  /** The scene it was produced beside; empty for images. */
  scene: string;
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

/** One rewrite kept in the history of the current text: the Enhanced pair and the source and record it was made with. */
export interface EnhancedRewrite {
  prompt: string;
  /** The Enhanced scene of a video; empty for images. */
  scene: string;
  from: EnhancedSource;
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

/**
 * One line of an audio enhance stream. The written texts are in `fields`, a mapping of field name to text
 * (`prompt` for the `sound` task, `style` for the `style` task), so a task that writes several fields uses the same frame.
 * Only the caption tasks stream `text` frames; a writer task sends `status`, then `done`.
 */
export type AudioEnhanceFrame =
  | { type: 'status'; phase: 'downloading' | 'loading' | 'generating' | 'generating_cpu' }
  | { type: 'text'; task: AudioEnhanceTask; text: string; fields: Record<string, string> }
  | { type: 'done'; task: AudioEnhanceTask | AudioWriterTask; fields: Record<string, string>; clamped: boolean; warning?: string; model?: string }
  | { type: 'error'; detail: string; /** A writer left the text as it was: a note, not a failure. */ no_change?: boolean };

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
export type EntryEnhance = null | true | Partial<Record<EnhanceAxisKey | 'colors' | 'note' | 'parts', string | string[]>>;

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
export type BrowsePurposeName = 'output_directory' | 'prompt_file' | 'prompt_file_folder' | 'checkpoint_file' | 'lora_file' | 'source_video' | 'reference_image' | 'audio_track';

export interface BrowsePurpose {
  name: BrowsePurposeName;
  kind: 'directory' | 'existing_file';
  label: string;
  accepted_extensions: string[];
  /** What the browser can preview of a chosen file; null when the purpose has no previews. */
  preview: BrowsePreviewKind | null;
}

/** The kinds of file the browser previews (`BrowsePurpose.preview` on the server). */
export type BrowsePreviewKind = 'image' | 'audio' | 'video';

/** The facts `GET /api/browse/info` returns: width and height for an image or video, the rest by kind. */
export interface BrowseInfo {
  width?: number | null;
  height?: number | null;
  seconds?: number | null;
  sample_rate?: number | null;
  channels?: number | null;
  fps?: number | null;
  frames?: number | null;
  /** Video only: seconds from the file start to the first frame. */
  start?: number | null;
  /** Audio only: 100 waveform peaks. */
  peaks?: number[];
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
  /** How many reference images the model takes; 0 means none. */
  max_reference_images: number;
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
  /** The model can upscale a clip made from an audio track; false hides the Upscale controls while a track is set. */
  supports_audio_track_upscale: boolean;
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
  requires_reference_image: boolean;
  /** The workflow continues a chosen video (extend). */
  requires_source_video: boolean;
  clear_fields: string[];
}

export interface WorkflowContract {
  values: WorkspaceWorkflow[];
  /** One entry per listed workflow; an audio workflow is listed only while a model of its kind runs on this machine. */
  definitions: Record<Workflow, WorkflowContractEntry> & Partial<Record<AudioWorkflow, WorkflowContractEntry>>;
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

/** The text a tab keeps for itself: what is typed, and the Enhanced text made from it. Image and Video each hold one; Video's two kinds share it. */
export interface TabText {
  prompt: string;
  scene: string;
  negativePrompt: string;
  jsonPromptEnabled: boolean;
  jsonPrompt: string;
  enhancedPrompt: string;
  enhancedScene: string;
  enhancedFrom: EnhancedSource | null;
  enhancedHistory: EnhancedRewrite[];
  enhancedScratched: number | null;
}

export interface DraftState {
  workflow: WorkspaceWorkflow;
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
  /** Image tab: the start image's path on the host; null when none is set or it was uploaded (`referenceImageUpload`). */
  referenceImagePath: string | null;
  /** Image tab: the file name of a start image uploaded from this device (its file is in `heldUploads` and the upload store); null when none. */
  referenceImageUpload: string | null;
  /** Video tab: the first frame's path on the host; null when none is set or it was uploaded (`firstFrameUpload`). */
  firstFramePath: string | null;
  /** Video tab: the file name of a first frame uploaded from this device; null when none. */
  firstFrameUpload: string | null;
  /** The reference images, in prompt order ("image 1" is the first). An image uploaded from this device has an empty path here and its file in the Workspace. */
  referenceImagePaths: string[];
  /** Image paths (start image, first frame, last frame, references) found gone; the slot stays and blocks Generate while it is used. */
  missingReferencePaths: string[];
  /** Video tab: the path of the image the clip ends on; null when none is chosen or it was uploaded (`endImageUpload`). It is kept while the first frame or the model's last frame is absent, and sent only while used. */
  endImagePath: string | null;
  /** Video tab: the file name of a last frame uploaded from this device; null when none. */
  endImageUpload: string | null;
  /** Video tab: the audio track's path on the host; null when none is set or it was uploaded (`audioTrackUpload`). Kept, but hidden and not sent, on a model without the capability. */
  audioTrackPath: string | null;
  /** Video tab: the file name of an audio track uploaded from this device (its file is in `heldUploads`); null when none. */
  audioTrackUpload: string | null;
  /** Video tab: the seconds into the track where the clip's sound starts. */
  audioTrackStart: number;
  /** Extend kind: the audio track's path on the host; null when none is set or it was uploaded (`extendAudioTrackUpload`). */
  extendAudioTrackPath: string | null;
  /** Extend kind: the file name of an audio track uploaded from this device; null when none. */
  extendAudioTrackUpload: string | null;
  /** Extend kind: the seconds into the track where the extension's sound starts. */
  extendAudioTrackStart: number;
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
  /** Cut the subject out and save a PNG with a transparent background. */
  postprocessTransparent: boolean;
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
  /** The last five versions of the Enhanced text, oldest first; browser-local. Written by Enhance, Scratch, a recent-prompt pick, Reset and the JSON capability gate. */
  enhancedHistory: EnhancedRewrite[];
  /** The position (from 0) of the version Scratch emptied the Enhanced text from, which the stepper's ‹ brings back; null otherwise. */
  enhancedScratched: number | null;
  /**
   * The text of the mode (image or video) that is not open; the fields from `prompt` to `enhancedHistory` (the same as
   * `TabText`) hold the open one's. Opening the other mode swaps the two. Empty until the first switch.
   */
  savedText: Partial<Record<WorkflowMode, TabText>>;
  /**
   * What the mode (image or video) that is not open kept: its Settings pane values, model with its LoRAs and quantization, and
   * applied preset. The fields themselves hold the open one's; opening the other mode swaps the two, so each mode comes back as
   * it was left. A mode not visited yet has none, and takes its model's defaults.
   */
  savedSettings: Partial<Record<WorkflowMode, Partial<DraftState>>>;
  /** Last-used enhancement options for images; null until the user picks any (backend defaults apply). */
  enhanceSettings: EnhanceSettings | null;
  /** Last-used enhancement options for video, with the Rewrite choice of each video workflow; null until any are picked. */
  enhanceSettingsVideo: EnhanceSettings | null;
  /** Enhance each image's prompt on the server at generation time. */
  enhanceAuto: boolean;
  /** The enhancer chosen in the popover; Smart falls back to Fast where it is unavailable. */
  enhanceModel: EnhancerTier;
  /** The enhancer Describe it's Fill in uses, chosen apart from `enhanceModel` (Enhance and the helpers); the same fallback to Fast. */
  describeTier: EnhancerTier;
  /** What image enhancement writes; JSON applies only to models that take JSON captions. */
  enhanceFormat: EnhanceFormat;
  /** Extend: the video to continue; null until one is chosen. Its facts are fetched again on load. */
  extendSourcePath: string | null;
  /** Extend: the new frames, a multiple of 8; null until a source sets the 2 s default. Never aligned to 8k+1. */
  extendFrames: number | null;
  /** Extend: where in the source to continue from, in seconds on its clock (the last frame kept); null continues from the end. The source owns it: another source resets it. */
  extendFrom: number | null;
  /** Extend: the source could not be found where Reuse recorded it, so it must be chosen again. */
  missingSourceVideo: boolean;
  /** Extend: the source whose Scene and Prompt were settled, and the text its prefill put there (to tell edits from prefill). */
  extendPrefill: { path: string; scene: string; prompt: string; audioTrack?: { path: string; start: number } | null } | null;
  /** Video tab: the kind last open (New or Extend); coming back to the tab restores it. */
  videoKind: VideoKind;
  /** Audio tab: the kind last open (Music, Speech or Sound); coming back to the tab restores it. */
  audioKind: AudioWorkflow;
  /** Audio tab: the model last chosen for each kind. Empty before one was chosen. */
  audioModels: Record<AudioWorkflow, string>;
  /** Audio tab, Music: the Style box. */
  audioStyle: string;
  /** Audio tab, Music: the Lyrics box. Empty or only `[section]` tags make an instrumental. */
  lyrics: string;
  /** Audio tab, Speech: the Text box. */
  speechText: string;
  /** Audio tab, Sound: the Prompt box. */
  soundPrompt: string;
  /** Audio tab, Sound: the length in seconds; 0 until the model's default fills it. */
  audioSeconds: number;
  /** Audio tab, Music: the tempo in beats per minute; null is Auto (the model chooses). */
  audioTempo: number | null;
  /** Audio tab, Music: the key as an ABC id such as `F#m`; null is Auto. */
  audioKey: string | null;
  /** Audio tab, Music of a model that sets its length: the length in seconds; 0 is the model's default. */
  musicSeconds: number;
  /** Audio tab, Sound: the type of the sound. */
  soundType: SoundType;
  /** Audio tab, Speech: how a built-in voice says the text; a recording ignores it. */
  speechDelivery: SpeechDeliveryValues;
  /** Audio tab, Speech: whether the take reads in a built-in voice (`builtin`), follows a one-off recording (`recording`) or follows a voice from the library (`saved`). */
  voiceMode: 'builtin' | 'recording' | 'saved';
  /** Audio tab, Speech: the slug of the saved voice a `saved` take follows; null when none is chosen. */
  savedVoice: string | null;
  /** Audio tab, Speech: the id of the built-in voice; null is the model's default voice. */
  speechSpeaker: string | null;
  /** Audio tab, Speech: the recording's path on the host; null when none is set or it was uploaded (`voiceUpload`). */
  voicePath: string | null;
  /** Audio tab, Speech: the file name of a recording uploaded from this device (its file is in `heldUploads`); null when none. */
  voiceUpload: string | null;
  /** Audio tab, Speech: what the recording says. */
  voiceText: string;
  /** Audio tab: the Enhanced text of each box and its earlier versions. */
  audioRewrites: Record<AudioBox, AudioRewrites>;
  /** Audio tab, Music: the versions of the lyrics the helper made, the user's own first (`mine`); empty before the helper ran. */
  lyricsVersions: LyricsVersion[];
  /** Audio tab, Music: the index in `lyricsVersions` of the version `lyrics` shows (it is out of step when `lyrics` was set by anything else). */
  lyricsCurrent: number;
  /** Audio tab: the Enhance options of each task, as last chosen; null is every default. */
  enhanceSettingsAudio: Record<AudioEnhanceTask, Partial<AudioEnhanceSettings> | null>;
  /** Audio tab, Music and Speech: the Describe it box of each kind. */
  audioDescribe: Record<DescribeKind, AudioDescribe>;
  /** The slug of the settings preset the form started from; null when none. Cleared on a mode switch, Reset, a loaded source's settings and when the preset is deleted. */
  appliedPreset: string | null;
}

/** Workspace prefill parameters; a name can repeat (`reference_path`), which makes its value a list. */
export type PrefillParams = Record<string, string | string[]>;

export type WorkspacePrefill = Partial<Omit<DraftState, 'historyCollapsed' | 'sidebarCollapsed' | 'lastGeneratedAt' | 'version'>>;

// ── Saved voices ──────────────────────────────────────────────────────────────

/** A voice in the library: a reference recording and the words it says. */
export interface SavedVoice {
  slug: string;
  name: string;
  /** What the recording says: the transcript a take follows it with. */
  text: string;
  origin: 'recording' | 'design';
  seconds: number;
  /** The description a designed voice was made from; null for a recording. */
  description: string | null;
  created_at: string;
}

/** One candidate of the voice designer: its stable id, its number (counted from 1 in the order made, across batches), its length, whether it can be saved (`reason` says why not) and the name of the voice it was saved as. */
export interface VoiceCandidate {
  id: string;
  number: number;
  seconds: number;
  saveable: boolean;
  reason: string | null;
  /** The name of the voice this candidate was saved as; null while unpicked. */
  saved_as: string | null;
}

/** One press of Make 3: the description and the preview line it was made from, and its candidates in the order they were made. */
export interface VoiceDesignBatch {
  batch_id: string;
  description: string;
  /** The preview line each candidate reads, which is saved with the voice. */
  text: string;
  created_at: string;
  candidates: VoiceCandidate[];
}

/** What the designer has made and kept until Clear voices: the batches, newest first, and how many saveable candidates are not yet saved. */
export interface VoiceDesignState {
  batches: VoiceDesignBatch[];
  unpicked: number;
}

// ── Settings presets ──────────────────────────────────────────────────────────

export type PresetMode = 'image' | 'video' | 'audio';

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
  /** `video` for a job that upscales an existing video file, `audio` for an audio job; absent on other jobs. */
  media_type?: 'image' | 'video' | 'audio';
  /** How a video file upscale runs; absent on other jobs and on jobs from an older server. */
  upscale_method?: UpscaleMethod;
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
  /** The project the job saves into (for an Extend, the folder beside its source); null for the top of the output folder, absent on an upscale. */
  save_project?: string | null;
  /** 1-based place in the queue; null once the job is the active one. */
  queue_position?: number | null;
  /** Short notes about how the job deviates from what was asked, e.g. "original settings unknown". */
  notices?: string[];
}

/** A small render that waits for the user to approve or discard it before its upscale. */
export interface JobReview {
  /** 1-based position of the generation under review within the job. */
  iteration: number;
  /** 1-based attempt at that generation: a repeat after a discard is a new attempt of the same iteration; a missing value is 1. */
  attempt?: number;
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
  /** An audio take's progress: the phase the model is in and how far it is; null before the first `audio_progress` of a take. */
  audio?: AudioProgress | null;
  /** A control the server accepted that has not taken effect yet (pause waits for the current image or video). */
  controlQueued?: JobControlAction | null;
  /** Client time (ms since epoch) a queued control took effect; a request made before it no longer counts as pending. */
  controlSettledAt?: number;
  /** 'reconnecting' while the event stream is down and the shown progress may be stale. */
  connection?: 'live' | 'reconnecting';
  /** Client time (ms since epoch) of the last event received. */
  lastEventAt?: number;
  /** Server clock minus client clock, in seconds, so live phase durations count from server timestamps. */
  serverOffset?: number;
}

/** Where an audio take is, from its `audio_progress` events. */
export interface AudioProgress {
  /** The phase the model is in, as it names it: `Plan`, `Song tokens`, `Synthesis`, `Decode`, `Speech` or `Sampling`. */
  phase: string;
  step: number;
  /** The phase's step count; null while the model cannot say (a song's tokens run until it ends). A speech part is one step: 0 before it is read and 1 after. */
  total: number | null;
  /** The part of a long speech text being read (1-based), and how many there are. */
  part: number;
  parts: number;
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

export interface AudioProgressEvent {
  type: 'audio_progress';
  job_id: string;
  phase: string;
  step: number;
  total: number | null;
  part: number;
  n_parts: number;
  /** Seconds since the take started, and the seconds left; the estimate is absent until the server has one. */
  elapsed_secs?: number;
  eta_secs?: number;
  run_index?: number;
  total_runs?: number;
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
  mode: 'image' | 'video' | 'audio';
  role: 'model' | 'enhancer' | 'matte';
  model: string;
}

/** The download a job is waiting on, taken from its newest `model_downloading` event. */
export interface JobDownload extends DownloadProgressFields {
  role: 'model' | 'enhancer' | 'matte';
  model: string;
}

export type SSEEvent =
  | StepEvent
  | AudioProgressEvent
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

// ── Projects ───────────────────────────────────────────────────────────────────

/** A project: a folder directly under the output folder. */
export interface Project {
  name: string;
  /** Absolute path of the folder on the host. */
  path: string;
}

/** One file a move renamed: its old ID and the asset as it now stands. */
export interface MovedAsset {
  from: string;
  asset: GalleryAsset;
}

/** What a move did: the files it moved, and the ones it could not with the reason. */
export interface MoveResult {
  moved: MovedAsset[];
  failed: { id: string; detail: string }[];
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

/** The answer to a Hugging Face LoRA import: it returns at once, and the file arrives on the download lane. */
export interface LoraImportResult extends Omit<ModelOperationResult, 'detail'> {
  name: string;
  outcome: 'queued' | 'running' | 'installed';
  download: DownloadItem | null;
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
  kind: 'model' | 'audio' | 'enhancer' | 'matte' | 'lora';
  name: string;
  label: string;
  state: 'queued' | 'running' | 'done' | 'failed' | 'cancelled';
  queue_position?: number;
  message: string | null;
  /** A LoRA item: the curated entry it installs, where it comes from, and the size the download expects. */
  curated_id?: string | null;
  source_repo?: string | null;
  source_file?: string | null;
  source_revision?: string | null;
  size_bytes?: number | null;
  /** A LoRA item: false when it downloads with the user's Hugging Face login (an Import), true when it reads anonymously (Discover). */
  anonymous?: boolean;
} & Partial<DownloadProgressFields>;

/** Response of `GET /api/models/downloads`: queued, running and recently finished items. */
export interface DownloadList {
  items: DownloadItem[];
}

/** One model a job or a Models-page row would download, from the plan routes. */
export interface JobDownloadPart {
  /** The name the dialog shows. */
  label: string;
  /** The model's id; the key of its licence records. */
  model: string;
  /** What is missing; `null` when the size is unknown. */
  bytes: number | null;
}

/** A licence a model comes with, as the server reads it from the packaged licence records. */
export interface ModelLicence {
  name: string;
  supplier: string;
  summary: string;
  /** The licence text pinned to a revision, or the model page of the licensor's original repository. */
  url: string;
  /** What the link to `url` is called: "Licence", "Model page" or "Dataset terms". */
  url_label: string;
  /** The credit line the licence requires beside the model, such as "Powered by Stability AI"; null for most. */
  attribution: string | null;
}

/** A licence record that covers a model of a job's plan. */
export interface JobDownloadLicence extends ModelLicence {
  /** The id of the plan part's model the record covers. */
  model: string;
}

/**
 * What a job would download before it runs (`POST /api/generate/plan`, `POST /api/upscale/plan`): the missing models in
 * the order the job fetches them. Empty `parts` means nothing is to be downloaded.
 */
export interface JobDownloadPlan {
  parts: JobDownloadPart[];
  /** `null` when any size is unknown. */
  total_bytes: number | null;
  /** Free space on the download drive against the estimate; `null` when nothing was checked. */
  space: { path: string; needed_bytes: number; free_bytes: number; enough: boolean } | null;
  /** A part could not be sized because Hugging Face could not be reached. */
  unchecked: boolean;
  licences: JobDownloadLicence[];
}

/** What a Models-page row would download, from `GET /api/models/download/plan`: one part, and the row's label. */
export interface RowDownloadPlan extends JobDownloadPlan {
  label: string;
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
  licences?: ModelLicence[];
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
  licences?: ModelLicence[];
}

/** An audio model on the Models page: one row per model, listing every kind it makes. `name` is the model's id. */
export interface AudioModelEntry extends ModelStatusFields {
  name: string;
  label: string;
  kinds: AudioWorkflow[];
  delete?: ModelDeleteInfo | null;
  disk_size_label?: string | null;
  /** Files are missing, so the Models page offers Download. */
  download_needed: boolean;
  licences?: ModelLicence[];
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
  /** The licence of the Fast and Smart tiers; empty for a custom model. */
  licences?: ModelLicence[];
}

/** The background removal models (the cut-out behind Transparent background): two repositories shown as one row. */
export interface BackgroundRemovalEntry {
  name: string;
  label: string;
  /** Whether both repositories are on disk; null when that could not be checked. */
  downloaded: boolean | null;
  /** Disk use of the downloads, when any. */
  disk_size_label: string | null;
  /** Size of the first-use download. */
  download_size_label: string | null;
  /** Whether a download is on disk to delete. */
  deletable: boolean;
  /** Files are missing, so the Models page offers Download. */
  download_needed: boolean;
  licences?: ModelLicence[];
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
  /** The audio models that run on this machine; empty with no audio preset, and then the page shows no panel. */
  audio_models?: AudioModelEntry[];
  enhancer_models?: EnhancerModelEntry[];
  background_removal?: BackgroundRemovalEntry;
  other_downloads?: OtherDownloads;
  loras: LoraEntry[];
  /** Whether this platform can save quantized copies of models (macOS). */
  stored_quants_supported?: boolean;
  huggingface_configured: boolean;
  huggingface_token_source: string | null;
}

/** A list of recent prompts: one per Workspace mode, and one per audio kind. */
export type RecentMode = Exclude<WorkflowMode, 'audio'> | AudioWorkflow;

/** One prompt the server accepted, as the recent prompts list returns it. */
export interface RecentPrompt {
  format: EnhanceFormat;
  /** The prompt, or the JSON caption text when *format* is `json`. */
  prompt: string;
  /** The video scene, or the lyrics of a music prompt; empty otherwise. */
  scene: string;
  negative_prompt: string;
  /** The list the entry is kept in. */
  mode: RecentMode;
  /** Server time of the submit, in epoch seconds. */
  used_at: number;
}

/** A curated entry's state on this machine (`inuse` is the Workspace's own: a chip uses it). */
export type CuratedState = 'available' | 'downloading' | 'installed' | 'name_taken';

/** One curated LoRA, from `GET /api/discover/curated`. */
export interface CuratedEntry {
  id: string;
  name: string;
  family: string;
  size_variant: string | null;
  version: string | null;
  display_name: string;
  author: string;
  description: string;
  section: 'style' | 'tool';
  what_it_does: string | null;
  workflow: Workflow | null;
  repo: string;
  revision: string;
  file: string;
  size_bytes: number;
  size_label: string;
  trigger: string | null;
  weight: number;
  licence: string | null;
  licence_label: string | null;
  non_commercial: boolean;
  /** The repository's Hugging Face download count when the list was last refreshed. */
  downloads: number;
  platforms: string[];
  note: string | null;
  previews: string[];
  preview_videos: string[];
  model_card_url: string;
  state: CuratedState;
  download: DownloadItem | null;
}

export interface CuratedList {
  family: string;
  size: string | null;
  version: string | null;
  hf_enabled: boolean;
  entries: CuratedEntry[];
}

export type HfSort = 'downloads' | 'likes' | 'trending' | 'newest';

/** One row of the Hugging Face listing. */
export interface HfItem {
  repo: string;
  name: string;
  author: string;
  downloads: number;
  likes: number;
  files: number;
  licence: string | null;
  thumbnail: string | null;
  curated_id: string | null;
}

export interface HfPage {
  items: HfItem[];
  next_cursor: string | null;
  base_models: string[];
}

export interface HfFile {
  file: string;
  size_bytes: number;
  size_label: string;
  sha256: string | null;
  step: number | null;
  marked: boolean;
  curated_id: string | null;
  save_as: string;
  renamed: boolean;
  state: 'available' | 'downloading' | 'installed';
}

export interface HfRepo {
  repo: string;
  name: string;
  author: string;
  revision: string;
  downloads: number;
  likes: number;
  licence: string | null;
  licence_label: string | null;
  base_model: string | null;
  instance_prompt: string | null;
  previews: string[];
  preview_videos: string[];
  model_card_url: string;
  preselected: string | null;
  files: HfFile[];
  other_files: HfFile[];
}

export interface HfCheck {
  status: 'match' | 'mismatch' | 'unknown';
  family: string | null;
  size_variant: string | null;
  expected: { family: string; size_variant: string | null };
}

/** The answer to `POST /api/discover/download`. */
export interface DiscoverDownload {
  name: string;
  status: 'queued' | 'running' | 'installed';
  renamed: boolean;
  item: DownloadItem | null;
  trigger: string | null;
  trigger_source: 'curated' | 'card' | null;
}

export interface LoraTriggerResult {
  name: string;
  trigger: string | null;
  trigger_source: LoraTriggerSource | null;
  user_trigger: string | null;
}
