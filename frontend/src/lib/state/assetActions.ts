import { parseQuery } from '$lib/prefillParams';
import { requestConfirm } from '$lib/components/molecules/confirm.svelte';
import { imageSupport } from '$lib/state/imageSlots';
import { workflowMode } from '$lib/state/promptEnhance';
import { stripMarks } from '$lib/state/speechDelivery';
import type { AssetUpscaleFactor, DraftState, GalleryAsset, JobWorkflow, PrefillParams, SpeechAudioModel, UpscaleFactor, UpscaleMethod, Workflow, WorkspaceContext } from '$lib/types';

/** Where an image goes: an image's start image, one more reference, a video's first frame, or a video's last frame. */
export type ReferenceTarget = 'image' | 'reference' | 'video' | 'last';

/** The targets that fill a single slot of a tab; a reference joins the list instead. */
export type SlotTarget = Exclude<ReferenceTarget, 'reference'>;

/** Where an audio take goes: the Video tab's audio track, the Speech tab's voice, or the library of saved voices. */
export type AudioUseTarget = 'audio_track' | 'voice' | 'save_voice';

/**
 * What the Use as menu shows for each target: whether it can be picked, and a hint (or the reason it can't).
 * `voiceMaxSeconds` is the longest take the Speech tab takes as a voice.
 */
export type UseAsState = Partial<Record<ReferenceTarget | AudioUseTarget, { disabled: boolean; hint?: string }>> & {
  /** The longest take, in seconds, a speech model takes as a voice. */
  voiceMaxSeconds?: number;
  /** The id of the speech model a take is saved as a voice for: the one that takes the longest voice. */
  voiceModel?: string;
};

/** Callbacks for the actions every asset surface offers; an absent callback hides its action. */
export interface AssetActionHandlers {
  onpreview?: (asset: GalleryAsset, trigger: HTMLElement) => void;
  onreuse?: (asset: GalleryAsset) => void;
  onreference?: (asset: GalleryAsset, target: ReferenceTarget) => void;
  onextend?: (asset: GalleryAsset) => void;
  /** An audio take becomes the Video tab's audio track or the Speech tab's voice. */
  onuseaudio?: (asset: GalleryAsset, target: AudioUseTarget) => void;
  onupscale?: (asset: GalleryAsset, factor: UpscaleFactor, method: UpscaleMethod) => void;
  ondelete?: (asset: GalleryAsset, options?: DeleteOptions) => void;
  /** Move the asset to a project, or out of every project with null. */
  onmove?: (asset: GalleryAsset, project: string | null) => void;
}

/** The Use as hints of the audio targets. */
export const AUDIO_TRACK_HINT = 'Video tab, from 0 s';
export const VOICE_HINT = 'Speech, text filled in';
export const SAVE_VOICE_HINT = 'Keep it in your voices';
const SPEECH_MODELS_UNKNOWN = 'Speech models not loaded';

const WORKFLOW_LABELS: Record<JobWorkflow, string> = {
  txt2img: 'Text to image',
  img2img: 'Image to image',
  txt2vid: 'Text to video',
  img2vid: 'Image to video',
  extend: 'Extend video',
  upscale: 'Upscale',
  music: 'Music',
  speech: 'Speech',
  sound: 'Sound',
  voice_design: 'New voice',
};

/** Describe a workflow value in words; unknown values are shown as they are. */
export function workflowLabel(workflow: string): string {
  return WORKFLOW_LABELS[workflow as JobWorkflow] ?? workflow;
}

/** Return the upscale factors an image offers, or none for videos and images without size data. */
export function upscaleFactors(asset: GalleryAsset): AssetUpscaleFactor[] {
  return asset.media_type === 'image' ? (asset.upscale?.factors ?? []) : [];
}

/** Return the SeedVR2 upscale options an image or video offers, or none where SeedVR2 is unavailable. */
export function seedvr2Options(asset: GalleryAsset): AssetUpscaleFactor[] {
  return asset.upscale?.seedvr2?.options ?? [];
}

/** Return the LTX upscale options a video offers on macOS, or none elsewhere. */
export function ltxOptions(asset: GalleryAsset): AssetUpscaleFactor[] {
  return asset.upscale?.ltx?.options ?? [];
}

/** Return the upscale factors an asset offers by any method, smallest first (the chord's keys and the button's tooltip). */
export function offeredUpscaleFactors(asset: GalleryAsset): UpscaleFactor[] {
  return [...new Set([...upscaleFactors(asset), ...seedvr2Options(asset), ...ltxOptions(asset)].map((option) => option.factor))].sort((a, b) => a - b);
}

/** Return the method the plain upscale shortcut runs: SeedVR2 where it is offered, LTX for a video on macOS, otherwise creative. */
export function plainUpscaleMethod(asset: GalleryAsset): UpscaleMethod {
  if (asset.upscale?.seedvr2) return 'seedvr2';
  return asset.upscale?.ltx ? 'ltx' : 'creative';
}

/** Return the upscale option for one factor and method, or undefined when the asset does not offer it. */
export function upscaleOption(asset: GalleryAsset, factor: UpscaleFactor, method: UpscaleMethod): AssetUpscaleFactor | undefined {
  const options = method === 'seedvr2' ? seedvr2Options(asset) : method === 'ltx' ? ltxOptions(asset) : upscaleFactors(asset);
  return options.find((option) => option.factor === factor);
}

export interface DeleteOptions {
  /** Ask before deleting; false skips the prompt (Shift+Delete in the viewer). */
  confirm?: boolean;
}

const REFERENCE_WORKFLOWS: Record<SlotTarget, Workflow> = {
  image: 'img2img',
  video: 'img2vid',
  last: 'img2vid',
};

const FALLBACK_REASON_TEXT: Record<string, string> = {
  workflow_media_mismatch: 'The saved workflow does not match this media type; the default workflow is used.',
  missing_reference_image: 'The original start image is unknown; reuse starts a text-to-media run instead.',
  model_not_configured: 'The original model is not configured here; the default model is used.',
  missing_source_video: 'The video this one continued has moved or been deleted; choose it again to extend.',
  extend_unavailable: 'Extending a video is not available here; reuse starts a text-to-video run instead.',
};

/**
 * Return the workspace prefill params carried by an asset's reuse URL; a name that repeats (`reference_path`) is a list, in order.
 *
 * `reuse` marks the params as a Reuse, which the Workspace applies exactly: what the asset did not record takes its default.
 * Like `use_as`, it is not a draft field.
 */
export function reuseParams(asset: GalleryAsset): PrefillParams {
  const query = (asset.reuse_workspace_url ?? '').replace(/^#\/[^?]*\??/, '');
  const params = parseQuery(query);
  if (query) params.reuse = 'true';
  return params;
}

/** Return whether an asset can be used as a start image or frame. */
export function canUseAsReference(asset: GalleryAsset): boolean {
  return asset.media_type === 'image' && Boolean(asset.file_path);
}

/**
 * Return the workspace prefill params that make an asset the start image (or a frame), carrying its prompt.
 *
 * For a video the prompt describes what the first frame shows, so it fills the Scene and leaves the Action for what happens;
 * for an image it is the prompt. A last frame is not the opening scene, so it carries no text.
 *
 * `use_as` names the slot the asset fills (`start` or `last`). The Workspace reads it to drop only that slot's held upload,
 * which tells a Use as from a Reuse when both arrive as address params; it is not a draft field.
 */
export function referenceParams(asset: GalleryAsset, target: SlotTarget): Record<string, string> {
  const path = asset.file_path ?? '';
  if (target === 'last') return { workflow: REFERENCE_WORKFLOWS[target], end_image_path: path, use_as: 'last' };
  const params: Record<string, string> = { workflow: REFERENCE_WORKFLOWS[target], image_path: path, use_as: 'start' };
  if (asset.prompt) params[target === 'video' ? 'scene' : 'prompt'] = asset.prompt;
  return params;
}

function modelLabel(options: readonly { id: string; label: string }[], id: string): string {
  return options.find((option) => option.id === id)?.label ?? id;
}

/**
 * Return what the Use as menu shows for each target, read from the model each tab opens on: the draft's model for the tab
 * it is in, else the model that mode last ran, else the context's current one. Without a context the image targets show no
 * hints and are not disabled, but a voice can't be judged (its length limit is the speech model's), so those are unavailable.
 */
export function useAsState(ctx: WorkspaceContext | null, state: DraftState): UseAsState {
  if (ctx === null) return { voice: { disabled: true, hint: SPEECH_MODELS_UNKNOWN }, save_voice: { disabled: true, hint: SPEECH_MODELS_UNKNOWN } };
  const ownMode = workflowMode(state.workflow);
  const imageModel = (ownMode === 'image' ? state.model : state.savedSettings.image?.model) || (ctx.current_image_model ?? '');
  const videoModel = (ownMode === 'video' ? state.model : state.savedSettings.video?.model) || (ctx.current_video_model ?? '');
  const imageName = modelLabel(ctx.image_models, imageModel);
  const videoName = modelLabel(ctx.video_models, videoModel);
  const imageSupports = imageSupport(ctx, imageModel, 'image');
  const videoSupports = imageSupport(ctx, videoModel, 'video');
  const result: UseAsState = {};

  Object.assign(result, audioUseState(ctx, videoModel, videoName));

  result.image = imageSupports.start
    ? { disabled: false }
    : { disabled: true, hint: `${imageName} can't use a start image` };

  const max = imageSupports.references;
  const used = state.referenceImagePaths.length;
  if (max === 0) result.reference = { disabled: true, hint: `${imageName} takes none` };
  else if (used >= max) result.reference = { disabled: true, hint: `all ${max} in use` };
  else result.reference = { disabled: false, hint: `as image ${used + 1}` };

  result.video = videoSupports.start ? { disabled: false } : { disabled: true, hint: `${videoName} can't start from an image` };
  if (!videoSupports.start) result.last = { disabled: true, hint: `${videoName} can't start from an image` };
  else if (!videoSupports.last) result.last = { disabled: true, hint: `${videoName} has no last frame` };
  else result.last = { disabled: false };
  return result;
}

/** Whether the video model's Video tab takes an audio track: the workflow offers the control and the model does not hide it. */
function videoTakesAudioTrack(ctx: WorkspaceContext, videoModel: string): boolean {
  const offered = ctx.workflow_contract.definitions.txt2vid?.visible_controls ?? [];
  const hidden = ctx.video_model_defaults?.[videoModel]?.hidden_controls ?? [];
  return offered.includes('audio_track') && offered.includes('audio_track_path') && !hidden.includes('audio_track');
}

/** The Use as state of an audio take: the Video model must follow a track, and a speech model must be there to follow a voice. */
function audioUseState(ctx: WorkspaceContext, videoModel: string, videoName: string): UseAsState {
  const cloning = (ctx.audio_models ?? []).filter((model): model is SpeechAudioModel => model.kind === 'speech');
  const state: UseAsState = {
    audio_track: videoTakesAudioTrack(ctx, videoModel) ? { disabled: false, hint: AUDIO_TRACK_HINT } : { disabled: true, hint: `${videoName || 'The video model'} can't follow an audio track` },
    voice: cloning.length > 0 ? { disabled: false, hint: VOICE_HINT } : { disabled: true, hint: 'No speech model here can use a voice' },
  };
  // Saving a take as a voice needs the same speech model and the same length as following it once.
  state.save_voice = { ...state.voice!, hint: cloning.length > 0 ? SAVE_VOICE_HINT : state.voice!.hint };
  if (cloning.length > 0) {
    const longest = cloning.reduce((best, model) => (model.limits.max_voice_seconds > best.limits.max_voice_seconds ? model : best));
    state.voiceMaxSeconds = longest.limits.max_voice_seconds;
    state.voiceModel = longest.id;
  }
  return state;
}

/** Return whether an audio take can be used as the Video tab's audio track: it is a file on the host. */
export function canUseAsAudioTrack(asset: GalleryAsset): boolean {
  return asset.media_type === 'audio' && Boolean(asset.file_path);
}

/** Return whether an audio take can be used as a voice: a speech take that is a file on the host. */
export function canUseAsVoice(asset: GalleryAsset): boolean {
  return asset.media_type === 'audio' && asset.kind === 'speech' && Boolean(asset.file_path);
}

/**
 * The reason a speech take is too long to be a voice, which the menu shows in place of the hint; undefined when it fits. The
 * limit is the speech model's (`useAs.voiceMaxSeconds`): without one, or for a take of unknown length, the take is allowed.
 */
export function voiceTooLongReason(asset: GalleryAsset, useAs: UseAsState = {}): string | undefined {
  const max = useAs.voiceMaxSeconds;
  return max !== undefined && asset.duration != null && asset.duration > max ? `Use a take of ${max} s or less as a voice` : undefined;
}

/**
 * Return the workspace prefill params that make an audio take the Video tab's audio track, from its first second.
 *
 * `use_as=audio_track` marks a Use as: the Workspace sets only the track, drops only that tab's held upload and keeps the
 * rest of the draft, as for an image slot. The workflow `txt2vid` opens the Video tab; with a first frame set it follows
 * from the images as usual.
 */
export function audioTrackParams(asset: GalleryAsset): Record<string, string> {
  return { workflow: 'txt2vid', audio_track_path: asset.file_path ?? '', audio_track_start: '0', use_as: 'audio_track' };
}

/**
 * Return the workspace prefill params that make a speech take the Speech tab's voice, with its words as "what the recording says".
 *
 * `use_as=voice` marks a Use as: only the voice, its transcript and the workflow change. `voice_clone=true` is present only
 * for a take that was itself made from a recording (`has_reference_voice`): such a voice is the speaker's, so the Speech tab
 * shows the consent tick for it. A take made with no reference voice (*Any*) carries no such param and needs no tick. The
 * consent itself is never part of a URL.
 */
export function voiceParams(asset: GalleryAsset): Record<string, string> {
  const params: Record<string, string> = { workflow: 'speech', voice_path: asset.file_path ?? '', voice_text: asset.prompt ?? '', use_as: 'voice' };
  if (asset.has_reference_voice === true) params.voice_clone = 'true';
  return params;
}

/** What the Save as voice dialog starts from: the take's file, its words without Delivery marks, and whether the tick is asked for. */
export interface SaveVoiceStart {
  text: string;
  consentNeeded: boolean;
}

/**
 * Return what saving a speech take as a voice starts from. The words are the take's prompt without its Delivery *marks*, which
 * steer the delivery and are not said, and without the spaces they leave behind. The tick is asked for only when the take was
 * itself made from a recording (`has_reference_voice`): such a voice is the speaker's, and a take in a built-in voice is nobody's.
 */
export function saveVoiceStart(asset: GalleryAsset, marks: readonly string[]): SaveVoiceStart {
  // A mark leaves its spaces behind: runs shrink to one, a space before punctuation (`harbour [sad].`) goes, and the ends are trimmed.
  const text = stripMarks(asset.prompt ?? '', marks).replace(/[ \t]{2,}/g, ' ').replace(/[ \t]+([.,!?;:])/g, '$1').trim();
  return { text, consentNeeded: asset.has_reference_voice === true };
}

/** Return whether a video can be extended right now. */
export function canExtend(asset: GalleryAsset): boolean {
  return asset.extend?.available === true;
}

/** Return why a video cannot be extended, or null when it can or the asset does not offer extending. */
export function extendReason(asset: GalleryAsset): string | null {
  return asset.extend && !asset.extend.available ? asset.extend.reason : null;
}

/**
 * Return the workspace prefill params that make a video the source of an extension.
 *
 * The text is left out on purpose: the server describes the source when the Workspace inspects it, and that answer fills
 * Scene (a recorded scene) or Prompt (a single recorded prompt) or leaves both empty. The gallery's own prompt may be only
 * a display fallback such as the file name, and must never reach the boxes.
 */
export function extendParams(asset: GalleryAsset): Record<string, string> {
  const params: Record<string, string> = { workflow: 'extend', source_video: asset.file_path ?? '' };
  const model = asset.reuse_state?.resolved_model;
  if (model) params.model = model;
  params.lora = asset.lora ?? '';
  return params;
}

/** Ask the user to approve deleting one asset; resolves true when they confirm. */
export function confirmDeleteAsset(asset: GalleryAsset): Promise<boolean> {
  return requestConfirm({ question: `Delete "${asset.filename}"?`, info: 'This cannot be undone.', confirmLabel: 'Delete' });
}

/** Describe a backend reuse fallback reason in words. */
export function describeFallbackReason(reason: string): string {
  return FALLBACK_REASON_TEXT[reason] ?? reason;
}

/** Return the width/height aspect ratio of an asset, clamped to a displayable range. */
export function assetAspect(asset: GalleryAsset, min = 0.5, max = 2): number {
  if (!asset.width || !asset.height) return 1;
  return Math.min(max, Math.max(min, asset.width / asset.height));
}
