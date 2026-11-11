import { trackOf, trackSet } from '$lib/state/audioTrack';
import { audioModelById } from '$lib/state/audioDraft';
import { imageSupport } from '$lib/state/imageSlots';
import { isAudioWorkflow, workspaceTab } from '$lib/state/workspaceTab';
import type { AudioWorkflow, DraftState, ImageModelDefaults, PromptEnhancerContract, VideoModelDefaults, WorkspaceContext } from '$lib/types';

/** Which workspace controls apply to the current workflow and model. */
export interface WorkspaceCapabilities {
  isImageMode: boolean;
  /** The Audio tab is open: no image, no video, only the fields of its kind. */
  isAudio: boolean;
  /** The Edit kind of the Image tab is open (`edit`): the prompt, model, Quant, LoRAs, Batch, Steps and Seed, and the Image to edit section. */
  isEdit: boolean;
  /** The settings Preset: the workflow lists it (New and Video, not Audio or Edit). */
  showPreset: boolean;
  /** The kind the Audio tab shows (its workflow), or null on any other tab. */
  audioKind: AudioWorkflow | null;
  /** A sound, or a song on a model that sets its length (its row does not hide `length`): the Length (s) row. */
  showLength: boolean;
  /** Music, on a model that sings lyrics: the Lyrics box. */
  showLyrics: boolean;
  /** Music: the Tempo row. */
  showTempo: boolean;
  /** Music: the Key row. */
  showKey: boolean;
  /** Speech on a model that steers its delivery: the Delivery section, which holds one line instead of its controls for a voice that follows a recording. */
  offersDelivery: boolean;
  /** Speech in a built-in voice: the Delivery controls. A voice that follows a recording reads as recorded, so it has none. */
  showDelivery: boolean;
  /** Sound: the Sound Type row. */
  showSoundType: boolean;
  /** Speech, on a model that can follow a recording: the Voice section. */
  showVoice: boolean;
  supportsNegativePrompt: boolean;
  supportsJsonPrompt: boolean;
  supportsFirstSigma: boolean;
  dimensionMin: number;
  dimensionMax: number | null;
  dimensionStep: number;
  showPromptSource: boolean;
  showPromptInline: boolean;
  /** The prompts come from a prompts file: the draft's source is File and the workflow offers a prompt source (the Audio tab and Edit take their text from their own boxes, whatever New last chose). */
  promptFileMode: boolean;
  /** Video workflows: the Scene box (the Default scene in prompt-file mode). */
  showScene: boolean;
  showPromptFileControls: boolean;
  /** The workflow continues a chosen video (extend): it has a Source video section in place of Size. */
  showSourceVideo: boolean;
  /** Extend only: the length of the new frames, in place of Duration. */
  showExtendLength: boolean;
  showNegativePrompt: boolean;
  showEnhance: boolean;
  showEnhanceAuto: boolean;
  enhancer: PromptEnhancerContract | null;
  enhanceMaxWords: number;
  /** The start image slot: img2img's image, or a video's first frame. It shows in both image workflows of a tab, not on a model that cannot start from an image. */
  showRefImage: boolean;
  /** How many references the model takes; 0 when the list is not shown. */
  maxReferences: number;
  /** The model can end a clip on a given image: the Last frame slot is offered, whether or not a first frame is set. */
  takesLastFrame: boolean;
  showDimensions: boolean;
  showRuns: boolean;
  showFrameCount: boolean;
  showSteps: boolean;
  showGuidance: boolean;
  showI2IStrength: boolean;
  showSeed: boolean;
  showScheduler: boolean;
  showPostprocessSharpen: boolean;
  showPostprocessContrast: boolean;
  showPostprocessSaturation: boolean;
  showPostprocessTransparent: boolean;
  showImageUpscale: boolean;
  showUpscaleDenoise: boolean;
  showUpscaleSteps: boolean;
  showUpscaleGuidance: boolean;
  showUpscaleSharpen: boolean;
  showUpscaleReview: boolean;
  showVideoUpscaleReview: boolean;
  /** The Audio switch and the enhancer's Sound options: off while an audio track is in effect, which is the clip's sound. */
  showAudio: boolean;
  /** The Audio track section: a video workflow on a model that follows a track. */
  showAudioTrack: boolean;
  /** An audio track is set on a model that follows one: the clip's sound is exactly that track. */
  audioTrackInEffect: boolean;
  showLowMemory: boolean;
  showVideoUpscale: boolean;
  showVideoUpscaleFactor: boolean;
}

/** Derive control visibility from the backend workflow contract and the current model's capability flags. */
export function workspaceCapabilities(context: WorkspaceContext | null, state: DraftState): WorkspaceCapabilities {
  const definition = context?.workflow_contract.definitions[state.workflow];
  const isImageMode = definition?.mode === 'image';
  const isAudio = definition?.mode === 'audio';
  const videoDefaults = context?.video_model_defaults?.[state.model] as VideoModelDefaults | undefined;
  // A model that lacks a control (low memory, upscale, Lyrics, ...) does not show it, whatever the workflow offers. An audio model's are those of its row for the open kind.
  const audioKind = isAudio && isAudioWorkflow(state.workflow) ? state.workflow : null;
  const audioModel = audioKind === null ? null : audioModelById(context, state.model, audioKind);
  const hidden = new Set(isImageMode ? [] : isAudio ? (audioModel?.hidden_controls ?? []) : (videoDefaults?.hidden_controls ?? []));
  const visible = new Set((definition?.visible_controls ?? []).filter((control) => !hidden.has(control)));
  const imageDefaults = isImageMode
    ? ((context?.image_model_defaults?.[state.model] ?? context?.defaults) as ImageModelDefaults | undefined)
    : undefined;
  // Permissive fallbacks keep models without explicit capability flags unaffected.
  const supportsUpscale = imageDefaults?.supports_upscale ?? true;
  const supportsScheduler = imageDefaults?.supports_scheduler ?? true;
  // A control shows when the workflow lists it and the model supports it.
  const supportsNegativePrompt = visible.has('negative_prompt') && (imageDefaults?.supports_negative_prompt ?? false);
  const enhancer = context?.prompt_enhancer ?? null;
  // What the model takes for images is `imageSupport`'s; the workflow's contract decides which of it the workflow shows.
  const images = imageSupport(context, state.model, isImageMode ? 'image' : 'video');
  const showRefImage = (visible.has('reference_image') || visible.has('reference_image_path')) && images.start;
  const maxReferences = visible.has('references') ? images.references : 0;
  const showImageUpscale = isImageMode && visible.has('image_upscale_enabled') && supportsUpscale;
  const takesLastFrame = (visible.has('end_image') || visible.has('end_image_path')) && images.last;
  // A track set on a model without the capability stays in the draft but is hidden, not sent and not counted.
  const showAudioTrack = !isImageMode && visible.has('audio_track') && visible.has('audio_track_path');
  const tab = workspaceTab(state.workflow);
  const audioTrackInEffect = showAudioTrack && (tab === 'video' || tab === 'extend') && trackSet(trackOf(state, tab));
  // A model that cannot upscale a clip made from a track (H3) loses the Upscale controls while one is in effect.
  const upscaleWithTrack = !audioTrackInEffect || videoDefaults?.supports_audio_track_upscale !== false;

  return {
    isImageMode,
    isAudio,
    isEdit: state.workflow === 'edit',
    showPreset: visible.has('preset'),
    audioKind,
    showLength: visible.has('length'),
    showLyrics: visible.has('lyrics'),
    showTempo: visible.has('tempo'),
    showKey: visible.has('key'),
    offersDelivery: visible.has('delivery'),
    showDelivery: visible.has('delivery') && state.voiceMode === 'builtin',
    showSoundType: visible.has('sound_type'),
    showVoice: visible.has('voice') && visible.has('voice_path') && visible.has('voice_text'),
    supportsNegativePrompt,
    supportsJsonPrompt: visible.has('json_prompt') && (imageDefaults?.supports_json_prompt ?? false),
    supportsFirstSigma: visible.has('first_sigma') && (imageDefaults?.supports_first_sigma ?? false),
    dimensionMin: imageDefaults?.dimension_min ?? 16,
    dimensionMax: imageDefaults?.dimension_max ?? null,
    dimensionStep: imageDefaults?.dimension_step ?? 16,
    showPromptSource: visible.has('prompt_source') && (context?.prompt_sources.length ?? 0) > 0,
    showPromptInline: visible.has('prompt_inline'),
    promptFileMode: state.promptSource === 'file' && visible.has('prompt_source'),
    showScene: visible.has('prompt_scene'),
    showSourceVideo: visible.has('source_video'),
    showExtendLength: visible.has('extend_length'),
    showPromptFileControls: ['prompt_file_path', 'prompt_file_option', 'prompt_file_preview', 'prompt_file_edit'].some((id) => visible.has(id)),
    showNegativePrompt: supportsNegativePrompt,
    showEnhance: enhancer !== null && visible.has('prompt_enhance'),
    showEnhanceAuto: enhancer !== null && visible.has('prompt_enhance_auto'),
    enhancer,
    enhanceMaxWords: (isImageMode ? imageDefaults?.enhance_max_words : videoDefaults?.enhance_max_words)
      ?? enhancer?.default_max_words
      ?? 300,
    showRefImage,
    maxReferences,
    takesLastFrame,
    showDimensions: visible.has('ratio') || visible.has('size') || visible.has('custom_dimensions'),
    showRuns: visible.has('runs'),
    showFrameCount: visible.has('frame_count'),
    showSteps: visible.has('steps'),
    showGuidance: visible.has('guidance'),
    showI2IStrength: visible.has('image_strength'),
    showSeed: visible.has('seed'),
    showScheduler: isImageMode && visible.has('scheduler') && supportsScheduler && (context?.scheduler_options.length ?? 0) > 0,
    showPostprocessSharpen: isImageMode && visible.has('postprocess_sharpen'),
    showPostprocessContrast: isImageMode && visible.has('postprocess_contrast'),
    showPostprocessSaturation: isImageMode && visible.has('postprocess_saturation'),
    showPostprocessTransparent: isImageMode && visible.has('postprocess_transparent'),
    showImageUpscale,
    showUpscaleDenoise: showImageUpscale && visible.has('image_upscale_denoise'),
    showUpscaleSteps: showImageUpscale && visible.has('image_upscale_steps'),
    showUpscaleGuidance: showImageUpscale && visible.has('image_upscale_guidance'),
    showUpscaleSharpen: showImageUpscale && visible.has('image_upscale_sharpen'),
    showUpscaleReview: showImageUpscale && visible.has('image_upscale_review'),
    showAudio: visible.has('audio') && !audioTrackInEffect,
    showAudioTrack,
    audioTrackInEffect,
    showLowMemory: visible.has('low_memory'),
    showVideoUpscale: visible.has('video_upscale_enabled') && upscaleWithTrack,
    showVideoUpscaleFactor: visible.has('video_upscale_factor') && upscaleWithTrack,
    showVideoUpscaleReview: visible.has('video_upscale_enabled') && visible.has('video_upscale_review') && upscaleWithTrack,
  };
}
