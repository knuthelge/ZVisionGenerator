import type { DraftState, ImageModelDefaults, PromptEnhancerContract, VideoModelDefaults, WorkspaceContext } from '$lib/types';

/** Which workspace controls apply to the current workflow and model. */
export interface WorkspaceCapabilities {
  isImageMode: boolean;
  supportsNegativePrompt: boolean;
  supportsJsonPrompt: boolean;
  supportsFirstSigma: boolean;
  dimensionMin: number;
  dimensionMax: number | null;
  dimensionStep: number;
  showPromptSource: boolean;
  showPromptInline: boolean;
  showPromptFileControls: boolean;
  showNegativePrompt: boolean;
  showEnhance: boolean;
  showEnhanceAuto: boolean;
  enhancer: PromptEnhancerContract | null;
  enhanceMaxWords: number;
  showRefImage: boolean;
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
  showImageUpscale: boolean;
  showUpscaleDenoise: boolean;
  showUpscaleSteps: boolean;
  showUpscaleGuidance: boolean;
  showUpscaleSharpen: boolean;
  showAudio: boolean;
  showLowMemory: boolean;
  showVideoUpscale: boolean;
  showVideoUpscaleFactor: boolean;
}

/** Derive control visibility from the backend workflow contract and the current model's capability flags. */
export function workspaceCapabilities(context: WorkspaceContext | null, state: DraftState): WorkspaceCapabilities {
  const definition = context?.workflow_contract.definitions[state.workflow];
  const visible = new Set(definition?.visible_controls ?? []);
  const isImageMode = definition?.mode === 'image';
  const imageDefaults = isImageMode
    ? ((context?.image_model_defaults?.[state.model] ?? context?.defaults) as ImageModelDefaults | undefined)
    : undefined;
  // Permissive fallbacks keep models without explicit capability flags unaffected.
  const supportsImg2img = imageDefaults?.supports_img2img ?? true;
  const supportsUpscale = imageDefaults?.supports_upscale ?? true;
  const supportsScheduler = imageDefaults?.supports_scheduler ?? true;
  const supportsNegativePrompt = isImageMode && (imageDefaults?.supports_negative_prompt ?? false);
  const enhancer = context?.prompt_enhancer ?? null;
  const videoDefaults = context?.video_model_defaults?.[state.model] as VideoModelDefaults | undefined;
  const showImageUpscale = isImageMode && visible.has('image_upscale_enabled') && supportsUpscale;

  return {
    isImageMode,
    supportsNegativePrompt,
    supportsJsonPrompt: isImageMode && (imageDefaults?.supports_json_prompt ?? false),
    supportsFirstSigma: isImageMode && (imageDefaults?.supports_first_sigma ?? false),
    dimensionMin: imageDefaults?.dimension_min ?? 16,
    dimensionMax: imageDefaults?.dimension_max ?? null,
    dimensionStep: imageDefaults?.dimension_step ?? 16,
    showPromptSource: visible.has('prompt_source') && (context?.prompt_sources.length ?? 0) > 0,
    showPromptInline: visible.has('prompt_inline'),
    showPromptFileControls: ['prompt_file_path', 'prompt_file_option', 'prompt_file_preview', 'prompt_file_edit'].some((id) => visible.has(id)),
    showNegativePrompt: visible.has('negative_prompt') && supportsNegativePrompt,
    showEnhance: enhancer !== null && visible.has('prompt_enhance'),
    showEnhanceAuto: enhancer !== null && visible.has('prompt_enhance_auto'),
    enhancer,
    enhanceMaxWords: (isImageMode ? imageDefaults?.enhance_max_words : videoDefaults?.enhance_max_words)
      ?? enhancer?.default_max_words
      ?? 300,
    showRefImage: (visible.has('reference_image') || visible.has('reference_image_path')) && supportsImg2img,
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
    showImageUpscale,
    showUpscaleDenoise: showImageUpscale && visible.has('image_upscale_denoise'),
    showUpscaleSteps: showImageUpscale && visible.has('image_upscale_steps'),
    showUpscaleGuidance: showImageUpscale && visible.has('image_upscale_guidance'),
    showUpscaleSharpen: showImageUpscale && visible.has('image_upscale_sharpen'),
    showAudio: visible.has('audio'),
    showLowMemory: visible.has('low_memory'),
    showVideoUpscale: visible.has('video_upscale_enabled'),
    showVideoUpscaleFactor: visible.has('video_upscale_factor'),
  };
}
