import type { GalleryAsset, GalleryPage, ImageModelDefaults, JobSnapshot, VideoModelDefaults, WorkspaceContext } from '$lib/types';

// Fixtures of the Workspace page tests: a server context with one image and one video model, and the job and asset records a page shows.

export function makeImageDefaults(overrides: Partial<ImageModelDefaults> = {}): ImageModelDefaults {
  return {
    ratio: '2:3',
    size: 'm',
    steps: 28,
    guidance: 6.2,
    width: 832,
    height: 1216,
    scheduler: 'beta',
    supports_negative_prompt: true,
    supports_quantize: true,
    quantize: null,
    image_strength: 0.5,
    postprocess: { sharpen: 0.8, contrast: false, saturation: false },
    upscale: {
      enabled: false,
      factor: null,
      denoise: null,
      steps: null,
      guidance: null,
      sharpen: true,
      save_pre: false,
    },
    supports_img2img: true,
    supports_upscale: true,
    supports_json_prompt: false,
    supports_first_sigma: false,
    supports_scheduler: true,
    max_reference_images: 0,
    dimension_min: 16,
    dimension_max: null,
    dimension_step: 16,
    ...overrides,
  };
}

export function makeIdeogramDefaults(overrides: Partial<ImageModelDefaults> = {}): ImageModelDefaults {
  return makeImageDefaults({
    ratio: '16:9',
    size: 'l',
    steps: 20,
    guidance: 7,
    width: 1664,
    height: 928,
    supports_negative_prompt: false,
    supports_img2img: false,
    supports_upscale: false,
    supports_json_prompt: true,
    supports_first_sigma: true,
    supports_scheduler: true,
    dimension_min: 256,
    dimension_max: 2048,
    dimension_step: 16,
    ...overrides,
  });
}

export function makeVideoDefaults(overrides: Partial<VideoModelDefaults> = {}): VideoModelDefaults {
  return {
    ratio: '16:9',
    size: 'm',
    steps: 8,
    width: 704,
    height: 448,
    frame_count: 49,
    audio: true,
    low_memory: true,
    supports_i2v: true,
    supports_quantize: false,
    quantize: null,
    max_steps: 8,
    fps: 24,
    frame_rule: { alignment: 8, offset: 1, round: 'nearest' },
    supports_extend: true,
    extend_frame_step: 8,
    upscale: {
      enabled: false,
      factor: 2,
      steps: null,
    },
    ...overrides,
  };
}

export function makeContext(overrides: Partial<WorkspaceContext> = {}): WorkspaceContext {
  const imageDefaults = makeImageDefaults();
  const alternateImageDefaults = makeImageDefaults({
    ratio: '16:9',
    size: 'l',
    steps: 12,
    guidance: 3.5,
    width: 1216,
    height: 832,
    supports_negative_prompt: false,
    supports_quantize: false,
  });
  const videoDefaults = makeVideoDefaults();

  return {
    image_models: [
      { id: 'zit', label: 'zit', type: 'image' },
      { id: 'flux-lite', label: 'flux-lite', type: 'image' },
    ],
    video_models: [{ id: 'ltx-8', label: 'ltx-8', type: 'video' }],
    loras: [],
    history_assets: [],
    active_job: null,
    defaults: imageDefaults,
    video_defaults: videoDefaults,
    image_model_defaults: {
      zit: imageDefaults,
      'flux-lite': alternateImageDefaults,
    },
    video_model_defaults: {
      'ltx-8': videoDefaults,
    },
    current_image_model: 'zit',
    current_video_model: 'ltx-8',
    config: {
      gallery_page_size: 20,
      startup_view: 'workspace',
    },
    output_dir: '/tmp/output',
    quantize_options: [4, 8],
    image_ratios: ['2:3', '16:9'],
    video_ratios: ['16:9'],
    image_size_options: { '2:3': ['m'], '16:9': ['l'] },
    image_size_dimensions: {},
    video_size_options: { '16:9': ['m'] },
    scheduler_options: ['beta'],
    workflow_contract: {
      values: ['txt2img', 'img2img', 'txt2vid', 'img2vid'],
      definitions: {
        txt2img: {
          mode: 'image',
          model_kind: 'image',
          visible_controls: [
            'workflow', 'model', 'quantize', 'loras', 'prompt_source', 'prompt_inline', 'negative_prompt',
            'prompt_file_path', 'prompt_file_option', 'prompt_file_preview', 'prompt_file_edit',
            'references', 'references_clear',
            'ratio', 'size', 'custom_dimensions', 'runs', 'steps', 'guidance', 'seed', 'scheduler',
            'postprocess_sharpen', 'postprocess_contrast', 'postprocess_saturation',
            'image_upscale_enabled', 'image_upscale_factor', 'image_upscale_denoise', 'image_upscale_steps',
            'image_upscale_guidance', 'image_upscale_sharpen', 'image_upscale_review'
          ],
          supports_reference_image: false,
          requires_reference_image: false,
          requires_source_video: false,
          clear_fields: ['image_path', 'image_strength', 'frames', 'audio', 'low_memory'],
        },
        img2img: {
          mode: 'image',
          model_kind: 'image',
          visible_controls: [
            'workflow', 'model', 'quantize', 'loras', 'prompt_source', 'prompt_inline', 'negative_prompt',
            'prompt_file_path', 'prompt_file_option', 'prompt_file_preview', 'prompt_file_edit',
            'reference_image', 'reference_image_path', 'reference_image_clear',
            'references', 'references_clear',
            'ratio', 'size', 'custom_dimensions', 'runs', 'steps', 'guidance', 'image_strength', 'seed', 'scheduler',
            'postprocess_sharpen', 'postprocess_contrast', 'postprocess_saturation',
            'image_upscale_enabled', 'image_upscale_factor', 'image_upscale_denoise', 'image_upscale_steps',
            'image_upscale_guidance', 'image_upscale_sharpen', 'image_upscale_review'
          ],
          supports_reference_image: true,
          requires_reference_image: true,
          requires_source_video: false,
          clear_fields: ['frames', 'audio', 'low_memory'],
        },
        txt2vid: {
          mode: 'video',
          model_kind: 'video',
          visible_controls: [
            'workflow', 'model', 'loras', 'prompt_source', 'prompt_inline', 'prompt_scene',
            'prompt_file_path', 'prompt_file_option', 'prompt_file_preview', 'prompt_file_edit',
            'ratio', 'size', 'custom_dimensions', 'runs', 'frame_count', 'steps', 'seed', 'audio', 'low_memory',
            'video_upscale_enabled', 'video_upscale_factor', 'video_upscale_review'
          ],
          supports_reference_image: false,
          requires_reference_image: false,
          requires_source_video: false,
          clear_fields: ['negative_prompt', 'guidance', 'image_path', 'image_strength', 'quantize'],
        },
        img2vid: {
          mode: 'video',
          model_kind: 'video',
          visible_controls: [
            'workflow', 'model', 'loras', 'prompt_source', 'prompt_inline', 'prompt_scene',
            'prompt_file_path', 'prompt_file_option', 'prompt_file_preview', 'prompt_file_edit',
            'reference_image', 'reference_image_path', 'reference_image_clear',
            'ratio', 'size', 'custom_dimensions', 'runs', 'frame_count', 'steps', 'seed', 'audio', 'low_memory',
            'video_upscale_enabled', 'video_upscale_factor', 'video_upscale_review'
          ],
          supports_reference_image: true,
          requires_reference_image: true,
          requires_source_video: false,
          clear_fields: ['negative_prompt', 'guidance', 'quantize'],
        },
        extend: {
          mode: 'video',
          model_kind: 'video',
          visible_controls: ['workflow', 'model', 'loras', 'prompt_source', 'prompt_scene', 'prompt_inline', 'source_video', 'source_video_path', 'source_video_clear', 'extend_length', 'runs', 'steps', 'seed', 'audio', 'low_memory'],
          supports_reference_image: false,
          requires_reference_image: false,
          requires_source_video: true,
          clear_fields: ['negative_prompt', 'guidance', 'image_path', 'image_strength', 'quantize', 'ratio', 'size', 'width', 'height', 'frames', 'video_upscale_enabled', 'video_upscale_factor'],
        },
      },
      field_precedence: {
        defaults: ['cli', 'model_variant', 'model_family', 'global'],
        dimensions: 'explicit_width_height_overrides_ratio_size',
      },
    },
    prompt_sources: ['inline', 'file'],
    default_prompt_source: 'inline',
    prompt_file: {
      accepted_extensions: ['.yaml', '.yml'],
      selection_required: true,
      trust_boundary: {
        scope: 'server_host_only',
        manual_entry: 'submitted_value_kept_until_backend_validation',
        picker: 'in_app_browser',
        read_write: 'yaml_files_only',
      },
      help: {
        path: 'Prompt file path.',
        option_required: 'Select an active prompt option before generating.',
        option_optional: 'Select an active prompt option from the file.',
        empty_options: 'This prompt file has no active prompt options.',
        stale_selection: 'The previously selected prompt option is no longer active.',
        loaded: 'Prompt file loaded.',
        ignored_negative_video: 'Negative prompt entries are ignored for video workflows.',
        ignored_negative_unsupported: 'The current image model ignores negative prompt entries.',
        ignored_scene_image: 'Scene entries are ignored for image workflows.',
      },
    },
    ...overrides,
  };
}

export function makeAsset(overrides: Partial<GalleryAsset> = {}): GalleryAsset {
  return {
    id: 'output/result.png',
    url: '/media/output/result.png',
    thumbnail_url: '/media/output/result.png',
    filename: 'result.png',
    created_at: '2026-04-30T12:00:00Z',
    workflow: 'txt2img',
    prompt: 'Test completed output',
    model: 'zit',
    media_type: 'image',
    reuse_workspace_url: '#/workspace?workflow=txt2img',
    ...overrides,
  };
}

export function makeGalleryPage(assets: GalleryAsset[] = []): GalleryPage {
  return {
    assets,
    page: 1,
    total_pages: 1,
    total_count: assets.length,
  };
}

/** Build a job snapshot with the given status. */
export function jobSnap(jobId: string, status: JobSnapshot['status'], extra: Partial<JobSnapshot> = {}): JobSnapshot {
  return {
    id: jobId, job_id: jobId, workflow: 'txt2img', job_type: 'Text to Image', status, created_at: '2026-10-04T10:00:00Z', completed_at: null,
    event_count: 1, last_event: { type: 'job_started' }, supported_controls: ['quit'], paused: false, result_path: null,
    prompt: `prompt of ${jobId}`, model: 'zit', runs: 1, ...extra,
  };
}


/** Add the prompt enhancer to the image workflows of *context*. */
export function withEnhancer(context: WorkspaceContext): WorkspaceContext {
  context.workflow_contract.definitions.txt2img.visible_controls.push('prompt_enhance', 'prompt_enhance_auto');
  context.prompt_enhancer = {
    matrix: {
      axes: [
        { key: 'style', label: 'Style', multi: false, video_only: false, audio_only: false, options: [{ slug: 'keep', label: 'Keep', description: '', group: null }, { slug: 'photo', label: 'Photographic', description: '', group: null }], default: ['keep'] },
        { key: 'details', label: 'Details', multi: true, video_only: false, audio_only: false, options: [{ slug: 'lighting', label: 'Lighting', description: '', group: null }], default: ['lighting'] },
        { key: 'length', label: 'Length', multi: false, video_only: false, audio_only: false, options: [{ slug: 'same', label: 'Same', description: '', group: null }, { slug: 'longer', label: 'Longer', description: '', group: null }], default: ['same'] },
        { key: 'motion', label: 'Motion', multi: true, video_only: true, audio_only: false, options: [{ slug: 'action', label: 'Action', description: '', group: null }], default: ['action'] },
      ],
      defaults: { style: 'keep', mood: 'keep', theme: 'keep', camera: 'keep', pace: 'keep', details: ['lighting'], length: 'same', motion: ['action'], sound: [], note: '' },
    },
    model: 'owner/llm',
    revision: null,
    downloaded: true,
    download_size_label: null,
    default_max_words: 300,
    error: null,
  };
  return context;
}

/** Add the prompt enhancer to the video workflows as well. */
export function withVideoEnhancer(context: WorkspaceContext): WorkspaceContext {
  withEnhancer(context);
  for (const workflow of ['txt2vid', 'img2vid'] as const) {
    context.workflow_contract.definitions[workflow].visible_controls.push('prompt_enhance', 'prompt_enhance_auto');
  }
  return context;
}
