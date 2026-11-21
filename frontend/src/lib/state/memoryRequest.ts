import type { DraftState, EditMode, ExtendSide, FrameRule } from '$lib/types';
import type { PictureSize } from './referenceSizes.svelte';
import { alignFrames } from './videoDuration';

/**
 * The body of `POST /api/memory`: what a job form submits for the job to be judged.
 *
 * `media` and `kind` pick the job; the other fields are the ones of its medium (see `memory_response` in `web/memory_api.py`).
 */
export interface MemoryRequest {
  media: 'image' | 'video';
  /** `image` or `edit` for an image; `clip` or `extension` for a video. */
  kind: 'image' | 'edit' | 'clip' | 'extension';
  model: string;
  /** Image: the quantisation level the job would use; absent for none. */
  quantize?: number;
  /** A preset size: the ratio and size names. Absent for a custom size and for an edit, whose canvas the server sizes. */
  ratio?: string;
  size?: string;
  /** A custom size in pixels. Absent for a preset size and for an edit. */
  width?: number;
  height?: number;
  /** Video: how many reference pictures the clip holds. */
  references?: number;
  /** Image: the size of each reference picture chosen from this device in pixels, `[width, height]`, measured by the browser. */
  reference_sizes?: [number, number][];
  /** Image: the path of each reference picture that is a file on the server's machine; the server reads its size. */
  reference_paths?: string[];
  /** The selected LoRAs as the job form submits them (`name:weight,...`); absent when none. */
  lora?: string;
  /** Edit: the picture's size in pixels. */
  source_width?: number;
  source_height?: number;
  /** Edit: a painted mask is sent (Paint). */
  masked?: boolean;
  /** Edit: the extension as `side=amount` (Extend). */
  extend?: string;
  /** Video: Low memory is on; LTX judges its placement by it. */
  low_memory?: boolean;
  /** Clip: the frame count the job would use. */
  frames?: number;
  /** Clip: the upscale factor, or null when the clip is not upscaled. Image: the factor of an upscale after the render; absent without one. */
  upscale?: number | null;
  /** Clip: an Image to Video clip, whose keyframe costs memory and time. Image: a start image is set (Image to Image). */
  image?: boolean;
  /** Clip and extension: the steps the job would run, for a model that offers the control. */
  steps?: number;
  /** Clip: the upscale pauses to review the small render first. */
  review?: boolean;
  /** Extension: the video it continues, and the new frames it adds. */
  source?: string;
  extend_frames?: number;
}

/** Which video controls the model offers, as they decide what the job form submits. */
export interface ClipMemoryControls {
  frameRule: FrameRule;
  /** The Low memory toggle is shown; without it the server's default (on) applies. */
  lowMemory: boolean;
  /** The Upscale toggle is shown. */
  upscale: boolean;
  /** The model offers a choice of step counts, which set its time; the steps are sent only then. */
  steps?: boolean;
  /** The Review toggle is shown; without it no review is submitted. */
  review?: boolean;
  /** How many reference images the job would send. */
  references?: number;
}

/** The size the form would submit: its custom width and height, or its ratio and size names; null while either is not complete yet. */
function requestSize(state: DraftState): Pick<MemoryRequest, 'ratio' | 'size' | 'width' | 'height'> | null {
  if (state.dimensionMode === 'custom') return state.width > 0 && state.height > 0 ? { width: state.width, height: state.height } : null;
  return state.ratio && state.size ? { ratio: state.ratio, size: state.size } : null;
}

/**
 * The memory check of the clip the form would submit, or null when there is nothing to check.
 *
 * Only Text to Video and Image to Video are checked here; an extension has its own request. A custom size that is not complete yet,
 * or a missing model or length, has nothing to check either.
 */
export function clipMemoryRequest(state: DraftState, controls: ClipMemoryControls): MemoryRequest | null {
  if (state.workflow !== 'txt2vid' && state.workflow !== 'img2vid') return null;
  if (!state.model || !(state.frameCount > 0)) return null;
  const size = requestSize(state);
  if (size === null) return null;
  const upscale = controls.upscale && state.videoUpscaleEnabled ? state.videoUpscaleFactor : null;
  const lora = state.loraString.trim();
  return {
    media: 'video',
    kind: 'clip',
    model: state.model,
    ...size,
    frames: alignFrames(state.frameCount, controls.frameRule),
    low_memory: controls.lowMemory ? state.lowMemory : true,
    upscale,
    ...(state.workflow === 'img2vid' ? { image: true } : {}),
    ...(controls.steps && state.steps > 0 ? { steps: state.steps } : {}),
    ...(lora ? { lora } : {}),
    ...(upscale !== null && controls.review && state.videoUpscaleReview ? { review: true } : {}),
    ...(controls.references ? { references: controls.references } : {}),
  };
}

/**
 * The memory check of the extension the form would submit, or null when there is nothing to check: another workflow, no model or no source.
 *
 * The server judges the window the model generates, which it works out from the source clip and the length asked for (the default
 * length when none is set yet).
 */
export function extensionMemoryRequest(state: DraftState, controls: Pick<ClipMemoryControls, 'lowMemory' | 'steps' | 'references'>): MemoryRequest | null {
  if (state.workflow !== 'extend' || !state.model || !state.extendSourcePath) return null;
  const lora = state.loraString.trim();
  return {
    media: 'video',
    kind: 'extension',
    model: state.model,
    source: state.extendSourcePath,
    low_memory: controls.lowMemory ? state.lowMemory : true,
    ...(state.extendFrames !== null ? { extend_frames: state.extendFrames } : {}),
    ...(controls.steps && state.steps > 0 ? { steps: state.steps } : {}),
    ...(lora ? { lora } : {}),
    ...(controls.references ? { references: controls.references } : {}),
  };
}

/** The facts of the open Edit that decide what it needs in memory. */
export interface ImageMemoryEditControls {
  mode: EditMode;
  /** The picture's size in pixels once decoded; null while unknown. */
  width: number | null;
  height: number | null;
  /** A painted mask is held (Paint). */
  painted: boolean;
  /** The side and amount set for Extend; null when no side is chosen. */
  extension: { side: ExtendSide; amount: number } | null;
}

/** What the image job form submits that the memory check needs. */
export interface ImageMemoryControls {
  /** The Quant control is shown; without it no quantisation is submitted. */
  quantize: boolean;
  /** The size of each reference picture chosen from this device, measured in the browser. */
  referenceSizes: readonly PictureSize[];
  /** The path of each reference picture that is a file on the server's machine. */
  referencePaths: readonly string[];
  /** The Upscale control is offered; without it no upscale is submitted. */
  upscale: boolean;
  /** The open Edit; null outside the Edit kind. */
  edit: ImageMemoryEditControls | null;
}

/**
 * The memory check of the image or edit the form would submit, or null when there is nothing to check.
 *
 * Nothing is checked for another workflow, without a model, for a custom size that is not complete yet, or for an edit
 * whose picture size is not known. An edit sends only the input of its open mode: the mask in Paint, the extension in Extend. An image
 * sends the upscale factor while Upscale is on and offered, and `image` when it starts from a picture: both change what it needs.
 */
export function imageMemoryRequest(state: DraftState, controls: ImageMemoryControls): MemoryRequest | null {
  const { workflow } = state;
  if (workflow !== 'txt2img' && workflow !== 'img2img' && workflow !== 'edit') return null;
  if (!state.model) return null;
  const quantize = controls.quantize && state.quantize !== null ? { quantize: state.quantize } : {};
  const lora = state.loraString.trim() ? { lora: state.loraString.trim() } : {};
  if (workflow === 'edit') {
    const edit = controls.edit;
    if (edit === null || !(edit.width !== null && edit.width > 0 && edit.height !== null && edit.height > 0)) return null;
    return {
      media: 'image',
      kind: 'edit',
      model: state.model,
      ...quantize,
      ...lora,
      source_width: edit.width,
      source_height: edit.height,
      ...(edit.mode === 'paint' && edit.painted ? { masked: true } : {}),
      ...(edit.mode === 'extend' && edit.extension ? { extend: `${edit.extension.side}=${edit.extension.amount}` } : {}),
    };
  }
  const size = requestSize(state);
  if (size === null) return null;
  return {
    media: 'image',
    kind: 'image',
    model: state.model,
    ...quantize,
    ...lora,
    ...size,
    ...(controls.referenceSizes.length > 0 ? { reference_sizes: controls.referenceSizes.map((size): [number, number] => [size.width, size.height]) } : {}),
    ...(controls.referencePaths.length > 0 ? { reference_paths: [...controls.referencePaths] } : {}),
    ...(controls.upscale && state.upscaleEnabled ? { upscale: state.upscaleFactor } : {}),
    ...(workflow === 'img2img' ? { image: true } : {}),
  };
}
