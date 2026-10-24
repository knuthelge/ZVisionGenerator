import type { ClipMemoryFit, DraftState, FrameRule, MemoryFitStatus, MemoryKind } from '$lib/types';
import { alignFrames } from './videoDuration';

/** What each fit level means in practice on each kind of memory; fits needs no explanation. */
export const MEMORY_FIT_CONSEQUENCES: Record<MemoryKind, Partial<Record<MemoryFitStatus, string>>> = {
  unified: {
    tight: 'It runs, but macOS compresses or swaps other memory to make room, so the Mac can slow down while generating.',
    too_large: 'More than MLX will try to fit (1.5× the recommendation): expect heavy swapping or an out-of-memory error.'
  },
  discrete: {
    tight: 'It runs, but slowly: the weights do not fit in system memory and partly stream from disk, or the GPU is nearly full.',
    too_large: "More than this GPU's memory: expect an out-of-memory error. Try a lower quant."
  }
};

/** What the memory note describes: a Text to Video or Image to Video clip, or the window an extension generates. */
export type MemoryFitContext = 'clip' | 'extension';

/** A memory note, shown for tight and too-large fits only. */
export interface MemoryNote {
  status: 'tight' | 'too_large';
  text: string;
}

const LEAD: Record<MemoryFitContext, Record<MemoryNote['status'], string>> = {
  clip: { tight: 'At this size and length: needs', too_large: 'At this size and length: needs' },
  extension: { tight: 'Tight: needs', too_large: 'Too large: needs' },
};
const CONSEQUENCE: Record<MemoryNote['status'], string> = {
  tight: 'The Mac may slow down while generating.',
  too_large: 'Expect heavy swapping or an out-of-memory error.',
};
/** The same on a CUDA GPU, by the memory the figures are about. */
const DISCRETE_RESOURCE: Record<'gpu' | 'system', string> = { gpu: 'GPU memory', system: 'system memory' };
const DISCRETE_CONSEQUENCE: Record<'gpu' | 'system', Record<MemoryNote['status'], string>> = {
  gpu: {
    tight: 'The GPU is nearly full, so it may fail on a longer or larger clip.',
    too_large: 'Expect an out-of-memory error. Use fewer frames or a smaller size.',
  },
  // The model's weights (plus the upscale's extra stage) fill system memory, whatever the clip's length or size.
  system: {
    tight: 'It runs, but slowly, and other apps may stall while it generates. A shorter or smaller clip does not change this.',
    too_large: 'Expect the job to be stopped for lack of memory. A shorter or smaller clip does not help; ltx-4 needs less than ltx-8, and Upscale adds about 4 GB.',
  },
};
/** MiniMax H3's system-memory too-large consequence: the advice about LTX levels and Upscale does not apply to it. */
const H3_SYSTEM_TOO_LARGE = 'Expect the job to be stopped for lack of memory. A shorter or smaller clip does not help; close other apps.';
/** System memory is about the model, not the clip, so its note does not lead with the size and length. */
const SYSTEM_LEAD = 'The model needs';

/** The memory note of a clip or an extension: shown for tight and too-large fits only, which never block Generate. */
export function memoryFitNote(fit: ClipMemoryFit | null, context: MemoryFitContext): MemoryNote | null {
  if (fit === null || (fit.status !== 'tight' && fit.status !== 'too_large')) return null;
  if (fit.kind === 'discrete') {
    const resource = fit.resource ?? 'gpu';
    const lead = resource === 'system' ? SYSTEM_LEAD : LEAD[context][fit.status];
    const consequence = fit.family === 'h3' && resource === 'system' && fit.status === 'too_large' ? H3_SYSTEM_TOO_LARGE : DISCRETE_CONSEQUENCE[resource][fit.status];
    return { status: fit.status, text: `${lead} ~${fit.required_gb} GB of ${fit.budget_gb} GB of ${DISCRETE_RESOURCE[resource]}. ${consequence}` };
  }
  return { status: fit.status, text: `${LEAD[context][fit.status]} ~${fit.required_gb} GB of ${fit.budget_gb} GB. ${CONSEQUENCE[fit.status]}` };
}

/** The estimated generation time of a clip or an extension, in whole minutes and labelled as an estimate; null where the server has none. */
export function clipTimeNote(fit: ClipMemoryFit | null): string | null {
  const seconds = fit?.expected_seconds;
  if (seconds === undefined || !(seconds > 0)) return null;
  return `About ${Math.max(1, Math.round(seconds / 60))} min (estimated).`;
}

/** The fields of `POST /api/video/memory-fit`: what the video job form submits for size and length. */
export interface ClipMemoryRequest {
  model: string;
  /** A preset size: the ratio and size names. Absent for a custom size. */
  ratio?: string;
  size?: string;
  /** A custom size in pixels. Absent for a preset size. */
  width?: number;
  height?: number;
  /** The frame count the job would use (8k + 1). */
  frames: number;
  low_memory: boolean;
  /** The upscale factor, or null when the clip is not upscaled. */
  upscale: number | null;
  /** Present (true) for an Image to Video clip, whose keyframe costs memory and time. */
  image?: boolean;
  /** The steps the job would run, for a model that offers the control. */
  steps?: number;
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
}

/**
 * The memory check of the clip the form would submit, or null when there is nothing to check.
 *
 * Only Text to Video and Image to Video are checked here; an extension has its own check with its source. A custom
 * size that is not complete yet, or a missing model or length, has nothing to check either.
 */
export function clipMemoryRequest(state: DraftState, controls: ClipMemoryControls): ClipMemoryRequest | null {
  if (state.workflow !== 'txt2vid' && state.workflow !== 'img2vid') return null;
  if (!state.model || !(state.frameCount > 0)) return null;
  let size: Pick<ClipMemoryRequest, 'ratio' | 'size' | 'width' | 'height'>;
  if (state.dimensionMode === 'custom') {
    if (!(state.width > 0 && state.height > 0)) return null;
    size = { width: state.width, height: state.height };
  } else {
    if (!state.ratio || !state.size) return null;
    size = { ratio: state.ratio, size: state.size };
  }
  return {
    model: state.model,
    ...size,
    frames: alignFrames(state.frameCount, controls.frameRule),
    low_memory: controls.lowMemory ? state.lowMemory : true,
    upscale: controls.upscale && state.videoUpscaleEnabled ? state.videoUpscaleFactor : null,
    ...(state.workflow === 'img2vid' ? { image: true } : {}),
    ...(controls.steps && state.steps > 0 ? { steps: state.steps } : {}),
  };
}
