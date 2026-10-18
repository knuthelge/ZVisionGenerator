import type { NumberSpec } from '$lib/actions/scrub';
import { MAX_SHARPEN_AMOUNT } from '$lib/state/draft.svelte';

/** Bounds and granularity of the numeric settings, shared by the inspector rows and the phone's setting popovers. */
export const SPEC = {
  runs: { step: 1, min: 1, max: 100 },
  steps: { step: 1, min: 1, max: 60 },
  guidance: { step: 0.1, min: 0, max: 10 },
  strength: { step: 0.01, min: 0, max: 1 },
  seed: { step: 1, min: 0, max: 2 ** 32 - 1 },
  firstSigma: { step: 0.001, min: 0.001, max: 2 },
  sharpen: { step: 0.05, min: 0, max: MAX_SHARPEN_AMOUNT },
  contrast: { step: 0.05, min: 0.5, max: 2 },
  saturation: { step: 0.05, min: 0.5, max: 2 },
  upscaleDenoise: { step: 0.01, min: 0, max: 1 },
  upscaleSteps: { step: 1, min: 1, max: 60 },
  upscaleGuidance: { step: 0.1, min: 0, max: 20 },
  loraWeight: { step: 0.1, min: -10, max: 10 },
} satisfies Record<string, NumberSpec>;

/** How far along its range a slider value sits, as a CSS percentage for the filled part of the track. */
export function rangeFill(value: number, spec: { min: number; max: number }): string {
  const share = (value - spec.min) / (spec.max - spec.min);
  return `${Math.min(1, Math.max(0, share)) * 100}%`;
}
