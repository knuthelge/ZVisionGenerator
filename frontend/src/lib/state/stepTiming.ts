/** The average step time of the current stage, measured on the client. */
export interface StepTiming {
  scope: string;
  step: number;
  /** Step and time the current measurement window started at; null while paused. */
  anchor: { step: number; at: number } | null;
  stepMs: number | null;
}

export interface StepSample {
  /** Identifies one stepped stage; a new scope discards the old measurement. */
  scope: string;
  step: number;
  paused: boolean;
  now: number;
}

/**
 * Fold one progress sample into the step timing.
 *
 * Step events can arrive unevenly (mflux reports a step before computing it, except on preview steps),
 * so the estimate averages every step since the stage started or the job last resumed.
 */
export function trackStepTiming(previous: StepTiming | null, sample: StepSample): StepTiming {
  const { scope, step, paused, now } = sample;
  if (!previous || previous.scope !== scope) {
    return { scope, step, anchor: paused ? null : { step, at: now }, stepMs: null };
  }
  if (paused) return { ...previous, step, anchor: null };
  if (step === previous.step) return previous;
  const anchor = previous.anchor;
  if (!anchor || step < anchor.step) return { ...previous, step, anchor: { step, at: now } };
  return { scope, step, anchor, stepMs: (now - anchor.at) / (step - anchor.step) };
}
