import { untrack } from 'svelte';
import type { ActiveJobState } from '$lib/types';
import { isPaused } from './jobView';
import { trackStepTiming, type StepTiming } from './stepTiming';

/**
 * Time the steps of the job *job* returns, for the step ghost: `ms` is how long the last step took, or null until one has
 * been timed. Call it while a component initialises; it keeps its measurement in an effect.
 */
export function useStepTiming(job: () => ActiveJobState): { readonly ms: number | null } {
  let timing = $state<StepTiming | null>(null);
  $effect(() => {
    const current = job();
    const sample = { scope: `${current.job_id}:${current.stageName}:${current.totalSteps}`, step: current.currentStep, paused: isPaused(current), now: performance.now() };
    timing = trackStepTiming(untrack(() => timing), sample);
  });
  return {
    get ms() {
      return timing?.stepMs ?? null;
    },
  };
}
