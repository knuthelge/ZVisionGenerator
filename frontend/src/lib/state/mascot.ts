import type { MascotMood } from '$lib/components/atoms/Mascot.svelte';
import type { ActiveJobState } from '$lib/types';

/** A short-lived reaction to a job finishing, failing, being stopped, or getting lost. */
export type MascotReaction = 'cheerful' | 'sad' | 'surprised';

export const REACTION_DURATION_MS: Record<MascotReaction, number> = {
  cheerful: 4000,
  sad: 5000,
  surprised: 2500,
};
export const GREETING_DURATION_MS = 2500;
export const TYPING_DURATION_MS = 1500;
export const DROWSY_AFTER_MS = 60_000;

export interface MascotSignals {
  job: ActiveJobState | null;
  reaction?: MascotReaction | null;
  loadError?: boolean;
  loading?: boolean;
  greeting?: boolean;
  typing?: boolean;
  drowsy?: boolean;
}

/**
 * Report whether a running job is painting. Once the first image has started,
 * the gaps between batch images still count, so the mascot does not flicker
 * back to thinking at every prompt boundary.
 */
function isPainting(job: ActiveJobState): boolean {
  if (job.stageName === 'enhancing_prompts') return false;
  if (job.totalSteps > 0 && job.currentStep > 0) return true;
  return job.outputs.length > 0 || (job.promptNumber ?? 1) > 1;
}

/** Pick the mood for an active job, or null when no job is in flight. */
function jobMood(job: ActiveJobState | null): MascotMood | null {
  switch (job?.status) {
    case 'queued':
    case 'pending':
      return 'thinking';
    case 'running':
      return isPainting(job) ? 'creating' : 'thinking';
    case 'paused':
      return 'paused';
    default:
      return null;
  }
}

/**
 * Pick the mascot mood from workspace signals. Reactions win until the next
 * job starts painting, then the active job, then workspace state, then the
 * user's own activity.
 */
export function mascotMood({ job, reaction, loadError, loading, greeting, typing, drowsy }: MascotSignals): MascotMood {
  const active = jobMood(job);
  if (reaction && active !== 'creating') return reaction;
  if (active) return active;
  if (loadError) return 'sad';
  if (loading) return 'thinking';
  if (greeting) return 'waving';
  if (typing) return 'curious';
  if (drowsy) return 'sleeping';
  return 'idle';
}
