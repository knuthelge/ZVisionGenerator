import type { MascotMood } from '$lib/components/atoms/Mascot.svelte';
import type { ActiveJobState } from '$lib/types';

/** A short-lived reaction to a job finishing, failing, or being stopped. */
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
 * Pick the mascot mood from workspace signals. Reactions win, then the active
 * job, then workspace state, then the user's own activity.
 */
export function mascotMood({ job, reaction, loadError, loading, greeting, typing, drowsy }: MascotSignals): MascotMood {
  if (reaction) return reaction;
  if (job?.status === 'queued' || job?.status === 'pending') return 'thinking';
  if (job?.status === 'running') {
    return job.totalSteps > 0 && job.currentStep > 0 ? 'creating' : 'thinking';
  }
  if (job?.status === 'paused') return 'sleeping';
  if (loadError) return 'sad';
  if (loading) return 'thinking';
  if (greeting) return 'waving';
  if (typing) return 'curious';
  if (drowsy) return 'sleeping';
  return 'idle';
}
