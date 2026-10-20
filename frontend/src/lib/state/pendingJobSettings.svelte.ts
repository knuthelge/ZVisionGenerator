import type { JobSettings } from '$lib/types';

// Reactive, so a Workspace that is already mounted notices a request made after it loaded.
let pending = $state<JobSettings | null>(null);

/** Ask the Workspace to refill its form from a finished job's settings once it has loaded. */
export function requestJobSettings(settings: JobSettings): void {
  pending = settings;
}

/** Whether a request is waiting; reading it inside an effect re-runs the effect when one arrives. */
export function hasPendingJobSettings(): boolean {
  return pending !== null;
}

/** Take the pending settings, if any; they are handed out once. */
export function takeJobSettings(): JobSettings | null {
  const settings = pending;
  pending = null;
  return settings;
}
