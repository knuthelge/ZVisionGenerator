import { formatSeconds } from '$lib/components/molecules/assetDetails';
import type { JobLogEntry } from '$lib/types';

/** How a logged job reads at a glance: `partial` is a job that finished or stopped with some runs failed. */
export type JobLogTone = 'completed' | 'failed' | 'cancelled' | 'partial';

/** Return the tone of an entry; a failed job is `failed`, and any other job with failed runs is `partial`. */
export function jobLogTone(entry: Pick<JobLogEntry, 'status' | 'failed_runs'>): JobLogTone {
  if (entry.status === 'failed') return 'failed';
  return entry.failed_runs > 0 ? 'partial' : entry.status;
}

/** Name a tone in words, for screen readers and tooltips. */
export function jobLogToneLabel(tone: JobLogTone): string {
  return { completed: 'Completed', failed: 'Failed', cancelled: 'Cancelled', partial: 'Some runs failed' }[tone];
}

/** Return whether an entry has anything to show when expanded: an error text or failed runs. */
export function hasErrorDetail(entry: Pick<JobLogEntry, 'error' | 'failed_runs'>): boolean {
  return Boolean(entry.error?.trim()) || entry.failed_runs > 0;
}

/** Return the entry's duration as `12.3 s` or `2m 05s`, or an empty string when none was recorded. */
export function jobLogDuration(entry: Pick<JobLogEntry, 'duration_secs'>): string {
  return entry.duration_secs > 0 ? formatSeconds(entry.duration_secs) : '';
}

/** Return the epoch seconds an entry is dated by: when it finished, else when it was queued. */
export function jobLogTime(entry: Pick<JobLogEntry, 'completed_at' | 'created_at'>): number | null {
  return entry.completed_at ?? entry.created_at;
}
