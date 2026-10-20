import { formatSeconds } from '$lib/components/molecules/assetDetails';
import type { JobLogEntry } from '$lib/types';

/** How a logged job reads at a glance: `partial` is a job that finished or stopped with some runs failed. */
export type JobLogTone = 'completed' | 'failed' | 'cancelled' | 'partial';

const MINUTE_S = 60;
const HOUR_S = 3600;
const DAY_S = 86400;
const RELATIVE_LIMIT_DAYS = 7;

/** Return the tone of an entry; a failed job is `failed`, and any other job with failed runs is `partial`. */
export function jobLogTone(entry: Pick<JobLogEntry, 'status' | 'failed_runs'>): JobLogTone {
  if (entry.status === 'failed') return 'failed';
  return entry.failed_runs > 0 ? 'partial' : entry.status;
}

/** Name a tone in words, for screen readers and tooltips. */
export function jobLogToneLabel(tone: JobLogTone): string {
  return { completed: 'Completed', failed: 'Failed', cancelled: 'Cancelled', partial: 'Some runs failed' }[tone];
}

/** Return the first non-blank line of a prompt, trimmed. */
export function promptFirstLine(prompt: string): string {
  return prompt.split(/\r?\n/).map((line) => line.trim()).find(Boolean) ?? '';
}

/** Return whether an entry has anything to show when expanded: an error text or failed runs. */
export function hasErrorDetail(entry: Pick<JobLogEntry, 'error' | 'failed_runs'>): boolean {
  return Boolean(entry.error?.trim()) || entry.failed_runs > 0;
}

/** Describe how long ago *epochSecs* was at *nowMs*: "just now", "5 min ago", "3 h ago", "2 d ago", or the date after a week. */
export function timeAgo(epochSecs: number, nowMs: number): string {
  const elapsed = nowMs / 1000 - epochSecs;
  if (elapsed < MINUTE_S) return 'just now';
  if (elapsed < HOUR_S) return `${Math.floor(elapsed / MINUTE_S)} min ago`;
  if (elapsed < DAY_S) return `${Math.floor(elapsed / HOUR_S)} h ago`;
  const days = Math.floor(elapsed / DAY_S);
  if (days <= RELATIVE_LIMIT_DAYS) return `${days} d ago`;
  return new Date(epochSecs * 1000).toLocaleDateString();
}

/** Return the entry's duration as `12.3 s` or `2m 05s`, or an empty string when none was recorded. */
export function jobLogDuration(entry: Pick<JobLogEntry, 'duration_secs'>): string {
  return entry.duration_secs > 0 ? formatSeconds(entry.duration_secs) : '';
}

/** Return the epoch seconds an entry is dated by: when it finished, else when it was queued. */
export function jobLogTime(entry: Pick<JobLogEntry, 'completed_at' | 'created_at'>): number | null {
  return entry.completed_at ?? entry.created_at;
}
