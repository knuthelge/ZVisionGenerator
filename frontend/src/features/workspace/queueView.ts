import { workflowLabel } from '$lib/state/assetActions';
import { isVoiceDesignJob } from '$lib/state/jobView';
import type { JobSnapshot } from '$lib/types';

/** The height of one expanded queue row, in px. The queue folds only when more than this much of it would be hidden. */
export const QUEUE_ROW_PX = 54;

/**
 * Decide whether the queue folds to its one-line head.
 *
 * *overflow* is how far the column's content (the card, then the expanded queue) runs past the column. A smaller
 * overflow than a whole row scrolls instead, with the last row's edge peeking out to say so.
 */
export function shouldFoldQueue(overflowPx: number): boolean {
  return overflowPx > QUEUE_ROW_PX;
}

/** A queued job's one meta line, the parts joined by "·" and the last cut first: workflow, runs, model, then size and steps. */
export function queueMetaParts(job: JobSnapshot): string[] {
  const parts = [workflowLabel(job.workflow)];
  // A design job is not a run of a workflow: its meta says how many voices.
  if (!isVoiceDesignJob(job)) parts.push(`${job.runs} ${job.runs === 1 ? 'run' : 'runs'}`);
  parts.push(job.model_label || job.model);
  if (job.meta) parts.push(...String(job.meta).split(' · '));
  return parts.filter(Boolean);
}
