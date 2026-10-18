import type { ActiveJobState } from '$lib/types';
import { isReviewing, iterations, jobPhase } from './jobView';

/** One short line of job state for a strip or a bar: Ready, Step 12 of 32, Review, 2 queued. */
export function jobStatusLine(job: ActiveJobState | null, queuedCount: number): { text: string; tone: 'idle' | 'running' | 'review' } {
  if (job && isReviewing(job)) return { text: 'Review', tone: 'review' };
  if (job && (job.status === 'running' || job.status === 'paused' || job.status === 'queued')) {
    const phase = jobPhase(job);
    if (phase === 'queue') return { text: 'Queued', tone: 'running' };
    if (phase === 'enhance') return { text: job.status === 'paused' ? 'Paused' : 'Enhancing…', tone: 'running' };
    if (phase === 'load') return { text: 'Loading model…', tone: 'running' };
    if (job.status === 'paused') return { text: 'Paused', tone: 'running' };
    const items = iterations(job);
    const position = items.total > 1 ? `${items.index + 1}/${items.total} · ` : '';
    if (job.totalSteps > 0) return { text: `${position}Step ${job.currentStep} of ${job.totalSteps}`, tone: 'running' };
    return { text: `${position}Generating…`, tone: 'running' };
  }
  if (queuedCount > 0) return { text: `${queuedCount} queued`, tone: 'idle' };
  return { text: 'Ready', tone: 'idle' };
}
