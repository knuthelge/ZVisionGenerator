import { api } from './client';
import { parseQuery } from '$lib/prefillParams';
import type { WorkspaceContext, JobContext, JobList, JobSnapshot, GalleryPage, PrefillParams } from '$lib/types';

export function getWorkspaceContext(params?: Record<string, string>): Promise<WorkspaceContext> {
  const query = params ? '?' + new URLSearchParams(params).toString() : '';
  return api.get<WorkspaceContext>(`/api/workspace${query}`);
}

export function getWorkspaceCoreContext(): Promise<WorkspaceContext> {
  return getWorkspaceContext({ include_history: 'false' });
}

export function submitGenerate(formData: FormData): Promise<JobContext> {
  return api.post<JobContext>('/api/generate', formData);
}

export function getJobSnapshot(jobId: string): Promise<JobSnapshot> {
  return api.get<JobSnapshot>(`/jobs/${jobId}`);
}

/** URL of a job's latest live preview; the version busts the browser cache when a new preview lands. */
export function jobPreviewUrl(jobId: string, version: number): string {
  return `/jobs/${jobId}/preview?v=${version}`;
}

/** URL of the small render of a video that waits for review; `iteration` and `attempt` make each review load fresh. */
export function jobReviewUrl(jobId: string, iteration: number, attempt: number): string {
  return `/jobs/${jobId}/review?v=${iteration}-${attempt}`;
}

export function cancelJob(jobId: string): Promise<void> {
  return api.post(`/api/jobs/${jobId}/cancel`);
}

export function listJobs(): Promise<JobList> {
  return api.get<JobList>('/api/jobs');
}

/** Remove every queued job; the active job keeps running. */
export function clearQueue(): Promise<{ cancelled: string[] }> {
  return api.delete<{ cancelled: string[] }>('/api/jobs/queue');
}

export function parseUrlPrefill(): PrefillParams {
  // The hash is the one form the app emits (#/workspace?...), and the one the Workspace clears after applying it.
  return parseQuery(window.location.hash.replace(/^#\/[^?]*\??/, ''));
}

/** Remember that the first-run dialog was skipped, so the Workspace stops offering it. */
export function skipFirstRun(): Promise<{ skipped: boolean }> {
  return api.post<{ skipped: boolean }>('/api/first-run/skip');
}

/** Ask which of the host-local reference paths are no longer files; the answer lists those paths. */
export async function checkReferencePaths(paths: string[]): Promise<string[]> {
  const { missing } = await api.post<{ missing: string[] }>('/api/references/check', { paths });
  return missing;
}
