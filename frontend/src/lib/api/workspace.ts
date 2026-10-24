import { api } from './client';
import type { WorkspaceContext, JobContext, JobList, JobSnapshot, GalleryPage } from '$lib/types';

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

/** URL of the small render of a video that waits for review; `iteration` makes each video of a batch load fresh. */
export function jobReviewUrl(jobId: string, iteration: number): string {
  return `/jobs/${jobId}/review?v=${iteration}`;
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

export function parseUrlPrefill(): Record<string, string> {
  const params: Record<string, string> = {};
  // Hash takes precedence over query string (per router design)
  const hashSearch = window.location.hash.replace(/^#\/[^?]*\??/, '');
  const querySearch = window.location.search.replace(/^\?/, '');
  const source = hashSearch || querySearch;
  if (!source) return params;
  new URLSearchParams(source).forEach((value, key) => {
    params[key] = value;
  });
  return params;
}

/** Remember that the first-run dialog was skipped, so the Workspace stops offering it. */
export function skipFirstRun(): Promise<{ skipped: boolean }> {
  return api.post<{ skipped: boolean }>('/api/first-run/skip');
}
