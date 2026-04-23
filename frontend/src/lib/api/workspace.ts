import { api } from './client';
import type { WorkspaceContext, JobContext, GalleryPage } from '$lib/types';

export function getWorkspaceContext(params?: Record<string, string>): Promise<WorkspaceContext> {
  const query = params ? '?' + new URLSearchParams(params).toString() : '';
  return api.get<WorkspaceContext>(`/api/workspace${query}`);
}

export function submitGenerate(formData: FormData): Promise<JobContext> {
  return fetch('/api/generate', {
    method: 'POST',
    headers: { Accept: 'application/json' },
    body: formData
  }).then(async r => {
    if (!r.ok) throw new Error(`Generate failed: ${await r.text()}`);
    return r.json() as Promise<JobContext>;
  });
}

export function cancelJob(jobId: string): Promise<void> {
  return api.post(`/api/jobs/${jobId}/cancel`);
}

export function getHistory(page: number = 1): Promise<GalleryPage> {
  return api.get<GalleryPage>(`/api/history?page=${page}`);
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
