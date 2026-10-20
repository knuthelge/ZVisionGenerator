import { api } from './client';
import type { JobLogFilter, JobLogPage } from '$lib/types';

/** Fetch one newest-first page of the finished-job log, narrowed to *status*. */
export function getJobLog(page: number = 1, status: JobLogFilter = 'all'): Promise<JobLogPage> {
  const params = new URLSearchParams({ page: String(page), status });
  return api.get<JobLogPage>(`/api/jobs/log?${params}`);
}

/** Remove one entry from the job log. */
export function deleteJobLogEntry(jobId: string): Promise<{ deleted: string }> {
  return api.delete(`/api/jobs/log/${encodeURIComponent(jobId)}`);
}

/** Remove every entry from the job log; resolves to how many there were. */
export function clearJobLog(): Promise<{ deleted_count: number }> {
  return api.delete('/api/jobs/log');
}
