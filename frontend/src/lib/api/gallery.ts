import { api } from './client';
import type { DeleteResult, GalleryPage, JobContext, JobDownloadPlan, MoveResult, UpscaleFactor, UpscaleMethod } from '$lib/types';

/** One page of the Gallery; `project` narrows it to that project's files (null or absent lists everything) and `starred` to the starred ones. */
export function getGallery(
  page: number = 1,
  filter?: string,
  sortOrder?: string,
  project?: string | null,
  starred: boolean = false
): Promise<GalleryPage> {
  const params = new URLSearchParams({ page: String(page) });
  if (filter && filter !== 'all') params.set('filter', filter);
  if (sortOrder) params.set('sort_order', sortOrder);
  if (project) params.set('project', project);
  if (starred) params.set('starred', 'true');
  return api.get<GalleryPage>(`/api/gallery?${params}`);
}

/** The IDs of every file a view matches (the same filter, project and starred narrowing as `getGallery`), not just one page. */
export function getGalleryIds(filter?: string, project?: string | null, starred: boolean = false): Promise<string[]> {
  const params = new URLSearchParams();
  if (filter && filter !== 'all') params.set('filter', filter);
  if (project) params.set('project', project);
  if (starred) params.set('starred', 'true');
  return api.get<{ ids: string[] }>(`/api/gallery/ids?${params}`).then((reply) => reply.ids);
}

/** Delete many files in one request; reports what was removed (with the facts the lists need) and what failed. */
export function deleteAssets(assetIds: string[]): Promise<DeleteResult> {
  return api.post<DeleteResult>('/api/gallery/delete', { asset_ids: assetIds });
}

/** Star or unstar one file; resolves to its ID and its new state. */
export function setStarred(assetId: string, starred: boolean): Promise<{ id: string; starred: boolean }> {
  return api.post<{ id: string; starred: boolean }>('/api/gallery/star', { asset_id: assetId, starred });
}

/**
 * Download files as one zip by submitting a hidden form, so the browser saves the attachment itself.
 *
 * The form is built, submitted and removed synchronously, so call this directly from the click or tap handler: Safari on
 * iPhone and iPad drops a download that starts after an await. The IDs travel as one field joined by newlines.
 */
export function downloadAssets(assetIds: readonly string[]): void {
  const form = document.createElement('form');
  form.method = 'post';
  form.action = '/api/gallery/download';
  form.hidden = true;
  const field = document.createElement('input');
  field.type = 'hidden';
  field.name = 'asset_ids';
  field.value = assetIds.join('\n');
  form.append(field);
  document.body.append(form);
  try {
    form.submit();
  } finally {
    form.remove();
  }
}

/** Move files into a project, or to the top of the output folder with null; reports what moved (with its new ID) and what failed. */
export function moveAssets(assetIds: string[], project: string | null): Promise<MoveResult> {
  return api.post<MoveResult>('/api/gallery/move', { asset_ids: assetIds, project });
}

/** Queue a job that upscales an existing image; resolves to the job, like a generate. */
export function submitUpscale(assetId: string, factor: UpscaleFactor, method: UpscaleMethod): Promise<JobContext> {
  return api.post<JobContext>('/api/upscale', { asset_id: assetId, factor, method });
}

/** What `submitUpscale` of the same file would download: the missing models, their sizes and the free space. */
export function planUpscale(assetId: string, factor: UpscaleFactor, method: UpscaleMethod): Promise<JobDownloadPlan> {
  return api.post<JobDownloadPlan>('/api/upscale/plan', { asset_id: assetId, factor, method });
}

/** The waveform peaks (100 values, 0 to 255) of an audio file; the server decodes the file on the first request only. */
export function getAudioPeaks(assetId: string): Promise<number[]> {
  return api.get<{ id: string; peaks: number[] }>(`/api/gallery-peaks?id=${encodeURIComponent(assetId)}`).then((reply) => reply.peaks);
}
