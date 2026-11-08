import { api } from './client';
import type { GalleryPage, JobContext, JobDownloadPlan, MoveResult, UpscaleFactor, UpscaleMethod } from '$lib/types';

/** One page of the Gallery; `project` narrows it to that project's files, and null or absent lists everything. */
export function getGallery(page: number = 1, filter?: string, sortOrder?: string, project?: string | null): Promise<GalleryPage> {
  const params = new URLSearchParams({ page: String(page) });
  if (filter && filter !== 'all') params.set('filter', filter);
  if (sortOrder) params.set('sort_order', sortOrder);
  if (project) params.set('project', project);
  return api.get<GalleryPage>(`/api/gallery?${params}`);
}

export function deleteAsset(assetId: string): Promise<void> {
  return api.delete(`/api/gallery/${encodeURIComponent(assetId)}`);
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
