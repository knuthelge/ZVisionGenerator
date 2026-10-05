import { api } from './client';
import type { GalleryPage, JobContext, UpscaleFactor } from '$lib/types';

export function getGallery(page: number = 1, filter?: string, sortOrder?: string): Promise<GalleryPage> {
  const params = new URLSearchParams({ page: String(page) });
  if (filter && filter !== 'all') params.set('filter', filter);
  if (sortOrder) params.set('sort_order', sortOrder);
  return api.get<GalleryPage>(`/api/gallery?${params}`);
}

export function deleteAsset(assetId: string): Promise<void> {
  return api.delete(`/api/gallery/${encodeURIComponent(assetId)}`);
}

/** Queue a job that upscales an existing image; resolves to the job, like a generate. */
export function submitUpscale(assetId: string, factor: UpscaleFactor): Promise<JobContext> {
  return api.post<JobContext>('/api/upscale', { asset_id: assetId, factor });
}
