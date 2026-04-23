import { api } from './client';
import type { GalleryPage } from '$lib/types';

export function getGallery(page: number = 1, filter?: string): Promise<GalleryPage> {
  const params = new URLSearchParams({ page: String(page) });
  if (filter) params.set('filter', filter);
  return api.get<GalleryPage>(`/api/gallery?${params}`);
}

export function deleteAsset(path: string): Promise<void> {
  return api.delete(`/api/gallery/${encodeURIComponent(path)}`);
}
