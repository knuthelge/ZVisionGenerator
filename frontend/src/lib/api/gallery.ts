import { api } from './client';
import type { GalleryPage } from '$lib/types';

export function getGallery(page: number = 1, filter?: string, sortOrder?: string): Promise<GalleryPage> {
  const params = new URLSearchParams({ page: String(page) });
  if (filter && filter !== 'all') params.set('filter', filter);
  if (sortOrder) params.set('sort_order', sortOrder);
  return api.get<GalleryPage>(`/api/gallery?${params}`);
}

export function deleteAsset(path: string): Promise<void> {
  return api.delete(`/api/gallery/${encodeURIComponent(path)}`);
}
