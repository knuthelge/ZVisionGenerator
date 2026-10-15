import { api } from './client';
import type { BrowseListing, BrowsePlaces, BrowsePurposeName } from '$lib/types';

/** List one host folder for *purpose*; an empty path opens at the purpose's own place. */
export function listDirectory(purpose: BrowsePurposeName, path: string, options: { hidden?: boolean; all?: boolean } = {}): Promise<BrowseListing> {
  const params = new URLSearchParams({ purpose, path });
  if (options.hidden) params.set('hidden', 'true');
  if (options.all) params.set('all', 'true');
  return api.get<BrowseListing>(`/api/browse?${params}`);
}

export function browsePlaces(purpose: BrowsePurposeName): Promise<BrowsePlaces> {
  return api.get<BrowsePlaces>(`/api/browse/places?${new URLSearchParams({ purpose })}`);
}

export function createFolder(purpose: BrowsePurposeName, path: string, name: string): Promise<{ path: string }> {
  return api.post<{ path: string }>('/api/browse/folder', { purpose, path, name });
}
