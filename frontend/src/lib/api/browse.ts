import { api } from './client';
import type { BrowseInfo, BrowseListing, BrowsePlaces, BrowsePurposeName } from '$lib/types';

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

function browseParams(purpose: BrowsePurposeName, path: string): URLSearchParams {
  return new URLSearchParams({ purpose, path });
}

/** Return the URL of the WebP preview of an image or video; `frame: 'last'` asks for a video's last frame instead of the one 10% in, `at` for the frame the second lies in (on the clip's own clock; not with `frame`), and `size: 'large'` for the larger picture the preview pane shows. */
export function browsePreviewUrl(purpose: BrowsePurposeName, path: string, options: { frame?: 'last'; at?: number; size?: 'large' } = {}): string {
  const params = browseParams(purpose, path);
  if (options.frame) params.set('frame', options.frame);
  if (options.at !== undefined) params.set('at', String(options.at));
  if (options.size) params.set('size', options.size);
  return `/api/browse/preview?${params}`;
}

/** Return the URL of an audio or video file itself, for a player. */
export function browseMediaUrl(purpose: BrowsePurposeName, path: string): string {
  return `/api/browse/media?${browseParams(purpose, path)}`;
}

/** Return the URL of the contact sheet of a video: one JPEG of evenly spaced frames. */
export function browseSpriteUrl(purpose: BrowsePurposeName, path: string): string {
  return `/api/browse/sprite?${browseParams(purpose, path)}`;
}

/** Fetch the facts of an image, audio or video file; abort with *signal* when the selection moves on. */
export function browseInfo(purpose: BrowsePurposeName, path: string, signal?: AbortSignal): Promise<BrowseInfo> {
  return api.get<BrowseInfo>(`/api/browse/info?${browseParams(purpose, path)}`, signal);
}
