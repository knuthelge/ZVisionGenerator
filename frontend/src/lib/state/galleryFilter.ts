import type { GalleryAsset } from '$lib/types';

/**
 * The Gallery's media filter: all files, one kind of media, or (in All files only) the files that sit in no project. The
 * last one is a filter value, not a project name, because a project folder can be named anything.
 */
export type MediaFilter = 'all' | 'image' | 'video' | 'audio' | 'no_project';

/** Whether a view with this filter shows the asset. */
export function filterAdmits(filter: MediaFilter, asset: Pick<GalleryAsset, 'media_type' | 'project'>): boolean {
  if (filter === 'all') return true;
  if (filter === 'no_project') return asset.project === null;
  return asset.media_type === filter;
}
