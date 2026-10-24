import type { GalleryAsset } from '$lib/types';

/**
 * Return a refreshed page one followed by the loaded assets that came after it, so a refresh does not snap a
 * scrolled list back to its first page.
 *
 * The tail starts after page one's last asset in the loaded list; when that asset is not loaded (page one is all
 * new), only page one is returned.
 */
export function keepLoadedTail(pageOne: GalleryAsset[], loaded: GalleryAsset[]): GalleryAsset[] {
  const pageIds = new Set(pageOne.map((asset) => asset.id));
  const last = pageOne.at(-1);
  const lastIndex = last ? loaded.findIndex((asset) => asset.id === last.id) : -1;
  if (lastIndex < 0) return pageOne;
  return [...pageOne, ...loaded.slice(lastIndex + 1).filter((asset) => !pageIds.has(asset.id))];
}
