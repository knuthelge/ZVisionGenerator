import { filterAdmits, type MediaFilter } from '$lib/state/galleryFilter';
import type { GalleryView } from '$lib/state/gallery.svelte';
import { inProject } from '$lib/state/projectPick.svelte';
import type { GalleryAsset, MovedAsset } from '$lib/types';

/**
 * Apply a move to one loaded list. A moved file that still belongs to the pick replaces its old entry in place (the
 * move keeps its modified time, so its position holds). One that no longer belongs leaves the list, and lowers the
 * count of a list whose media filter admits it whether or not that list had loaded it, as a delete does; Not in a project
 * lowers its count only for the files it had loaded.
 *
 * @param filter The media filter of the list.
 * @param pick The project the list is scoped to, or null for All files.
 */
export function applyMovesToView(view: GalleryView, filter: MediaFilter, moves: readonly MovedAsset[], pick: string | null): GalleryView {
  const byOldId = new Map(moves.map((move) => [move.from, move.asset]));
  // A move never changes a file's media, so a media filter knows whether it counted the file. Not in a project is about
  // the project, which the move changed: it counts what it loaded.
  const loaded = new Set(view.assets.map((asset) => asset.id));
  const counted = ({ from, asset }: MovedAsset): boolean => (filter === 'no_project' ? loaded.has(from) : filterAdmits(filter, asset));
  const leaving = moves.filter((move) => counted(move) && !belongsToView(move.asset, filter, pick));
  const assets = view.assets.flatMap((asset) => {
    const moved = byOldId.get(asset.id);
    if (!moved) return [asset];
    return belongsToView(moved, filter, pick) ? [moved] : [];
  });
  if (leaving.length === 0 && assets.every((asset, index) => asset === view.assets[index])) return view;
  return { ...view, assets, totalCount: view.totalCount === null ? null : Math.max(0, view.totalCount - leaving.length) };
}

/** Whether a list scoped to *pick* with this filter shows an asset. */
export function belongsToView(asset: Pick<GalleryAsset, 'media_type' | 'project'>, filter: MediaFilter, pick: string | null): boolean {
  return inProject(asset, pick) && filterAdmits(filter, asset);
}
