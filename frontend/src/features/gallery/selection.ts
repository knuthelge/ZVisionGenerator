import type { MediaFilter } from '$lib/state/galleryFilter';
import type { GalleryAsset } from '$lib/types';

/** What a click on a tile's media does in Select mode: toggle the tile, or add the range from the anchor. */
export type TileClickIntent = 'toggle' | 'range';

/** Decide what a Select-mode click on a tile's media does: it toggles the tile, and with Shift adds the range from the anchor. */
export function tileClickIntent(shiftKey: boolean): TileClickIntent {
  return shiftKey ? 'range' : 'toggle';
}

/**
 * The IDs from the anchor to the target, inclusive, in grid order whichever comes first. Without a loaded anchor, or
 * for a target that is not loaded, it is the target alone.
 *
 * @param orderedIds The loaded tiles' IDs in grid order.
 * @param anchorId The last tile toggled, or null.
 * @param target The tile Shift-clicked.
 */
export function rangeIds(orderedIds: readonly string[], anchorId: string | null, target: string): string[] {
  const to = orderedIds.indexOf(target);
  const from = anchorId === null ? -1 : orderedIds.indexOf(anchorId);
  if (to === -1 || from === -1) return [target];
  return orderedIds.slice(Math.min(from, to), Math.max(from, to) + 1);
}

/** A copy of the selection with one ID added, or removed when it was in. */
export function withToggled(selected: ReadonlySet<string>, id: string): Set<string> {
  const next = new Set(selected);
  if (!next.delete(id)) next.add(id);
  return next;
}

/** A copy of the selection with the IDs added. It never removes one. */
export function withAdded(selected: ReadonlySet<string>, ids: Iterable<string>): Set<string> {
  const next = new Set(selected);
  for (const id of ids) next.add(id);
  return next;
}

/**
 * Where the selected files sit when they all share it: a project name, or null for no project. Undefined for a mix, for
 * an empty selection, and when a selected file's place is unknown, which checks no row of the Move menu.
 *
 * A selected file that is not loaded sits where the view says: in the picked project, or in no project with the Not in a
 * project filter. With All files and any other filter its place is unknown.
 *
 * @param selectedIds The selected IDs, loaded or not.
 * @param loaded The loaded tiles.
 * @param pick The project the view is scoped to, or null for All files.
 * @param filter The view's media filter.
 */
export function selectionProject(
  selectedIds: ReadonlySet<string>,
  loaded: readonly Pick<GalleryAsset, 'id' | 'project'>[],
  pick: string | null,
  filter: MediaFilter
): string | null | undefined {
  const byId = new Map(loaded.map((asset) => [asset.id, asset]));
  const viewProject: string | null | undefined = pick !== null ? pick : filter === 'no_project' ? null : undefined;
  const places = new Set<string | null | undefined>();
  for (const id of selectedIds) places.add(byId.has(id) ? byId.get(id)!.project : viewProject);
  if (places.size !== 1) return undefined;
  return [...places][0];
}

const MEDIA_LABELS: Record<MediaFilter, string> = {
  all: 'All media',
  image: 'Images',
  video: 'Videos',
  audio: 'Audio',
  no_project: 'Not in a project',
};

/**
 * The Select mode's description line: the view being selected from, then how to select. It carries no count, which the
 * header and the phone bar already show.
 *
 * @param coarse Whether the pointer is coarse (touch), where the instruction says Tap.
 */
export function modeLine(view: { project: string | null; starred: boolean; filter: MediaFilter; coarse: boolean }): string {
  const parts = [
    ...(view.project === null ? [] : [view.project]),
    ...(view.starred ? ['Starred'] : []),
    MEDIA_LABELS[view.filter],
    `${view.coarse ? 'Tap' : 'Click'} files to add or remove`,
  ];
  return parts.join(' · ');
}
