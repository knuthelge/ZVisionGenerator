import type { GalleryAsset } from '$lib/types';

export type MediaFilter = 'all' | 'image' | 'video';
export type SortOrder = 'newest' | 'oldest';

/** What the Gallery showed when it was left: the loaded pages, their query, and how far it was scrolled. */
export interface GalleryView {
  assets: GalleryAsset[];
  page: number;
  totalPages: number;
  totalCount: number;
  filter: MediaFilter;
  sort: SortOrder;
  scrollTop: number;
}

// The Gallery draws the remembered view the moment it mounts and refreshes page one in the background, so a return
// shows the grid where it was instead of a spinner (pages remember).
let _view: GalleryView | null = null;

/** The view a previous visit left, or null before the first load. */
export function rememberedGallery(): GalleryView | null {
  return _view;
}

/** Keep a view for the next visit. */
export function rememberGallery(view: GalleryView): void {
  _view = view;
}

/** Drop the remembered view (a load that failed, and tests). */
export function forgetGallery(): void {
  _view = null;
}
