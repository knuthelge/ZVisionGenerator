import type { AppConfig, GalleryAsset, ModelInventory, WorkspaceContext } from '$lib/types';

/**
 * A value one visit of a page loads and the next visit draws at once, refreshing it behind the frame.
 *
 * Pages remember: a page component is rebuilt on every visit, so what it loaded lives here, outside it. The page
 * adopts the remembered value on mount, then asks the server again and replaces it.
 */
export interface Remembered<T> {
  /** The remembered value, or null before the first load. */
  get(): T | null;
  set(value: T): void;
  /** Drop the value: a load that failed, and tests. */
  forget(): void;
}

function remembered<T>(): Remembered<T> {
  let value: T | null = null;
  return {
    get: () => value,
    set: (next) => { value = next; },
    forget: () => { value = null; },
  };
}

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

export const workspaceMemory = remembered<WorkspaceContext>();
export const galleryMemory = remembered<GalleryView>();
export const inventoryMemory = remembered<ModelInventory>();
export const configMemory = remembered<AppConfig>();
