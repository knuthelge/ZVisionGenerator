import type { AppConfig, ModelInventory, WorkspaceContext } from '$lib/types';
import type { MediaFilter, SortOrder } from '$lib/state/gallery.svelte';

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

/** How the Gallery page was left: its query and how far it was scrolled. The list itself lives in the gallery store. */
export interface GalleryVisit {
  filter: MediaFilter;
  sort: SortOrder;
  scrollTop: number;
}

export const workspaceMemory = remembered<WorkspaceContext>();
export const galleryMemory = remembered<GalleryVisit>();
export const inventoryMemory = remembered<ModelInventory>();
export const configMemory = remembered<AppConfig>();
