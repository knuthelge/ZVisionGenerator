import type { GalleryAsset } from '$lib/types';
import { getGallery } from '$lib/api/gallery';
import { keepLoadedTail } from '$lib/state/pageMerge';

export type MediaFilter = 'all' | 'image' | 'video';
export type SortOrder = 'newest' | 'oldest';

/** What a Gallery view shows: the media it keeps and the order it runs in. */
export interface GalleryQuery {
  filter: MediaFilter;
  sort: SortOrder;
}

/** The whole Gallery, newest first: the Workspace strip, the stage's history and the Gallery page's first view. */
export const DEFAULT_QUERY: GalleryQuery = { filter: 'all', sort: 'newest' };

/** The loaded pages of one query, as the Gallery page shows them. */
export interface GalleryView {
  assets: GalleryAsset[];
  /** The last page loaded. */
  page: number;
  totalPages: number;
  /** Unknown (null) until page one has been fetched; a seed carries no paging info. */
  totalCount: number | null;
}

export function queryKey(query: GalleryQuery): string {
  return `${query.filter}|${query.sort}`;
}

const DEFAULT_KEY = queryKey(DEFAULT_QUERY);
const EMPTY_VIEW: GalleryView = Object.freeze({ assets: [], page: 1, totalPages: 1, totalCount: null });

// One list per query. The default one is the strip's and the stage's, and the store loads it; the Gallery page
// loads the others and commits them here, so every page shows the same list and a visit comes back where it was.
let _views = $state<Record<string, GalleryView>>({});
let _loading = $state(false);
let _loadingMore = $state(false);
let _pageSize = $state<number | null>(null);
// Bumped by every refresh so an older load-more response is dropped.
let _generation = 0;

function defaultView(): GalleryView {
  return _views[DEFAULT_KEY] ?? EMPTY_VIEW;
}

function setDefault(patch: Partial<GalleryView>): void {
  _views[DEFAULT_KEY] = { ...defaultView(), ...patch };
}

function appendNew(assets: GalleryAsset[]): number {
  const current = defaultView();
  const loadedIds = new Set(current.assets.map((asset) => asset.id));
  const fresh = assets.filter((asset) => !loadedIds.has(asset.id));
  if (fresh.length > 0) setDefault({ assets: [...current.assets, ...fresh] });
  return fresh.length;
}

export const galleryStore = {
  /** The list a query shows; empty until it has been loaded. */
  view(query: GalleryQuery = DEFAULT_QUERY): GalleryView {
    return _views[queryKey(query)] ?? EMPTY_VIEW;
  },

  /** Commit what a query shows: page one and whatever the page kept after it. */
  setView(query: GalleryQuery, view: GalleryView): void {
    _views[queryKey(query)] = view;
  },

  /** Drop every list (tests). */
  reset(): void {
    _views = {};
    _loading = false;
    _loadingMore = false;
    _pageSize = null;
    _generation += 1;
  },

  // The default view: the Gallery newest first, as the strip and the stage show it.
  get assets(): GalleryAsset[] { return defaultView().assets; },
  get loading(): boolean { return _loading; },
  get loadingMore(): boolean { return _loadingMore; },
  get totalCount(): number | null { return defaultView().totalCount; },
  get hasMore(): boolean {
    const { assets, totalCount } = defaultView();
    return totalCount !== null && _pageSize !== null && assets.length < totalCount;
  },

  /** Start from the assets the Workspace context carried, before page one is known. */
  seed(assets: GalleryAsset[]): void {
    setDefault({ assets });
  },

  /**
   * Put a finished job's outputs at the front of every newest-first list whose filter admits them. An oldest-first
   * list would need its last page for them, so it is dropped and its next visit reloads it.
   */
  mergeOutputs(assets: GalleryAsset[]): void {
    if (assets.length === 0) return;
    const newestFirst = [...assets].reverse();
    const next: Record<string, GalleryView> = {};
    for (const key of new Set([DEFAULT_KEY, ...Object.keys(_views)])) {
      const [filter, sort] = key.split('|') as [MediaFilter, SortOrder];
      if (sort !== 'newest') continue;
      const view = _views[key] ?? EMPTY_VIEW;
      const incoming = newestFirst.filter((asset) => filter === 'all' || asset.media_type === filter);
      const incomingIds = new Set(incoming.map((asset) => asset.id));
      const added = incoming.filter((asset) => !view.assets.some((item) => item.id === asset.id)).length;
      next[key] = {
        ...view,
        assets: [...incoming, ...view.assets.filter((asset) => !incomingIds.has(asset.id))],
        totalCount: view.totalCount === null ? null : view.totalCount + added,
      };
    }
    _views = next;
  },

  /** Take deleted assets out of every list, and out of each list's count. */
  removeAssets(removed: readonly GalleryAsset[]): void {
    if (removed.length === 0) return;
    const removedIds = new Set(removed.map((asset) => asset.id));
    for (const key of Object.keys(_views)) {
      const view = _views[key];
      const filter = key.split('|')[0] as MediaFilter;
      const assets = view.assets.filter((asset) => !removedIds.has(asset.id));
      // A list is shorter by every removed asset its filter admits, whether or not it had loaded it yet.
      const gone = removed.filter((asset) => filter === 'all' || asset.media_type === filter).length;
      if (gone === 0 && assets.length === view.assets.length) continue;
      _views[key] = { ...view, assets, totalCount: view.totalCount === null ? null : Math.max(0, view.totalCount - gone) };
    }
  },

  /** Reload page one of the default view, keeping the pages loaded after it. */
  async refresh(): Promise<void> {
    _loading = true;
    const generation = ++_generation;
    try {
      const page = await getGallery(1, DEFAULT_QUERY.filter, DEFAULT_QUERY.sort);
      if (generation !== _generation) return;
      // Keep tiles loaded past page one so a refresh does not snap a scrolled strip back to the start.
      const current = defaultView();
      const assets = keepLoadedTail(page.assets, current.assets);
      // With the tail kept, the last loaded page stays what it was, so the Gallery page continues from there.
      setDefault({ assets, page: assets.length > page.assets.length ? current.page : page.page, totalPages: page.total_pages, totalCount: page.total_count });
      // Page one is full whenever more pages follow, so its length is the server's page size.
      if (page.total_pages > 1) _pageSize = page.assets.length;
    } catch {
      // ignore refresh errors
    } finally {
      if (generation === _generation) _loading = false;
    }
  },

  /** Append the next page of older assets to the default view, skipping pages that hold only assets already shown. */
  async loadMore(): Promise<void> {
    if (_loading || _loadingMore || !galleryStore.hasMore || _pageSize === null) return;
    _loadingMore = true;
    const generation = _generation;
    // Pages are fixed-size slices, so the loaded count says where the next unseen asset sits.
    let pageNumber = Math.floor(defaultView().assets.length / _pageSize) + 1;
    try {
      for (;;) {
        const page = await getGallery(pageNumber, DEFAULT_QUERY.filter, DEFAULT_QUERY.sort);
        if (generation !== _generation) return;
        setDefault({ totalCount: page.total_count, page: page.page, totalPages: page.total_pages });
        if (appendNew(page.assets) > 0 || pageNumber >= page.total_pages) return;
        pageNumber += 1;
      }
    } catch {
      // ignore load-more errors; the next scroll retries
    } finally {
      _loadingMore = false;
    }
  },
};
