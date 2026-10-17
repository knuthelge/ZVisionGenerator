import type { GalleryAsset, GalleryPage } from '$lib/types';
import { getHistory } from '$lib/api/workspace';

let _history = $state<GalleryAsset[]>([]);
let _loading = $state(false);
let _loadingMore = $state(false);
// Unknown (null) until page one has been fetched; the seed carries no paging info.
let _totalCount = $state<number | null>(null);
let _pageSize = $state<number | null>(null);
// Bumped by every refresh so an older load-more response is dropped.
let _generation = 0;

function appendNew(assets: GalleryAsset[]): number {
  const loadedIds = new Set(_history.map((asset) => asset.id));
  const fresh = assets.filter((asset) => !loadedIds.has(asset.id));
  if (fresh.length > 0) _history = [..._history, ...fresh];
  return fresh.length;
}

// Keep tiles loaded past page one so a refresh does not snap a scrolled strip back to the start.
function mergePageOne(page: GalleryPage): GalleryAsset[] {
  const pageIds = new Set(page.assets.map((asset) => asset.id));
  const last = page.assets.at(-1);
  const lastIndex = last ? _history.findIndex((asset) => asset.id === last.id) : -1;
  if (lastIndex < 0) return page.assets;
  return [...page.assets, ..._history.slice(lastIndex + 1).filter((asset) => !pageIds.has(asset.id))];
}

export const historyStore = {
  get assets(): GalleryAsset[] { return _history; },
  get loading(): boolean { return _loading; },
  get loadingMore(): boolean { return _loadingMore; },
  get totalCount(): number | null { return _totalCount; },
  get hasMore(): boolean { return _totalCount !== null && _pageSize !== null && _history.length < _totalCount; },

  seedHistory(assets: GalleryAsset[]): void {
    _history = assets;
  },

  mergeOutputs(assets: GalleryAsset[]): void {
    if (assets.length === 0) return;
    const newestFirst = [...assets].reverse();
    const incomingIds = new Set(newestFirst.map((asset) => asset.id));
    const added = newestFirst.filter((asset) => !_history.some((item) => item.id === asset.id)).length;
    _history = [...newestFirst, ..._history.filter((asset) => !incomingIds.has(asset.id))];
    if (_totalCount !== null) _totalCount += added;
  },

  removeAssets(ids: Iterable<string>): void {
    const removed = new Set(ids);
    if (removed.size === 0) return;
    const before = _history.length;
    _history = _history.filter((asset) => !removed.has(asset.id));
    if (_totalCount !== null) _totalCount = Math.max(0, _totalCount - (before - _history.length));
  },

  async refreshHistory(): Promise<void> {
    _loading = true;
    const generation = ++_generation;
    try {
      const page = await getHistory(1);
      if (generation !== _generation) return;
      _history = mergePageOne(page);
      _totalCount = page.total_count;
      // Page one is full whenever more pages follow, so its length is the server's page size.
      if (page.total_pages > 1) _pageSize = page.assets.length;
    } catch {
      // ignore refresh errors
    } finally {
      if (generation === _generation) _loading = false;
    }
  },

  /** Append the next page of older assets, skipping pages that hold only assets already shown. */
  async loadMore(): Promise<void> {
    if (_loading || _loadingMore || !historyStore.hasMore || _pageSize === null) return;
    _loadingMore = true;
    const generation = _generation;
    // Pages are fixed-size slices, so the loaded count says where the next unseen asset sits.
    let pageNumber = Math.floor(_history.length / _pageSize) + 1;
    try {
      for (;;) {
        const page = await getHistory(pageNumber);
        if (generation !== _generation) return;
        _totalCount = page.total_count;
        if (appendNew(page.assets) > 0 || pageNumber >= page.total_pages) return;
        pageNumber += 1;
      }
    } catch {
      // ignore load-more errors; the next scroll retries
    } finally {
      _loadingMore = false;
    }
  }
};
