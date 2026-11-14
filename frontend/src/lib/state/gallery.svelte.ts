import type { GalleryAsset, MovedAsset, RemovedAsset } from '$lib/types';
import { getGallery } from '$lib/api/gallery';
import { keepLoadedTail } from '$lib/state/pageMerge';
import { inProject, projectPick } from '$lib/state/projectPick.svelte';
import { filterAdmits, viewAdmits, type MediaFilter } from '$lib/state/galleryFilter';
import { applyMovesToView } from '$lib/state/viewMoves';

export type { MediaFilter };
export type SortOrder = 'newest' | 'oldest';

/** What a Gallery view shows: the media it keeps, the order it runs in and whether it keeps starred files only. */
export interface GalleryQuery {
  filter: MediaFilter;
  sort: SortOrder;
  /** The ★ toggle. */
  starred: boolean;
}

/** The whole Gallery, newest first: the Workspace strip, the stage's history and the Gallery page's first view. */
export const DEFAULT_QUERY: GalleryQuery = { filter: 'all', sort: 'newest', starred: false };

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
  return `${query.filter}|${query.sort}|${query.starred ? 1 : 0}`;
}

/** The query a key stands for: the inverse of `queryKey`. */
export function parseQueryKey(key: string): GalleryQuery {
  const [filter, sort, starred] = key.split('|');
  return { filter: filter as MediaFilter, sort: sort as SortOrder, starred: starred === '1' };
}

const DEFAULT_KEY = queryKey(DEFAULT_QUERY);
const EMPTY_VIEW: GalleryView = Object.freeze({ assets: [], page: 1, totalPages: 1, totalCount: null });

// One list per query, all scoped to the picked project (the picker resets them when the pick changes). The default one
// is the strip's and the stage's, and the store loads it; the Gallery page loads the others and commits them here, so
// every page shows the same list and a visit comes back where it was.
let _views = $state<Record<string, GalleryView>>({});
// Page one of the whole Gallery, newest first, for what ignores the pick (recent LoRAs, the last seed). Only
// kept apart while a project is picked; with All files the default view is that list.
let _recent = $state<GalleryAsset[]>([]);
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

/** Fetch the unscoped recent list; a stale or failed response changes nothing. */
async function loadRecent(generation: number): Promise<void> {
  try {
    const page = await getGallery(1, DEFAULT_QUERY.filter, DEFAULT_QUERY.sort);
    if (generation === _generation) _recent = page.assets;
  } catch {
    // ignore: the recent list stays as it was
  }
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

  /**
   * Drop every list and the paging state, and drop responses still on their way. The picker calls it when the pick
   * changes. The unscoped recent list stays, as it does not follow the pick.
   */
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
  /**
   * The newest assets of the whole Gallery whatever project is picked: where the recent LoRAs and the last seed come
   * from. With All files that is the default view; with a project it is its own unfiltered request, made by `refresh`.
   */
  get allRecent(): GalleryAsset[] { return projectPick.current === null ? defaultView().assets : _recent; },
  get hasMore(): boolean {
    const { assets, totalCount } = defaultView();
    return totalCount !== null && _pageSize !== null && assets.length < totalCount;
  },

  /** Start from the assets the Workspace context carried, before page one is known. */
  seed(assets: GalleryAsset[]): void {
    setDefault({ assets });
  },

  /**
   * Put a finished job's outputs at the front of every newest-first list whose filter admits them (a starred list
   * admits only starred files, so a new output does not join one). Outputs from
   * another project than the pick are left out. An oldest-first list would need its last page for them, so it is
   * dropped and its next visit reloads it.
   */
  mergeOutputs(assets: GalleryAsset[]): void {
    const pick = projectPick.current;
    const scoped = assets.filter((asset) => inProject(asset, pick));
    if (scoped.length === 0) return;
    const newestFirst = [...scoped].reverse();
    const next: Record<string, GalleryView> = {};
    for (const key of new Set([DEFAULT_KEY, ...Object.keys(_views)])) {
      const { filter, sort, starred } = parseQueryKey(key);
      if (sort !== 'newest') continue;
      const view = _views[key] ?? EMPTY_VIEW;
      const incoming = newestFirst.filter((asset) => viewAdmits(filter, starred, asset));
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

  /**
   * Take deleted assets out of every list, and out of each list's count. A list drops the count of every removed file
   * its view admits (media filter and, in a starred list, the star), whether or not it had loaded that file.
   */
  removeAssets(removed: readonly RemovedAsset[]): void {
    if (removed.length === 0) return;
    const removedIds = new Set(removed.map((asset) => asset.id));
    _recent = _recent.filter((asset) => !removedIds.has(asset.id));
    for (const key of Object.keys(_views)) {
      const view = _views[key];
      const { filter, starred } = parseQueryKey(key);
      const assets = view.assets.filter((asset) => !removedIds.has(asset.id));
      // A list is shorter by every removed asset its view admits, whether or not it had loaded it yet.
      const gone = removed.filter((asset) => viewAdmits(filter, starred, asset)).length;
      if (gone === 0 && assets.length === view.assets.length) continue;
      _views[key] = { ...view, assets, totalCount: view.totalCount === null ? null : Math.max(0, view.totalCount - gone) };
    }
  },

  /** Update every loaded list after files moved between projects: see `applyMovesToView`. */
  applyMoves(moves: readonly MovedAsset[]): void {
    if (moves.length === 0) return;
    const pick = projectPick.current;
    // A file moved out of every project now belongs to the Not in a project list, at a place only its next load knows.
    const joinsUnsorted = moves.some((move) => move.asset.project === null);
    for (const key of Object.keys(_views)) {
      const { filter, starred } = parseQueryKey(key);
      if (filter === 'no_project' && joinsUnsorted) delete _views[key];
      else _views[key] = applyMovesToView(_views[key], filter, moves, pick, starred);
    }
    // The recent list ignores the pick, so a moved file stays in it under its new ID.
    const byOldId = new Map(moves.map((move) => [move.from, move.asset]));
    if (_recent.some((asset) => byOldId.has(asset.id))) _recent = _recent.map((asset) => byOldId.get(asset.id) ?? asset);
  },

  /**
   * Record a file's new star in every list that holds it. Unstarring takes the file out of the starred lists, and out of
   * their counts (a list that never loaded it counts it when its media filter admits the file, which is known from a copy
   * loaded anywhere; with no copy at all that list is dropped, to reload). Starring drops the starred lists, since the
   * file's place in them is only known to the server.
   */
  setStarred(id: string, starred: boolean): void {
    const copies = [...Object.values(_views).flatMap((view) => view.assets), ..._recent];
    const known = copies.find((asset) => asset.id === id);
    const update = (asset: GalleryAsset): GalleryAsset => (asset.id === id && (asset.starred ?? false) !== starred ? { ...asset, starred } : asset);
    _recent = _recent.map(update);
    for (const key of Object.keys(_views)) {
      const view = _views[key];
      const query = parseQueryKey(key);
      if (query.starred && starred) {
        delete _views[key];
      } else if (query.starred) {
        const loaded = view.assets.some((asset) => asset.id === id);
        if (!loaded && !known) {
          delete _views[key];
          continue;
        }
        const counted = loaded || (known?.starred === true && filterAdmits(query.filter, known));
        _views[key] = {
          ...view,
          assets: view.assets.filter((asset) => asset.id !== id),
          totalCount: view.totalCount === null || !counted ? view.totalCount : Math.max(0, view.totalCount - 1),
        };
      } else if (view.assets.some((asset) => asset.id === id)) {
        _views[key] = { ...view, assets: view.assets.map(update) };
      }
    }
  },

  /**
   * Reload page one of the default view for the picked project, keeping the pages loaded after it. With a project
   * picked it also loads the unscoped recent list, in the same round.
   */
  async refresh(): Promise<void> {
    _loading = true;
    const generation = ++_generation;
    const pick = projectPick.current;
    const recentLoad = pick === null ? null : loadRecent(generation);
    try {
      const page = await getGallery(1, DEFAULT_QUERY.filter, DEFAULT_QUERY.sort, pick);
      if (generation !== _generation) return;
      // Keep tiles loaded past page one so a refresh does not snap a scrolled strip back to the start.
      const current = defaultView();
      const assets = keepLoadedTail(page.assets, current.assets);
      // With the tail kept, the last loaded page stays what it was, so the Gallery page continues from there.
      setDefault({ assets, page: assets.length > page.assets.length ? current.page : page.page, totalPages: page.total_pages, totalCount: page.total_count });
      // Page one is full whenever more pages follow, so its length is the server's page size.
      if (page.total_pages > 1) _pageSize = page.assets.length;
      // Without a pick the default view is the whole Gallery. Keeping its page one lets a later pick start with
      // recent LoRAs and a last seed already in hand.
      if (pick === null) _recent = page.assets;
    } catch {
      // ignore refresh errors
    } finally {
      if (generation === _generation) _loading = false;
    }
    await recentLoad;
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
        const page = await getGallery(pageNumber, DEFAULT_QUERY.filter, DEFAULT_QUERY.sort, projectPick.current);
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
