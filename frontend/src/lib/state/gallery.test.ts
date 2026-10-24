import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { GalleryAsset, GalleryPage } from '$lib/types';

const getGallery = vi.fn<(page?: number, filter?: string, sort?: string) => Promise<GalleryPage>>();
vi.mock('$lib/api/gallery', () => ({ getGallery: (page?: number, filter?: string, sort?: string) => getGallery(page, filter, sort) }));

type Store = typeof import('./gallery.svelte').galleryStore;

const PAGE_SIZE = 3;

function asset(id: number, media_type: 'image' | 'video' = 'image'): GalleryAsset {
  return { id: `a${id}`, media_type } as GalleryAsset;
}

/** Serve newest-first pages over `count` assets numbered from the newest. */
function serve(count: number): void {
  const all = Array.from({ length: count }, (_, i) => asset(i));
  getGallery.mockImplementation(async (page = 1) => ({
    assets: all.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE),
    page,
    total_pages: Math.max(1, Math.ceil(count / PAGE_SIZE)),
    total_count: count,
  }));
}

const ids = (store: Store): string[] => store.assets.map((item) => item.id);

let store: Store;

beforeEach(async () => {
  vi.resetModules();
  getGallery.mockReset();
  store = (await import('./gallery.svelte')).galleryStore;
});

describe('galleryStore paging', () => {
  it('has nothing more to load before page one is known', () => {
    store.seed([asset(0), asset(1)]);
    expect(store.hasMore).toBe(false);
    expect(store.totalCount).toBeNull();
  });

  it('appends the next page after page one, from the whole gallery newest first', async () => {
    serve(7);
    await store.refresh();
    expect(getGallery).toHaveBeenLastCalledWith(1, 'all', 'newest');
    expect(store.hasMore).toBe(true);
    expect(store.totalCount).toBe(7);

    await store.loadMore();
    expect(ids(store)).toEqual(['a0', 'a1', 'a2', 'a3', 'a4', 'a5']);
    expect(getGallery).toHaveBeenLastCalledWith(2, 'all', 'newest');

    await store.loadMore();
    expect(ids(store)).toEqual(['a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6']);
    expect(store.hasMore).toBe(false);

    getGallery.mockClear();
    await store.loadMore();
    expect(getGallery).not.toHaveBeenCalled();
  });

  it('fills the rest of a partly loaded page without duplicates', async () => {
    serve(7);
    await store.refresh();
    store.seed([asset(0), asset(1), asset(2), asset(3)]);

    await store.loadMore();
    expect(ids(store)).toEqual(['a0', 'a1', 'a2', 'a3', 'a4', 'a5']);
  });

  it('moves past a page that holds only assets already shown', async () => {
    serve(7);
    await store.refresh();
    getGallery.mockClear();
    // Newer assets landed on the server, pushing the loaded ones onto page two.
    getGallery.mockImplementationOnce(async () => ({ assets: [asset(0), asset(1), asset(2)], page: 2, total_pages: 3, total_count: 7 }));
    getGallery.mockImplementationOnce(async () => ({ assets: [asset(6)], page: 3, total_pages: 3, total_count: 7 }));

    await store.loadMore();
    expect(getGallery.mock.calls.map(([page]) => page)).toEqual([2, 3]);
    expect(ids(store)).toEqual(['a0', 'a1', 'a2', 'a6']);
  });

  it('keeps older loaded tiles when page one is refreshed', async () => {
    serve(7);
    await store.refresh();
    await store.loadMore();

    await store.refresh();
    expect(ids(store)).toEqual(['a0', 'a1', 'a2', 'a3', 'a4', 'a5']);
    // The Gallery page continues from the last loaded page, not from page one again.
    expect(store.view().page).toBe(2);
  });

  it('tracks the total as outputs arrive and assets are deleted', async () => {
    serve(7);
    await store.refresh();

    store.mergeOutputs([asset(9)]);
    expect(store.totalCount).toBe(8);
    store.mergeOutputs([asset(9)]);
    expect(store.totalCount).toBe(8);

    store.removeAssets([asset(9), asset(0)]);
    expect(store.totalCount).toBe(6);
  });

  it('drops a load-more response that a refresh overtook', async () => {
    serve(7);
    await store.refresh();
    let release: (page: GalleryPage) => void = () => {};
    getGallery.mockImplementationOnce(() => new Promise((resolve) => { release = resolve; }));

    const loading = store.loadMore();
    expect(store.loadingMore).toBe(true);
    await store.refresh();
    release({ assets: [asset(42)], page: 2, total_pages: 3, total_count: 7 });
    await loading;

    expect(ids(store)).not.toContain('a42');
    expect(store.loadingMore).toBe(false);
  });
});

describe('galleryStore views', () => {
  const videos = { filter: 'video' as const, sort: 'newest' as const };

  it('keeps one list per query, empty until the page commits it', () => {
    expect(store.view(videos).assets).toEqual([]);
    store.setView(videos, { assets: [asset(1), asset(3)], page: 1, totalPages: 2, totalCount: 5 });
    expect(store.view(videos).assets.map((item) => item.id)).toEqual(['a1', 'a3']);
    expect(store.view().assets).toEqual([]);
  });

  it('removes a deleted asset from every list and count, counting it in lists that had not loaded it yet', () => {
    store.seed([asset(1, 'video'), asset(2)]);
    store.setView(videos, { assets: [asset(1, 'video'), asset(3, 'video')], page: 1, totalPages: 2, totalCount: 4 });
    store.removeAssets([asset(1, 'video')]);
    expect(ids(store)).toEqual(['a2']);
    expect(store.view(videos).assets.map((item) => item.id)).toEqual(['a3']);
    expect(store.view(videos).totalCount).toBe(3);

    // A video deleted from a page the videos list had not loaded still shortens that list; an image does not.
    store.removeAssets([asset(7, 'video'), asset(8)]);
    expect(store.view(videos).totalCount).toBe(2);
    expect(store.view(videos).assets.map((item) => item.id)).toEqual(['a3']);
  });

  it('puts new outputs at the front of every newest-first list whose filter admits them', () => {
    const images = { filter: 'image' as const, sort: 'newest' as const };
    const oldest = { filter: 'all' as const, sort: 'oldest' as const };
    store.setView(videos, { assets: [asset(3, 'video')], page: 1, totalPages: 1, totalCount: 1 });
    store.setView(images, { assets: [asset(4)], page: 1, totalPages: 1, totalCount: 1 });
    store.setView(oldest, { assets: [asset(4), asset(3, 'video')], page: 1, totalPages: 1, totalCount: 2 });

    store.mergeOutputs([asset(1, 'video'), asset(0)]);
    expect(ids(store)).toEqual(['a0', 'a1']);
    expect(store.view(videos).assets.map((item) => item.id)).toEqual(['a1', 'a3']);
    expect(store.view(videos).totalCount).toBe(2);
    expect(store.view(images).assets.map((item) => item.id)).toEqual(['a0', 'a4']);
    expect(store.view(images).totalCount).toBe(2);
    // An oldest-first list would need its last page, so it is dropped and its next visit reloads it.
    expect(store.view(oldest).assets).toEqual([]);
  });
});
