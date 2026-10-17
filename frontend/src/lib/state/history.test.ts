import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { GalleryAsset, GalleryPage } from '$lib/types';

const getHistory = vi.fn<(page?: number) => Promise<GalleryPage>>();
vi.mock('$lib/api/workspace', () => ({ getHistory: (page?: number) => getHistory(page) }));

type Store = typeof import('./history.svelte').historyStore;

const PAGE_SIZE = 3;

function asset(id: number): GalleryAsset {
  return { id: `a${id}` } as GalleryAsset;
}

/** Serve newest-first pages over `count` assets numbered from the newest. */
function serve(count: number): void {
  const all = Array.from({ length: count }, (_, i) => asset(i));
  getHistory.mockImplementation(async (page = 1) => ({
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
  getHistory.mockReset();
  store = (await import('./history.svelte')).historyStore;
});

describe('historyStore paging', () => {
  it('has nothing more to load before page one is known', () => {
    store.seedHistory([asset(0), asset(1)]);
    expect(store.hasMore).toBe(false);
    expect(store.totalCount).toBeNull();
  });

  it('appends the next page after page one', async () => {
    serve(7);
    await store.refreshHistory();
    expect(store.hasMore).toBe(true);
    expect(store.totalCount).toBe(7);

    await store.loadMore();
    expect(ids(store)).toEqual(['a0', 'a1', 'a2', 'a3', 'a4', 'a5']);
    expect(getHistory).toHaveBeenLastCalledWith(2);

    await store.loadMore();
    expect(ids(store)).toEqual(['a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6']);
    expect(store.hasMore).toBe(false);

    getHistory.mockClear();
    await store.loadMore();
    expect(getHistory).not.toHaveBeenCalled();
  });

  it('fills the rest of a partly loaded page without duplicates', async () => {
    serve(7);
    await store.refreshHistory();
    store.seedHistory([asset(0), asset(1), asset(2), asset(3)]);

    await store.loadMore();
    expect(ids(store)).toEqual(['a0', 'a1', 'a2', 'a3', 'a4', 'a5']);
  });

  it('moves past a page that holds only assets already shown', async () => {
    serve(7);
    await store.refreshHistory();
    getHistory.mockClear();
    // Newer assets landed on the server, pushing the loaded ones onto page two.
    getHistory.mockImplementationOnce(async () => ({ assets: [asset(0), asset(1), asset(2)], page: 2, total_pages: 3, total_count: 7 }));
    getHistory.mockImplementationOnce(async () => ({ assets: [asset(6)], page: 3, total_pages: 3, total_count: 7 }));

    await store.loadMore();
    expect(getHistory.mock.calls.map(([page]) => page)).toEqual([2, 3]);
    expect(ids(store)).toEqual(['a0', 'a1', 'a2', 'a6']);
  });

  it('keeps older loaded tiles when page one is refreshed', async () => {
    serve(7);
    await store.refreshHistory();
    await store.loadMore();

    await store.refreshHistory();
    expect(ids(store)).toEqual(['a0', 'a1', 'a2', 'a3', 'a4', 'a5']);
  });

  it('tracks the total as outputs arrive and assets are deleted', async () => {
    serve(7);
    await store.refreshHistory();

    store.mergeOutputs([asset(9)]);
    expect(store.totalCount).toBe(8);
    store.mergeOutputs([asset(9)]);
    expect(store.totalCount).toBe(8);

    store.removeAssets(['a9', 'a0']);
    expect(store.totalCount).toBe(6);
  });

  it('drops a load-more response that a refresh overtook', async () => {
    serve(7);
    await store.refreshHistory();
    let release: (page: GalleryPage) => void = () => {};
    getHistory.mockImplementationOnce(() => new Promise((resolve) => { release = resolve; }));

    const loading = store.loadMore();
    expect(store.loadingMore).toBe(true);
    await store.refreshHistory();
    release({ assets: [asset(42)], page: 2, total_pages: 3, total_count: 7 });
    await loading;

    expect(ids(store)).not.toContain('a42');
    expect(store.loadingMore).toBe(false);
  });
});
