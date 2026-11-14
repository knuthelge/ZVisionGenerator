import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { GalleryAsset, GalleryPage } from '$lib/types';

const getGallery = vi.fn<(page?: number, filter?: string, sort?: string, project?: string | null) => Promise<GalleryPage>>();
vi.mock('$lib/api/gallery', () => ({
  getGallery: (page?: number, filter?: string, sort?: string, project?: string | null) => getGallery(page, filter, sort, project),
}));

type Store = typeof import('./gallery.svelte').galleryStore;

const PAGE_SIZE = 3;

function asset(id: number, media_type: 'image' | 'video' | 'audio' = 'image', project: string | null = null): GalleryAsset {
  return { id: `a${id}`, media_type, project } as GalleryAsset;
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
let pickProject: (name: string | null) => void;

beforeEach(async () => {
  vi.resetModules();
  localStorage.clear();
  getGallery.mockReset();
  store = (await import('./gallery.svelte')).galleryStore;
  const { projectPick } = await import('./projectPick.svelte');
  pickProject = (name) => projectPick.set(name);
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
    expect(getGallery).toHaveBeenLastCalledWith(1, 'all', 'newest', null);
    expect(store.hasMore).toBe(true);
    expect(store.totalCount).toBe(7);

    await store.loadMore();
    expect(ids(store)).toEqual(['a0', 'a1', 'a2', 'a3', 'a4', 'a5']);
    expect(getGallery).toHaveBeenLastCalledWith(2, 'all', 'newest', null);

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
  const videos = { filter: 'video' as const, sort: 'newest' as const, starred: false };

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
    const images = { filter: 'image' as const, sort: 'newest' as const, starred: false };
    const oldest = { filter: 'all' as const, sort: 'oldest' as const, starred: false };
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

describe('galleryStore audio filter', () => {
  it('keeps new audio only in the audio list and the whole list, never in the image list', () => {
    const audio = { filter: 'audio' as const, sort: 'newest' as const, starred: false };
    const images = { filter: 'image' as const, sort: 'newest' as const, starred: false };
    store.setView(audio, { assets: [asset(3, 'audio')], page: 1, totalPages: 1, totalCount: 1 });
    store.setView(images, { assets: [asset(4)], page: 1, totalPages: 1, totalCount: 1 });

    store.mergeOutputs([asset(1, 'audio'), asset(0)]);
    expect(ids(store)).toEqual(['a0', 'a1']);
    expect(store.view(audio).assets.map((item) => item.id)).toEqual(['a1', 'a3']);
    expect(store.view(images).assets.map((item) => item.id)).toEqual(['a0', 'a4']);
  });
});

describe('galleryStore project scope', () => {
  const view = (assets: GalleryAsset[], totalCount: number | null = assets.length) => ({ assets, page: 1, totalPages: 1, totalCount });

  it('asks for the picked project on refresh and on load more', async () => {
    pickProject('A');
    getGallery.mockImplementation(async (page = 1, _filter, _sort, project) => ({
      assets: project === 'A' ? [asset(page * 10, 'image', 'A'), asset(page * 10 + 1, 'image', 'A')] : [asset(99)],
      page,
      total_pages: 3,
      total_count: 6,
    }));
    await store.refresh();
    expect(getGallery).toHaveBeenCalledWith(1, 'all', 'newest', 'A');
    await store.loadMore();
    expect(getGallery).toHaveBeenLastCalledWith(2, 'all', 'newest', 'A');
    expect(ids(store)).toEqual(['a10', 'a11', 'a20', 'a21']);
  });

  it('makes one request with All files, and the default view is the recent list', async () => {
    serve(4);
    await store.refresh();
    expect(getGallery).toHaveBeenCalledTimes(1);
    expect(store.allRecent).toBe(store.assets);
    expect(store.allRecent.map((item) => item.id)).toEqual(['a0', 'a1', 'a2']);
  });

  it('makes a second, unfiltered request with a project and keeps its assets as the recent list', async () => {
    pickProject('A');
    getGallery.mockImplementation(async (_page, _filter, _sort, project) => ({
      assets: project === 'A' ? [asset(1, 'image', 'A')] : [asset(7, 'image', 'B'), asset(1, 'image', 'A')],
      page: 1,
      total_pages: 1,
      total_count: project === 'A' ? 1 : 2,
    }));
    await store.refresh();
    expect(getGallery).toHaveBeenCalledTimes(2);
    expect(getGallery).toHaveBeenCalledWith(1, 'all', 'newest', undefined);
    expect(ids(store)).toEqual(['a1']);
    expect(store.allRecent.map((item) => item.id)).toEqual(['a7', 'a1']);
  });

  it('ignores a failed recent request and keeps the scoped list', async () => {
    pickProject('A');
    getGallery.mockImplementation(async (_page, _filter, _sort, project) => {
      if (project === 'A') return { assets: [asset(1, 'image', 'A')], page: 1, total_pages: 1, total_count: 1 };
      throw new Error('offline');
    });
    await store.refresh();
    expect(ids(store)).toEqual(['a1']);
    expect(store.allRecent).toEqual([]);
  });

  it('drops a recent response that a newer refresh overtook', async () => {
    pickProject('A');
    let release: (page: GalleryPage) => void = () => {};
    getGallery.mockImplementationOnce(async () => ({ assets: [asset(1, 'image', 'A')], page: 1, total_pages: 1, total_count: 1 }));
    getGallery.mockImplementationOnce(() => new Promise((resolve) => { release = resolve; }));
    const first = store.refresh();
    getGallery.mockImplementation(async (_page, _filter, _sort, project) => ({
      assets: project === 'A' ? [asset(2, 'image', 'A')] : [asset(8)],
      page: 1,
      total_pages: 1,
      total_count: 1,
    }));
    await store.refresh();
    release({ assets: [asset(42)], page: 1, total_pages: 1, total_count: 1 });
    await first;
    expect(store.allRecent.map((item) => item.id)).toEqual(['a8']);
  });

  it('keeps the recent list through a reset, as it does not follow the pick', async () => {
    serve(3);
    await store.refresh();
    pickProject('A');
    store.reset();
    expect(store.assets).toEqual([]);
    expect(store.allRecent.map((item) => item.id)).toEqual(['a0', 'a1', 'a2']);
  });

  it('leaves out outputs from another project, and admits all with All files', () => {
    pickProject('A');
    store.setView({ filter: 'all', sort: 'newest', starred: false }, view([asset(5, 'image', 'A')]));
    store.mergeOutputs([asset(1, 'image', 'B'), asset(2, 'image', 'A'), asset(3, 'image', null)]);
    expect(ids(store)).toEqual(['a2', 'a5']);
    expect(store.totalCount).toBe(2);

    pickProject(null);
    store.mergeOutputs([asset(1, 'image', 'B'), asset(3, 'image', null)]);
    expect(ids(store)).toEqual(['a3', 'a1', 'a2', 'a5']);
  });

  it('replaces a moved asset in place with All files, keeping the count', () => {
    store.setView({ filter: 'all', sort: 'newest', starred: false }, view([asset(1), asset(2), asset(3)], 9));
    store.applyMoves([{ from: 'a2', from_path: '/o/a2', asset: { ...asset(2), id: 'B/a2', project: 'B' } }]);
    expect(ids(store)).toEqual(['a1', 'B/a2', 'a3']);
    expect(store.totalCount).toBe(9);
  });

  it('removes a file moved out of the picked project and lowers every count that admits it', () => {
    pickProject('A');
    const videos = { filter: 'video' as const, sort: 'newest' as const, starred: false };
    const images = { filter: 'image' as const, sort: 'newest' as const, starred: false };
    store.setView({ filter: 'all', sort: 'newest', starred: false }, view([asset(1, 'image', 'A'), asset(2, 'video', 'A'), asset(3, 'image', 'A')], 8));
    store.setView(videos, view([asset(2, 'video', 'A')], 3));
    store.setView(images, view([asset(1, 'image', 'A'), asset(3, 'image', 'A')], 5));

    store.applyMoves([{ from: 'a2', from_path: '/o/a2', asset: { ...asset(2, 'video'), id: 'B/a2', project: 'B' } }]);
    expect(ids(store)).toEqual(['a1', 'a3']);
    expect(store.totalCount).toBe(7);
    expect(store.view(videos).assets).toEqual([]);
    expect(store.view(videos).totalCount).toBe(2);
    // The image list does not admit a video: untouched.
    expect(store.view(images).totalCount).toBe(5);
  });

  it('keeps a file moved into the picked project under its new ID, and counts one it never loaded', () => {
    pickProject('A');
    store.setView({ filter: 'all', sort: 'newest', starred: false }, view([asset(1, 'image', 'A'), asset(2, 'image', 'A')], 6));
    store.applyMoves([
      { from: 'a2', from_path: '/o/a2', asset: { ...asset(2), id: 'A/a2', project: 'A' } },
      { from: 'a9', from_path: '/o/a9', asset: { ...asset(9), id: 'B/a9', project: 'B' } },
    ]);
    expect(ids(store)).toEqual(['a1', 'A/a2']);
    expect(store.totalCount).toBe(5);
  });

  it('keeps a moved file in the unscoped recent list under its new ID', async () => {
    pickProject('A');
    getGallery.mockImplementation(async (_page, _filter, _sort, project) => ({
      assets: project === 'A' ? [asset(1, 'image', 'A')] : [asset(1, 'image', 'A'), asset(2, 'image', 'B')],
      page: 1,
      total_pages: 1,
      total_count: 1,
    }));
    await store.refresh();
    store.applyMoves([{ from: 'a1', from_path: '/o/a1', asset: { ...asset(1), id: 'B/a1', project: 'B' } }]);
    expect(store.allRecent.map((item) => item.id)).toEqual(['B/a1', 'a2']);
  });
});

describe('galleryStore Not in a project list', () => {
  const unsorted = { filter: 'no_project' as const, sort: 'newest' as const, starred: false };
  const view = (assets: GalleryAsset[], totalCount: number | null = assets.length) => ({ assets, page: 1, totalPages: 1, totalCount });

  it('takes new outputs only when they sit in no project', () => {
    store.setView(unsorted, view([asset(1)], 1));
    store.mergeOutputs([asset(2, 'image', null), asset(3, 'image', 'A')]);
    expect(store.view(unsorted).assets.map((item) => item.id)).toEqual(['a2', 'a1']);
    expect(store.view(unsorted).totalCount).toBe(2);
  });

  it('lowers its count when a file in no project is deleted, whatever the media', () => {
    store.setView(unsorted, view([asset(1), asset(2, 'video')], 5));
    store.removeAssets([asset(2, 'video'), asset(3, 'image', 'A')]);
    expect(store.view(unsorted).assets.map((item) => item.id)).toEqual(['a1']);
    expect(store.view(unsorted).totalCount).toBe(4);
  });

  it('drops a file moved into a project, and lowers the count', () => {
    store.setView(unsorted, view([asset(1), asset(2), asset(3)], 7));
    store.applyMoves([{ from: 'a2', from_path: '/o/a2', asset: { ...asset(2), id: 'B/a2', project: 'B' } }]);
    expect(store.view(unsorted).assets.map((item) => item.id)).toEqual(['a1', 'a3']);
    expect(store.view(unsorted).totalCount).toBe(6);
  });

  it('lowers its count only for the files it had loaded', () => {
    store.setView(unsorted, view([asset(1), asset(2)], 7));
    store.applyMoves([
      { from: 'a2', from_path: '/o/a2', asset: { ...asset(2), id: 'B/a2', project: 'B' } },
      { from: 'a9', from_path: '/o/a9', asset: { ...asset(9), id: 'B/a9', project: 'B' } },
    ]);
    expect(store.view(unsorted).assets.map((item) => item.id)).toEqual(['a1']);
    expect(store.view(unsorted).totalCount).toBe(6);
  });

  it('is dropped, to load again, when a file moves out of every project', () => {
    store.setView(unsorted, view([asset(1)], 1));
    store.setView({ filter: 'all', sort: 'newest', starred: false }, view([asset(1), { ...asset(2), id: 'B/a2', project: 'B' }], 2));
    store.applyMoves([{ from: 'B/a2', from_path: '/o/B/a2', asset: asset(2) }]);
    expect(store.view(unsorted).totalCount).toBeNull();
    expect(ids(store)).toEqual(['a1', 'a2']);
  });
});

describe('galleryStore starred lists', () => {
  const starredAll = { filter: 'all' as const, sort: 'newest' as const, starred: true };
  const starredImages = { filter: 'image' as const, sort: 'newest' as const, starred: true };
  const plain = { filter: 'all' as const, sort: 'newest' as const, starred: false };
  const star = (id: number, media_type: 'image' | 'video' | 'audio' = 'image', project: string | null = null, starred = true): GalleryAsset => ({ ...asset(id, media_type, project), starred });
  const view = (assets: GalleryAsset[], totalCount: number | null = assets.length) => ({ assets, page: 1, totalPages: 1, totalCount });

  it('keeps the starred and the plain list of one filter apart', () => {
    store.setView(plain, view([asset(1), asset(2)]));
    store.setView(starredAll, view([star(2)]));
    expect(store.view(plain).assets).toHaveLength(2);
    expect(store.view(starredAll).assets.map((item) => item.id)).toEqual(['a2']);
  });

  it('never gains a merged new output in a starred list', () => {
    store.setView(plain, view([asset(1)]));
    store.setView(starredAll, view([star(2)]));
    store.mergeOutputs([asset(0)]);
    expect(store.view(plain).assets.map((item) => item.id)).toEqual(['a0', 'a1']);
    expect(store.view(starredAll).assets.map((item) => item.id)).toEqual(['a2']);
    expect(store.view(starredAll).totalCount).toBe(1);
  });

  it('lowers a starred list\'s count for a starred file it admits, and not for an unstarred one', () => {
    store.setView(starredImages, view([star(1)], 4));
    store.removeAssets([star(8), star(9, 'video'), star(7, 'image', null, false)]);
    expect(store.view(starredImages).totalCount).toBe(3);
    store.removeAssets([star(1)]);
    expect(store.view(starredImages).assets).toEqual([]);
    expect(store.view(starredImages).totalCount).toBe(2);
  });

  it('keeps a moved starred file in a starred list under its new ID and drops it from a list it left', () => {
    pickProject('A');
    store.setView(starredAll, view([star(1, 'image', 'A'), star(2, 'image', 'A')], 5));
    store.applyMoves([
      { from: 'a1', from_path: '/o/A/a1', asset: { ...star(1, 'image', 'A'), id: 'A/a1' } },
      { from: 'a2', from_path: '/o/A/a2', asset: { ...star(2, 'image', 'B'), id: 'B/a2' } },
    ]);
    expect(store.view(starredAll).assets.map((item) => item.id)).toEqual(['A/a1']);
    expect(store.view(starredAll).totalCount).toBe(4);
  });

  it('does not count an unstarred file leaving a project in a starred list', () => {
    pickProject('A');
    store.setView(starredAll, view([star(1, 'image', 'A')], 3));
    store.applyMoves([{ from: 'a9', from_path: '/o/A/a9', asset: { ...star(9, 'image', 'B', false), id: 'B/a9' } }]);
    expect(store.view(starredAll).totalCount).toBe(3);
  });

  describe('setStarred', () => {
    it('replaces the file in every list and the recent list', () => {
      store.setView(plain, view([asset(1), asset(2)]));
      store.setStarred('a2', true);
      expect(store.view(plain).assets.map((item) => item.starred ?? false)).toEqual([false, true]);
    });

    it('drops the cached starred lists on star, since the file\'s place in them is the server\'s to say', () => {
      store.setView(plain, view([asset(1)]));
      store.setView(starredAll, view([star(2)]));
      store.setView(starredImages, view([star(2)]));
      store.setStarred('a1', true);
      expect(store.view(starredAll).totalCount).toBeNull();
      expect(store.view(starredImages).totalCount).toBeNull();
      expect(store.view(plain).assets[0].starred).toBe(true);
    });

    it('takes an unstarred file out of the starred lists at once and lowers their counts', () => {
      store.setView(plain, view([star(1), star(2)], 6));
      store.setView(starredAll, view([star(1), star(2)], 4));
      store.setView(starredImages, view([star(1)], 3));
      store.setStarred('a1', false);
      expect(store.view(starredAll).assets.map((item) => item.id)).toEqual(['a2']);
      expect(store.view(starredAll).totalCount).toBe(3);
      expect(store.view(starredImages).assets).toEqual([]);
      expect(store.view(starredImages).totalCount).toBe(2);
      expect(store.view(plain).assets.map((item) => item.starred)).toEqual([false, true]);
      expect(store.view(plain).totalCount).toBe(6);
    });

    it('lowers a starred list\'s count for a file it never loaded when its media filter admits it', () => {
      store.setView(plain, view([star(1, 'video')], 6));
      store.setView(starredImages, view([star(5)], 3));
      store.setView(starredAll, view([star(5)], 4));
      store.setStarred('a1', false);
      // The file is a video: the starred images list never counted it.
      expect(store.view(starredImages).totalCount).toBe(3);
      expect(store.view(starredAll).totalCount).toBe(3);
    });

    it('drops a starred list to reload when the file is known nowhere', () => {
      store.setView(starredAll, view([star(5)], 4));
      store.setStarred('unknown', false);
      expect(store.view(starredAll).totalCount).toBeNull();
    });
  });
});
