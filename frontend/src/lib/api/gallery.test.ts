import { afterEach, describe, expect, it, vi } from 'vitest';

import { deleteAssets, downloadAssets, getGallery, getGalleryIds, setStarred } from './gallery';

function stubFetch(body: unknown) {
  const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: () => Promise.resolve(body) });
  vi.stubGlobal('fetch', fetchMock);
  return fetchMock;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('getGallery', () => {
  it('asks for the starred files only when starred is set', async () => {
    const fetchMock = stubFetch({ assets: [], page: 1, total_pages: 1, total_count: 0 });
    await getGallery(2, 'image', 'newest', 'A', true);
    const params = new URL(fetchMock.mock.calls[0][0] as string, 'http://x').searchParams;
    expect(params.get('page')).toBe('2');
    expect(params.get('filter')).toBe('image');
    expect(params.get('project')).toBe('A');
    expect(params.get('starred')).toBe('true');
  });
});

describe('getGalleryIds', () => {
  it('sends the view narrowing and resolves to the IDs', async () => {
    const fetchMock = stubFetch({ ids: ['a/1.png', 'a/2.png'] });
    await expect(getGalleryIds('image', 'a', true)).resolves.toEqual(['a/1.png', 'a/2.png']);
    const url = new URL(fetchMock.mock.calls[0][0] as string, 'http://x');
    expect(url.pathname).toBe('/api/gallery/ids');
    expect(url.searchParams.get('filter')).toBe('image');
    expect(url.searchParams.get('project')).toBe('a');
    expect(url.searchParams.get('starred')).toBe('true');
  });

  it('leaves the defaults out of the query: starred of the listing, the all filter, an empty project and a false starred of the IDs', async () => {
    const listing = stubFetch({ assets: [], page: 1, total_pages: 1, total_count: 0 });
    await getGallery(1);
    expect(new URL(listing.mock.calls[0][0] as string, 'http://x').searchParams.has('starred')).toBe(false);

    const ids = stubFetch({ ids: [] });
    await getGalleryIds('all', null, false);
    expect(ids.mock.calls[0][0]).toBe('/api/gallery/ids?');
  });
});

describe('deleteAssets', () => {
  it('posts every ID in one request and resolves to the result', async () => {
    const result = { deleted: [{ id: 'a.png', media_type: 'image', project: null, starred: false, file_path: '/out/a.png' }], failed: [] };
    const fetchMock = stubFetch(result);
    await expect(deleteAssets(['a.png', 'b.png'])).resolves.toEqual(result);
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe('/api/gallery/delete');
    expect(init.method).toBe('POST');
    expect(JSON.parse(init.body as string)).toEqual({ asset_ids: ['a.png', 'b.png'] });
  });
});

describe('setStarred', () => {
  it('posts the ID and the new state', async () => {
    const fetchMock = stubFetch({ id: 'a/1.png', starred: true });
    await expect(setStarred('a/1.png', true)).resolves.toEqual({ id: 'a/1.png', starred: true });
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe('/api/gallery/star');
    expect(init.method).toBe('POST');
    expect(JSON.parse(init.body as string)).toEqual({ asset_id: 'a/1.png', starred: true });
  });
});

describe('downloadAssets', () => {
  it('submits a form post with the IDs joined by newlines, then removes the form', () => {
    let seen: { method: string; action: string; fields: [string, string][]; attached: boolean } | null = null;
    const submit = vi.spyOn(HTMLFormElement.prototype, 'submit').mockImplementation(function (this: HTMLFormElement) {
      seen = {
        method: this.method,
        action: new URL(this.action).pathname,
        fields: [...new FormData(this).entries()] as [string, string][],
        attached: document.body.contains(this)
      };
    });
    downloadAssets(['a/1.png', 'b.mp4', 'c.wav']);
    expect(submit).toHaveBeenCalledTimes(1);
    expect(seen).toEqual({ method: 'post', action: '/api/gallery/download', fields: [['asset_ids', 'a/1.png\nb.mp4\nc.wav']], attached: true });
    expect(document.querySelector('form[action="/api/gallery/download"]')).toBeNull();
    submit.mockRestore();
  });
});
