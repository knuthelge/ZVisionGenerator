import { afterEach, describe, expect, it, vi } from 'vitest';

import { getGallery, moveAssets } from './gallery';
import { createProject, listProjects } from './projects';

function stubFetch(body: unknown) {
  const fetchMock = vi.fn(async () => new Response(JSON.stringify(body), { status: 200 }));
  vi.stubGlobal('fetch', fetchMock);
  return fetchMock;
}

function lastCall(fetchMock: ReturnType<typeof stubFetch>): { url: URL; init: RequestInit } {
  const [url, init] = fetchMock.mock.calls.at(-1) as unknown as [string, RequestInit];
  return { url: new URL(url, 'http://localhost'), init };
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('project API', () => {
  it('lists the projects array of the answer', async () => {
    const fetchMock = stubFetch({ projects: [{ name: 'Night Bus', path: '/out/Night Bus' }] });
    await expect(listProjects()).resolves.toEqual([{ name: 'Night Bus', path: '/out/Night Bus' }]);
    expect(lastCall(fetchMock).url.pathname).toBe('/api/projects');
  });

  it('creates with a POST of the name', async () => {
    const fetchMock = stubFetch({ name: 'Zoo', path: '/out/Zoo' });
    await expect(createProject('Zoo')).resolves.toEqual({ name: 'Zoo', path: '/out/Zoo' });
    const { url, init } = lastCall(fetchMock);
    expect(url.pathname).toBe('/api/projects');
    expect(init).toMatchObject({ method: 'POST', body: JSON.stringify({ name: 'Zoo' }) });
  });
});

describe('gallery API with projects', () => {
  const page = { assets: [], page: 1, total_pages: 1, total_count: 0 };

  it('sets the project only when one is given', async () => {
    const fetchMock = stubFetch(page);
    await getGallery(1, 'all', 'newest', 'Night Bus');
    expect(lastCall(fetchMock).url.searchParams.get('project')).toBe('Night Bus');
    await getGallery(1, 'all', 'newest', null);
    expect(lastCall(fetchMock).url.searchParams.has('project')).toBe(false);
    await getGallery(1, 'all', 'newest');
    expect(lastCall(fetchMock).url.searchParams.has('project')).toBe(false);
  });

  it('moves with a POST of the IDs and the target, null for the top of the output folder', async () => {
    const fetchMock = stubFetch({ moved: [], failed: [] });
    await expect(moveAssets(['a.png', 'b.png'], null)).resolves.toEqual({ moved: [], failed: [] });
    const { url, init } = lastCall(fetchMock);
    expect(url.pathname).toBe('/api/gallery/move');
    expect(init).toMatchObject({ method: 'POST', body: JSON.stringify({ asset_ids: ['a.png', 'b.png'], project: null }) });
  });
});
