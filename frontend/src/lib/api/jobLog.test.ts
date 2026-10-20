import { afterEach, describe, expect, it, vi } from 'vitest';

import { clearJobLog, deleteJobLogEntry, getJobLog } from './jobLog';

function stubFetch(body: unknown) {
  const fetchMock = vi.fn(async () => new Response(JSON.stringify(body), { status: 200 }));
  vi.stubGlobal('fetch', fetchMock);
  return fetchMock;
}

const lastCall = (fetchMock: ReturnType<typeof vi.fn>) => fetchMock.mock.calls.at(-1) as unknown as [string, RequestInit];

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('job log API', () => {
  it('requests a page with its status filter', async () => {
    const fetchMock = stubFetch({ entries: [], page: 2, total_pages: 3, total_count: 120, page_size: 50 });
    await expect(getJobLog(2, 'failed')).resolves.toMatchObject({ page: 2, total_count: 120 });

    const [url, init] = lastCall(fetchMock);
    const parsed = new URL(url, 'http://localhost');
    expect(parsed.pathname).toBe('/api/jobs/log');
    expect(parsed.searchParams.get('page')).toBe('2');
    expect(parsed.searchParams.get('status')).toBe('failed');
    expect(init.method ?? 'GET').toBe('GET');
  });

  it('defaults to page 1 of every outcome', async () => {
    const fetchMock = stubFetch({});
    await getJobLog();
    const parsed = new URL(lastCall(fetchMock)[0], 'http://localhost');
    expect(parsed.searchParams.get('page')).toBe('1');
    expect(parsed.searchParams.get('status')).toBe('all');
  });

  it('deletes one entry by its encoded id', async () => {
    const fetchMock = stubFetch({ deleted: 'a/b c' });
    await expect(deleteJobLogEntry('a/b c')).resolves.toEqual({ deleted: 'a/b c' });
    const [url, init] = lastCall(fetchMock);
    expect(new URL(url, 'http://localhost').pathname).toBe('/api/jobs/log/a%2Fb%20c');
    expect(init.method).toBe('DELETE');
  });

  it('clears the whole log', async () => {
    const fetchMock = stubFetch({ deleted_count: 4 });
    await expect(clearJobLog()).resolves.toEqual({ deleted_count: 4 });
    const [url, init] = lastCall(fetchMock);
    expect(new URL(url, 'http://localhost').pathname).toBe('/api/jobs/log');
    expect(init.method).toBe('DELETE');
  });
});
