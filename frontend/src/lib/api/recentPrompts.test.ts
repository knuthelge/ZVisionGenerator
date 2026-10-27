import { afterEach, describe, expect, it, vi } from 'vitest';

import { clearRecentPrompts, getRecentPrompts } from './recentPrompts';

function stubFetch(body: unknown) {
  const fetchMock = vi.fn(async () => new Response(JSON.stringify(body), { status: 200 }));
  vi.stubGlobal('fetch', fetchMock);
  return fetchMock;
}

const lastCall = (fetchMock: ReturnType<typeof vi.fn>) => fetchMock.mock.calls.at(-1) as unknown as [string, RequestInit];

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('recent prompts API', () => {
  it('asks for the list of one mode with the encoded filter', async () => {
    const entry = { format: 'text', prompt: 'a fox', scene: '', negative_prompt: '', mode: 'image', used_at: 1 };
    const fetchMock = stubFetch({ entries: [entry] });

    await expect(getRecentPrompts('red & fox', 'video')).resolves.toEqual({ entries: [entry] });

    const [url, init] = lastCall(fetchMock);
    const parsed = new URL(url, 'http://localhost');
    expect(parsed.pathname).toBe('/api/recent-prompts');
    expect(parsed.searchParams.get('q')).toBe('red & fox');
    expect(parsed.searchParams.get('mode')).toBe('video');
    expect(init.method ?? 'GET').toBe('GET');
  });

  it('clears the list of one mode', async () => {
    const fetchMock = stubFetch({ deleted_count: 3 });
    await expect(clearRecentPrompts('image')).resolves.toEqual({ deleted_count: 3 });
    const [url, init] = lastCall(fetchMock);
    const parsed = new URL(url, 'http://localhost');
    expect(parsed.pathname).toBe('/api/recent-prompts');
    expect(parsed.searchParams.get('mode')).toBe('image');
    expect(init.method).toBe('DELETE');
  });
});
