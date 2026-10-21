import { afterEach, describe, expect, it, vi } from 'vitest';

import { appendPromptEntry } from './promptFiles';

function stubFetch(body: unknown) {
  const fetchMock = vi.fn(async () => new Response(JSON.stringify(body), { status: 200 }));
  vi.stubGlobal('fetch', fetchMock);
  return fetchMock;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('prompt file API', () => {
  it('appends an entry with a POST of the given fields', async () => {
    const appended = { path: '/p.yaml', revision: 'r1', entry_id: 's0.e3', set_name: 'prompts' };
    const fetchMock = stubFetch(appended);
    const data = { path: '/p.yaml', prompt: 'a', scene: 'b' };

    await expect(appendPromptEntry(data)).resolves.toEqual(appended);

    const [url, init] = fetchMock.mock.calls.at(-1) as unknown as [string, RequestInit];
    expect(new URL(url, 'http://localhost').pathname).toBe('/api/prompt-files/append');
    expect(init.method).toBe('POST');
    expect(JSON.parse(init.body as string)).toEqual(data);
  });
});
