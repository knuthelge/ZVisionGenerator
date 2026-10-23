import { afterEach, describe, expect, it, vi } from 'vitest';

import { createPreset, deletePreset, listPresets, renamePreset, updatePreset } from './presets';

function stubFetch(body: unknown) {
  const fetchMock = vi.fn(async () => new Response(JSON.stringify(body), { status: 200 }));
  vi.stubGlobal('fetch', fetchMock);
  return fetchMock;
}

function lastCall(fetchMock: ReturnType<typeof stubFetch>): { path: string; init: RequestInit } {
  const [url, init] = fetchMock.mock.calls.at(-1) as unknown as [string, RequestInit];
  return { path: new URL(url, 'http://localhost').pathname, init };
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('preset API', () => {
  it('lists the presets array of the answer', async () => {
    const preset = { slug: 'soft', name: 'Soft', mode: 'image', model: null, settings: {} };
    const fetchMock = stubFetch({ presets: [preset] });

    await expect(listPresets()).resolves.toEqual([preset]);
    expect(lastCall(fetchMock)).toMatchObject({ path: '/api/presets', init: { method: 'GET' } });
  });

  it('creates with a POST of the whole body', async () => {
    const body = { name: 'Soft', mode: 'image' as const, model: 'zit', settings: { steps: 9 } };
    const fetchMock = stubFetch({ slug: 'soft', ...body });

    await createPreset(body);

    const { path, init } = lastCall(fetchMock);
    expect(path).toBe('/api/presets');
    expect(init.method).toBe('POST');
    expect(JSON.parse(init.body as string)).toEqual(body);
  });

  it('updates with a PUT to the mode and slug, the body without the mode', async () => {
    const body = { name: 'Soft', model: 'zit', settings: { steps: 9 } };
    const fetchMock = stubFetch({ slug: 'soft', mode: 'image', ...body });

    await updatePreset('image', 'soft', body);

    const { path, init } = lastCall(fetchMock);
    expect(path).toBe('/api/presets/image/soft');
    expect(init.method).toBe('PUT');
    expect(JSON.parse(init.body as string)).toEqual(body);
  });

  it('renames with a POST of the new name and deletes with a DELETE', async () => {
    const fetchMock = stubFetch({ slug: 'new-name', name: 'New name', mode: 'video', model: null, settings: {} });
    await renamePreset('video', 'old', 'New name');
    let call = lastCall(fetchMock);
    expect(call.path).toBe('/api/presets/video/old/rename');
    expect(call.init.method).toBe('POST');
    expect(JSON.parse(call.init.body as string)).toEqual({ name: 'New name' });

    stubFetch({ status: 'ok' });
    const deleteMock = stubFetch({ status: 'ok' });
    await expect(deletePreset('video', 'old')).resolves.toEqual({ status: 'ok' });
    call = lastCall(deleteMock);
    expect(call).toMatchObject({ path: '/api/presets/video/old', init: { method: 'DELETE' } });
  });
});
