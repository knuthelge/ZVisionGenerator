import { afterEach, describe, expect, it, vi } from 'vitest';

import { makeAsset } from '../../test-utils/workspace';
import { saveAdjusted, type AdjustBody } from './adjust';

afterEach(() => {
  vi.unstubAllGlobals();
});

const body: AdjustBody = { asset_id: 'a.png', save: 'copy', rotate: 0, flip: false, crop: null, exposure: 0, contrast: 0, highlights: 0, shadows: 0, saturation: 0, vibrance: 0, warmth: 0, mix_red: 0, mix_orange: 0, mix_yellow: 0, mix_green: 0, mix_aqua: 0, mix_blue: 0, mix_purple: 0, mix_magenta: 0 };

describe('saveAdjusted', () => {
  it('posts the body as JSON to /api/adjust and returns the asset of the answer', async () => {
    const saved = makeAsset({ id: 'a_adjusted.png' });
    const fetchMock = vi.fn(async () => new Response(JSON.stringify({ asset: saved }), { status: 200, headers: { 'content-type': 'application/json' } }));
    vi.stubGlobal('fetch', fetchMock);

    expect(await saveAdjusted(body)).toEqual(saved);

    const [url, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe('/api/adjust');
    expect(init.method).toBe('POST');
    expect(JSON.parse(init.body as string)).toEqual(body);
  });

  it('rejects with the server\'s explanation when it refuses', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response(JSON.stringify({ detail: 'Nothing to save: no adjustment is set.' }), { status: 422 })));
    await expect(saveAdjusted(body)).rejects.toMatchObject({ status: 422, detail: 'Nothing to save: no adjustment is set.' });
  });
});
