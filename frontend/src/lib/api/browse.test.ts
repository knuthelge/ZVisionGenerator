import { afterEach, describe, expect, it, vi } from 'vitest';

import { browseInfo, browseMediaUrl, browsePreviewUrl } from './browse';

afterEach(() => {
  vi.unstubAllGlobals();
});

function query(url: string): { pathname: string; params: URLSearchParams } {
  const parsed = new URL(url, 'http://localhost');
  return { pathname: parsed.pathname, params: parsed.searchParams };
}

describe('browse media URLs', () => {
  it('builds the preview URL with the purpose and the path', () => {
    const { pathname, params } = query(browsePreviewUrl('reference_image', '/home/user/fox.png'));

    expect(pathname).toBe('/api/browse/preview');
    expect(Object.fromEntries(params)).toEqual({ purpose: 'reference_image', path: '/home/user/fox.png' });
  });

  it('asks for a video last frame only when told to', () => {
    expect(query(browsePreviewUrl('source_video', '/v.mp4')).params.has('frame')).toBe(false);
    expect(query(browsePreviewUrl('source_video', '/v.mp4', { frame: 'last' })).params.get('frame')).toBe('last');
  });

  it('asks for the large picture only when told to', () => {
    expect(query(browsePreviewUrl('reference_image', '/a.png')).params.has('size')).toBe(false);
    expect(query(browsePreviewUrl('reference_image', '/a.png', { size: 'large' })).params.get('size')).toBe('large');
  });

  it('encodes Windows paths and odd characters in the query', () => {
    const url = browseMediaUrl('audio_track', 'C:\\Users\\me\\a&b #1.wav');

    expect(url).not.toContain('#');
    expect(query(url).pathname).toBe('/api/browse/media');
    expect(query(url).params.get('path')).toBe('C:\\Users\\me\\a&b #1.wav');
  });
});

describe('browseInfo', () => {
  it('fetches the facts for the purpose and path and passes the abort signal on', async () => {
    const fetchMock = vi.fn(async () => new Response(JSON.stringify({ width: 640, height: 480 }), { status: 200 }));
    vi.stubGlobal('fetch', fetchMock);
    const controller = new AbortController();

    await expect(browseInfo('reference_image', '/fox.png', controller.signal)).resolves.toEqual({ width: 640, height: 480 });

    const [url, init] = fetchMock.mock.calls.at(-1) as unknown as [string, RequestInit];
    expect(query(url).pathname).toBe('/api/browse/info');
    expect(Object.fromEntries(query(url).params)).toEqual({ purpose: 'reference_image', path: '/fox.png' });
    expect(init.signal).toBe(controller.signal);
  });
});
