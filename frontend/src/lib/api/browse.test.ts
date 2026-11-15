import { afterEach, describe, expect, it, vi } from 'vitest';

import { fileVersions } from '$lib/state/fileVersions.svelte';
import { browseInfo, browseMediaUrl, browsePreviewUrl, browseSpriteUrl } from './browse';

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

  it('asks for the frame at a second only when told to', () => {
    expect(query(browsePreviewUrl('source_video', '/v.mp4')).params.has('at')).toBe(false);
    expect(query(browsePreviewUrl('source_video', '/v.mp4', { at: 3.208333 })).params.get('at')).toBe('3.208333');
    expect(query(browsePreviewUrl('source_video', '/v.mp4', { at: 0 })).params.get('at')).toBe('0');
  });

  it('asks for the large picture only when told to', () => {
    expect(query(browsePreviewUrl('reference_image', '/a.png')).params.has('size')).toBe(false);
    expect(query(browsePreviewUrl('reference_image', '/a.png', { size: 'large' })).params.get('size')).toBe('large');
  });

  it('builds the contact sheet URL with the purpose and the path', () => {
    const { pathname, params } = query(browseSpriteUrl('source_video', '/home/user/a&b.mp4'));

    expect(pathname).toBe('/api/browse/sprite');
    expect(Object.fromEntries(params)).toEqual({ purpose: 'source_video', path: '/home/user/a&b.mp4' });
  });

  it('encodes Windows paths and odd characters in the query', () => {
    const url = browseMediaUrl('audio_track', 'C:\\Users\\me\\a&b #1.wav');

    expect(url).not.toContain('#');
    expect(query(url).pathname).toBe('/api/browse/media');
    expect(query(url).params.get('path')).toBe('C:\\Users\\me\\a&b #1.wav');
  });
});

describe('browse picture URLs of a rewritten file', () => {
  it('carry a version once the file was rewritten, for the preview and the media URL, and only for that file', () => {
    const path = '/rewritten/fox.png';
    const other = '/rewritten/owl.png';
    const before = browsePreviewUrl('reference_image', path);
    const otherBefore = browsePreviewUrl('reference_image', other);
    expect(query(before).params.has('v')).toBe(false);

    fileVersions.bump(path);

    expect(query(browsePreviewUrl('reference_image', path)).params.get('v')).toBe('1');
    expect(query(browseMediaUrl('reference_image', path)).params.get('v')).toBe('1');
    expect(query(browsePreviewUrl('reference_image', path, { size: 'large' })).params.get('v')).toBe('1');
    expect(browsePreviewUrl('reference_image', other)).toBe(otherBefore);
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
