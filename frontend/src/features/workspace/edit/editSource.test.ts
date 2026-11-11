import { describe, expect, it } from 'vitest';

import { sourceOfPath } from './editSource';

const query = (url: string) => {
  const parsed = new URL(url, 'http://localhost');
  return { pathname: parsed.pathname, params: parsed.searchParams };
};

describe('the picture of an image chosen on the host', () => {
  it('is drawn from the file itself, and shown as a small preview in the thumbnail', () => {
    const source = sourceOfPath('/home/user/fox.png');

    const picture = query(source.url);
    expect(picture.pathname).toBe('/api/browse/media');
    expect(picture.params.get('purpose')).toBe('reference_image');
    expect(picture.params.get('path')).toBe('/home/user/fox.png');

    const thumb = query(source.thumbUrl);
    expect(thumb.pathname).toBe('/api/browse/preview');
    expect(thumb.params.has('size')).toBe(false);
  });
});
