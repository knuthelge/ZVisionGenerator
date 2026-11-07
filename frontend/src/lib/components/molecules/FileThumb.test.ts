import { flushSync } from 'svelte';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const browseApiMocks = vi.hoisted(() => ({
  browsePreviewUrl: vi.fn(),
}));

vi.mock('$lib/api/browse', () => browseApiMocks);

import { stubBrowsePreviewUrl } from '../../../test-utils/pathBrowser';
import { mountComponent, settle } from '../../../test-utils/mount';
import { reactiveProps } from '../../../test-utils/reactiveProps.svelte';
import FileThumb from './FileThumb.svelte';

describe('FileThumb', () => {
  beforeEach(() => {
    for (const mock of Object.values(browseApiMocks)) mock.mockReset();
    browseApiMocks.browsePreviewUrl.mockImplementation(stubBrowsePreviewUrl);
  });

  const thumb = () => document.body.querySelector<HTMLElement>('[data-testid="file-thumb"]')!;
  const image = () => thumb().querySelector('img');

  it('shows the picture of an image', () => {
    mountComponent(FileThumb, { purpose: 'reference_image', path: '/pics/fox.png', kind: 'image' });

    expect(image()?.getAttribute('src')).toBe('/preview/reference_image/pics/fox.png');
    expect(image()?.getAttribute('loading')).toBe('lazy');
    expect(thumb().querySelector('[data-testid="play-badge"]')).toBeNull();
  });

  it('shows a video\'s still with a play badge, and its last frame without one', () => {
    mountComponent(FileThumb, { purpose: 'source_video', path: '/clips/a.mp4', kind: 'video' });
    mountComponent(FileThumb, { purpose: 'source_video', path: '/clips/a.mp4', kind: 'video', frame: 'last' });

    const [still, last] = Array.from(document.body.querySelectorAll<HTMLElement>('[data-testid="file-thumb"]'));
    expect(still.querySelector('img')?.getAttribute('src')).toBe('/preview/source_video/clips/a.mp4');
    expect(still.querySelector('[data-testid="play-badge"]')).not.toBeNull();
    expect(last.querySelector('img')?.getAttribute('src')).toBe('/preview/source_video/clips/a.mp4?frame=last');
    expect(last.querySelector('[data-testid="play-badge"]')).toBeNull();
  });

  it('shows the fallback icon it was given, and the file icon otherwise, when the picture cannot load', async () => {
    mountComponent(FileThumb, { purpose: 'reference_image', path: '/pics/a.png', kind: 'image' });
    mountComponent(FileThumb, { purpose: 'source_video', path: '/clips/a.mp4', kind: 'video', frame: 'last', fallbackIcon: 'film' });
    const [plain, film] = Array.from(document.body.querySelectorAll<HTMLElement>('[data-testid="file-thumb"]'));
    const iconOf = (box: HTMLElement) => box.querySelector('svg')?.innerHTML;
    plain.querySelector('img')!.dispatchEvent(new Event('error'));
    film.querySelector('img')!.dispatchEvent(new Event('error'));
    await settle();

    expect(plain.querySelector('img')).toBeNull();
    expect(iconOf(plain)).toBeTruthy();
    expect(iconOf(film)).toBeTruthy();
    expect(iconOf(film)).not.toBe(iconOf(plain));
  });

  it('shows an icon and requests nothing for audio', async () => {
    mountComponent(FileThumb, { purpose: 'audio_track', path: '/music/take.flac', kind: 'audio' });
    await settle();

    expect(thumb().querySelector('svg')).not.toBeNull();
    expect(thumb().querySelector('img')).toBeNull();
    expect(browseApiMocks.browsePreviewUrl).not.toHaveBeenCalled();
  });

  it('asks for the small picture', () => {
    mountComponent(FileThumb, { purpose: 'reference_image', path: '/pics/fox.png', kind: 'image' });

    expect(browseApiMocks.browsePreviewUrl.mock.calls[0][2]?.size).toBeUndefined();
  });

  it('starts over when it is given another file', async () => {
    const props = reactiveProps({ purpose: 'reference_image', path: '/pics/fox.png', kind: 'image' });
    mountComponent(FileThumb, props);
    flushSync(() => image()!.dispatchEvent(new Event('error')));
    expect(image()).toBeNull();

    props.path = '/pics/owl.png';
    flushSync();

    expect(image()?.getAttribute('src')).toBe('/preview/reference_image/pics/owl.png');
  });
});
