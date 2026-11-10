import { flushSync } from 'svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { sourceInfo } from '$lib/state/extend.fixtures';
import { videoSprite } from '$lib/state/videoSprite.svelte';
import { mountComponent, settle } from '../../test-utils/mount';
import { reactiveProps } from '../../test-utils/reactiveProps.svelte';
import ExtendSourceThumb from './ExtendSourceThumb.svelte';

vi.mock('$lib/api/browse', async () => (await import('../../test-utils/workspaceMocks')).browseApiMocks);

const PATH = '/clips/walk.mp4';
const info = sourceInfo({ fps: 24, frame_count: 144, video_start: 0, extend_frames: 48 });

describe('ExtendSourceThumb', () => {
  beforeEach(() => {
    videoSprite.reset();
    // A sheet of 11 x 11 tiles of 160 x 90 loads at once; the thumb is 76 x 44.
    class SheetImage {
      naturalWidth = 1760;
      naturalHeight = 990;
      onload: (() => void) | null = null;
      onerror: (() => void) | null = null;
      set src(_url: string) { queueMicrotask(() => this.onload?.()); }
    }
    vi.stubGlobal('Image', SheetImage);
    vi.spyOn(HTMLElement.prototype, 'clientWidth', 'get').mockReturnValue(76);
    vi.spyOn(HTMLElement.prototype, 'clientHeight', 'get').mockReturnValue(44);
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  // The Extend bar next to the thumb asks for the sheet; the thumb only draws from it.
  const withSheet = async () => {
    videoSprite.ensure(PATH);
    await settle();
  };
  const at = (frame: number) => `/preview/source_video/clips/walk.mp4?at=${Math.round((frame / 24) * 1e6) / 1e6}`;
  const thumb = () => document.body.querySelector<HTMLElement>('[data-testid="extend-source-thumb"]')!;
  const picture = () => thumb().querySelector('img')?.getAttribute('src');
  const tile = () => thumb().querySelector<HTMLElement>('[data-testid="extend-thumb-tile"]');

  function show(props: Record<string, unknown>) {
    const reactive = reactiveProps<Record<string, unknown>>({ path: PATH, info, extendFrom: null, dragKeep: null, ...props });
    mountComponent(ExtendSourceThumb, reactive);
    return reactive;
  }

  const last = '/preview/source_video/clips/walk.mp4?frame=last';

  it('shows the last frame of the clip when it is kept whole, and when the source is not described yet', () => {
    show({});
    expect(picture()).toBe(last);

    document.body.innerHTML = '';
    show({ info: null });
    expect(picture()).toBe(last);
  });

  it('asks for the last frame once when the source is described after the pick', () => {
    const props = show({ info: null });
    const before = thumb().querySelector('img');
    expect(picture()).toBe(last);

    props.info = info;
    flushSync();

    expect(thumb().querySelector('img')).toBe(before);
    expect(picture()).toBe(last);
  });

  it('shows the exact frame at the point, asked for by its time', () => {
    show({ extendFrom: 76 / 24 });
    expect(picture()).toBe(at(76));
  });

  it('shows a tile of the contact sheet while the handle is dragged, and asks for no other picture', async () => {
    await withSheet();
    const props = show({ extendFrom: 76 / 24 });
    const before = picture();
    // Settle on the exact frame first, as it has arrived by the time a drag starts.
    thumb().querySelector('img')!.dispatchEvent(new Event('load'));
    flushSync();
    expect(tile()).toBeNull();

    props.dragKeep = 100;
    flushSync();

    expect(tile()).not.toBeNull();
    expect(tile()!.style.backgroundImage).toContain('/sprite/source_video/clips/walk.mp4');
    expect(picture()).toBe(before);
  });

  it('keeps the tile after the release until the exact frame has arrived', async () => {
    await withSheet();
    const props = show({ extendFrom: 76 / 24 });
    thumb().querySelector('img')!.dispatchEvent(new Event('load'));
    flushSync();

    props.dragKeep = 100;
    flushSync();
    props.extendFrom = 99 / 24;
    props.dragKeep = null;
    flushSync();

    expect(picture()).toBe(at(99));
    expect(tile()).not.toBeNull();

    thumb().querySelector('img')!.dispatchEvent(new Event('load'));
    flushSync();
    expect(tile()).toBeNull();
  });

  it('keeps the picture it had while dragging without a contact sheet', () => {
    const props = show({ extendFrom: 76 / 24 });
    props.dragKeep = 100;
    flushSync();

    expect(tile()).toBeNull();
    expect(picture()).toBe(at(76));
  });
});
