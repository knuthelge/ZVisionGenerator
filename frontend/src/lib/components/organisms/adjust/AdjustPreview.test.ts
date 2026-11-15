import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { flushSync } from 'svelte';

import { NEUTRAL_LIGHT } from '$lib/state/adjustMath';
import type { Geometry } from '$lib/state/adjustGeometry';
import { fakeGl } from '../../../../test-utils/fakeGl';
import { mountComponent, settle } from '../../../../test-utils/mount';
import { reactiveProps } from '../../../../test-utils/reactiveProps.svelte';
import AdjustPreview from './AdjustPreview.svelte';

class FakeImage {
  src = '';
  naturalWidth = 800;
  naturalHeight = 600;
  decode(): Promise<void> {
    return Promise.resolve();
  }
}

const GEOMETRY: Geometry = { rotate: 0, flip: false, crop: null };

describe('AdjustPreview with WebGL2', () => {
  let gl: ReturnType<typeof fakeGl>;

  beforeEach(() => {
    vi.useFakeTimers();
    vi.stubGlobal('Image', FakeImage);
    vi.spyOn(Element.prototype, 'clientWidth', 'get').mockReturnValue(640);
    vi.spyOn(Element.prototype, 'clientHeight', 'get').mockReturnValue(480);
    gl = fakeGl();
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(gl.gl as never);
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  function mountPreview(props: Record<string, unknown> = {}) {
    const onloaded = vi.fn();
    const mounted = mountComponent(AdjustPreview, {
      src: '/media/a.png',
      light: { ...NEUTRAL_LIGHT },
      geometry: GEOMETRY,
      region: 'crop',
      original: false,
      onloaded,
      onfail: vi.fn(),
      ...props,
    });
    return { ...mounted, onloaded };
  }

  it('reports the picture size, draws it on the canvas and shows no notice', async () => {
    const { onloaded } = mountPreview();
    await settle(20);
    expect(onloaded).toHaveBeenCalledWith({ width: 800, height: 600 });
    expect(document.querySelector('.adjust-notice')).toBeNull();
    expect(document.querySelector('[data-testid="adjust-fallback"]')).toBeNull();
    expect(document.querySelector<HTMLCanvasElement>('.pic-canvas')!.hidden).toBe(false);
    expect(gl.calls('drawArrays').length).toBeGreaterThan(0);
  });

  it('draws once for several changes made within one animation frame', async () => {
    const props = reactiveProps({
      src: '/media/a.png',
      light: { ...NEUTRAL_LIGHT },
      geometry: GEOMETRY,
      region: 'crop',
      original: false,
      onloaded: vi.fn(),
      onfail: vi.fn(),
    });
    mountComponent(AdjustPreview, props);
    await settle(20);
    const before = gl.calls('drawArrays').length;
    props.light = { ...NEUTRAL_LIGHT, exposure: 10 };
    flushSync();
    props.light = { ...NEUTRAL_LIGHT, exposure: 20 };
    flushSync();
    props.original = true;
    flushSync();
    await settle(20);
    expect(gl.calls('drawArrays').length).toBe(before + 1);
  });

  it('falls back to the turned picture when the context is lost and draws again when it is restored', async () => {
    mountPreview();
    await settle(20);
    const canvas = document.querySelector<HTMLCanvasElement>('.pic-canvas')!;
    const lost = new Event('webglcontextlost', { cancelable: true });
    canvas.dispatchEvent(lost);
    flushSync();
    expect(lost.defaultPrevented).toBe(true);
    expect(document.querySelector('.adjust-notice')).not.toBeNull();
    expect(document.querySelector('[data-testid="adjust-fallback"]')).not.toBeNull();
    expect(canvas.hidden).toBe(true);

    const drawn = gl.calls('drawArrays').length;
    canvas.dispatchEvent(new Event('webglcontextrestored'));
    await settle(20);
    expect(document.querySelector('.adjust-notice')).toBeNull();
    expect(canvas.hidden).toBe(false);
    expect(gl.calls('drawArrays').length).toBeGreaterThan(drawn);
  });

  it('uploads the picture scaled to the size it is shown at, and again only when that size changes', async () => {
    const props = reactiveProps({
      src: '/media/a.png',
      light: { ...NEUTRAL_LIGHT },
      geometry: GEOMETRY,
      region: 'crop',
      original: false,
      onloaded: vi.fn(),
      onfail: vi.fn(),
    });
    mountComponent(AdjustPreview, props);
    await settle(20);
    const uploads = (): unknown[][] => gl.calls('texImage2D').filter((call) => call.length === 6);
    // 800x600 in a 640x480 box shows at 640 px wide: a 640x480 texture, never the picture element itself.
    expect(uploads()).toHaveLength(1);
    expect(uploads()[0][5]).not.toBeInstanceOf(FakeImage);
    expect(gl.calls('drawImage').at(-1)?.slice(1)).toEqual([0, 0, 640, 480]);

    props.light = { ...NEUTRAL_LIGHT, exposure: 20 };
    await settle(20);
    expect(uploads()).toHaveLength(1);

    // Turned a quarter: the frame is 600x800 and fits the box at 360x480, so a 512x384 texture is enough.
    props.geometry = { ...GEOMETRY, rotate: 90 };
    await settle(20);
    expect(uploads()).toHaveLength(2);
    expect(gl.calls('drawImage').at(-1)?.slice(1)).toEqual([0, 0, 512, 384]);
  });

  it('uploads the picture again after the context is restored', async () => {
    mountPreview();
    await settle(20);
    const canvas = document.querySelector<HTMLCanvasElement>('.pic-canvas')!;
    const before = gl.calls('texImage2D').filter((call) => call.length === 6).length;
    canvas.dispatchEvent(new Event('webglcontextlost', { cancelable: true }));
    canvas.dispatchEvent(new Event('webglcontextrestored'));
    await settle(20);
    expect(gl.calls('texImage2D').filter((call) => call.length === 6)).toHaveLength(before + 1);
  });

  it('releases the GL context when it goes away', async () => {
    const { unmount } = mountPreview();
    await settle(20);
    await unmount();
    expect(gl.calls('loseContext')).toHaveLength(1);
  });
});
