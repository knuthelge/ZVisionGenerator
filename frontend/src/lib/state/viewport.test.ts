import { afterEach, describe, expect, it, vi } from 'vitest';
import { viewportBox, watchViewport } from './viewport';

describe('viewportBox', () => {
  it('fills the visible area', () => {
    expect(viewportBox({ height: 512.4, offsetTop: 0, scale: 1 })).toEqual({ height: 512, top: 0 });
  });

  it('follows the visible area down when the browser scrolled it to show a field', () => {
    expect(viewportBox({ height: 400, offsetTop: 210.6, scale: 1 })).toEqual({ height: 400, top: 211 });
  });

  it('keeps the last size while the page is pinch-zoomed', () => {
    expect(viewportBox({ height: 200, offsetTop: 90, scale: 2 })).toBeNull();
  });
});

/** A visual viewport the test can resize, with the listeners it was given. */
function fakeViewport(height: number) {
  const listeners = new Map<string, Set<() => void>>();
  const viewport = {
    height,
    offsetTop: 0,
    scale: 1,
    addEventListener: (type: string, listener: () => void) => {
      if (!listeners.has(type)) listeners.set(type, new Set());
      listeners.get(type)!.add(listener);
    },
    removeEventListener: (type: string, listener: () => void) => { listeners.get(type)?.delete(listener); },
    fire: (type: string) => listeners.get(type)?.forEach((listener) => listener()),
  };
  return viewport;
}

describe('watchViewport', () => {
  let stop: () => void = () => {};

  afterEach(() => {
    stop();
    document.documentElement.style.removeProperty('--app-height');
    document.documentElement.style.removeProperty('--app-top');
    document.body.innerHTML = '';
    vi.restoreAllMocks();
  });

  function watch(viewport: ReturnType<typeof fakeViewport>, scrollY = 0) {
    const scrollTo = vi.fn();
    const win = { visualViewport: viewport, scrollY, scrollX: 0, scrollTo, document, addEventListener: vi.fn(), removeEventListener: vi.fn() };
    stop = watchViewport(win as unknown as Window);
    return { scrollTo };
  }

  it('sizes the app to the visible area and resizes with it', () => {
    const viewport = fakeViewport(800);
    watch(viewport);
    expect(document.documentElement.style.getPropertyValue('--app-height')).toBe('800px');
    viewport.height = 450;
    viewport.fire('resize');
    expect(document.documentElement.style.getPropertyValue('--app-height')).toBe('450px');
  });

  it('puts a scrolled page back at the top when no field is being edited', () => {
    const { scrollTo } = watch(fakeViewport(800), 120);
    expect(scrollTo).toHaveBeenCalledWith(0, 0);
  });

  it('leaves the page where it is while a field is being edited', () => {
    const field = document.createElement('textarea');
    document.body.append(field);
    field.focus();
    const { scrollTo } = watch(fakeViewport(400), 120);
    expect(scrollTo).not.toHaveBeenCalled();
  });
});
