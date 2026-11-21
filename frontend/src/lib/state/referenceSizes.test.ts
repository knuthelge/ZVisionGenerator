import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { referenceSizes } from './referenceSizes.svelte';
import type { ReferenceRow } from './references';

/** A stand-in for `Image` that answers at once with the size its source names: `size:WxH`, or an error for anything else. */
class FakeImage {
  naturalWidth = 0;
  naturalHeight = 0;
  onload: (() => void) | null = null;
  onerror: (() => void) | null = null;
  set src(url: string) {
    const match = /size[:-](\d+)x(\d+)/.exec(url);
    queueMicrotask(() => {
      if (!match) return this.onerror?.();
      this.naturalWidth = Number(match[1]);
      this.naturalHeight = Number(match[2]);
      this.onload?.();
    });
  }
}

const row = (path: string, file: File | null = null): ReferenceRow => ({ path, file });

describe('referenceSizes', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.stubGlobal('Image', FakeImage);
    URL.createObjectURL = vi.fn((file: Blob) => `blob:${(file as File).name}`);
    URL.revokeObjectURL = vi.fn();
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  const upload = (name: string, modified = 1) => row('', new File(['x'], name, { lastModified: modified }));

  it('is ready at once when there are no references, and skips an empty row', () => {
    expect(referenceSizes.of([])).toEqual({ sizes: [], paths: [], ready: true });
    expect(referenceSizes.of([row('')])).toEqual({ sizes: [], paths: [], ready: true });
  });

  it('sends a file on the server by its path, reads nothing and is ready, so the picture never travels here', () => {
    const created = vi.fn();
    vi.stubGlobal('Image', class { constructor() { created(); } });

    expect(referenceSizes.of([row('/pics/a.png'), row('/pics/b.png')])).toEqual({ sizes: [], paths: ['/pics/a.png', '/pics/b.png'], ready: true });
    expect(created).not.toHaveBeenCalled();
  });

  it('measures an upload once from an object URL, is not ready until it has, then gives its size and lets the URL go', async () => {
    const rows = [upload('size-1000x500')];

    expect(referenceSizes.of(rows)).toEqual({ sizes: [], paths: [], ready: false });
    await vi.advanceTimersByTimeAsync(0);

    expect(referenceSizes.of(rows)).toEqual({ sizes: [{ width: 1000, height: 500 }], paths: [], ready: true });
    expect(URL.createObjectURL).toHaveBeenCalledOnce();
    expect(URL.revokeObjectURL).toHaveBeenCalledOnce();
  });

  it('keeps the order of the uploads and sorts the paths out beside them', async () => {
    const rows = [upload('a-size-100x200'), row('/pics/b.png'), upload('c-size-300x400')];
    referenceSizes.of(rows);
    await vi.advanceTimersByTimeAsync(0);

    expect(referenceSizes.of(rows)).toEqual({ sizes: [{ width: 100, height: 200 }, { width: 300, height: 400 }], paths: ['/pics/b.png'], ready: true });
  });

  it('leaves out an upload the browser cannot read and does not read it again, however long it stays', async () => {
    const broken = upload('broken-picture');
    referenceSizes.of([broken]);
    await vi.advanceTimersByTimeAsync(0);

    expect(referenceSizes.of([broken])).toEqual({ sizes: [], paths: [], ready: true });
    await vi.advanceTimersByTimeAsync(10 * 60_000);

    expect(referenceSizes.of([broken])).toEqual({ sizes: [], paths: [], ready: true });
    expect(URL.createObjectURL).toHaveBeenCalledTimes(1);
  });

  it('reads a changed file again: it has a new key, unlike the one that failed', async () => {
    referenceSizes.of([upload('swap-picture', 1)]);
    await vi.advanceTimersByTimeAsync(0);
    expect(referenceSizes.of([upload('swap-picture', 1)])).toEqual({ sizes: [], paths: [], ready: true });

    const changed = upload('swap-picture-size-640x480', 2);
    expect(referenceSizes.of([changed])).toEqual({ sizes: [], paths: [], ready: false });
    await vi.advanceTimersByTimeAsync(0);

    expect(referenceSizes.of([changed])).toEqual({ sizes: [{ width: 640, height: 480 }], paths: [], ready: true });
    expect(URL.createObjectURL).toHaveBeenCalledTimes(2);
  });
});
