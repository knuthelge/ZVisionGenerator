import { SvelteMap } from 'svelte/reactivity';

import type { ReferenceRow } from './references';

/** A picture's size in pixels. */
export interface PictureSize {
  width: number;
  height: number;
}

/** How long a picture the browser could not read is left out before it is tried again. */
export const UNREADABLE_RETRY_MS = 30_000;

/** The sizes of the pictures looked at. A picture still being read, or one whose read failed, has no entry. */
const _sizes = new SvelteMap<string, PictureSize>();
const _loading = new Set<string>();
// A read that failed, when it did. It is never kept as a size: after `UNREADABLE_RETRY_MS` the picture is read again.
const _failedAt = new SvelteMap<string, number>();

/** What identifies a picture chosen from this device. */
function keyOf(file: File): string {
  return `file:${file.name}:${file.size}:${file.lastModified}`;
}

/** Load the picture at `url` and report its size, as the browser sees it (after the EXIF turn); null when it cannot be decoded. */
function readSize(url: string): Promise<PictureSize | null> {
  return new Promise((resolve) => {
    const image = new Image();
    image.onload = () => resolve(image.naturalWidth > 0 ? { width: image.naturalWidth, height: image.naturalHeight } : null);
    image.onerror = () => resolve(null);
    image.src = url;
  });
}

async function load(key: string, file: File): Promise<void> {
  const url = URL.createObjectURL(file);
  try {
    const size = await readSize(url);
    if (size) {
      _sizes.set(key, size);
      _failedAt.delete(key);
    } else {
      _failedAt.set(key, Date.now());
    }
  } finally {
    URL.revokeObjectURL(url);
    _loading.delete(key);
  }
}

/** What the memory check is told of the reference pictures a submit sends. */
export interface ReferenceFacts {
  /** The size of each picture chosen from this device, measured here: the file is local, so nothing travels. A picture the browser cannot read is left out (the job refuses it). */
  sizes: PictureSize[];
  /** The path of each picture that is a file on the server's machine: the server reads its header, so the picture never travels to this browser. */
  paths: string[];
  /** Every upload has been looked at. */
  ready: boolean;
}

/** The real size of each reference picture, which the memory check judges them at; read once per picture. */
export const referenceSizes = {
  /**
   * Sort the rows a submit sends into the sizes measured here and the paths the server measures. Reading starts for an upload not yet looked at, and a
   * reactive caller is run again when its size arrives. An upload whose read failed is left out for `UNREADABLE_RETRY_MS`, then read again.
   */
  of(rows: readonly ReferenceRow[]): ReferenceFacts {
    const facts: ReferenceFacts = { sizes: [], paths: [], ready: true };
    for (const row of rows) {
      if (!row.file) {
        if (row.path !== '') facts.paths.push(row.path);
        continue;
      }
      const key = keyOf(row.file);
      const size = _sizes.get(key);
      if (size) {
        facts.sizes.push(size);
        continue;
      }
      const failedAt = _failedAt.get(key);
      if (failedAt !== undefined && Date.now() - failedAt < UNREADABLE_RETRY_MS) continue;
      facts.ready = false;
      if (!_loading.has(key)) {
        _loading.add(key);
        void load(key, row.file);
      }
    }
    return facts;
  },
};
