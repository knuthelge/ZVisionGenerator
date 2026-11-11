import { browseMediaUrl, browsePreviewUrl } from '$lib/api/browse';
import { fileName } from '$lib/components/molecules/assetDetails';
import type { GalleryAsset } from '$lib/types';
import { detectCutout } from './paintCanvas';

/** The image being edited: where it comes from, how to draw it and how big it is. */
export interface EditSource {
  /** Changes with every choice, so a late answer about an earlier image is told apart. */
  id: number;
  name: string;
  /** The file's path on the server's machine; null for a file uploaded from this device. */
  path: string | null;
  file: File | null;
  /** The picture the painter draws, at its full size. */
  url: string;
  /** A small picture for the section's thumbnail. */
  thumbUrl: string;
  /** The real size in pixels; null until it is known. */
  width: number | null;
  height: number | null;
  /** The picture has transparent pixels, so Extend does not apply. Null until it has been looked at. */
  cutout: boolean | null;
}

let _nextId = 1;

/** A file on the server's machine, chosen with the browser; the painter draws the file itself, the thumbnail is a preview. Its size is read from the loaded picture (`readPicture`). */
export function sourceOfPath(path: string): EditSource {
  return {
    id: _nextId++,
    name: fileName(path),
    path,
    file: null,
    url: browseMediaUrl('reference_image', path),
    thumbUrl: browsePreviewUrl('reference_image', path),
    width: null,
    height: null,
    cutout: null,
  };
}

/** A file chosen from this device. */
export function sourceOfFile(file: File): EditSource {
  const url = URL.createObjectURL(file);
  return { id: _nextId++, name: file.name, path: null, file, url, thumbUrl: url, width: null, height: null, cutout: null };
}

/**
 * An image from the gallery. Its size is read from the loaded picture (`readPicture`), as for a browsed file: the size the gallery
 * recorded can be the raw one, before the picture is turned upright, or out of date. A cut-out the gallery marks as one
 * (`transparent`) is known; another is looked at later.
 */
export function sourceOfAsset(asset: GalleryAsset): EditSource {
  return {
    id: _nextId++,
    name: asset.filename,
    path: asset.file_path ?? null,
    file: null,
    url: asset.url,
    thumbUrl: asset.thumbnail_url || asset.url,
    width: null,
    height: null,
    cutout: asset.transparent === true ? true : null,
  };
}

/** Load the picture at `url`; null when the browser cannot decode it. */
function loadImage(url: string): Promise<HTMLImageElement | null> {
  return new Promise((resolve) => {
    const image = new Image();
    image.onload = () => resolve(image.naturalWidth > 0 ? image : null);
    image.onerror = () => resolve(null);
    image.src = url;
  });
}

/**
 * What a loaded picture tells: its size (the browser reports `naturalWidth` and `naturalHeight` after applying the EXIF orientation,
 * so a photo taken sideways reads upright) and whether it has transparent pixels (a cut-out, looked for on a small copy). Null when
 * the browser cannot decode it.
 */
export async function readPicture(source: EditSource): Promise<{ width: number; height: number; cutout: boolean | null } | null> {
  const image = await loadImage(source.url);
  return image ? { width: image.naturalWidth, height: image.naturalHeight, cutout: detectCutout(image) } : null;
}
