import type { AdjustBody, AdjustSave } from '$lib/api/adjust';
import { normalizedCrop, type Aspect, type Geometry, type Size } from '$lib/state/adjustGeometry';
import { isNeutralLight, NEUTRAL_LIGHT, type LightSettings } from '$lib/state/adjustMath';
import type { GalleryAsset } from '$lib/types';

/** What the Adjust editor holds: the sliders, the crop, rotation and flip, and the chosen shape chip. */
export interface AdjustDraft {
  light: LightSettings;
  geometry: Geometry;
  aspect: Aspect;
}

/** A draft that changes nothing: neutral sliders, no rotation, no flip, no crop and a free shape. */
export function neutralDraft(): AdjustDraft {
  return {
    light: { ...NEUTRAL_LIGHT },
    geometry: { rotate: 0, flip: false, crop: null },
    aspect: { preset: 'free', ratio: null },
  };
}

/**
 * Whether the draft differs from the original picture. The shape chip does not count: it only steers the crop box.
 * A crop that covers the whole frame is no change, which needs the displayed frame to tell; without it any crop counts.
 */
export function draftChanged(draft: AdjustDraft, frame?: Size): boolean {
  const { geometry, light } = draft;
  const crop = frame ? normalizedCrop(geometry.crop, frame) : geometry.crop;
  return !isNeutralLight(light) || geometry.rotate !== 0 || geometry.flip || crop !== null;
}

/**
 * Build the body of the adjust request. A crop that covers the whole displayed frame is sent as null.
 *
 * @param frame The displayed frame's size in pixels: the upright size, swapped for a quarter turn (`frameSize`).
 */
export function adjustBody(asset: Pick<GalleryAsset, 'id'>, draft: AdjustDraft, save: AdjustSave, frame: Size): AdjustBody {
  const { light, geometry } = draft;
  return {
    asset_id: asset.id,
    save,
    rotate: geometry.rotate,
    flip: geometry.flip,
    crop: normalizedCrop(geometry.crop, frame),
    exposure: light.exposure,
    contrast: light.contrast,
    highlights: light.highlights,
    shadows: light.shadows,
    saturation: light.saturation,
    vibrance: light.vibrance,
    warmth: light.warmth,
    mix_red: light.mix_red,
    mix_orange: light.mix_orange,
    mix_yellow: light.mix_yellow,
    mix_green: light.mix_green,
    mix_aqua: light.mix_aqua,
    mix_blue: light.mix_blue,
    mix_purple: light.mix_purple,
    mix_magenta: light.mix_magenta,
  };
}

/** Whether an asset can be adjusted: an image whose file the server can find. */
export function canAdjust(asset: Pick<GalleryAsset, 'media_type' | 'file_path'>): boolean {
  return asset.media_type === 'image' && !!asset.file_path;
}

/** Whether an adjusted picture can replace the file: only a PNG, which the server rewrites without loss. */
export function canReplace(asset: Pick<GalleryAsset, 'filename'>): boolean {
  return /\.png$/i.test(asset.filename);
}

/** An adjusted copy the Gallery viewer must show even though the loaded list does not hold it, and the asset it sits beside. */
export interface PinnedCopy {
  asset: GalleryAsset;
  /** The id of the source it goes right after; null puts it first. */
  besideId: string | null;
}

/**
 * Return the list with the pinned copy in it: right after `besideId` (first when that is null or not in the list). The
 * list itself comes back when nothing is pinned or it already holds the copy, so an unchanged list keeps its identity.
 */
export function withAdjustedCopy(assets: readonly GalleryAsset[], pinned: PinnedCopy | null): readonly GalleryAsset[] {
  if (pinned === null || assets.some((asset) => asset.id === pinned.asset.id)) return assets;
  const at = pinned.besideId === null ? -1 : assets.findIndex((asset) => asset.id === pinned.besideId);
  return [...assets.slice(0, at + 1), pinned.asset, ...assets.slice(at + 1)];
}
