import { api } from './client';
import type { LightSettings } from '$lib/state/adjustMath';
import type { Crop, Rotate } from '$lib/state/adjustGeometry';
import type { GalleryAsset } from '$lib/types';

/** Whether Adjust keeps the picture as a new file next to the original or writes over the original. */
export type AdjustSave = 'copy' | 'replace';

/** The body of `POST /api/adjust`: the file, how to save it and the whole adjustment (neutral values are sent too). */
export interface AdjustBody extends LightSettings {
  asset_id: string;
  save: AdjustSave;
  rotate: Rotate;
  flip: boolean;
  /** In pixels of the displayed frame; null for the whole frame. */
  crop: Crop | null;
}

/** Apply an adjustment on the server and return the asset it saved: the new copy, or the replaced file as it now is. */
export async function saveAdjusted(body: AdjustBody): Promise<GalleryAsset> {
  return (await api.post<{ asset: GalleryAsset }>('/api/adjust', body)).asset;
}
