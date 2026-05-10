import type { GalleryAsset } from '$lib/types';
import { getHistory } from '$lib/api/workspace';

let _history = $state<GalleryAsset[]>([]);
let _loading = $state(false);

export const historyStore = {
  get assets(): GalleryAsset[] { return _history; },
  get loading(): boolean { return _loading; },

  seedHistory(assets: GalleryAsset[]): void {
    _history = assets;
  },

  async refreshHistory(): Promise<void> {
    _loading = true;
    try {
      const page = await getHistory(1);
      _history = page.assets;
    } catch {
      // ignore refresh errors
    } finally {
      _loading = false;
    }
  }
};
