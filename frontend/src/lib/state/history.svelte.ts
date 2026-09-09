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

  mergeOutputs(assets: GalleryAsset[]): void {
    if (assets.length === 0) return;
    const newestFirst = [...assets].reverse();
    const incomingIds = new Set(newestFirst.map((asset) => asset.id));
    _history = [...newestFirst, ..._history.filter((asset) => !incomingIds.has(asset.id))];
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
