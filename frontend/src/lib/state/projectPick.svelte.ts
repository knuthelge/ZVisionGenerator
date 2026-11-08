import type { GalleryAsset } from '$lib/types';

/** Where this browser keeps its pick; the server knows nothing of it, so two browsers can pick differently. */
const STORAGE_KEY = 'ziv-project-v1';

function readStored(): string | null {
  try {
    return localStorage.getItem(STORAGE_KEY) || null;
  } catch {
    return null;
  }
}

function writeStored(name: string | null): void {
  try {
    if (name === null) localStorage.removeItem(STORAGE_KEY);
    else localStorage.setItem(STORAGE_KEY, name);
  } catch {
    // Storage is unavailable (private mode, quota): the pick lives for this page only.
  }
}

let _current = $state<string | null>(readStored());

/**
 * The project this browser works in: a name, or null for All files. Only the projects store changes it, through
 * `choose`, which also reloads what the pick scopes.
 */
export const projectPick = {
  get current(): string | null { return _current; },

  set(name: string | null): void {
    _current = name || null;
    writeStored(_current);
  },
};

/** Whether an asset belongs to a pick: everything belongs to All files (null), otherwise only that project's files. */
export function inProject(asset: Pick<GalleryAsset, 'project'>, pick: string | null): boolean {
  return pick === null || asset.project === pick;
}
