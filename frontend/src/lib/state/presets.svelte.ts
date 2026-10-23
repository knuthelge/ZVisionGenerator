// Module-level store of the saved settings presets, using Svelte 5 runes (the pattern of downloads.svelte.ts).
// The Workspace reads the list when it mounts and again each time the Preset menu opens, so presets saved or removed
// elsewhere (another browser) show up. The mutations throw: the caller shows the server's reason.

import { createPreset, deletePreset, listPresets, renamePreset, updatePreset } from '$lib/api/presets';
import type { PresetMode, SettingsPreset, SettingsPresetBody, SettingsPresetUpdate } from '$lib/types';

function byName(a: SettingsPreset, b: SettingsPreset): number {
  return a.name.toLowerCase().localeCompare(b.name.toLowerCase()) || a.slug.localeCompare(b.slug);
}

let _list = $state<SettingsPreset[]>([]);

function isPreset(preset: SettingsPreset, mode: PresetMode, slug: string): boolean {
  return preset.mode === mode && preset.slug === slug;
}

/** Replace the preset with the same mode and slug, or add it. */
function upsert(preset: SettingsPreset): void {
  _list = [..._list.filter((other) => !isPreset(other, preset.mode, preset.slug)), preset].sort(byName);
}

export const presetsStore = {
  /** Every preset of both modes, by name. */
  get list(): SettingsPreset[] { return _list; },

  /** Read the list. A failed read keeps the list as it was. */
  async load(): Promise<void> {
    try {
      _list = [...(await listPresets())].sort(byName);
    } catch {
      // The menu still works from what it has; the next open reads again.
    }
  },

  /** Create a preset; the server derives its slug. */
  async create(body: SettingsPresetBody): Promise<SettingsPreset> {
    const created = await createPreset(body);
    upsert(created);
    return created;
  },

  /** Overwrite the preset under *slug* of *mode*. */
  async update(mode: PresetMode, slug: string, body: SettingsPresetUpdate): Promise<SettingsPreset> {
    const updated = await updatePreset(mode, slug, body);
    upsert(updated);
    return updated;
  },

  /** Rename a preset; it moves to its new slug. */
  async rename(mode: PresetMode, slug: string, name: string): Promise<SettingsPreset> {
    const renamed = await renamePreset(mode, slug, name);
    _list = _list.filter((other) => !isPreset(other, mode, slug));
    upsert(renamed);
    return renamed;
  },

  /** Delete a preset. */
  async remove(mode: PresetMode, slug: string): Promise<void> {
    await deletePreset(mode, slug);
    _list = _list.filter((other) => !isPreset(other, mode, slug));
  },
};
