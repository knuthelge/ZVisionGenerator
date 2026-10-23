import { api } from './client';
import type { PresetMode, SettingsPreset, SettingsPresetBody, SettingsPresetUpdate } from '$lib/types';

// Slugs are unique within a mode only, so every route of one preset names its mode.
function presetPath(mode: PresetMode, slug: string): string {
  return `/api/presets/${mode}/${encodeURIComponent(slug)}`;
}

/** Return every saved preset of both modes. */
export async function listPresets(): Promise<SettingsPreset[]> {
  const { presets } = await api.get<{ presets: SettingsPreset[] }>('/api/presets');
  return presets;
}

/** Create a preset; the server derives its slug from the name. */
export function createPreset(body: SettingsPresetBody): Promise<SettingsPreset> {
  return api.post<SettingsPreset>('/api/presets', body);
}

/** Overwrite the preset under *slug* of *mode*. */
export function updatePreset(mode: PresetMode, slug: string, body: SettingsPresetUpdate): Promise<SettingsPreset> {
  return api.put<SettingsPreset>(presetPath(mode, slug), body);
}

/** Give a preset a new name; the answer carries its new slug. */
export function renamePreset(mode: PresetMode, slug: string, name: string): Promise<SettingsPreset> {
  return api.post<SettingsPreset>(`${presetPath(mode, slug)}/rename`, { name });
}

export function deletePreset(mode: PresetMode, slug: string): Promise<{ status: string }> {
  return api.delete<{ status: string }>(presetPath(mode, slug));
}
