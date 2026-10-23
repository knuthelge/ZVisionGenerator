import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { SettingsPreset } from '$lib/types';

const api = vi.hoisted(() => ({
  listPresets: vi.fn<() => Promise<SettingsPreset[]>>(),
  createPreset: vi.fn<(body: unknown) => Promise<SettingsPreset>>(),
  updatePreset: vi.fn<(mode: SettingsPreset['mode'], slug: string, body: unknown) => Promise<SettingsPreset>>(),
  renamePreset: vi.fn<(mode: SettingsPreset['mode'], slug: string, name: string) => Promise<SettingsPreset>>(),
  deletePreset: vi.fn<(mode: SettingsPreset['mode'], slug: string) => Promise<{ status: string }>>(),
}));

vi.mock('$lib/api/presets', () => api);

const preset = (slug: string, name: string, mode: SettingsPreset['mode'] = 'image'): SettingsPreset => ({ slug, name, mode, model: null, settings: {} });
const body = (name: string, model: string | null = null) => ({ name, mode: 'image' as const, model, settings: {} });

// A fresh module per test gives each an empty list.
let presetsStore: typeof import('./presets.svelte').presetsStore;

beforeEach(async () => {
  vi.resetModules();
  for (const mock of Object.values(api)) mock.mockReset();
  ({ presetsStore } = await import('./presets.svelte'));
});

describe('presetsStore', () => {
  it('loads the list sorted by name, both modes', async () => {
    api.listPresets.mockResolvedValue([preset('zeta', 'zeta'), preset('alpha-clip', 'Alpha clip', 'video'), preset('beta', 'Beta')]);

    await presetsStore.load();

    expect(presetsStore.list.map((item) => item.slug)).toEqual(['alpha-clip', 'beta', 'zeta']);
  });

  it('keeps the list when a read fails', async () => {
    api.listPresets.mockResolvedValueOnce([preset('beta', 'Beta')]);
    await presetsStore.load();
    api.listPresets.mockRejectedValueOnce(new Error('offline'));

    await presetsStore.load();

    expect(presetsStore.list.map((item) => item.slug)).toEqual(['beta']);
  });

  it('adds a created preset and replaces an updated one under its slug', async () => {
    api.createPreset.mockResolvedValueOnce(preset('soft', 'Soft'));
    await presetsStore.create(body('Soft'));
    api.updatePreset.mockResolvedValueOnce({ ...preset('soft', 'Soft'), model: 'zit' });
    await presetsStore.update('image', 'soft', { name: 'Soft', model: 'zit', settings: {} });

    expect(presetsStore.list).toEqual([{ ...preset('soft', 'Soft'), model: 'zit' }]);
  });

  it('moves a renamed preset to its new slug, and drops a deleted one', async () => {
    api.listPresets.mockResolvedValue([preset('old-name', 'Old name'), preset('other', 'Other')]);
    await presetsStore.load();
    api.renamePreset.mockResolvedValue(preset('new-name', 'New name'));
    await presetsStore.rename('image', 'old-name', 'New name');
    expect(presetsStore.list.map((item) => item.slug)).toEqual(['new-name', 'other']);

    api.deletePreset.mockResolvedValue({ status: 'ok' });
    await presetsStore.remove('image', 'other');
    expect(presetsStore.list.map((item) => item.slug)).toEqual(['new-name']);
  });

  it('keeps same-slug presets of the other mode apart', async () => {
    api.listPresets.mockResolvedValue([preset('look', 'Look'), preset('look', 'Look', 'video')]);
    await presetsStore.load();
    api.updatePreset.mockResolvedValue({ ...preset('look', 'Look', 'video'), model: 'ltx-2' });
    await presetsStore.update('video', 'look', { name: 'Look', model: 'ltx-2', settings: {} });
    expect(presetsStore.list.map((item) => [item.mode, item.model])).toEqual([['image', null], ['video', 'ltx-2']]);

    api.deletePreset.mockResolvedValue({ status: 'ok' });
    await presetsStore.remove('image', 'look');
    expect(presetsStore.list.map((item) => item.mode)).toEqual(['video']);
  });

  it('lets a failed create, rename or delete reach the caller and leaves the list as it was', async () => {
    api.listPresets.mockResolvedValue([preset('keep', 'Keep')]);
    await presetsStore.load();
    api.createPreset.mockRejectedValue(new Error('409'));
    api.renamePreset.mockRejectedValue(new Error('409'));
    api.deletePreset.mockRejectedValue(new Error('500'));

    await expect(presetsStore.create(body('Keep'))).rejects.toThrow();
    await expect(presetsStore.rename('image', 'keep', 'Other')).rejects.toThrow();
    await expect(presetsStore.remove('image', 'keep')).rejects.toThrow();

    expect(presetsStore.list.map((item) => item.slug)).toEqual(['keep']);
  });
});
