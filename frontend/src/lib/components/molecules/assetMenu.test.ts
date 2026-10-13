import { describe, expect, it, vi } from 'vitest';

import type { AssetUpscaleFactor, GalleryAsset } from '$lib/types';

import { upscaleEntries } from './assetMenu';

function option(factor: 2 | 4, overrides: Partial<AssetUpscaleFactor> = {}): AssetUpscaleFactor {
  return { factor, width: 832 * factor, height: 1216 * factor, allowed: true, reason: null, warning: null, ...overrides };
}

function makeAsset(upscale: GalleryAsset['upscale']): GalleryAsset {
  return {
    id: 'out/a.png', url: '/media/out/a.png', thumbnail_url: '/media/out/a.png', filename: 'a.png', created_at: '2026-10-03T12:00:00Z',
    workflow: 'txt2img', prompt: 'a fox', model: 'zit', media_type: 'image', reuse_workspace_url: '', upscale,
  };
}

const seedvr2 = (options: AssetUpscaleFactor[], overrides = {}) => ({
  options, model: 'SeedVR2 7B q8', downloaded: true, download_size_label: null, ...overrides,
});

const kinds = (entries: ReturnType<typeof upscaleEntries>) => entries.map((entry) => (entry.kind === 'item' ? entry.id : entry.kind));

describe('upscaleEntries', () => {
  it('lists the SeedVR2 items, a separator, then the creative items', () => {
    const asset = makeAsset({ factors: [option(2), option(4)], seedvr2: seedvr2([option(2), option(4)]) });
    const entries = upscaleEntries(asset, vi.fn());

    expect(kinds(entries)).toEqual(['upscale-seedvr2-2', 'upscale-seedvr2-4', 'separator', 'upscale-creative-2', 'upscale-creative-4']);
    const labels = entries.flatMap((entry) => (entry.kind === 'item' ? [entry.label] : []));
    expect(labels).toEqual(['2× → 1664×2432', '4× → 3328×4864', '2× creative → 1664×2432', '4× creative → 3328×4864']);
  });

  it('offers only the creative items, with no separator, where SeedVR2 is unavailable', () => {
    const entries = upscaleEntries(makeAsset({ factors: [option(2), option(4)], seedvr2: null }), vi.fn());
    expect(kinds(entries)).toEqual(['upscale-creative-2', 'upscale-creative-4']);
  });

  it('reports each item with its method', () => {
    const onupscale = vi.fn();
    const asset = makeAsset({ factors: [option(2), option(4)], seedvr2: seedvr2([option(2), option(4)]) });
    for (const entry of upscaleEntries(asset, onupscale)) if (entry.kind === 'item') entry.onselect?.();

    expect(onupscale.mock.calls.map(([, factor, method]) => [factor, method])).toEqual([[2, 'seedvr2'], [4, 'seedvr2'], [2, 'creative'], [4, 'creative']]);
  });

  it('keeps a warned SeedVR2 item enabled, marks its label and puts the warning in its tooltip', () => {
    const warned = option(4, { warning: 'May need more memory than this Mac has.' });
    const [, second] = upscaleEntries(makeAsset({ factors: [], seedvr2: seedvr2([option(2), warned]) }), vi.fn());

    expect(second).toMatchObject({ id: 'upscale-seedvr2-4', label: '4× → 3328×4864 · high memory', title: 'May need more memory than this Mac has.' });
    expect(second).not.toMatchObject({ disabled: true });
  });

  it('names the first-use download in the SeedVR2 tooltips only while the model is not downloaded', () => {
    const warned = option(4, { warning: 'May need more memory than this Mac has.' });
    const pending = seedvr2([option(2), warned], { downloaded: false, download_size_label: '17.0 GB' });
    const [first, second, , creative] = upscaleEntries(makeAsset({ factors: [option(2)], seedvr2: pending }), vi.fn());

    expect(first).toMatchObject({ title: 'First use downloads 17.0 GB' });
    expect(second).toMatchObject({ title: 'May need more memory than this Mac has.\nFirst use downloads 17.0 GB' });
    expect(creative).toMatchObject({ title: undefined });
    const [downloaded] = upscaleEntries(makeAsset({ factors: [], seedvr2: seedvr2([option(2)]) }), vi.fn());
    expect(downloaded).toMatchObject({ title: undefined });
  });

  it('disables a creative item over a limit and explains it in the tooltip', () => {
    const blocked = option(4, { allowed: false, reason: 'Over the limit.' });
    const entries = upscaleEntries(makeAsset({ factors: [option(2), blocked], seedvr2: null }), vi.fn());

    expect(entries[1]).toMatchObject({ id: 'upscale-creative-4', disabled: true, title: 'Over the limit.' });
    expect(entries[0]).toMatchObject({ disabled: false });
  });
});
