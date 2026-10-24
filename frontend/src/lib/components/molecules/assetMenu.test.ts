import { describe, expect, it, vi } from 'vitest';

import type { AssetUpscaleFactor, GalleryAsset } from '$lib/types';

import { assetMenuEntries, referenceEntries, upscaleEntries } from './assetMenu';

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
    const sizes = [option(2), option(4), option(2), option(4)].map((entry) => `${entry.width}×${entry.height}`);
    expect(labels.map((label, index) => label.includes(sizes[index]))).toEqual([true, true, true, true]);
    expect(new Set(labels).size).toBe(4);
  });

  it('offers a video one SeedVR2 item, marked when it is over the memory estimate, and no creative ones', () => {
    const video = { ...makeAsset({ factors: [option(2)], seedvr2: seedvr2([option(2, { warning: 'May need more memory.' })]) }), media_type: 'video' as const };
    const entries = upscaleEntries(video, vi.fn());

    expect(kinds(entries)).toEqual(['upscale-seedvr2-2']);
    expect(entries[0]).toMatchObject({ title: 'May need more memory.' });
    expect(kinds(assetMenuEntries(video, { onupscale: vi.fn() }))).toEqual(['heading', 'upscale-seedvr2-2', 'separator', 'download']);
    expect(upscaleEntries({ ...video, upscale: null }, vi.fn())).toEqual([]);
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

  it('marks a one-frame-batch warning differently from a memory warning, by the warning code', () => {
    const label = (warning_code: 'memory' | 'one_frame_batch') => {
      const [entry] = upscaleEntries(makeAsset({ factors: [], seedvr2: seedvr2([option(2, { warning: 'w', warning_code })]) }), vi.fn());
      return entry.kind === 'item' ? entry.label : '';
    };

    expect(label('one_frame_batch')).not.toBe(label('memory'));
  });

  it('keeps a warned SeedVR2 item enabled, marks its label and puts the warning in its tooltip', () => {
    const warning = 'a warning from the server';
    const [unwarned] = upscaleEntries(makeAsset({ factors: [], seedvr2: seedvr2([option(4)]) }), vi.fn());
    const [warned] = upscaleEntries(makeAsset({ factors: [], seedvr2: seedvr2([option(4, { warning })]) }), vi.fn());

    expect(warned).toMatchObject({ id: 'upscale-seedvr2-4', title: warning });
    expect(warned).not.toMatchObject({ disabled: true });
    expect(unwarned).toMatchObject({ title: undefined });
    expect(warned.kind === 'item' && unwarned.kind === 'item' && warned.label.startsWith(unwarned.label) && warned.label !== unwarned.label).toBe(true);
  });

  it('names the first-use download in the SeedVR2 tooltips only while the model is not downloaded', () => {
    const warning = 'a warning from the server';
    const sizeLabel = '17.0 GB';
    const pending = seedvr2([option(2), option(4, { warning })], { downloaded: false, download_size_label: sizeLabel });
    const [first, second, , creative] = upscaleEntries(makeAsset({ factors: [option(2)], seedvr2: pending }), vi.fn());

    expect(first).toMatchObject({ title: expect.stringContaining(sizeLabel) });
    expect(second).toMatchObject({ title: expect.stringContaining(sizeLabel) });
    expect(second).toMatchObject({ title: expect.stringContaining(warning) });
    expect(first).not.toMatchObject({ title: expect.stringContaining(warning) });
    expect(creative).toMatchObject({ title: undefined });
    const [downloaded] = upscaleEntries(makeAsset({ factors: [], seedvr2: seedvr2([option(2)]) }), vi.fn());
    expect(downloaded).toMatchObject({ title: undefined });
  });

  it('disables a creative item over a limit and explains it in the tooltip', () => {
    const blocked = option(4, { allowed: false, reason: 'Over the limit.' });
    const entries = upscaleEntries(makeAsset({ factors: [option(2), blocked], seedvr2: null }), vi.fn());

    expect(entries[1]).toMatchObject({ id: 'upscale-creative-4', disabled: true, title: blocked.reason });
    expect(entries[0]).toMatchObject({ disabled: false });
  });
});

describe('referenceEntries', () => {
  const asset = makeAsset(undefined);
  const ids = (entries: ReturnType<typeof referenceEntries>) => entries.flatMap((entry) => (entry.kind === 'item' ? [entry.id] : []));
  const labels = (entries: ReturnType<typeof referenceEntries>) => entries.flatMap((entry) => (entry.kind === 'item' ? [entry.label] : []));

  it('offers an image and a video target by default', () => {
    expect(ids(referenceEntries(asset, vi.fn()))).toEqual(['reference-image', 'reference-video']);
    expect(labels(referenceEntries(asset, vi.fn()))).toEqual(['For image (img2img)', 'For video (img2vid)']);
  });

  it('adds the last frame, and names the video target the first frame, when the workspace has the slot', () => {
    const entries = referenceEntries(asset, vi.fn(), {}, ['image', 'video', 'last']);
    expect(ids(entries)).toEqual(['reference-image', 'reference-video', 'reference-last-frame']);
    expect(labels(entries)).toEqual(['For image (img2img)', 'For first frame (img2vid)', 'For last frame (img2vid)']);
  });

  it('reports the chosen target', () => {
    const onreference = vi.fn();
    for (const entry of referenceEntries(asset, onreference, {}, ['image', 'video', 'last'])) if (entry.kind === 'item') entry.onselect?.();
    expect(onreference.mock.calls.map(([, target]) => target)).toEqual(['image', 'video', 'last']);
  });

  it('carries the targets through the asset menu', () => {
    const usable = { ...asset, file_path: '/outputs/out/a.png' };
    const menu = assetMenuEntries(usable, { onreference: vi.fn() }, false, {}, ['image', 'video', 'last']);
    expect(ids(menu)).toContain('reference-last-frame');
    expect(ids(assetMenuEntries(usable, { onreference: vi.fn() }))).not.toContain('reference-last-frame');
  });
});
