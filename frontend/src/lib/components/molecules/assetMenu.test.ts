import { describe, expect, it, vi } from 'vitest';

import type { AssetUpscaleFactor, GalleryAsset } from '$lib/types';

import { makeAudioAsset, makeSpeechAsset } from '../../../test-utils/audio';
import { assetMenuEntries, audioUseEntries, referenceEntries, upscaleEntries } from './assetMenu';

function option(factor: 2 | 4, overrides: Partial<AssetUpscaleFactor> = {}): AssetUpscaleFactor {
  return { factor, width: 832 * factor, height: 1216 * factor, allowed: true, reason: null, warning: null, ...overrides };
}

function makeAsset(upscale: GalleryAsset['upscale']): GalleryAsset {
  return {
    id: 'out/a.png', url: '/media/out/a.png', thumbnail_url: '/media/out/a.png', filename: 'a.png', created_at: '2026-10-03T12:00:00Z', project: null,
    workflow: 'txt2img', prompt: 'a fox', model: 'zit', media_type: 'image', reuse_workspace_url: '', upscale,
  };
}

const seedvr2 = (options: AssetUpscaleFactor[], overrides = {}) => ({
  options, model: 'SeedVR2 7B q8', downloaded: true, download_size_label: null, ...overrides,
});

const kinds = (entries: ReturnType<typeof upscaleEntries>) => entries.map((entry) => (entry.kind === 'item' ? entry.id : entry.kind));

describe('upscaleEntries', () => {
  it('lists the one LTX 2x item of a video on macOS, which upscales with the ltx method', () => {
    const onupscale = vi.fn();
    const video = { ...makeAsset({ factors: [], seedvr2: null, ltx: seedvr2([option(2)], { model: 'LTX-2.3 4-bit' }) }), media_type: 'video' as const };

    const entries = upscaleEntries(video, onupscale);
    expect(kinds(entries)).toEqual(['upscale-ltx-2']);
    expect(entries[0]).toMatchObject({ label: '2× → 1664×2432', title: undefined });
    if (entries[0].kind === 'item') entries[0].onselect?.();
    expect(onupscale).toHaveBeenCalledWith(video, 2, 'ltx');
  });

  it('marks a warned LTX item and names its first-use download', () => {
    const warning = 'May need more memory than this computer has: it may be slow or fail.';
    const ltx = seedvr2([option(2, { warning, warning_code: 'memory' })], { downloaded: false, download_size_label: '26 GB' });

    const [entry] = upscaleEntries({ ...makeAsset({ factors: [], seedvr2: null, ltx }), media_type: 'video' }, vi.fn());

    expect(entry).toMatchObject({ id: 'upscale-ltx-2', label: '2× → 1664×2432 · high memory', title: `${warning}\nFirst use downloads 26 GB` });
  });

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

  it('lists Start image, Reference, First frame and Last frame with hints', () => {
    const entries = referenceEntries(asset, vi.fn(), {
      reference: { disabled: false, hint: 'next slot' },
      last: { disabled: true, hint: 'LTX has no last frame' },
    });

    expect(ids(entries)).toEqual(['reference-image', 'reference-reference', 'reference-video', 'reference-last-frame']);
    expect(entries.flatMap((entry) => (entry.kind === 'item' ? [entry.label] : []))).toEqual(['Start image', 'Reference', 'First frame', 'Last frame']);
    expect(entries[1]).toMatchObject({ hint: 'next slot', disabled: false, title: undefined });
    expect(entries[3]).toMatchObject({ hint: 'LTX has no last frame', disabled: true, title: 'LTX has no last frame' });
    expect(entries[0]).toMatchObject({ disabled: false, hint: undefined });
  });

  it('reports the chosen target', () => {
    const onreference = vi.fn();
    for (const entry of referenceEntries(asset, onreference)) if (entry.kind === 'item') entry.onselect?.();
    expect(onreference.mock.calls.map(([, target]) => target)).toEqual(['image', 'reference', 'video', 'last']);
  });

  it('carries the use-as state through the asset menu', () => {
    const usable = { ...asset, file_path: '/outputs/out/a.png' };
    const menu = assetMenuEntries(usable, { onreference: vi.fn() }, false, { image: { disabled: true, hint: 'Ideogram 4 can\'t use a start image' } });
    expect(menu.find((entry) => entry.kind === 'item' && entry.id === 'reference-image')).toMatchObject({ disabled: true });
    expect(ids(menu)).toContain('reference-last-frame');
  });
});

describe('audioUseEntries', () => {
  const items = (entries: ReturnType<typeof audioUseEntries>) => entries.flatMap((entry) => (entry.kind === 'item' ? [entry] : []));

  it('offers Saved voice next to Voice for a speech take, each with its hint', () => {
    const entries = items(audioUseEntries(makeSpeechAsset(), vi.fn(), { voice: { disabled: false, hint: 'Speech, text filled in' }, save_voice: { disabled: false, hint: 'Keep it in your voices' } }));

    expect(entries.map((entry) => [entry.id, entry.label, entry.hint, entry.disabled])).toEqual([
      ['use-audio-track', 'Audio track', 'Video tab, from 0 s', false],
      ['use-voice', 'Voice', 'Speech, text filled in', false],
      ['use-save-voice', 'Saved voice', 'Keep it in your voices', false],
    ]);
  });

  it('offers no voice item for music or sound', () => {
    expect(items(audioUseEntries(makeAudioAsset(), vi.fn())).map((entry) => entry.id)).toEqual(['use-audio-track']);
    expect(items(audioUseEntries(makeAudioAsset({ kind: 'sound' }), vi.fn())).map((entry) => entry.id)).toEqual(['use-audio-track']);
  });

  it('greys Voice and Saved voice for a take longer than the model takes, with no reason line and the reason as the tooltip', () => {
    const [, voice, saved] = items(audioUseEntries(makeSpeechAsset({ duration: 44 }), vi.fn(), { voiceMaxSeconds: 30 }));

    for (const entry of [voice, saved]) {
      expect(entry).toMatchObject({ disabled: true, hint: undefined, title: 'Use a take of 30 s or less as a voice' });
    }
    expect([voice.label, saved.label]).toEqual(['Voice', 'Saved voice']);
  });

  it('greys both for a machine where no speech model can follow a voice, with the same tooltip and no hint', () => {
    const none = { disabled: true, hint: 'No speech model here can use a voice' };
    const [, voice, saved] = items(audioUseEntries(makeSpeechAsset(), vi.fn(), { voice: none, save_voice: none }));

    for (const entry of [voice, saved]) expect(entry).toMatchObject({ disabled: true, hint: undefined, title: none.hint });
  });

  it('reports the chosen target', () => {
    const onuseaudio = vi.fn();
    for (const entry of items(audioUseEntries(makeSpeechAsset(), onuseaudio))) entry.onselect?.();

    expect(onuseaudio.mock.calls.map(([, target]) => target)).toEqual(['audio_track', 'voice', 'save_voice']);
  });

  it('puts them in the asset menu under Use as, and offers no upscale or extend for a take', () => {
    const menu = assetMenuEntries(makeSpeechAsset(), { onreuse: vi.fn(), onuseaudio: vi.fn(), onupscale: vi.fn(), onextend: vi.fn(), ondelete: vi.fn() });

    expect(menu.map((entry) => (entry.kind === 'item' ? entry.id : entry.kind === 'heading' ? entry.label : entry.kind))).toEqual([
      'reuse', 'separator', 'Use as', 'use-audio-track', 'use-voice', 'use-save-voice', 'separator', 'download', 'separator', 'delete',
    ]);
  });
});

describe('assetMenuEntries Move', () => {
  const ids = (entries: ReturnType<typeof assetMenuEntries>) => entries.map((entry) => (entry.kind === 'item' ? entry.id : entry.kind));

  it('offers Move to project directly above Download, only when the owner can open the list', () => {
    const asset = makeAsset(undefined);
    const withMove = ids(assetMenuEntries(asset, { ondelete: vi.fn() }, false, {}, vi.fn()));
    expect(withMove).toEqual(['move', 'download', 'separator', 'delete']);
    expect(ids(assetMenuEntries(asset, { ondelete: vi.fn() }))).toEqual(['download', 'separator', 'delete']);
  });

  it('opens the project list when chosen, and is disabled while the file is being deleted', () => {
    const openMove = vi.fn();
    const [move] = assetMenuEntries(makeAsset(undefined), {}, false, {}, openMove);
    if (move.kind === 'item') move.onselect?.();
    expect(openMove).toHaveBeenCalledOnce();

    const [busy] = assetMenuEntries(makeAsset(undefined), {}, true, {}, openMove);
    expect(busy).toMatchObject({ id: 'move', disabled: true });
  });
});
