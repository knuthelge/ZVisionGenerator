import { describe, expect, it, vi } from 'vitest';

import type { GalleryAsset } from '$lib/types';
import { makeAudioAsset, makeSpeechAsset } from '../../../test-utils/audio';
import { assetPanel } from './assetMenu';
import { chromeFor, phoneActs, viewerActs } from './viewerActions';

const handlers = () => ({ onreuse: vi.fn(), onreference: vi.fn(), onextend: vi.fn(), onedit: vi.fn(), onupscale: vi.fn(), onuseaudio: vi.fn(), ondelete: vi.fn(), onstar: vi.fn() });
const option = (factor: 2 | 4) => ({ factor, width: 832 * factor, height: 1216 * factor, allowed: true, reason: null, warning: null });

const image: GalleryAsset = {
  id: 'out/a.png', url: '/media/out/a.png', thumbnail_url: '/media/out/a.png', filename: 'a.png', created_at: '2026-10-03T12:00:00Z', project: null, workflow: 'txt2img',
  prompt: 'a fox', model: 'zit', media_type: 'image', file_path: '/outputs/out/a.png', has_reusable_config: true, reuse_workspace_url: '',
  upscale: { factors: [option(2), option(4)], seedvr2: null },
};
const video: GalleryAsset = {
  ...image, id: 'out/c.mp4', filename: 'c.mp4', media_type: 'video', extend: { available: true, reason: null },
  upscale: { factors: [], seedvr2: { options: [option(2)], model: 'SeedVR2', downloaded: true, download_size_label: null } },
};

function actsFor(asset: GalleryAsset) {
  const panel = assetPanel(asset, handlers(), { openMove: vi.fn() });
  return viewerActs({ asset, panel, upscaleKeys: asset.media_type === 'video' ? '2' : '2 or 4' });
}

const ids = (acts: readonly { id: string }[]) => acts.map((act) => act.id);

describe('viewerActs', () => {
  it('lists an image as Reuse, Edit, Use as, Upscale, then Download, Move, Copy prompt and Delete', () => {
    const acts = actsFor(image);

    expect(ids(acts.make)).toEqual(['reuse', 'edit', 'use-as', 'upscale']);
    expect(ids(acts.file)).toEqual(['download', 'move', 'copy-prompt']);
    expect(acts.del?.id).toBe('delete');
    expect(acts.make.filter((act) => act.menu).map((act) => act.menu)).toEqual(['useas', 'upscale']);
  });

  it('gives a video Extend, no Use as, and one direct Upscale with no menu and its key in the tooltip', () => {
    const acts = actsFor(video);

    expect(ids(acts.make)).toEqual(['reuse', 'extend', 'upscale']);
    expect(acts.make[2].title).toContain('X then 2');
    expect(acts.make[2].menu).toBeUndefined();
  });

  it('carries a warned direct upscale\'s warning and tooltip, and none for an unwarned one', () => {
    const warned = {
      ...video,
      upscale: { factors: [], seedvr2: { options: [{ ...option(2), warning: 'May need more memory.', warning_code: 'memory' as const }], model: 'SeedVR2', downloaded: false, download_size_label: '17 GB' } },
    };
    const [, , upscale] = actsFor(warned).make;
    expect(upscale.item.warn).toBeTruthy();
    expect(upscale.title).toContain('May need more memory.');
    expect(upscale.title).toContain('17 GB');
    expect(upscale.item.id).toBe('upscale-seedvr2-2');
    expect(actsFor(video).make[2].item.warn).toBeUndefined();
  });

  it('gives every audio take Audio track beside Reuse, and Use as only to a speech take', () => {
    expect(ids(actsFor(makeAudioAsset()).make)).toEqual(['reuse', 'use-audio-track']);
    expect(ids(actsFor(makeAudioAsset({ kind: 'sound' })).make)).toEqual(['reuse', 'use-audio-track']);
    expect(ids(actsFor(makeSpeechAsset()).make)).toEqual(['reuse', 'use-audio-track', 'use-as']);
  });

  it('closes up a kind without an action, and keeps File and Delete the same for every kind', () => {
    for (const asset of [image, video, makeAudioAsset(), makeSpeechAsset()]) {
      const acts = actsFor(asset);
      expect(ids(acts.file)).toEqual(['download', 'move', 'copy-prompt']);
      expect(acts.del?.item.danger).toBe(true);
    }
  });

  it('marks the end of the Reuse label that gives way first, and says why it is off', () => {
    const [reuse] = actsFor(image).make;
    expect(reuse.item.primary).toBe(true);
    expect(reuse.item.label.endsWith(reuse.tail ?? '\0')).toBe(true);

    const off = actsFor({ ...image, has_reusable_config: false }).make[0];
    // Off for a reason a tap explains: greyed, never disabled outright.
    expect(off.item.disabled).toBe(true);
    expect(off.disabled).toBeUndefined();
    expect(off.title).toBeTruthy();
    expect(off.title).not.toBe(reuse.title);
  });

  it('omits Move without an owner for the list, and the actions without handlers', () => {
    const panel = assetPanel(image, {});
    const acts = viewerActs({ asset: image, panel, upscaleKeys: '2' });

    expect(acts.make).toEqual([]);
    expect(ids(acts.file)).toEqual(['download', 'copy-prompt']);
    expect(acts.del).toBeNull();
  });
});

describe('phoneActs', () => {
  it('has four slots, Reuse, the kind\'s own action, Download and More, for every kind', () => {
    expect(ids(phoneActs(actsFor(image)))).toEqual(['reuse', 'edit', 'download', 'more']);
    expect(ids(phoneActs(actsFor(video)))).toEqual(['reuse', 'extend', 'download', 'more']);
    expect(ids(phoneActs(actsFor(makeAudioAsset())))).toEqual(['reuse', 'use-audio-track', 'download', 'more']);
  });

  it('draws More with filled dots', () => {
    expect(phoneActs(actsFor(image)).at(-1)).toMatchObject({ item: { icon: 'more-dots' }, solid: true });
  });
});

describe('chromeFor', () => {
  it.each([
    ['a mouse on any window', { coarse: false, compact: false, short: false }, 'toolbar'],
    ['a mouse in a narrow window', { coarse: false, compact: true, short: false }, 'toolbar'],
    ['a mouse in a short window', { coarse: false, compact: true, short: true }, 'toolbar'],
    ['a finger on a tablet', { coarse: true, compact: false, short: false }, 'dock'],
    ['a finger on a phone', { coarse: true, compact: true, short: false }, 'phone'],
    ['a finger on a phone held sideways', { coarse: true, compact: true, short: true }, 'rail'],
  ])('is the %s', (_name, layout, expected) => {
    expect(chromeFor(layout)).toBe(expected);
  });
});
