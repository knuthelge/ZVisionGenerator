import { describe, expect, it } from 'vitest';

import type { GalleryAsset, LoraInfo, WorkspaceContext } from '$lib/types';
import { familyLabel, formatLoraString, groupLorasForModel, loraLabel, loraMatches, modelLoraFamily, parseLoraString, recentLoraNames } from './loras';

const installed = [
  { name: 'film-grain', path: '/Users/me/.ziv/loras/film-grain.safetensors' },
  { name: 'ink-sketch', path: '/Users/me/.ziv/loras/ink-sketch.safetensors' },
];

describe('parseLoraString', () => {
  it('reads names and weights', () => {
    expect(parseLoraString('film-grain:0.7,ink-sketch', installed)).toEqual([
      { name: 'film-grain', weight: 0.7 },
      { name: 'ink-sketch', weight: 1 },
    ]);
  });

  it('names a recorded path by its installed LoRA', () => {
    expect(parseLoraString('/Users/me/.ziv/loras/film-grain.safetensors:1, /Users/me/.ziv/loras/ink-sketch.safetensors:-0.5', installed)).toEqual([
      { name: 'film-grain', weight: 1 },
      { name: 'ink-sketch', weight: -0.5 },
    ]);
  });

  it('keeps a Hugging Face repo or a file elsewhere that shares an installed name', () => {
    expect(parseLoraString('owner/ink-sketch:0.8,/old/place/ink-sketch.safetensors:0.4', installed)).toEqual([
      { name: 'owner/ink-sketch', weight: 0.8 },
      { name: '/old/place/ink-sketch.safetensors', weight: 0.4 },
    ]);
  });

  it('keeps the first entry when a LoRA is listed twice', () => {
    expect(parseLoraString('ink-sketch:1,/Users/me/.ziv/loras/ink-sketch.safetensors:0.4', installed)).toEqual([{ name: 'ink-sketch', weight: 1 }]);
  });

  it('keeps a path with no installed LoRA, including a Windows path', () => {
    expect(parseLoraString('C:\\loras\\gone.safetensors:0.5', installed)).toEqual([{ name: 'C:\\loras\\gone.safetensors', weight: 0.5 }]);
  });

  it('reads the list as the server does: spaces ignored, an empty weight is 1', () => {
    expect(parseLoraString(' film-grain : 0.5 , ink-sketch:', installed)).toEqual([
      { name: 'film-grain', weight: 0.5 },
      { name: 'ink-sketch', weight: 1 },
    ]);
  });

  it('returns nothing for an empty list', () => {
    expect(parseLoraString('', installed)).toEqual([]);
  });
});

describe('formatLoraString', () => {
  it('sends a reused path as the installed LoRA it matched', () => {
    const chips = parseLoraString('/Users/me/.ziv/loras/ink-sketch.safetensors:0.4,/gone/x.safetensors:1', installed);
    expect(formatLoraString(chips)).toBe('ink-sketch:0.4,/gone/x.safetensors:1');
  });
});

describe('loraLabel', () => {
  it('shortens a path to the file name without the extension', () => {
    expect(loraLabel('/a/b/gone.safetensors')).toBe('gone');
    expect(loraLabel('film-grain')).toBe('film-grain');
  });
});

const lora = (name: string, family: string | null, size_variant: string | null = null): LoraInfo => ({
  name, path: `/loras/${name}.safetensors`, family, size_variant,
});
const names = (loras: LoraInfo[]): string[] => loras.map((item) => item.name);

describe('modelLoraFamily', () => {
  const ctx = {
    image_model_defaults: { klein4: { family: 'flux2_klein', size_variant: '4b' }, zit: { family: 'zimage' }, mystery: { family: null } },
    video_model_defaults: { 'ltx-v': { family: 'ltx' } },
  } as unknown as WorkspaceContext;

  it('reads an image model family and size', () => {
    expect(modelLoraFamily(ctx, 'klein4')).toEqual({ family: 'flux2_klein', sizeVariant: '4b' });
    expect(modelLoraFamily(ctx, 'zit')).toEqual({ family: 'zimage', sizeVariant: null });
  });

  it('reads a video model family with no size', () => {
    expect(modelLoraFamily(ctx, 'ltx-v')).toEqual({ family: 'ltx', sizeVariant: null });
  });

  it('reports an unknown family, and null for a model the context does not list', () => {
    expect(modelLoraFamily(ctx, 'mystery')).toEqual({ family: null, sizeVariant: null });
    expect(modelLoraFamily(ctx, 'absent')).toBeNull();
  });
});

describe('loraMatches', () => {
  const klein4 = { family: 'flux2_klein', sizeVariant: '4b' };

  it('needs equal families', () => {
    expect(loraMatches(lora('a', 'zimage'), klein4)).toBe(false);
    expect(loraMatches(lora('a', null), klein4)).toBe(false);
  });

  it('rejects a different size of the same family', () => {
    expect(loraMatches(lora('a', 'flux2_klein', '9b'), klein4)).toBe(false);
    expect(loraMatches(lora('a', 'flux2_klein', '4b'), klein4)).toBe(true);
  });

  it('matches when either size is null', () => {
    expect(loraMatches(lora('a', 'flux2_klein', null), klein4)).toBe(true);
    expect(loraMatches(lora('a', 'flux2_klein', '9b'), { family: 'flux2_klein', sizeVariant: null })).toBe(true);
  });
});

describe('recentLoraNames', () => {
  const installedLoras = [lora('one', 'zimage'), lora('two', 'zimage'), lora('three', 'flux1')];
  const asset = (loraValue: string | null, family: string | null): GalleryAsset => ({ lora: loraValue, details: { model_family: family } }) as GalleryAsset;

  it('lists installed LoRAs used with the family, newest first and once each', () => {
    const assets = [asset('two:0.8', 'zimage'), asset('three:1', 'flux1'), asset('one:1,two:1', 'zimage'), asset('gone:1', 'zimage')];
    expect(recentLoraNames(assets, 'zimage', installedLoras)).toEqual(['two', 'one']);
  });

  it('reads recorded paths as the installed LoRA they point to', () => {
    expect(recentLoraNames([asset('/loras/one.safetensors:1', 'zimage')], 'zimage', installedLoras)).toEqual(['one']);
  });

  it('is empty for an unknown family or assets with no LoRA', () => {
    expect(recentLoraNames([asset('one:1', 'zimage')], null, installedLoras)).toEqual([]);
    expect(recentLoraNames([asset(null, 'zimage')], 'zimage', installedLoras)).toEqual([]);
  });
});

describe('groupLorasForModel', () => {
  const zimage = { family: 'zimage', sizeVariant: null };

  it('shows only the matches, by name, and folds the rest, unknown-family LoRAs included', () => {
    const loras = [lora('z-match', 'zimage'), lora('mystery', null), lora('a-match', 'zimage'), lora('b-ltx', 'ltx'), lora('a-flux', 'flux1')];
    const { main, other } = groupLorasForModel(loras, zimage, []);
    expect(names(main)).toEqual(['a-match', 'z-match']);
    expect(names(other)).toEqual(['a-flux', 'b-ltx', 'mystery']);
  });

  it('leads with recently used matches, newest first, before the other matches', () => {
    const loras = [lora('a', 'zimage'), lora('b', 'zimage'), lora('c', 'zimage'), lora('d', 'zimage')];
    expect(names(groupLorasForModel(loras, zimage, ['c', 'a']).main)).toEqual(['c', 'a', 'b', 'd']);
  });

  it('ignores recent names that are not matches', () => {
    const loras = [lora('a', 'zimage'), lora('x', 'ltx')];
    const { main, other } = groupLorasForModel(loras, zimage, ['x']);
    expect(names(main)).toEqual(['a']);
    expect(names(other)).toEqual(['x']);
  });

  it('sends a 9B LoRA to Other for a Klein 4B model', () => {
    const loras = [lora('nine', 'flux2_klein', '9b'), lora('four', 'flux2_klein', '4b')];
    const { main, other } = groupLorasForModel(loras, { family: 'flux2_klein', sizeVariant: '4b' }, []);
    expect(names(main)).toEqual(['four']);
    expect(names(other)).toEqual(['nine']);
  });

  it('matches a LoRA with no size against a sized model', () => {
    const { main, other } = groupLorasForModel([lora('any', 'flux2_klein', null)], { family: 'flux2_klein', sizeVariant: '9b' }, []);
    expect(names(main)).toEqual(['any']);
    expect(other).toEqual([]);
  });

  it('gives a flat sorted list when the model family is unknown', () => {
    const loras = [lora('b', 'ltx'), lora('c', null), lora('a', 'zimage')];
    for (const model of [null, { family: null, sizeVariant: null }]) {
      const { main, other } = groupLorasForModel(loras, model, ['b']);
      expect(names(main)).toEqual(['a', 'b', 'c']);
      expect(other).toEqual([]);
    }
  });

  it('treats LoRAs without the family fields as unknown, so they fold under Other', () => {
    const { main, other } = groupLorasForModel([{ name: 'plain', path: '/p' }], zimage, []);
    expect(main).toEqual([]);
    expect(names(other)).toEqual(['plain']);
  });

  it('leaves main empty when nothing matches', () => {
    const { main, other } = groupLorasForModel([lora('x', 'ltx')], zimage, []);
    expect(main).toEqual([]);
    expect(names(other)).toEqual(['x']);
  });
});

describe('familyLabel', () => {
  it('names the known families', () => {
    expect(familyLabel('zimage')).toBe('Z-Image');
    expect(familyLabel('flux2_klein', '4b')).toBe('Klein 4B');
    expect(familyLabel('flux2_klein', '9b')).toBe('Klein 9B');
    expect(familyLabel('flux2_klein')).toBe('Klein');
    expect(familyLabel('flux1')).toBe('Flux 1');
    expect(familyLabel('krea2')).toBe('Krea 2');
    expect(familyLabel('ideogram4')).toBe('Ideogram 4');
    expect(familyLabel('moss')).toBe('MOSS-Image');
    expect(familyLabel('ltx')).toBe('LTX');
    expect(familyLabel('h3')).toBe('MiniMax H3');
  });

  it('falls back to the family for an unlisted size, and shows an unknown family as it is', () => {
    expect(familyLabel('flux2_klein', '20b')).toBe('Klein');
    expect(familyLabel('brand_new', '2b')).toBe('brand_new');
  });
});
