import { describe, expect, it } from 'vitest';

import { formatLoraString, loraLabel, parseLoraString } from './loras';

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

  it('matches a moved file by its name', () => {
    expect(parseLoraString('/old/place/ink-sketch.safetensors:0.4', installed)).toEqual([{ name: 'ink-sketch', weight: 0.4 }]);
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
    const chips = parseLoraString('/old/place/ink-sketch.safetensors:0.4,/gone/x.safetensors:1', installed);
    expect(formatLoraString(chips)).toBe('ink-sketch:0.4,/gone/x.safetensors:1');
  });
});

describe('loraLabel', () => {
  it('shortens a path to the file name without the extension', () => {
    expect(loraLabel('/a/b/gone.safetensors')).toBe('gone');
    expect(loraLabel('film-grain')).toBe('film-grain');
  });
});
