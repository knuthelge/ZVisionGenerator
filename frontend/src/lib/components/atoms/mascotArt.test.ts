import { describe, expect, it } from 'vitest';

import { bubbleCells, noteConfetti, notePixels, risingNotes, TILE_STEP_COUNT, tileCells, tileStep, zGlyphs } from './mascotArt';

describe('tileStep', () => {
  it('moves from the first step to the last as the job progresses', () => {
    const steps = [0, 0.2, 0.4, 0.6, 0.84].map((progress) => tileStep(progress, 0.85));

    expect(steps[0]).toBe(0);
    expect(steps).toEqual([...steps].sort((a, b) => a - b));
    expect(steps.at(-1)).toBe(TILE_STEP_COUNT - 1);
  });

  it('stays on the last step once the job is past the point where he adds the finishing touches', () => {
    expect(tileStep(1, 0.85)).toBe(TILE_STEP_COUNT - 1);
  });
});

describe('tileCells', () => {
  it('draws the same picture for the same step', () => {
    expect(tileCells(3)).toEqual(tileCells(3));
  });

  it('looks different at each stage of the denoise, and different again once finished', () => {
    const pictures = [0, 4, 7].map((step) => JSON.stringify(tileCells(step)));

    expect(new Set([...pictures, JSON.stringify(tileCells(null))]).size).toBe(4);
  });
});

describe('notePixels', () => {
  it('lights the cells of each glyph from the corner it is given', () => {
    expect(notePixels('eighth', 10, 20, 2)).toHaveLength(12);
    expect(notePixels('beamed', 10, 20, 2)).toHaveLength(23);
    expect(notePixels('eighth', 10, 20, 2)[0]).toEqual({ x: 14, y: 20 });
  });

  it('shares its layout with the Z glyphs, so a Z keeps its cells', () => {
    expect(zGlyphs().map((glyph) => glyph.pixels.length)).toEqual([13, 13, 13]);
    expect(zGlyphs()[0]!.pixels[0]).toEqual({ x: 146, y: 50 });
  });
});

describe('risingNotes', () => {
  it('has three notes in three colours, spread over the loop', () => {
    const notes = risingNotes();

    expect(notes).toHaveLength(3);
    expect(new Set(notes.map((note) => note.fill)).size).toBe(3);
    expect(notes.map((note) => note.delay)).toEqual([0, -1, -2]);
    expect(notes.every((note) => note.pixels.length > 0)).toBe(true);
  });

  it('is the same every time', () => {
    expect(risingNotes()).toEqual(risingNotes());
  });
});

describe('noteConfetti', () => {
  it('scatters eight notes, the same every time', () => {
    expect(noteConfetti()).toHaveLength(8);
    expect(noteConfetti()).toEqual(noteConfetti());
  });
});

describe('bubbleCells', () => {
  it('draws a 6 x 6 picture by default and a different one for a note', () => {
    expect(bubbleCells()).toHaveLength(36);
    expect(bubbleCells('note')).toHaveLength(36);
    expect(bubbleCells('note')).not.toEqual(bubbleCells('picture'));
    expect(bubbleCells('note')).toEqual(bubbleCells('note'));
  });
});
