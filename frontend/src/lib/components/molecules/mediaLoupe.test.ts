import { describe, expect, it } from 'vitest';

import { LOUPE_EDGE, loupeCentre, loupeImage } from './mediaLoupe';

describe('loupeImage', () => {
  it('scales the long side of a landscape tile to 168 px and picks the tile out of the sheet', () => {
    const image = loupeImage({ url: '/sheet.jpg', tileWidth: 160, tileHeight: 90 }, 13);

    expect(image.width).toBeCloseTo(168);
    expect(image.height).toBeCloseTo(94.5);
    // Tile 13 is column 2, row 1 of an 11-wide grid.
    expect(image.offsetX).toBeCloseTo(-2 * 168);
    expect(image.offsetY).toBeCloseTo(-1 * 94.5);
    expect(image.sheetWidth).toBeCloseTo(168 * 11);
  });

  it('keeps the short side of a very wide tile at 56 px or more', () => {
    const image = loupeImage({ url: '/sheet.jpg', tileWidth: 160, tileHeight: 30 }, 0);

    expect(image.height).toBeGreaterThanOrEqual(56);
    expect(image.width / image.height).toBeCloseTo(160 / 30);
  });
});

describe('loupeCentre', () => {
  it('follows the pointer, kept clear of both sides', () => {
    expect(loupeCentre(400, 1000)).toBe(400);
    expect(loupeCentre(10, 1000)).toBe(LOUPE_EDGE);
    expect(loupeCentre(990, 1000)).toBe(1000 - LOUPE_EDGE);
  });

  it('centres in a player too narrow to keep clear of both sides', () => {
    expect(loupeCentre(10, 150)).toBe(75);
  });
});
