import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { SPRITE_COLUMNS, SPRITE_MAX_TILES, coverTile, spriteCell, spriteTile, spriteTiles, videoSprite } from './videoSprite.svelte';

class FakeImage {
  static made: FakeImage[] = [];
  naturalWidth = 1760;
  naturalHeight = 990;
  onload: (() => void) | null = null;
  onerror: (() => void) | null = null;
  src = '';

  constructor() {
    FakeImage.made.push(this);
  }
}

beforeEach(() => {
  FakeImage.made = [];
  vi.stubGlobal('Image', FakeImage);
  videoSprite.reset();
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('spriteTiles', () => {
  it('has one tile per frame up to 121 frames, then 121', () => {
    expect(SPRITE_MAX_TILES).toBe(121);
    expect(spriteTiles(1)).toBe(1);
    expect(spriteTiles(73)).toBe(73);
    expect(spriteTiles(121)).toBe(121);
    expect(spriteTiles(5000)).toBe(121);
  });
});

describe('spriteTile', () => {
  it('is the frame itself while the clip has a tile for every frame', () => {
    for (let frame = 0; frame < 73; frame += 1) expect(spriteTile(frame, 73)).toBe(frame);
  });

  it('spreads a long clip evenly over its tiles, first frame to last', () => {
    expect(spriteTile(0, 1000)).toBe(0);
    expect(spriteTile(999, 1000)).toBe(120);
    expect(spriteTile(500, 1000)).toBe(Math.round((500 / 999) * 120));
  });

  it('shows a one-frame clip as tile 0', () => {
    expect(spriteTile(0, 1)).toBe(0);
  });

  it('keeps a frame outside the clip on the first or the last tile', () => {
    expect(spriteTile(-3, 100)).toBe(0);
    expect(spriteTile(250, 100)).toBe(99);
  });
});

describe('coverTile', () => {
  const sprite = { url: '/sheet.jpg', tileWidth: 160, tileHeight: 90 };

  it('scales a tile to cover a box and centres the crop', () => {
    // A 76 x 44 box: the tile scales by 44 / 90 at least and 76 / 160 at most; the larger wins.
    const cover = coverTile(sprite, 0, { width: 76, height: 44 });
    const scale = Math.max(76 / 160, 44 / 90);
    expect(cover.sheetWidth).toBeCloseTo(160 * scale * 11, 6);
    expect(cover.offsetX).toBeCloseTo((76 - 160 * scale) / 2, 6);
    expect(cover.offsetY).toBeCloseTo((44 - 90 * scale) / 2, 6);
  });

  it('moves to the column and row of the tile', () => {
    const first = coverTile(sprite, 0, { width: 160, height: 90 });
    const tile = coverTile(sprite, 13, { width: 160, height: 90 });
    expect(tile.offsetX - first.offsetX).toBeCloseTo(-2 * 160, 6);
    expect(tile.offsetY - first.offsetY).toBeCloseTo(-90, 6);
  });
});

describe('spriteCell', () => {
  it('reads tiles left to right, then top to bottom, in an 11 wide grid', () => {
    expect(SPRITE_COLUMNS).toBe(11);
    expect(spriteCell(0)).toEqual({ column: 0, row: 0 });
    expect(spriteCell(10)).toEqual({ column: 10, row: 0 });
    expect(spriteCell(11)).toEqual({ column: 0, row: 1 });
    expect(spriteCell(120)).toEqual({ column: 10, row: 10 });
  });
});

describe('videoSprite', () => {
  it('has no sheet until the picture has loaded, then knows the size of one tile', () => {
    expect(videoSprite.of('/a.mp4')).toBeNull();
    videoSprite.ensure('/a.mp4');
    expect(videoSprite.of('/a.mp4')).toBeNull();

    FakeImage.made[0].onload?.();

    expect(videoSprite.of('/a.mp4')).toEqual({ url: '/api/browse/sprite?purpose=source_video&path=%2Fa.mp4', tileWidth: 160, tileHeight: 90 });
  });

  it('fetches once per path', () => {
    videoSprite.ensure('/a.mp4');
    videoSprite.ensure('/a.mp4');
    videoSprite.ensure('/b.mp4');
    FakeImage.made[0].onload?.();
    videoSprite.ensure('/a.mp4');

    expect(FakeImage.made).toHaveLength(2);
    expect(videoSprite.of('/b.mp4')).toBeNull();
  });

  it('has no sheet for a video whose sheet failed to load, and asks again on the next view', () => {
    videoSprite.ensure('/a.mp4');
    FakeImage.made[0].onerror?.();

    expect(videoSprite.of('/a.mp4')).toBeNull();
    videoSprite.ensure('/a.mp4');
    expect(FakeImage.made).toHaveLength(2);
  });
});
