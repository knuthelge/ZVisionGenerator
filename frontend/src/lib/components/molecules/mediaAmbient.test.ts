import { describe, expect, it } from 'vitest';

import { ambientRegion, glowMask, glowShows, glowSpread, leavesBands, shownPicture } from './mediaAmbient';

describe('shownPicture', () => {
  it('fits a picture that is larger than the player to the player', () => {
    const wide = shownPicture({ width: 1000, height: 600 }, { width: 1920, height: 1080 }, false);
    expect(wide.width).toBeCloseTo(1000, 3);
    expect(wide.height).toBeCloseTo(562.5, 3);
    expect(shownPicture({ width: 1000, height: 600 }, { width: 480, height: 1280 }, false)).toEqual({ width: 225, height: 600 });
  });

  it('keeps a picture that already fits at its own size, unless it may be enlarged', () => {
    expect(shownPicture({ width: 1000, height: 600 }, { width: 400, height: 300 }, false)).toEqual({ width: 400, height: 300 });
    expect(shownPicture({ width: 1000, height: 600 }, { width: 400, height: 300 }, true)).toEqual({ width: 800, height: 600 });
  });

  it('answers an empty box while a size is not known', () => {
    expect(shownPicture({ width: 0, height: 0 }, { width: 400, height: 300 }, false)).toEqual({ width: 0, height: 0 });
    expect(shownPicture({ width: 1000, height: 600 }, { width: 0, height: 0 }, true)).toEqual({ width: 0, height: 0 });
  });
});

describe('leavesBands', () => {
  it('is false when the picture fills the pane', () => {
    expect(leavesBands({ width: 1280, height: 720 }, { width: 1280, height: 720 })).toBe(false);
  });

  it('is true when the picture is letterboxed or pillarboxed', () => {
    expect(leavesBands({ width: 1280, height: 900 }, { width: 1280, height: 720 })).toBe(true);
    expect(leavesBands({ width: 1280, height: 720 }, { width: 960, height: 720 })).toBe(true);
  });

  it('ignores a sliver of a pixel or two', () => {
    expect(leavesBands({ width: 1281, height: 720 }, { width: 1280, height: 720 })).toBe(false);
    expect(leavesBands({ width: 1284, height: 720 }, { width: 1280, height: 720 })).toBe(true);
  });

  it('is true while a size is not known', () => {
    expect(leavesBands({ width: 0, height: 0 }, { width: 1280, height: 720 })).toBe(true);
    expect(leavesBands({ width: 1280, height: 720 }, { width: 0, height: 0 })).toBe(true);
  });
});

describe('ambientRegion', () => {
  const reach = 100;

  it('lays out a canvas as wide as the pane and its reach, with the picture centred in it', () => {
    const region = ambientRegion({ width: 1000, height: 600 }, { width: 800, height: 600 }, reach, 24)!;

    // 1200 x 800 CSS pixels in 24 columns of 50 px: 16 rows; the picture is 800 x 600, so 16 x 12 of them.
    expect(region.columns).toBe(24);
    expect(region.rows).toBe(16);
    expect(region.picture).toEqual({ x: 4, y: 2, width: 16, height: 12 });
  });

  it('leaves the bands at the sides of a tall picture', () => {
    const region = ambientRegion({ width: 1000, height: 600 }, { width: 300, height: 600 }, reach, 24)!;

    expect(region.picture).toEqual({ x: 9, y: 2, width: 6, height: 12 });
  });

  it('leaves the bands above and below a wide picture', () => {
    const region = ambientRegion({ width: 600, height: 1000 }, { width: 600, height: 300 }, reach, 24)!;

    expect(region.rows).toBe(36);
    expect(region.picture).toEqual({ x: 3, y: 14, width: 18, height: 9 });
  });

  it('leaves a band all round a picture that is smaller than the pane', () => {
    const region = ambientRegion({ width: 1000, height: 600 }, { width: 400, height: 300 }, reach, 24)!;

    expect(region.picture).toEqual({ x: 8, y: 5, width: 8, height: 6 });
  });

  it('never makes the picture thinner than a pixel or bigger than the canvas', () => {
    const thin = ambientRegion({ width: 1000, height: 600 }, { width: 1, height: 600 }, reach, 24)!;
    const huge = ambientRegion({ width: 1000, height: 600 }, { width: 100000, height: 600 }, reach, 24)!;

    expect(thin.picture.width).toBe(1);
    expect(huge.picture.width).toBeLessThanOrEqual(huge.columns);
    expect(huge.picture.x + huge.picture.width).toBeLessThanOrEqual(huge.columns);
  });

  it('answers null while a size is not known', () => {
    expect(ambientRegion({ width: 0, height: 0 }, { width: 640, height: 360 }, reach, 24)).toBeNull();
    expect(ambientRegion({ width: 640, height: 360 }, { width: 0, height: 0 }, reach, 24)).toBeNull();
  });
});

describe('glowSpread', () => {
  it('spreads as far as the band the light lies in, and no further than the reach', () => {
    // A 960 x 720 picture in a 1280 x 720 pane: 160 px bands at the sides, none above or below.
    const wide = glowSpread({ width: 1280, height: 720 }, { width: 960, height: 720 }, 160);
    expect(wide.x).toBeCloseTo(160 / 2.6, 5);
    expect(wide.y).toBe(0);

    const far = glowSpread({ width: 2000, height: 720 }, { width: 960, height: 720 }, 160);
    expect(far.x).toBeCloseTo(160 / 2.6, 5);
  });

  it('spreads above and below no further than at the sides, so a picture filling the width gets no slab of light over and under it', () => {
    const upright = glowSpread({ width: 400, height: 900 }, { width: 400, height: 225 }, 160);

    expect(upright).toEqual({ x: 0, y: 0 });
  });

  it('spreads above and below as far as the sides allow', () => {
    const spread = glowSpread({ width: 800, height: 900 }, { width: 600, height: 400 }, 160);

    expect(spread.x).toBeCloseTo(100 / 2.6, 5);
    expect(spread.y).toBeCloseTo(100 / 2.6, 5);
  });

  it('is nothing where the picture covers the pane, and not below zero where it is larger', () => {
    expect(glowSpread({ width: 1280, height: 720 }, { width: 1280, height: 720 }, 160)).toEqual({ x: 0, y: 0 });
    expect(glowSpread({ width: 1000, height: 600 }, { width: 1200, height: 800 }, 160)).toEqual({ x: 0, y: 0 });
  });
});

describe('glowShows', () => {
  it('is false under 3 px on both axes, where there is no light to see', () => {
    expect(glowShows({ x: 2.9, y: 2.9 })).toBe(false);
    expect(glowShows({ x: 0, y: 0 })).toBe(false);
  });

  it('is true when either axis reaches 3 px', () => {
    expect(glowShows({ x: 3, y: 0 })).toBe(true);
    expect(glowShows({ x: 0, y: 3 })).toBe(true);
  });
});

describe('glowMask', () => {
  const decoded = (mask: string): string => decodeURIComponent(mask.slice('url("data:image/svg+xml,'.length, -2));

  it('is an image of the pane\'s size, for a CSS mask', () => {
    const mask = glowMask({ width: 1280, height: 720 }, { width: 960, height: 720 }, { x: 61.5, y: 0 });

    expect(mask.startsWith('url("data:image/svg+xml,')).toBe(true);
    expect(decoded(mask)).toContain("width='1280.0' height='720.0'");
  });

  it('draws the picture\'s box grown by one spread, softened by that spread', () => {
    const svg = decoded(glowMask({ width: 1280, height: 720 }, { width: 960, height: 720 }, { x: 61.5, y: 0 }));

    // The picture starts 160 px in: one spread of 61.5 px out from it is 98.5 px, and the box is wider by two spreads.
    expect(svg).toContain("x='98.5'");
    expect(svg).toContain("width='1083.0'");
    expect(svg).toContain("stdDeviation='61.5 0.0'");
  });

  it('rounds its corners by the smaller spread', () => {
    const svg = decoded(glowMask({ width: 800, height: 900 }, { width: 600, height: 400 }, { x: 40, y: 12 }));

    expect(svg).toContain("rx='12.0'");
  });
});
