import { describe, expect, it } from 'vitest';

import { shownPicture } from './mediaPicture';

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
