import { describe, expect, it } from 'vitest';
import {
  aspectLabel,
  canSwapRatio,
  cssFrameTransform,
  dragCrop,
  fitAspect,
  flipGeometry,
  frameSize,
  moveCrop,
  normalizedCrop,
  nudgeCrop,
  presetRatio,
  quadCorners,
  rotateGeometry,
  swapRatio,
  type Aspect,
  type AspectPreset,
  type Crop,
  type Geometry,
  type Handle,
  type Rotate,
  type Size,
} from './adjustGeometry';

const W = 300;
const H = 200;
const CROP: Crop = { x: 20, y: 10, width: 100, height: 60 };
const ROTATES: Rotate[] = [0, 90, 180, 270];

function geometry(rotate: Rotate, flip: boolean, crop: Crop | null = null): Geometry {
  return { rotate, flip, crop };
}

describe('frameSize', () => {
  it('swaps the sides for odd quarter turns only', () => {
    expect(frameSize(0, W, H)).toEqual({ width: 300, height: 200 });
    expect(frameSize(90, W, H)).toEqual({ width: 200, height: 300 });
    expect(frameSize(180, W, H)).toEqual({ width: 300, height: 200 });
    expect(frameSize(270, W, H)).toEqual({ width: 200, height: 300 });
  });
});

describe('rotateGeometry and flipGeometry', () => {
  it('rotates the crop right as (Hf-y-h, x, h, w)', () => {
    const g = rotateGeometry(geometry(0, false, CROP), 'right', W, H);
    expect(g.rotate).toBe(90);
    expect(g.crop).toEqual({ x: 200 - 10 - 60, y: 20, width: 60, height: 100 });
  });

  it('rotates the crop left as (y, Wf-x-w, h, w)', () => {
    const g = rotateGeometry(geometry(0, false, CROP), 'left', W, H);
    expect(g.rotate).toBe(270);
    expect(g.crop).toEqual({ x: 10, y: 300 - 20 - 100, width: 60, height: 100 });
  });

  it('uses the frame after earlier turns', () => {
    const g = rotateGeometry(geometry(90, false, { x: 10, y: 20, width: 50, height: 40 }), 'right', W, H);
    // Wf x Hf is 200 x 300 here.
    expect(g.rotate).toBe(180);
    expect(g.crop).toEqual({ x: 300 - 20 - 40, y: 10, width: 40, height: 50 });
  });

  it('changes the quarter turns the other way under flip', () => {
    expect(rotateGeometry(geometry(90, true), 'right', W, H).rotate).toBe(0);
    expect(rotateGeometry(geometry(0, true), 'right', W, H).rotate).toBe(270);
    expect(rotateGeometry(geometry(0, true), 'left', W, H).rotate).toBe(90);
    expect(rotateGeometry(geometry(270, true), 'left', W, H).rotate).toBe(0);
  });

  it('keeps flip on rotate and a null crop null', () => {
    const g = rotateGeometry(geometry(0, true), 'right', W, H);
    expect(g.flip).toBe(true);
    expect(g.crop).toBeNull();
  });

  it('mirrors the crop on flip as (Wf-x-w, y, w, h)', () => {
    const g = flipGeometry(geometry(0, false, CROP), W, H);
    expect(g).toEqual({ rotate: 0, flip: true, crop: { x: 180, y: 10, width: 100, height: 60 } });
    expect(flipGeometry(geometry(90, false, { x: 0, y: 0, width: 50, height: 50 }), W, H).crop?.x).toBe(150);
  });

  describe.each(ROTATES)('from rotate %i', (rotate) => {
    describe.each([false, true])('flip %s', (flip) => {
      const start = geometry(rotate, flip, { x: 7, y: 11, width: 31, height: 23 });

      it('right then left is the identity', () => {
        const back = rotateGeometry(rotateGeometry(start, 'right', W, H), 'left', W, H);
        expect(back).toEqual(start);
      });

      it('left then right is the identity', () => {
        const back = rotateGeometry(rotateGeometry(start, 'left', W, H), 'right', W, H);
        expect(back).toEqual(start);
      });

      it('four rights are the identity', () => {
        let g = start;
        for (let i = 0; i < 4; i += 1) g = rotateGeometry(g, 'right', W, H);
        expect(g).toEqual(start);
      });

      it('flip twice is the identity', () => {
        expect(flipGeometry(flipGeometry(start, W, H), W, H)).toEqual(start);
      });

      it('keeps the crop inside the new frame', () => {
        const g = rotateGeometry(start, 'right', W, H);
        const f = frameSize(g.rotate, W, H);
        expect(g.crop!.x + g.crop!.width).toBeLessThanOrEqual(f.width);
        expect(g.crop!.y + g.crop!.height).toBeLessThanOrEqual(f.height);
      });
    });
  });
});

describe('presets and ratios', () => {
  const landscape: Size = { width: 300, height: 200 };

  it('maps each chip to its ratio', () => {
    expect(presetRatio('free', landscape)).toBeNull();
    expect(presetRatio('original', landscape)).toBeCloseTo(1.5);
    expect(presetRatio('original', { width: 200, height: 300 })).toBeCloseTo(2 / 3);
    expect(presetRatio('1:1', landscape)).toBe(1);
    expect(presetRatio('4:5', landscape)).toBeCloseTo(0.8);
    expect(presetRatio('3:2', landscape)).toBeCloseTo(1.5);
    expect(presetRatio('16:9', landscape)).toBeCloseTo(16 / 9);
    expect(presetRatio('9:16', landscape)).toBeCloseTo(9 / 16);
  });

  it('swaps 16:9 and 9:16 by changing chips', () => {
    expect(swapRatio({ preset: '16:9', ratio: 16 / 9 })).toEqual({ preset: '9:16', ratio: 9 / 16 });
    expect(swapRatio({ preset: '9:16', ratio: 9 / 16 })).toEqual({ preset: '16:9', ratio: 16 / 9 });
  });

  it('keeps the chip and inverts the ratio for the others', () => {
    const swapped = swapRatio({ preset: '4:5', ratio: 0.8 });
    expect(swapped.preset).toBe('4:5');
    expect(swapped.ratio).toBeCloseTo(1.25);
    expect(swapRatio(swapped).ratio).toBeCloseTo(0.8);
    expect(swapRatio({ preset: 'original', ratio: 1.5 })).toEqual({ preset: 'original', ratio: 1 / 1.5 });
  });

  it('leaves Free and square shapes alone', () => {
    const free: Aspect = { preset: 'free', ratio: null };
    const square: Aspect = { preset: '1:1', ratio: 1 };
    expect(canSwapRatio(free)).toBe(false);
    expect(canSwapRatio(square)).toBe(false);
    expect(canSwapRatio({ preset: '3:2', ratio: 1.5 })).toBe(true);
    expect(swapRatio(free)).toBe(free);
    expect(swapRatio(square)).toBe(square);
  });

  it('labels the swapped forms', () => {
    expect(aspectLabel({ preset: 'free', ratio: null })).toBe('Free');
    expect(aspectLabel({ preset: 'original', ratio: 0.5 })).toBe('Original');
    expect(aspectLabel({ preset: '4:5', ratio: 0.8 })).toBe('4:5');
    expect(aspectLabel(swapRatio({ preset: '4:5', ratio: 0.8 }))).toBe('5:4');
    expect(aspectLabel(swapRatio({ preset: '3:2', ratio: 1.5 }))).toBe('2:3');
    expect(aspectLabel(swapRatio({ preset: '16:9', ratio: 16 / 9 }))).toBe('9:16');
    expect(aspectLabel({ preset: '1:1', ratio: 1 })).toBe('1:1');
  });
});

describe('fitAspect', () => {
  const presets: AspectPreset[] = ['original', '1:1', '4:5', '3:2', '16:9', '9:16'];
  const frames: Size[] = [
    { width: 300, height: 200 },
    { width: 200, height: 300 },
    { width: 1024, height: 1024 },
  ];

  describe.each(frames)('frame $width x $height', (frame) => {
    it.each(presets)('fills the frame along one side for %s and keeps the ratio', (preset) => {
      const ratio = presetRatio(preset, frame)!;
      const box = fitAspect(null, ratio, frame);
      expect(box.x).toBeGreaterThanOrEqual(0);
      expect(box.y).toBeGreaterThanOrEqual(0);
      expect(box.x + box.width).toBeLessThanOrEqual(frame.width);
      expect(box.y + box.height).toBeLessThanOrEqual(frame.height);
      expect(Number.isInteger(box.width) && Number.isInteger(box.height)).toBe(true);
      expect(box.width === frame.width || box.height === frame.height).toBe(true);
      expect(Math.abs(box.width / box.height - ratio)).toBeLessThan(0.02);
    });
  });

  it('centres on the frame without a crop', () => {
    expect(fitAspect(null, 1, { width: 300, height: 200 })).toEqual({ x: 50, y: 0, width: 200, height: 200 });
    expect(fitAspect(null, 1, { width: 200, height: 300 })).toEqual({ x: 0, y: 50, width: 200, height: 200 });
  });

  it('centres on the current box and clamps to the frame', () => {
    const frame = { width: 300, height: 200 };
    // Box centre is (240, 100): the 200 px square would start at 140 and overrun, so it clamps to 100.
    expect(fitAspect({ x: 200, y: 80, width: 80, height: 40 }, 1, frame)).toEqual({ x: 100, y: 0, width: 200, height: 200 });
    // Box centre is (150, 100): a centred square starts at 50.
    expect(fitAspect({ x: 100, y: 50, width: 100, height: 100 }, 1, frame)).toEqual({ x: 50, y: 0, width: 200, height: 200 });
  });

  it('shifts a full-width box vertically toward the current box', () => {
    const frame = { width: 300, height: 200 };
    const box = fitAspect({ x: 0, y: 150, width: 40, height: 40 }, 3, frame);
    expect(box).toEqual({ x: 0, y: 100, width: 300, height: 100 });
  });
});

describe('dragCrop', () => {
  const frame: Size = { width: 300, height: 200 };
  const start: Crop = { x: 100, y: 60, width: 100, height: 80 };
  const MIN = 20;
  const all: Handle[] = ['n', 's', 'e', 'w', 'ne', 'nw', 'se', 'sw'];

  function inside(c: Crop): boolean {
    return c.x >= 0 && c.y >= 0 && c.x + c.width <= frame.width && c.y + c.height <= frame.height;
  }

  describe('Free', () => {
    it('moves a single edge and keeps the opposite one', () => {
      expect(dragCrop(start, 'e', 30, 99, null, frame, MIN)).toEqual({ x: 100, y: 60, width: 130, height: 80 });
      expect(dragCrop(start, 'w', -30, 99, null, frame, MIN)).toEqual({ x: 70, y: 60, width: 130, height: 80 });
      expect(dragCrop(start, 'n', 99, -10, null, frame, MIN)).toEqual({ x: 100, y: 50, width: 100, height: 90 });
      expect(dragCrop(start, 's', 99, 10, null, frame, MIN)).toEqual({ x: 100, y: 60, width: 100, height: 90 });
    });

    it('moves both edges of a corner', () => {
      expect(dragCrop(start, 'se', 10, 20, null, frame, MIN)).toEqual({ x: 100, y: 60, width: 110, height: 100 });
      expect(dragCrop(start, 'nw', -10, -20, null, frame, MIN)).toEqual({ x: 90, y: 40, width: 110, height: 100 });
      expect(dragCrop(start, 'ne', 10, -20, null, frame, MIN)).toEqual({ x: 100, y: 40, width: 110, height: 100 });
      expect(dragCrop(start, 'sw', -10, 20, null, frame, MIN)).toEqual({ x: 90, y: 60, width: 110, height: 100 });
    });

    it.each(all)('clamps %s to the frame', (handle) => {
      const grown = dragCrop(start, handle, 9999, 9999, null, frame, MIN);
      const shrunk = dragCrop(start, handle, -9999, -9999, null, frame, MIN);
      expect(inside(grown)).toBe(true);
      expect(inside(shrunk)).toBe(true);
    });

    it.each(all)('never goes below the minimum for %s', (handle) => {
      for (const [dx, dy] of [
        [9999, 9999],
        [-9999, -9999],
        [9999, -9999],
        [-9999, 9999],
      ]) {
        const c = dragCrop(start, handle, dx, dy, null, frame, MIN);
        expect(c.width).toBeGreaterThanOrEqual(MIN);
        expect(c.height).toBeGreaterThanOrEqual(MIN);
        expect(inside(c)).toBe(true);
      }
    });

    it('stops at the minimum with the opposite edge fixed', () => {
      expect(dragCrop(start, 'e', -500, 0, null, frame, MIN)).toEqual({ x: 100, y: 60, width: 20, height: 80 });
      expect(dragCrop(start, 'w', 500, 0, null, frame, MIN)).toEqual({ x: 180, y: 60, width: 20, height: 80 });
    });

    it('does not drift: the result depends on the start state and the total delta only', () => {
      const once = dragCrop(start, 'se', 40, 25, null, frame, MIN);
      dragCrop(start, 'se', 500, 500, null, frame, MIN);
      expect(dragCrop(start, 'se', 40, 25, null, frame, MIN)).toEqual(once);
    });
  });

  describe('locked ratio', () => {
    const ratios = [1, 4 / 5, 3 / 2, 16 / 9, 9 / 16];
    const corners: Handle[] = ['ne', 'nw', 'se', 'sw'];
    const locked: Crop = { x: 100, y: 50, width: 90, height: 60 };

    it.each(ratios)('keeps ratio %f, the frame and the minimum from every corner', (ratio) => {
      const s: Crop = { x: 100, y: 50, width: Math.round(60 * ratio), height: 60 };
      for (const handle of corners) {
        for (const [dx, dy] of [
          [0, 0],
          [25, -13],
          [-40, 30],
          [9999, 9999],
          [-9999, -9999],
          [9999, -9999],
          [-9999, 9999],
        ]) {
          const c = dragCrop(s, handle, dx, dy, ratio, frame, MIN);
          expect(inside(c)).toBe(true);
          expect(c.width).toBeGreaterThanOrEqual(MIN);
          expect(c.height).toBeGreaterThanOrEqual(MIN);
          expect(Math.abs(c.width / c.height - ratio)).toBeLessThan(0.5 / c.height + 1e-9);
        }
      }
    });

    it('anchors the opposite corner', () => {
      const s: Crop = { x: 100, y: 60, width: 80, height: 80 };
      const se = dragCrop(s, 'se', 20, 20, 1, frame, MIN);
      expect(se).toEqual({ x: 100, y: 60, width: 100, height: 100 });
      const nw = dragCrop(s, 'nw', -20, -20, 1, frame, MIN);
      expect(nw).toEqual({ x: 80, y: 40, width: 100, height: 100 });
      const ne = dragCrop(s, 'ne', 20, -20, 1, frame, MIN);
      expect(ne).toEqual({ x: 100, y: 40, width: 100, height: 100 });
      const sw = dragCrop(s, 'sw', -20, 20, 1, frame, MIN);
      expect(sw).toEqual({ x: 80, y: 60, width: 100, height: 100 });
    });

    it('grows as far as the frame allows', () => {
      const s: Crop = { x: 100, y: 60, width: 80, height: 80 };
      expect(dragCrop(s, 'se', 9999, 9999, 1, frame, MIN)).toEqual({ x: 100, y: 60, width: 140, height: 140 });
      expect(dragCrop(s, 'nw', -9999, -9999, 1, frame, MIN)).toEqual({ x: 40, y: 0, width: 140, height: 140 });
    });

    it('shrinks to the minimum on the long side of a narrow ratio', () => {
      const s: Crop = { x: 100, y: 20, width: 45, height: 80 };
      const c = dragCrop(s, 'se', -9999, -9999, 9 / 16, frame, MIN);
      expect(c.width).toBeGreaterThanOrEqual(MIN);
      expect(c.height).toBeGreaterThanOrEqual(MIN);
      expect(c.x).toBe(100);
      expect(c.y).toBe(20);
    });

    it('ignores edge handles', () => {
      for (const handle of ['n', 's', 'e', 'w'] as Handle[]) {
        expect(dragCrop(locked, handle, 30, 30, 1.5, frame, MIN)).toBe(locked);
      }
    });

    it('does not drift', () => {
      const once = dragCrop(locked, 'se', 30, 12, 1.5, frame, MIN);
      dragCrop(locked, 'se', 9999, 9999, 1.5, frame, MIN);
      expect(dragCrop(locked, 'se', 30, 12, 1.5, frame, MIN)).toEqual(once);
    });
  });
});

describe('moveCrop', () => {
  const frame: Size = { width: 300, height: 200 };
  const start: Crop = { x: 100, y: 60, width: 100, height: 80 };

  it('moves and keeps the size', () => {
    expect(moveCrop(start, 10, -20, frame)).toEqual({ x: 110, y: 40, width: 100, height: 80 });
  });

  it('clamps to every edge', () => {
    expect(moveCrop(start, -9999, -9999, frame)).toEqual({ x: 0, y: 0, width: 100, height: 80 });
    expect(moveCrop(start, 9999, 9999, frame)).toEqual({ x: 200, y: 120, width: 100, height: 80 });
  });
});

describe('nudgeCrop', () => {
  const crop: Crop = { x: 100, y: 100, width: 50, height: 50 };

  it('moves by max(1, round(min(Wf, Hf) / 100)) per arrow press', () => {
    const frame = { width: 1000, height: 800 }; // step 8
    expect(nudgeCrop(crop, 'ArrowRight', false, frame)).toEqual({ ...crop, x: 108 });
    expect(nudgeCrop(crop, 'ArrowLeft', false, frame)).toEqual({ ...crop, x: 92 });
    expect(nudgeCrop(crop, 'ArrowDown', false, frame)).toEqual({ ...crop, y: 108 });
    expect(nudgeCrop(crop, 'ArrowUp', false, frame)).toEqual({ ...crop, y: 92 });
  });

  it('moves ten times as far with Shift', () => {
    const frame = { width: 1000, height: 800 };
    expect(nudgeCrop(crop, 'ArrowRight', true, frame)).toEqual({ ...crop, x: 180 });
  });

  it('steps at least one pixel on a small frame', () => {
    const frame = { width: 120, height: 120 };
    const small = { x: 10, y: 10, width: 20, height: 20 };
    expect(nudgeCrop(small, 'ArrowRight', false, frame).x).toBe(11);
    expect(nudgeCrop(small, 'ArrowRight', true, frame).x).toBe(20);
  });

  it('uses the short side of the frame', () => {
    expect(nudgeCrop(crop, 'ArrowRight', false, { width: 4000, height: 300 }).x).toBe(103);
  });

  it('clamps at the frame edge', () => {
    const frame = { width: 1000, height: 800 };
    const right = { x: 945, y: 0, width: 50, height: 50 };
    expect(nudgeCrop(right, 'ArrowRight', false, frame).x).toBe(950);
    expect(nudgeCrop(right, 'ArrowRight', true, frame).x).toBe(950);
    expect(nudgeCrop(right, 'ArrowUp', true, frame).y).toBe(0);
    expect(nudgeCrop({ x: 3, y: 0, width: 50, height: 50 }, 'ArrowLeft', true, frame).x).toBe(0);
  });

  it('never resizes and ignores other keys', () => {
    const frame = { width: 1000, height: 800 };
    const moved = nudgeCrop(crop, 'ArrowDown', true, frame);
    expect([moved.width, moved.height]).toEqual([50, 50]);
    expect(nudgeCrop(crop, 'a', false, frame)).toBe(crop);
  });
});

describe('normalizedCrop', () => {
  const frame: Size = { width: 300, height: 200 };

  it('turns a full-frame crop into null', () => {
    expect(normalizedCrop({ x: 0, y: 0, width: 300, height: 200 }, frame)).toBeNull();
    expect(normalizedCrop(null, frame)).toBeNull();
  });

  it('keeps a real crop', () => {
    const c = { x: 0, y: 0, width: 300, height: 199 };
    expect(normalizedCrop(c, frame)).toBe(c);
    expect(normalizedCrop(CROP, frame)).toBe(CROP);
  });
});

type Corners = [number, number][];

/** The eight states: quadCorners for the whole frame of a 300 x 200 image, hand-derived from the frame definition. */
const EXPECTED_CORNERS: { rotate: Rotate; flip: boolean; corners: Corners }[] = [
  { rotate: 0, flip: false, corners: [[0, 0], [1, 0], [1, 1], [0, 1]] },
  { rotate: 90, flip: false, corners: [[0, 1], [0, 0], [1, 0], [1, 1]] },
  { rotate: 180, flip: false, corners: [[1, 1], [0, 1], [0, 0], [1, 0]] },
  { rotate: 270, flip: false, corners: [[1, 0], [1, 1], [0, 1], [0, 0]] },
  { rotate: 0, flip: true, corners: [[1, 0], [0, 0], [0, 1], [1, 1]] },
  { rotate: 90, flip: true, corners: [[0, 0], [0, 1], [1, 1], [1, 0]] },
  { rotate: 180, flip: true, corners: [[0, 1], [1, 1], [1, 0], [0, 0]] },
  { rotate: 270, flip: true, corners: [[1, 1], [1, 0], [0, 0], [0, 1]] },
];

describe('quadCorners', () => {
  it.each(EXPECTED_CORNERS)('rotate $rotate flip $flip: frame', ({ rotate, flip, corners }) => {
    const q = quadCorners(geometry(rotate, flip), W, H, 'frame');
    q.forEach(([u, v], i) => {
      expect(u).toBeCloseTo(corners[i][0], 9);
      expect(v).toBeCloseTo(corners[i][1], 9);
    });
  });

  it('puts the top-left of a quarter turn at (0, 1)', () => {
    expect(quadCorners(geometry(90, false), W, H, 'frame')[0]).toEqual([0, 1]);
  });

  it('frame ignores the crop and equals crop when there is none', () => {
    const withCrop = geometry(0, false, CROP);
    expect(quadCorners(withCrop, W, H, 'frame')).toEqual(quadCorners(geometry(0, false), W, H, 'frame'));
    expect(quadCorners(geometry(90, true), W, H, 'crop')).toEqual(quadCorners(geometry(90, true), W, H, 'frame'));
  });

  it('crops an unrotated frame to the crop rectangle', () => {
    const q = quadCorners(geometry(0, false, CROP), W, H, 'crop');
    expect(q[0][0]).toBeCloseTo(20 / 300);
    expect(q[0][1]).toBeCloseTo(10 / 200);
    expect(q[2][0]).toBeCloseTo(120 / 300);
    expect(q[2][1]).toBeCloseTo(70 / 200);
  });

  it.each(EXPECTED_CORNERS)('crop follows the transform for rotate $rotate flip $flip', ({ rotate, flip }) => {
    // Cropping a rotated, flipped image equals the cropped region of the same transform of the whole frame.
    const frame = frameSize(rotate, W, H);
    const crop = { x: frame.width / 4, y: frame.height / 2, width: frame.width / 2, height: frame.height / 4 };
    const full = quadCorners(geometry(rotate, flip), W, H, 'frame');
    const part = quadCorners(geometry(rotate, flip, crop), W, H, 'crop');
    // Bilinear interpolation of the full quad at the crop corners.
    const at = (fx: number, fy: number): [number, number] => {
      const lerp = (a: readonly [number, number], b: readonly [number, number], t: number): [number, number] => [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
      ];
      const top = lerp(full[0], full[1], fx);
      const bottom = lerp(full[3], full[2], fx);
      return lerp(top, bottom, fy);
    };
    const expected = [at(0.25, 0.5), at(0.75, 0.5), at(0.75, 0.75), at(0.25, 0.75)];
    part.forEach(([u, v], i) => {
      expect(u).toBeCloseTo(expected[i][0], 9);
      expect(v).toBeCloseTo(expected[i][1], 9);
    });
  });
});

/** Parse `matrix(a, b, c, d, e, f)`. */
function parseMatrix(transform: string): number[] {
  const m = /^matrix\((.+)\)$/.exec(transform);
  expect(m).not.toBeNull();
  return m![1].split(',').map(Number);
}

describe('cssFrameTransform', () => {
  const EXPECTED_MATRIX: Record<string, string> = {
    '0 false': 'matrix(1, 0, 0, 1, 0, 0)',
    '90 false': 'matrix(0, 1, -1, 0, 200, 0)',
    '180 false': 'matrix(-1, 0, 0, -1, 300, 200)',
    '270 false': 'matrix(0, -1, 1, 0, 0, 300)',
    '0 true': 'matrix(-1, 0, 0, 1, 300, 0)',
    '90 true': 'matrix(0, 1, 1, 0, 0, 0)',
    '180 true': 'matrix(1, 0, 0, -1, 0, 200)',
    '270 true': 'matrix(0, -1, -1, 0, 200, 300)',
  };

  it.each(EXPECTED_CORNERS)('rotate $rotate flip $flip: whole frame at scale 1', ({ rotate, flip }) => {
    const frame = frameSize(rotate, W, H);
    const css = cssFrameTransform(geometry(rotate, flip), W, H, frame, 'frame');
    expect(css.transform).toBe(EXPECTED_MATRIX[`${rotate} ${flip}`]);
    expect(css.width).toBe(frame.width);
    expect(css.height).toBe(frame.height);
    expect(css.clip).toBe('none');
  });

  it.each(EXPECTED_CORNERS)('rotate $rotate flip $flip: sends the quad corners to the box corners', ({ rotate, flip }) => {
    const frame = frameSize(rotate, W, H);
    for (const region of ['frame', 'crop'] as const) {
      const crop = { x: 13, y: 7, width: Math.floor(frame.width / 2), height: Math.floor(frame.height / 3) };
      const g = geometry(rotate, flip, crop);
      const css = cssFrameTransform(g, W, H, { width: 600, height: 600 }, region);
      const [a, b, c, d, e, f] = parseMatrix(css.transform);
      const quad = quadCorners(g, W, H, region);
      const expected = [
        [0, 0],
        [css.width, 0],
        [css.width, css.height],
        [0, css.height],
      ];
      quad.forEach(([u, v], i) => {
        const sx = u * W;
        const sy = v * H;
        expect(a * sx + c * sy + e).toBeCloseTo(expected[i][0], 4);
        expect(b * sx + d * sy + f).toBeCloseTo(expected[i][1], 4);
      });
    }
  });

  it('fits the region into the box, keeping its aspect', () => {
    const css = cssFrameTransform(geometry(0, false), W, H, { width: 150, height: 150 }, 'frame');
    expect(css.width).toBeCloseTo(150);
    expect(css.height).toBeCloseTo(100);
    expect(css.transform).toBe('matrix(0.5, 0, 0, 0.5, 0, 0)');
  });

  it('scales, offsets and clips for a crop', () => {
    const g = geometry(0, false, CROP);
    const css = cssFrameTransform(g, W, H, { width: 200, height: 200 }, 'crop');
    expect(css.width).toBeCloseTo(200);
    expect(css.height).toBeCloseTo(120);
    expect(css.transform).toBe('matrix(2, 0, 0, 2, -40, -20)');
    expect(css.clip).toBe('inset(10px 180px 130px 20px)');
  });

  it('clips in source pixels when rotated and flipped', () => {
    const g = geometry(90, true, { x: 0, y: 0, width: 100, height: 50 });
    const css = cssFrameTransform(g, W, H, { width: 100, height: 50 }, 'crop');
    // Flip of a clockwise quarter turn is a transpose, so the frame's top-left strip is the source's top-left.
    expect(css.clip).toBe('inset(0px 250px 100px 0px)');
  });

  it('does not clip the frame region even with a crop set', () => {
    const css = cssFrameTransform(geometry(90, false, CROP), W, H, { width: 200, height: 300 }, 'frame');
    expect(css.clip).toBe('none');
  });

  it('never emits negative zero', () => {
    for (const { rotate, flip } of EXPECTED_CORNERS) {
      const css = cssFrameTransform(geometry(rotate, flip), W, H, frameSize(rotate, W, H), 'frame');
      expect(css.transform).not.toContain('-0,');
      expect(css.transform).not.toContain('-0)');
    }
  });
});
