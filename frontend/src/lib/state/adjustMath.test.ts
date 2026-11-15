import { describe, expect, it } from 'vitest';

import adjustCases from '../../../../tests/fixtures/adjust_cases.json';
import { KB, KG, KR, MIX_CENTRES, MIX_NAMES, NEUTRAL_LIGHT, SLIDER_MAX, SLIDER_MIN, TONE_AWAY, TONE_TOWARD, adjustPixel, isNeutralLight, lerpTable, mixFactor, previewPixel, previewTables, type LightSettings } from './adjustMath';

type Case = { name: string; settings: Partial<LightSettings>; pixels: number[][]; expected: number[][] };
const CASES = adjustCases as Case[];

const maxDiff = (a: readonly number[], b: readonly number[]): number => Math.max(...a.map((v, i) => Math.abs(v - b[i])));

// Mulberry32: a small seeded generator, so a failure reproduces.
function seeded(seed: number): () => number {
  let a = seed;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// tests/fixtures/adjust_cases.json is also checked by pytest against `adjust_light` in processing/adjust.py: both sides agree on the pixels.
describe('adjustPixel', () => {
  it.each(CASES)('matches the fixture within one step: $name', ({ settings, pixels, expected }) => {
    const light = { ...NEUTRAL_LIGHT, ...settings };
    pixels.forEach(([r, g, b], i) => {
      expect(maxDiff(adjustPixel(r, g, b, light), expected[i])).toBeLessThanOrEqual(1);
    });
  });

  it('returns the input exactly when every setting is 0', () => {
    for (let v = 0; v < 256; v += 5) expect(adjustPixel(v, 255 - v, (v * 7) % 256, NEUTRAL_LIGHT)).toEqual([v, 255 - v, (v * 7) % 256]);
  });

  it('keeps the contrast pivot fixed', () => {
    expect(adjustPixel(128, 128, 128, { ...NEUTRAL_LIGHT, contrast: 100 })).toEqual([128, 128, 128]);
  });

  it('returns contrast +c followed by -c within one step', () => {
    for (const c of [30, 70, 100]) {
      for (let v = 0; v < 256; v += 1) {
        const [up] = adjustPixel(v, v, v, { ...NEUTRAL_LIGHT, contrast: c });
        const [back] = adjustPixel(up, up, up, { ...NEUTRAL_LIGHT, contrast: -c });
        expect(Math.abs(back - v)).toBeLessThanOrEqual(1);
      }
    }
  });

  it('turns every pixel grey at saturation -100', () => {
    const [r, g, b] = adjustPixel(200, 40, 40, { ...NEUTRAL_LIGHT, saturation: -100 });
    expect(r).toBe(g);
    expect(g).toBe(b);
  });
});

describe('vibrance', () => {
  it('leaves greys alone and keeps a saturated colour while boosting a muted one', () => {
    const light = { ...NEUTRAL_LIGHT, vibrance: 100 };
    expect(adjustPixel(128, 128, 128, light)).toEqual([128, 128, 128]);
    expect(adjustPixel(200, 40, 40, light)).toEqual([200, 40, 40]);
    const [r, , b] = adjustPixel(150, 130, 120, light);
    expect(r - b).toBeGreaterThan(150 - 120);
  });

  it('gives black and white at saturation -100 whatever the vibrance', () => {
    const [r, g, b] = adjustPixel(150, 130, 120, { ...NEUTRAL_LIGHT, saturation: -100, vibrance: 100 });
    expect(r).toBe(g);
    expect(g).toBe(b);
  });

  it('puts the boost and the mute in the preview tables as one scalar', () => {
    expect(previewTables({ ...NEUTRAL_LIGHT, vibrance: 50 }).vib).toBeCloseTo(1);
    expect(previewTables({ ...NEUTRAL_LIGHT, vibrance: -50 }).vib).toBeCloseTo(-0.5);
    expect(previewTables(NEUTRAL_LIGHT).vib).toBe(0);
  });
});

describe('colour mixer', () => {
  // Rec. 709 linear Y is the luminance both sides keep; the tests read it back through a mix-only setting.
  const none = new Array<number>(8).fill(0);
  const only = (band: number, amount: number): number[] => none.map((_, i) => (i === band ? amount : 0));

  // A colour of Oklab hue `hue` degrees at chroma 0.1, as linear RGB (not clamped, so the hue is exact).
  const colourAt = (hue: number): [number, number, number] => {
    const a = 0.1 * Math.cos((hue * Math.PI) / 180);
    const b = 0.1 * Math.sin((hue * Math.PI) / 180);
    const l = (0.7 + 0.3963377774 * a + 0.2158037573 * b) ** 3;
    const m = (0.7 - 0.1055613458 * a - 0.0638541728 * b) ** 3;
    const s = (0.7 - 0.0894841775 * a - 1.291485548 * b) ** 3;
    return [4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s, -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s, -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s];
  };

  // The mixer's factor for a colour (given as linear RGB) whose saturation and vibrance factor is `base`.
  const factor = (c: readonly [number, number, number], mix: readonly number[], base = 1): number => mixFactor(c[0], c[1], c[2], KR * c[0] + KG * c[1] + KB * c[2], base, mix);

  it('has weights that sum to 1 at every hue: muting all eight bands fully gives a factor of 0', () => {
    const all = none.map(() => -1);
    for (let h = 0; h < 360; h += 0.1) expect(Math.abs(factor(colourAt(h), all))).toBeLessThan(1e-9);
  });

  it('keeps full weight for a quarter of the gap on each side of a band centre', () => {
    for (let band = 0; band < 8; band += 1) {
      const centre = MIX_CENTRES[band];
      const next = MIX_CENTRES[(band + 1) % 8] + (band === 7 ? 360 : 0);
      const previous = MIX_CENTRES[(band + 7) % 8] - (band === 0 ? 360 : 0);
      for (const h of [centre - 0.25 * (centre - previous), centre - 0.125 * (centre - previous), centre, centre + 0.125 * (next - centre), centre + 0.25 * (next - centre)]) {
        expect(factor(colourAt(h), only(band, -1))).toBeCloseTo(0, 9);
      }
      expect(factor(colourAt(centre + 0.5 * (next - centre)), only(band, -1))).toBeGreaterThan(0.4);
    }
  });

  it('has no seam: a single band changes smoothly with hue, across every centre and the wrap at 360', () => {
    for (let band = 0; band < 8; band += 1) {
      let previous = factor(colourAt(0), only(band, -1));
      for (let h = 0.05; h <= 360; h += 0.05) {
        const value = factor(colourAt(h), only(band, -1));
        expect(Math.abs(value - previous)).toBeLessThan(0.01);
        previous = value;
      }
    }
  });

  it('leaves a grey at factor 1 for any setting, and the result finite', () => {
    for (const v of [0, 1e-9, 0.001, 0.18, 0.5, 1]) {
      for (let band = 0; band < 8; band += 1) {
        for (const amount of [-1, 1]) expect(factor([v, v, v], only(band, amount))).toBe(1);
      }
    }
    for (let v = 0; v < 256; v += 17) {
      const out = adjustPixel(v, v, v, { ...NEUTRAL_LIGHT, mix_red: 100, mix_blue: -100, mix_green: 100 });
      expect(out.every(Number.isFinite)).toBe(true);
      expect(out).toEqual([v, v, v]);
    }
  });

  it('turns a colour on its band centre grey when that band is muted fully', () => {
    const orange = [196, 128, 80] as const;
    const [r, g, b] = adjustPixel(...orange, { ...NEUTRAL_LIGHT, mix_orange: -100 });
    expect(r).toBe(g);
    expect(g).toBe(b);
  });

  it('never clips or shifts the hue of a colour with a boost: every channel scales by the same factor', () => {
    const random = seeded(77);
    const boost = none.map(() => 1);
    for (let n = 0; n < 3000; n += 1) {
      const c = [random(), random(), random()].map((v) => v ** 2.2) as [number, number, number];
      const y = KR * c[0] + KG * c[1] + KB * c[2];
      const f = mixFactor(c[0], c[1], c[2], y, 1, boost);
      const out = c.map((v) => y + f * (v - y));
      const scales = out.map((o, i) => [o, c[i]]).filter(([, v]) => Math.abs(v - y) > 1e-3).map(([o, v]) => (o - y) / (v - y));
      expect(Math.max(...out)).toBeLessThanOrEqual(1 + 1e-9);
      expect(Math.min(...out)).toBeGreaterThanOrEqual(-1e-9);
      for (const sc of scales) expect(sc).toBeCloseTo(f, 9);
    }
  });

  it('gives a muted colour a stronger boost than a vivid one, and none to a near-grey', () => {
    const blue = (chroma: number): [number, number, number] => {
      const a = chroma * Math.cos((264 * Math.PI) / 180);
      const b = chroma * Math.sin((264 * Math.PI) / 180);
      const l = (0.6 + 0.3963377774 * a + 0.2158037573 * b) ** 3;
      const m = (0.6 - 0.1055613458 * a - 0.0638541728 * b) ** 3;
      const s = (0.6 - 0.0894841775 * a - 1.291485548 * b) ** 3;
      return [4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s, -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s, -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s];
    };
    const boost = only(5, 1);
    expect(factor(blue(0.02), boost)).toBeGreaterThan(factor(blue(0.1), boost));
    expect(factor(blue(0.02), boost)).toBeGreaterThan(3.5);
    expect(factor(blue(0.001), boost)).toBe(1);
  });

  it('keeps the bands in MIX_NAMES order and centres ascending', () => {
    expect(MIX_NAMES).toHaveLength(8);
    expect([...MIX_CENTRES]).toEqual([...MIX_CENTRES].sort((a, b) => a - b));
  });

  it('puts the amounts in the preview tables, and mixOn only when one is set', () => {
    expect(previewTables(NEUTRAL_LIGHT)).toMatchObject({ mixOn: false, mix: [0, 0, 0, 0, 0, 0, 0, 0] });
    const t = previewTables({ ...NEUTRAL_LIGHT, mix_orange: 50, mix_magenta: -100 });
    expect(t.mixOn).toBe(true);
    expect(t.mix).toEqual([0, 0.5, 0, 0, 0, 0, 0, -1]);
  });
});

describe('isNeutralLight', () => {
  it('is true only when every setting is 0', () => {
    expect(isNeutralLight(NEUTRAL_LIGHT)).toBe(true);
    for (const key of Object.keys(NEUTRAL_LIGHT) as (keyof LightSettings)[]) expect(isNeutralLight({ ...NEUTRAL_LIGHT, [key]: 1 })).toBe(false);
  });
});

describe('previewTables', () => {
  it('has the sizes the shader uploads', () => {
    const t = previewTables(NEUTRAL_LIGHT);
    expect(t.decode.length).toBe(256);
    expect(t.tone.length).toBe(4097);
    expect(t.shift.length).toBe(1025);
    expect(t.gains).toHaveLength(3);
    expect(t.sat).toBe(1);
  });

  it('keeps the tone curve non-decreasing for every extreme of highlights and shadows', () => {
    for (const shadows of [SLIDER_MIN, -50, 0, 50, SLIDER_MAX]) {
      for (const highlights of [SLIDER_MIN, -50, 0, 50, SLIDER_MAX]) {
        const { shift } = previewTables({ ...NEUTRAL_LIGHT, shadows, highlights });
        let previous = -Infinity;
        for (let j = 0; j <= 1024; j += 1) {
          const y2 = j / 1024 + shift[j];
          // Float32 table noise is far below one 8-bit step.
          expect(y2).toBeGreaterThanOrEqual(previous - 1e-6);
          previous = y2;
        }
        expect(shift[0]).toBeCloseTo(0, 6);
        expect(shift[1024]).toBeCloseTo(0, 6);
      }
    }
    expect(TONE_TOWARD).toBeGreaterThan(TONE_AWAY);
  });
});

describe('lerpTable', () => {
  it('interpolates linearly and clamps the last interval', () => {
    const table = new Float32Array([0, 1, 3]);
    expect(lerpTable(table, 0)).toBe(0);
    expect(lerpTable(table, 0.25)).toBeCloseTo(0.5);
    expect(lerpTable(table, 0.75)).toBeCloseTo(2);
    expect(lerpTable(table, 1)).toBe(3);
  });
});

describe('previewPixel', () => {
  it.each(CASES)('matches the fixture within one step: $name', ({ settings, pixels, expected }) => {
    const tables = previewTables({ ...NEUTRAL_LIGHT, ...settings });
    pixels.forEach(([r, g, b], i) => {
      expect(maxDiff(previewPixel(r, g, b, tables), expected[i])).toBeLessThanOrEqual(1);
    });
  });

  it('returns every grey level unchanged when every setting is 0', () => {
    const tables = previewTables(NEUTRAL_LIGHT);
    for (let v = 0; v < 256; v += 1) expect(previewPixel(v, v, v, tables)).toEqual([v, v, v]);
  });

  it('keeps black and white finite and within one step of adjustPixel for every contrast', () => {
    for (let contrast = SLIDER_MIN; contrast <= SLIDER_MAX; contrast += 1) {
      const light: LightSettings = { ...NEUTRAL_LIGHT, contrast };
      const tables = previewTables(light);
      for (const v of [0, 255]) {
        const preview = previewPixel(v, v, v, tables);
        expect(preview.every(Number.isFinite), `contrast ${contrast} on ${v}`).toBe(true);
        expect(maxDiff(preview, adjustPixel(v, v, v, light)), `contrast ${contrast} on ${v}`).toBeLessThanOrEqual(1);
      }
    }
  });

  it('stays within two steps of adjustPixel on random colours and settings, mixes included', () => {
    const random = seeded(20260515);
    const int = (min: number, max: number): number => min + Math.floor(random() * (max - min + 1));
    const setting = (): number => (random() < 0.3 ? 0 : int(SLIDER_MIN, SLIDER_MAX));
    let worst = 0;
    // 1000 setting sets x 20 colours = 20k samples; the tables are the slow part, so each set is built once.
    for (let n = 0; n < 1000; n += 1) {
      const light: LightSettings = {
        exposure: setting(),
        contrast: setting(),
        highlights: setting(),
        shadows: setting(),
        saturation: setting(),
        vibrance: setting(),
        warmth: setting(),
        mix_red: setting(),
        mix_orange: setting(),
        mix_yellow: setting(),
        mix_green: setting(),
        mix_aqua: setting(),
        mix_blue: setting(),
        mix_purple: setting(),
        mix_magenta: setting(),
      };
      const tables = previewTables(light);
      for (let k = 0; k < 20; k += 1) {
        const [r, g, b] = [int(0, 255), int(0, 255), int(0, 255)];
        const diff = maxDiff(previewPixel(r, g, b, tables), adjustPixel(r, g, b, light));
        worst = Math.max(worst, diff);
        expect(diff, `${r},${g},${b} ${JSON.stringify(light)}`).toBeLessThanOrEqual(2);
      }
    }
    expect(worst).toBeLessThanOrEqual(2);
  });
});
