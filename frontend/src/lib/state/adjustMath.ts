// The colour maths of the viewer's Adjust editor. It mirrors `processing/adjust.py` constant for constant: `adjustPixel` is the exact float64 reference, and `previewTables` with `previewPixel` is the separable table pipeline the WebGL preview shader composes, so the saved copy and the preview agree to within one 8-bit step. Alpha is never read.

/** Fifteen integer settings, each in [SLIDER_MIN, SLIDER_MAX]; 0 leaves that quality unchanged. */
export type LightSettings = {
  exposure: number;
  contrast: number;
  highlights: number;
  shadows: number;
  saturation: number;
  vibrance: number;
  warmth: number;
  mix_red: number;
  mix_orange: number;
  mix_yellow: number;
  mix_green: number;
  mix_aqua: number;
  mix_blue: number;
  mix_purple: number;
  mix_magenta: number;
};

/** The lookup tables and scalars one frame of the preview needs. `tone` and `shift` are sampled with `lerpTable`. */
export type PreviewTables = {
  /** `lin(i / 255)` for i = 0..255. */
  decode: Float32Array;
  /** Per-channel linear-light gains from exposure and warmth. */
  gains: [number, number, number];
  /** Saturation factor, `1 + saturation / 100`. */
  sat: number;
  /** Vibrance strength, `kv * vibrance / 100` with `kv` the boost above 0 and the mute below. */
  vib: number;
  /** `contrast(encode(j / 4096))` for j = 0..4096, indexed by linear light. */
  tone: Float32Array;
  /** `D(j / 1024)` for j = 0..1024, the highlight and shadow shift, indexed by encoded luma. */
  shift: Float32Array;
  /** The eight colour-mixer amounts divided by 100, in `MIX_NAMES` order. */
  mix: number[];
  /** Whether any mixer amount is non-zero, so the hue and chroma of each pixel need computing. */
  mixOn: boolean;
};

export const NEUTRAL_LIGHT: Readonly<LightSettings> = Object.freeze({
  exposure: 0,
  contrast: 0,
  highlights: 0,
  shadows: 0,
  saturation: 0,
  vibrance: 0,
  warmth: 0,
  mix_red: 0,
  mix_orange: 0,
  mix_yellow: 0,
  mix_green: 0,
  mix_aqua: 0,
  mix_blue: 0,
  mix_purple: 0,
  mix_magenta: 0,
});

export const SLIDER_MIN = -100;
export const SLIDER_MAX = 100;
export const SLIDER_STEP = 1;

export const KR = 0.2126;
export const KG = 0.7152;
export const KB = 0.0722;
export const EV_MAX = 2;
export const CONTRAST_MAX = 1.5;
export const WARMTH_STOPS = 0.5;
export const TONE_TOWARD = 1.5;
export const TONE_AWAY = 1;
export const VIBRANCE_BOOST = 2;
export const VIBRANCE_MUTE = 1;
export const VIBRANCE_FALLOFF = 2;
export const MIX_NAMES = ['red', 'orange', 'yellow', 'green', 'aqua', 'blue', 'purple', 'magenta'] as const;
/** The band centres in degrees of Oklab hue: the hues of sRGB #FF0000, #FF8000, #FFFF00, #00FF00, #00FFFF, #0000FF, #8000FF and #FF00FF, rounded; orange is 55, nudged from #FF8000's 52.98 so skin tones sit on its centre. */
export const MIX_CENTRES = [29, 55, 110, 142, 195, 264, 294, 328] as const;
/** Each side of a band centre keeps full weight for this fraction of the gap to its neighbouring centre. */
export const MIX_FLAT = 0.25;
/** The Oklab chroma range over which a boost fades in, so a near-grey is never tinted. */
export const MIX_FADE_START = 0.002;
export const MIX_FADE_END = 0.006;
/** The boost gain at low chroma and at and above `MIX_VIVID_CHROMA`: +100 multiplies the saturation by 1 plus the gain. */
export const MIX_BOOST_MUTED = 3;
export const MIX_BOOST_VIVID = 1;
export const MIX_VIVID_CHROMA = 0.2;
/** The `p` of the soft minimum that keeps the mixer's boost inside the sRGB gamut. */
export const MIX_GAMUT_SOFTNESS = 4;
export const MIX_MUTE = 1;
/** Below this Oklab chroma the hue is undefined and the mixer does nothing. */
export const MIX_MIN_CHROMA = 1e-6;

/** The band edges for the weights: the eight centres, then the first one a full turn on. */
const MIX_EDGES = [...MIX_CENTRES, MIX_CENTRES[0] + 360];

/** Entries minus one of the preview tables. */
export const DECODE_SIZE = 256;
export const TONE_INTERVALS = 4096;
export const SHIFT_INTERVALS = 1024;

/** The eight colour-mixer settings of `l`, in `MIX_NAMES` order. */
export function mixSettings(l: Readonly<LightSettings>): number[] {
  return MIX_NAMES.map((name) => l[`mix_${name}`]);
}

export function isNeutralLight(l: Readonly<LightSettings>): boolean {
  return (
    l.exposure === 0 &&
    l.contrast === 0 &&
    l.highlights === 0 &&
    l.shadows === 0 &&
    l.saturation === 0 &&
    l.vibrance === 0 &&
    l.warmth === 0 &&
    mixSettings(l).every((v) => v === 0)
  );
}

function clamp01(x: number): number {
  return Math.min(1, Math.max(0, x));
}

function lin(v: number): number {
  return v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
}

function enc(l: number): number {
  return l <= 0.0031308 ? 12.92 * l : 1.055 * Math.pow(l, 1 / 2.4) - 0.055;
}

function contrastCurve(p: number, contrast: number): number {
  if (contrast === 0) return p;
  const b = (CONTRAST_MAX * Math.abs(contrast)) / 100;
  const centred = 2 * p - 1;
  if (contrast > 0) return 0.5 + (0.5 * Math.tanh(b * centred)) / Math.tanh(b);
  return 0.5 + (0.5 * Math.atanh(centred * Math.tanh(b))) / b;
}

/** The highlight and shadow shift for an encoded luma `y`. */
function toneShift(y: number, highlights: number, shadows: number): number {
  const ks = shadows > 0 ? TONE_TOWARD : TONE_AWAY;
  const kh = highlights < 0 ? TONE_TOWARD : TONE_AWAY;
  return (shadows / 100) * ks * y * Math.pow(1 - y, 3) + (highlights / 100) * kh * Math.pow(y, 3) * (1 - y);
}

function channelGains(l: Readonly<LightSettings>): [number, number, number] {
  const k = l.warmth / 100;
  const g0 = Math.pow(2, WARMTH_STOPS * k);
  const g2 = Math.pow(2, -WARMTH_STOPS * k);
  const n = KR * g0 + KG + KB * g2;
  const x = Math.pow(2, (EV_MAX * l.exposure) / 100);
  return [(x * g0) / n, x / n, (x * g2) / n];
}

/** The vibrance strength `kv * v / 100`: the boost applies above 0, the milder mute below. */
function vibranceStrength(vibrance: number): number {
  return ((vibrance > 0 ? VIBRANCE_BOOST : VIBRANCE_MUTE) * vibrance) / 100;
}

/** A pixel's chroma relative to its brightness in linear light: 0 is grey, 1 a pure primary. */
function chromaRatio(r: number, g: number, b: number): number {
  const mx = Math.max(r, g, b);
  return mx > 0 ? (mx - Math.min(r, g, b)) / mx : 0;
}

function smoothstep(edge0: number, edge1: number, x: number): number {
  const u = clamp01((x - edge0) / (edge1 - edge0));
  return u * u * (3 - 2 * u);
}

/** The largest saturation factor at which no channel of a linear RGB pixel with luminance `y` clips; infinity for a grey. */
function gamutFactor(lr: number, lg: number, lb: number, y: number): number {
  let limit = Infinity;
  for (const v of [lr, lg, lb]) {
    if (v - y > 1e-9) limit = Math.min(limit, (1 - y) / (v - y));
    else if (v - y < -1e-9) limit = Math.min(limit, y / (y - v));
  }
  return limit;
}

/**
 * The saturation factor of a pixel of linear RGB once the colour mixer has changed `base`, the factor of saturation and vibrance.
 *
 * The hue and chroma are Oklab's, read from the pixel as given. The band amount blends the two bands whose centres bracket the hue:
 * full weight on each side of a centre, a raised cosine between, so the weights sum to 1 everywhere. A boost adds a factor that is
 * strongest on muted colours and fades in with chroma, so a near-grey is never tinted; a soft minimum keeps it inside the room left
 * before a channel clips, so the mixer never clips a colour or shifts its hue. A mute scales `base` down and does not fade.
 * Where the hue is undefined (chroma at or below `MIX_MIN_CHROMA`) the result is `base`.
 *
 * @param y The pixel's luminance.
 * @param mix The eight band amounts divided by 100, in `MIX_NAMES` order.
 */
export function mixFactor(lr: number, lg: number, lb: number, y: number, base: number, mix: readonly number[]): number {
  const q1 = Math.cbrt(0.4122214708 * lr + 0.5363325363 * lg + 0.0514459929 * lb);
  const q2 = Math.cbrt(0.2119034982 * lr + 0.6806995451 * lg + 0.1073969566 * lb);
  const q3 = Math.cbrt(0.0883024619 * lr + 0.2817188376 * lg + 0.6299787005 * lb);
  const a = 1.9779984951 * q1 - 2.428592205 * q2 + 0.4505937099 * q3;
  const b = 0.0259040371 * q1 + 0.7827717662 * q2 - 0.808675766 * q3;
  const chroma = Math.hypot(a, b);
  if (!(chroma > MIX_MIN_CHROMA)) return base;
  let hue = (((Math.atan2(b, a) * 180) / Math.PI) % 360 + 360) % 360;
  if (hue < MIX_CENTRES[0]) hue += 360;
  let band = 0;
  for (let k = 1; k < MIX_NAMES.length; k += 1) if (hue >= MIX_EDGES[k]) band = k;
  const t = (hue - MIX_EDGES[band]) / (MIX_EDGES[band + 1] - MIX_EDGES[band]);
  const weight = 0.5 + 0.5 * Math.cos(Math.PI * clamp01((t - MIX_FLAT) / (1 - 2 * MIX_FLAT)));
  const amount = weight * mix[band] + (1 - weight) * mix[(band + 1) % MIX_NAMES.length];
  if (amount <= 0) return base * (1 + MIX_MUTE * amount);
  const gain = MIX_BOOST_VIVID + (MIX_BOOST_MUTED - MIX_BOOST_VIVID) * (1 - clamp01(chroma / MIX_VIVID_CHROMA)) ** 2;
  const excess = amount * gain * smoothstep(MIX_FADE_START, MIX_FADE_END, chroma) * base;
  const headroom = Math.max(gamutFactor(lr, lg, lb, y) - base, 0);
  if (!(excess > 0)) return base;
  if (headroom === Infinity) return base + excess;
  return base + (excess * headroom) / (excess ** MIX_GAMUT_SOFTNESS + headroom ** MIX_GAMUT_SOFTNESS) ** (1 / MIX_GAMUT_SOFTNESS);
}

function quantize(p: number): number {
  return Math.floor(p * 255 + 0.5);
}

/** Apply the fifteen settings to one 8-bit pixel in float64: the reference the saved copy follows. Returns the input when every setting is 0. */
export function adjustPixel(r8: number, g8: number, b8: number, l: Readonly<LightSettings>): [number, number, number] {
  if (isNeutralLight(l)) return [r8, g8, b8];
  const [gr, gg, gb] = channelGains(l);
  const lr = clamp01(lin(r8 / 255) * gr);
  const lg = clamp01(lin(g8 / 255) * gg);
  const lb = clamp01(lin(b8 / 255) * gb);
  const y = KR * lr + KG * lg + KB * lb;
  const s = chromaRatio(lr, lg, lb);
  const mix = mixSettings(l);
  const base = (1 + l.saturation / 100) * (1 + vibranceStrength(l.vibrance) * Math.pow(1 - s, VIBRANCE_FALLOFF));
  const factor = mix.some((v) => v !== 0) ? mixFactor(lr, lg, lb, y, base, mix.map((v) => v / 100)) : base;
  const sr = clamp01(y + factor * (lr - y));
  const sg = clamp01(y + factor * (lg - y));
  const sb = clamp01(y + factor * (lb - y));
  const pr = contrastCurve(enc(sr), l.contrast);
  const pg = contrastCurve(enc(sg), l.contrast);
  const pb = contrastCurve(enc(sb), l.contrast);
  const shift = toneShift(KR * pr + KG * pg + KB * pb, l.highlights, l.shadows);
  return [quantize(clamp01(pr + shift)), quantize(clamp01(pg + shift)), quantize(clamp01(pb + shift))];
}

/** Build the tables `previewPixel` and the preview shader share for these settings. */
export function previewTables(l: Readonly<LightSettings>): PreviewTables {
  const decode = new Float32Array(DECODE_SIZE);
  for (let i = 0; i < DECODE_SIZE; i += 1) decode[i] = lin(i / 255);
  const tone = new Float32Array(TONE_INTERVALS + 1);
  for (let j = 0; j <= TONE_INTERVALS; j += 1) tone[j] = clamp01(contrastCurve(enc(j / TONE_INTERVALS), l.contrast));
  const shift = new Float32Array(SHIFT_INTERVALS + 1);
  for (let j = 0; j <= SHIFT_INTERVALS; j += 1) shift[j] = toneShift(j / SHIFT_INTERVALS, l.highlights, l.shadows);
  const mix = mixSettings(l);
  return {
    decode,
    gains: channelGains(l),
    sat: 1 + l.saturation / 100,
    vib: vibranceStrength(l.vibrance),
    tone,
    shift,
    mix: mix.map((v) => v / 100),
    mixOn: mix.some((v) => v !== 0),
  };
}

/** Sample an evenly spaced table over [0, 1] with linear interpolation. `x` is clamped to [0, 1], like a clamped sampler coordinate. */
export function lerpTable(table: Float32Array, x: number): number {
  const intervals = table.length - 1;
  const f = clamp01(x) * intervals;
  const i = Math.min(Math.floor(f), intervals - 1);
  return table[i] + (table[i + 1] - table[i]) * (f - i);
}

/** Apply the tables to one 8-bit pixel exactly as the preview shader composes them, then quantize. */
export function previewPixel(r8: number, g8: number, b8: number, t: PreviewTables): [number, number, number] {
  const lr = clamp01(t.decode[r8] * t.gains[0]);
  const lg = clamp01(t.decode[g8] * t.gains[1]);
  const lb = clamp01(t.decode[b8] * t.gains[2]);
  const y = KR * lr + KG * lg + KB * lb;
  const w = (1 - chromaRatio(lr, lg, lb)) ** 2;
  const base = t.sat * (1 + t.vib * w);
  const f = t.mixOn ? mixFactor(lr, lg, lb, y, base, t.mix) : base;
  const pr = lerpTable(t.tone, clamp01(y + f * (lr - y)));
  const pg = lerpTable(t.tone, clamp01(y + f * (lg - y)));
  const pb = lerpTable(t.tone, clamp01(y + f * (lb - y)));
  const shift = lerpTable(t.shift, KR * pr + KG * pg + KB * pb);
  return [quantize(clamp01(pr + shift)), quantize(clamp01(pg + shift)), quantize(clamp01(pb + shift))];
}
