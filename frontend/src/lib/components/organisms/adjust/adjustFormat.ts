import { aspectLabel, type Aspect, type Geometry, type Size } from '$lib/state/adjustGeometry';
import { SLIDER_MAX, SLIDER_MIN, type LightSettings } from '$lib/state/adjustMath';
import { LIGHT_TOOLS, MIX_TOOLS } from './adjustTools';

/** The value a slider row shows: `+24`, `−8` (a true minus sign) and `0`. */
export function signedLabel(value: number): string {
  if (value > 0) return `+${value}`;
  if (value < 0) return `−${-value}`;
  return '0';
}

/** The value a screen reader says: `+24`, `-8` (ASCII, so every reader says "minus") and `0`. */
export function signedValueText(value: number): string {
  return value > 0 ? `+${value}` : String(value);
}

/** The `--lo` and `--hi` fill stops, in percent, of a centre-zero range at `value`: the fill runs from the middle to the thumb. */
export function centredFill(value: number): { lo: number; hi: number } {
  const clamped = Math.min(SLIDER_MAX, Math.max(SLIDER_MIN, value));
  const thumb = 50 + clamped / 2;
  return clamped >= 0 ? { lo: 50, hi: thumb } : { lo: thumb, hi: 50 };
}

/** What a side-panel section header says about the Crop section: the shape and size of a crop, the size alone for a turn or flip, else "Full picture". */
export function cropSummary(aspect: Aspect, geometry: Geometry, output: Size | null, cropped: boolean): { text: string; set: boolean } {
  const size = output ? `${output.width} × ${output.height}` : '';
  if (cropped && size) return { text: `${aspectLabel(aspect)} · ${size}`, set: true };
  if ((geometry.rotate !== 0 || geometry.flip) && size) return { text: size, set: true };
  return { text: 'Full picture', set: false };
}

/** What the Light & colour header says: how many of its seven sliders are off zero. */
export function lightSummary(light: Readonly<LightSettings>): { text: string; set: boolean } {
  const count = LIGHT_TOOLS.filter((tool) => light[tool.key] !== 0).length;
  return count > 0 ? { text: `${count} changed`, set: true } : { text: 'No changes', set: false };
}

/** What the Colour mixer header says: the band and its value when one is set ("Blue +30"), else how many colours are. */
export function mixSummary(light: Readonly<LightSettings>): { text: string; set: boolean } {
  const set = MIX_TOOLS.filter((tool) => light[tool.key] !== 0);
  if (set.length === 0) return { text: 'No changes', set: false };
  if (set.length === 1) return { text: `${set[0].label} ${signedLabel(light[set[0].key])}`, set: true };
  return { text: `${set.length} colours changed`, set: true };
}
