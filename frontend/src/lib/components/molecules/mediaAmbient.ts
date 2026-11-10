/** The ambient light shows only in the bands a picture leaves around itself, so it is worth drawing only when a band is there, and it is drawn from the picture's edges outward. */

/** A band thinner than this many CSS pixels, on both axes, does not count: the picture covers the pane. */
export const AMBIENT_TOLERANCE = 2;

export interface Box {
  width: number;
  height: number;
}

/**
 * Return the box a picture of the natural size *natural* takes inside *player*, fitted whole: a picture that already fits is not
 * enlarged unless *upscale* says it may be. A size that is not known yet (0) answers an empty box.
 */
export function shownPicture(player: Box, natural: Box, upscale: boolean): Box {
  if (!(player.width > 0 && player.height > 0 && natural.width > 0 && natural.height > 0)) return { width: 0, height: 0 };
  const fit = Math.min(player.width / natural.width, player.height / natural.height);
  const scale = upscale ? fit : Math.min(1, fit);
  return { width: natural.width * scale, height: natural.height * scale };
}

/**
 * Return whether the picture *shown*, as it sits in *pane*, leaves a band of the pane showing.
 * A size that is not known yet (0) answers `true`, since nothing says the light is hidden.
 */
export function leavesBands(pane: Box, shown: Box, tolerance = AMBIENT_TOLERANCE): boolean {
  if (!(pane.width > 0 && pane.height > 0 && shown.width > 0 && shown.height > 0)) return true;
  return pane.width - shown.width > tolerance || pane.height - shown.height > tolerance;
}

/** The pixels the ambient light is drawn in: the canvas size, and where the picture sits inside it. */
export interface AmbientRegion {
  columns: number;
  rows: number;
  picture: { x: number; y: number; width: number; height: number };
}

/**
 * Return how to lay out the ambient light for the picture *shown*, centred in *pane*: a canvas *columns* wide that covers the pane
 * and *reach* more on every side, with the picture where it sits on screen, so the light fills the bands from the picture's own edges.
 * Return `null` while a size is not known.
 */
export function ambientRegion(pane: Box, shown: Box, reach: number, columns: number): AmbientRegion | null {
  if (!(pane.width > 0 && pane.height > 0 && shown.width > 0 && shown.height > 0)) return null;
  const regionWidth = pane.width + reach * 2;
  const regionHeight = pane.height + reach * 2;
  const unit = columns / regionWidth;
  const width = Math.min(columns, Math.max(1, Math.round(shown.width * unit)));
  const rows = Math.max(1, Math.round(regionHeight * unit));
  const height = Math.min(rows, Math.max(1, Math.round(shown.height * unit)));
  return {
    columns,
    rows,
    picture: {
      x: Math.round((columns - width) / 2),
      y: Math.round((rows - height) / 2),
      width,
      height,
    },
  };
}

/** The softness of the light's falloff, as the fraction of the spread that is the blur's deviation: about a third of it. */
const SPREAD_DIVISOR = 2.6;

/** How far the light's mask spreads past the picture on each axis, in CSS pixels. */
export interface GlowSpread {
  x: number;
  y: number;
}

/**
 * Return how far the light spreads past the picture *shown* in *pane*: on each axis no further than *reach* or the band it lies in.
 * Above and below it spreads no further than it does at the sides, so a picture that fills the width (a phone held upright) gets no flat slab of light over and under it.
 */
export function glowSpread(pane: Box, shown: Box, reach: number): GlowSpread {
  const side = Math.max(0, (pane.width - shown.width) / 2);
  const top = Math.max(0, (pane.height - shown.height) / 2);
  return { x: Math.min(reach, side) / SPREAD_DIVISOR, y: Math.min(reach, top, side) / SPREAD_DIVISOR };
}

/** Whether a spread is wide enough to show: under 3 px on both axes there is no light to see. */
export function glowShows(spread: GlowSpread): boolean {
  return spread.x >= 3 || spread.y >= 3;
}

/**
 * Return the CSS `mask-image` for the light: a softly blurred rounded box, the picture *shown* grown by one *spread*, laid in *pane*.
 * The light is near full strength where it meets the picture and gone by about three spreads out, with no line where it ends.
 */
export function glowMask(pane: Box, shown: Box, spread: GlowSpread): string {
  const { x, y } = spread;
  const left = (pane.width - shown.width) / 2 - x;
  const top = (pane.height - shown.height) / 2 - y;
  const r = (n: number): string => n.toFixed(1);
  const svg = `<svg xmlns='http://www.w3.org/2000/svg' width='${r(pane.width)}' height='${r(pane.height)}' viewBox='0 0 ${r(pane.width)} ${r(pane.height)}'><filter id='f' x='-1' y='-1' width='3' height='3'><feGaussianBlur stdDeviation='${r(x)} ${r(y)}'/></filter><rect x='${r(left)}' y='${r(top)}' width='${r(shown.width + 2 * x)}' height='${r(shown.height + 2 * y)}' rx='${r(Math.min(x, y))}' fill='white' filter='url(#f)'/></svg>`;
  return `url("data:image/svg+xml,${encodeURIComponent(svg)}")`;
}
