/** How far the viewer's image is zoomed and panned: a scale, and an offset of its centre in screen px. */
export interface Zoom {
  scale: number;
  x: number;
  y: number;
}

/** The image fitted to the stage, as it opens. */
export const FIT: Zoom = { scale: 1, x: 0, y: 0 };

/** The furthest a pinch may zoom in. */
export const MAX_ZOOM = 8;

/** Below this a pinch settles back to the fitted image. */
const SNAP_SCALE = 1.05;

interface Size {
  width: number;
  height: number;
}

function clamp(value: number, min: number, max: number): number {
  // `|| 0` turns -0 into 0, so a centred image reads as exactly centred.
  return Math.min(max, Math.max(min, value)) || 0;
}

/**
 * Zoom to `scale`, keeping the point `(px, py)` (screen px from the stage's centre) over the same spot of the image.
 *
 * The scale stays between the fitted image and `max`.
 */
export function zoomAt(zoom: Zoom, scale: number, px: number, py: number, max = MAX_ZOOM): Zoom {
  const next = clamp(scale, 1, Math.max(1, max));
  const ratio = next / zoom.scale;
  return { scale: next, x: px - (px - zoom.x) * ratio, y: py - (py - zoom.y) * ratio };
}

/** Keep a zoomed image over the stage: it pans only as far as its own edges, and a fitted one stays centred. */
export function clampPan(zoom: Zoom, media: Size, stage: Size): Zoom {
  const maxX = Math.max(0, (media.width * zoom.scale - stage.width) / 2);
  const maxY = Math.max(0, (media.height * zoom.scale - stage.height) / 2);
  return { scale: zoom.scale, x: clamp(zoom.x, -maxX, maxX), y: clamp(zoom.y, -maxY, maxY) };
}

/** Settle a zoom after a pinch ends: barely zoomed comes back to the fitted image. */
export function settleZoom(zoom: Zoom): Zoom {
  return zoom.scale < SNAP_SCALE ? FIT : zoom;
}

/** Return the scale that shows the image at its real pixels, given its fitted size; at least a 2× look for small images. */
export function actualPixelsScale(natural: Size, fitted: Size, minimum = 1): number {
  if (fitted.width <= 0 || natural.width <= 0) return minimum;
  return Math.max(minimum, natural.width / fitted.width);
}
