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
 * Zoom to `scale`, keeping the point `(px, py)` (screen px from the fitted image's centre) over the same spot of the image.
 *
 * The scale stays between the fitted image and `max`.
 */
export function zoomAt(zoom: Zoom, scale: number, px: number, py: number, max = MAX_ZOOM): Zoom {
  const next = clamp(scale, 1, Math.max(1, max));
  const ratio = next / zoom.scale;
  return { scale: next, x: px - (px - zoom.x) * ratio, y: py - (py - zoom.y) * ratio };
}

/** What covers the stage's edges while the controls show, in px: the top bar, the foot (dock and strip), a phone's side rail. */
export interface Cover {
  top: number;
  right: number;
  bottom: number;
}

/** How far the stage reaches from the fitted image's centre each way, in px. */
export interface Reach {
  left: number;
  right: number;
  top: number;
  bottom: number;
}

/** Return where the fitted image's centre sits from the stage's centre: the stage runs under the controls, the image clears them. */
export function fitOffset(cover: Cover): { x: number; y: number } {
  return { x: -cover.right / 2 || 0, y: (cover.top - cover.bottom) / 2 };
}

/** Return the stage's reach around the fitted image: all of it with the controls hidden, else the part they leave clear. */
export function stageReach(stage: Size, cover: Cover, hidden: boolean): Reach {
  const offset = fitOffset(cover);
  const cx = stage.width / 2 + offset.x;
  const cy = stage.height / 2 + offset.y;
  const clear = hidden ? { top: 0, right: 0, bottom: 0 } : cover;
  return { left: cx, right: stage.width - clear.right - cx, top: cy - clear.top, bottom: stage.height - clear.bottom - cy };
}

/** An image larger than the view pans until its edge meets the view's; a smaller one stays as near its fitted place as the view allows. */
function clampAxis(offset: number, half: number, before: number, after: number): number {
  return half * 2 >= before + after ? clamp(offset, after - half, half - before) : clamp(0, half - before, after - half);
}

/** Keep a zoomed image over the stage's reach: it pans only as far as its own edges, and a fitted one stays in place. */
export function clampPan(zoom: Zoom, media: Size, reach: Reach): Zoom {
  return {
    scale: zoom.scale,
    x: clampAxis(zoom.x, (media.width * zoom.scale) / 2, reach.left, reach.right),
    y: clampAxis(zoom.y, (media.height * zoom.scale) / 2, reach.top, reach.bottom),
  };
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
