import type { Rect } from './editOps';

/**
 * The painter's view of the picture. `z` is the zoom relative to Fit (1 shows the whole canvas), and `cx, cy` is the point of
 * the picture, in image pixels, at the middle of the canvas area. Everything else is derived, so zooming and panning are
 * plain arithmetic on three numbers.
 */
export interface View {
  z: number;
  cx: number;
  cy: number;
}

export interface Point {
  x: number;
  y: number;
}

export interface Size {
  w: number;
  h: number;
}

export const ZOOM_MIN = 1;
export const ZOOM_MAX = 4;
/** One press of the zoom buttons: 25% of Fit. */
const ZOOM_STEP = 0.25;

/** The air kept around the picture at Fit. */
const FIT_PADDING = 16;

/** The view that shows the whole canvas. */
export function fitView(content: Rect): View {
  return { z: ZOOM_MIN, cx: content.x + content.w / 2, cy: content.y + content.h / 2 };
}

/** Screen pixels per image pixel at Fit. */
export function fitScale(area: Size, content: Rect): number {
  if (content.w <= 0 || content.h <= 0) return 1;
  const w = Math.max(1, area.w - FIT_PADDING * 2);
  const h = Math.max(1, area.h - FIT_PADDING * 2);
  return Math.min(w / content.w, h / content.h);
}

/** Screen pixels per image pixel in this view. */
export function scaleOf(view: View, fit: number): number {
  return fit * view.z;
}

/** Where an image point lands on the canvas area. */
export function toScreen(view: View, fit: number, area: Size, point: Point): Point {
  const s = scaleOf(view, fit);
  return { x: area.w / 2 + (point.x - view.cx) * s, y: area.h / 2 + (point.y - view.cy) * s };
}

/** The image point under a point of the canvas area. */
export function toImage(view: View, fit: number, area: Size, point: Point): Point {
  const s = scaleOf(view, fit);
  return { x: view.cx + (point.x - area.w / 2) / s, y: view.cy + (point.y - area.h / 2) / s };
}

/** A zoom within its range. */
export const clampZoom = (z: number): number => Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, z));

/** Keep the middle of the area on the canvas, so the picture can never be panned out of sight; Fit is always centred. */
export function clampView(view: View, content: Rect): View {
  const z = clampZoom(view.z);
  if (z === ZOOM_MIN) return fitView(content);
  return {
    z,
    cx: Math.min(content.x + content.w, Math.max(content.x, view.cx)),
    cy: Math.min(content.y + content.h, Math.max(content.y, view.cy)),
  };
}

/** Zoom to `z` keeping the image point under `anchor` (a point of the canvas area) where it is. */
export function zoomAbout(view: View, fit: number, area: Size, content: Rect, anchor: Point, z: number): View {
  const next = clampZoom(z);
  const under = toImage(view, fit, area, anchor);
  const s = fit * next;
  return clampView({ z: next, cx: under.x - (anchor.x - area.w / 2) / s, cy: under.y - (anchor.y - area.h / 2) / s }, content);
}

/** Move the picture by a drag of `dx, dy` screen pixels. */
export function panBy(view: View, fit: number, content: Rect, dx: number, dy: number): View {
  const s = scaleOf(view, fit);
  return clampView({ ...view, cx: view.cx - dx / s, cy: view.cy - dy / s }, content);
}

/** The state of a two-finger gesture when it began. */
export interface PinchStart {
  view: View;
  mid: Point;
  distance: number;
}

/** The view for two fingers now: the zoom follows their distance and the picture follows their midpoint. */
export function pinchView(start: PinchStart, mid: Point, distance: number, fit: number, area: Size, content: Rect): View {
  const z = clampZoom(start.view.z * (distance / Math.max(1, start.distance)));
  const grabbed = toImage(start.view, fit, area, start.mid);
  const s = fit * z;
  return clampView({ z, cx: grabbed.x - (mid.x - area.w / 2) / s, cy: grabbed.y - (mid.y - area.h / 2) / s }, content);
}

/** The zoom one button press away, snapped to the 25% steps. */
export function stepZoom(z: number, direction: 1 | -1): number {
  const steps = Math.round(z / ZOOM_STEP);
  const snapped = Math.abs(steps * ZOOM_STEP - z) < 1e-6 ? steps : direction > 0 ? Math.floor(z / ZOOM_STEP) : Math.ceil(z / ZOOM_STEP);
  return clampZoom((snapped + direction) * ZOOM_STEP);
}

/** The zoom as the percentage the painter shows: Fit is 100%. */
export function zoomLabel(z: number): string {
  return `${Math.round(z * 100)}%`;
}

/** The brush's circle on screen, in screen pixels: the true size at the current zoom. */
export function ringDiameter(brush: number, view: View, fit: number): number {
  return brush * scaleOf(view, fit);
}
