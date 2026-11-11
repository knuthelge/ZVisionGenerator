import { describe, expect, it } from 'vitest';

import {
  ZOOM_MAX, ZOOM_MIN, clampView, fitScale, fitView, panBy, pinchView, ringDiameter, scaleOf, stepZoom, toImage, toScreen, zoomAbout, zoomLabel,
} from './paintView';

const AREA = { w: 1000, h: 800 };
const CONTENT = { x: 0, y: 0, w: 832, h: 1216 };

describe('the painter view', () => {
  it('fits the whole canvas inside the area with its padding', () => {
    const fit = fitScale(AREA, CONTENT);
    expect(fit).toBeCloseTo((800 - 32) / 1216, 6);
    expect(CONTENT.w * fit).toBeLessThanOrEqual(AREA.w);
  });

  it('maps a point there and back', () => {
    const fit = fitScale(AREA, CONTENT);
    const view = { z: 2.5, cx: 300, cy: 700 };
    const point = { x: 123, y: 456 };
    const back = toImage(view, fit, AREA, toScreen(view, fit, AREA, point));
    expect(back.x).toBeCloseTo(point.x, 6);
    expect(back.y).toBeCloseTo(point.y, 6);
  });

  it('centres the picture at Fit', () => {
    const fit = fitScale(AREA, CONTENT);
    const view = fitView(CONTENT);
    const middle = toScreen(view, fit, AREA, { x: 416, y: 608 });
    expect([middle.x, middle.y]).toEqual([500, 400]);
  });

  it('zooms about a point, which stays under the pointer', () => {
    const fit = fitScale(AREA, CONTENT);
    const anchor = { x: 560, y: 300 };
    const before = fitView(CONTENT);
    const grabbed = toImage(before, fit, AREA, anchor);
    const after = zoomAbout(before, fit, AREA, CONTENT, anchor, 3);
    const now = toScreen(after, fit, AREA, grabbed);
    expect(now.x).toBeCloseTo(anchor.x, 4);
    expect(now.y).toBeCloseTo(anchor.y, 4);
    expect(after.z).toBe(3);
  });

  it('never zooms out past Fit or in past 400%', () => {
    const fit = fitScale(AREA, CONTENT);
    expect(zoomAbout(fitView(CONTENT), fit, AREA, CONTENT, { x: 10, y: 10 }, 0.2).z).toBe(ZOOM_MIN);
    expect(zoomAbout(fitView(CONTENT), fit, AREA, CONTENT, { x: 10, y: 10 }, 9).z).toBe(ZOOM_MAX);
  });

  it('returns to the centre at Fit whatever the pan was', () => {
    expect(clampView({ z: 1, cx: 5, cy: 5 }, CONTENT)).toEqual(fitView(CONTENT));
  });

  it('keeps the middle of the area on the canvas when panned too far', () => {
    const view = clampView({ z: 2, cx: 5000, cy: -5000 }, CONTENT);
    expect([view.cx, view.cy]).toEqual([832, 0]);
  });

  it('pans by the distance of the drag', () => {
    const fit = fitScale(AREA, CONTENT);
    const view = { z: 2, cx: 400, cy: 600 };
    const moved = panBy(view, fit, CONTENT, 100, -50);
    expect(moved.cx).toBeCloseTo(400 - 100 / scaleOf(view, fit), 6);
    expect(moved.cy).toBeCloseTo(600 + 50 / scaleOf(view, fit), 6);
  });

  it('follows two fingers: the zoom by their distance, the picture by their midpoint', () => {
    const fit = fitScale(AREA, CONTENT);
    const start = { view: fitView(CONTENT), mid: { x: 500, y: 400 }, distance: 100 };
    const spread = pinchView(start, { x: 500, y: 400 }, 250, fit, AREA, CONTENT);
    expect(spread.z).toBeCloseTo(2.5, 6);
    const grabbed = toImage(start.view, fit, AREA, { x: 500, y: 400 });
    const moved = pinchView(start, { x: 560, y: 420 }, 250, fit, AREA, CONTENT);
    const where = toScreen(moved, fit, AREA, grabbed);
    expect([Math.round(where.x), Math.round(where.y)]).toEqual([560, 420]);
  });

  it('steps the zoom buttons by 25% of Fit and stops at the ends', () => {
    expect(stepZoom(1, 1)).toBe(1.25);
    expect(stepZoom(1.25, -1)).toBe(1);
    expect(stepZoom(1, -1)).toBe(1);
    expect(stepZoom(4, 1)).toBe(4);
    expect(stepZoom(1.3, 1)).toBe(1.5);
    expect(stepZoom(1.3, -1)).toBe(1.25);
    expect(zoomLabel(1.25)).toBe('125%');
  });

  it('draws the brush ring at its true size for the zoom', () => {
    const fit = fitScale(AREA, CONTENT);
    expect(ringDiameter(64, fitView(CONTENT), fit)).toBeCloseTo(64 * fit, 6);
    expect(ringDiameter(64, { z: 2, cx: 0, cy: 0 }, fit)).toBeCloseTo(128 * fit, 6);
  });
});
