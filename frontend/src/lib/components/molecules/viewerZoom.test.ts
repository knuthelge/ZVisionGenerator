import { describe, expect, it } from 'vitest';
import { FIT, MAX_ZOOM, actualPixelsScale, clampPan, settleZoom, zoomAt } from './viewerZoom';

describe('zoomAt', () => {
  it('keeps the point under the finger in place', () => {
    const zoomed = zoomAt(FIT, 2, 100, 50);
    // The image point that sat at (100, 50) before must sit there after: x + scale * m = p with m = 100.
    expect(zoomed).toEqual({ scale: 2, x: -100, y: -50 });
  });

  it('zooms around the centre without moving it', () => {
    expect(zoomAt(FIT, 3, 0, 0)).toEqual({ scale: 3, x: 0, y: 0 });
  });

  it('never zooms out past the fitted image or in past the limit', () => {
    expect(zoomAt(FIT, 0.5, 0, 0).scale).toBe(1);
    expect(zoomAt(FIT, 100, 0, 0).scale).toBe(MAX_ZOOM);
  });
});

describe('clampPan', () => {
  const media = { width: 400, height: 300 };
  const stage = { width: 400, height: 600 };

  it('pans only as far as the zoomed image reaches', () => {
    // At 2× the image is 800 wide over a 400 stage: 200 px either way; 600 high over 600: no vertical pan.
    expect(clampPan({ scale: 2, x: 500, y: 80 }, media, stage)).toEqual({ scale: 2, x: 200, y: 0 });
  });

  it('keeps a fitted image centred', () => {
    expect(clampPan({ scale: 1, x: 40, y: -40 }, media, stage)).toEqual({ scale: 1, x: 0, y: 0 });
  });
});

describe('settleZoom', () => {
  it('snaps a barely zoomed image back to fit', () => {
    expect(settleZoom({ scale: 1.02, x: 5, y: 5 })).toBe(FIT);
  });

  it('keeps a real zoom', () => {
    const zoom = { scale: 2, x: 5, y: 5 };
    expect(settleZoom(zoom)).toBe(zoom);
  });
});

describe('actualPixelsScale', () => {
  it('shows a large image at its real pixels', () => {
    expect(actualPixelsScale({ width: 2048, height: 2048 }, { width: 512, height: 512 })).toBe(4);
  });

  it('never goes below the minimum for an image smaller than the stage', () => {
    expect(actualPixelsScale({ width: 256, height: 256 }, { width: 512, height: 512 }, 2)).toBe(2);
  });
});
