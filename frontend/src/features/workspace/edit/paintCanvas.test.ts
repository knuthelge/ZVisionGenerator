import { afterEach, describe, expect, it, vi } from 'vitest';

import { drawStroke, maskUpload } from './paintCanvas';

/** A context that records the style a stroke is drawn with, as a stand-in for a canvas. */
function recordingContext() {
  const seen = { strokeStyle: '', fillStyle: '', operation: '' };
  const noop = () => {};
  const ctx = {
    save: noop, restore: noop, beginPath: noop, arc: noop, fill: noop, moveTo: noop, lineTo: noop, stroke: noop,
    set strokeStyle(value: string) { seen.strokeStyle = value; },
    set fillStyle(value: string) { seen.fillStyle = value; },
    set globalCompositeOperation(value: string) { seen.operation = value; },
    lineCap: '', lineJoin: '', lineWidth: 0,
  };
  return { ctx: ctx as unknown as CanvasRenderingContext2D, seen };
}

describe('drawStroke', () => {
  it('paints in white, so a canvas painted completely is an opaque white mask the server reads as fully painted', () => {
    const { ctx, seen } = recordingContext();
    drawStroke(ctx, { erase: false, size: 20, points: [10, 10, 40, 40] }, 1);
    expect([seen.strokeStyle, seen.fillStyle, seen.operation]).toEqual(['#fff', '#fff', 'source-over']);
  });

  it('erases by removing alpha', () => {
    const { ctx, seen } = recordingContext();
    drawStroke(ctx, { erase: true, size: 20, points: [10, 10, 40, 40] }, 1);
    expect(seen.operation).toBe('destination-out');
  });
});

describe('maskUpload', () => {
  /** A mask of `width × height` pixels whose alpha channel is `alphas`, as a stand-in for a canvas (jsdom draws none). */
  function maskOf(width: number, height: number, alphas: number[]) {
    const data = new Uint8ClampedArray(alphas.flatMap((alpha) => [255, 255, 255, alpha]));
    const put = vi.fn();
    const mask = { width, height, getContext: () => ({ getImageData: () => ({ data, width, height }) }) } as unknown as HTMLCanvasElement;
    const crisp = { width: 0, height: 0, getContext: () => ({ putImageData: put }), toBlob: (done: (blob: Blob) => void) => done(new Blob(['png'], { type: 'image/png' })) };
    const create = document.createElement.bind(document);
    vi.spyOn(document, 'createElement').mockImplementation((tag: string) => (tag === 'canvas' ? (crisp as unknown as HTMLCanvasElement) : create(tag)));
    return { mask, put, crisp };
  }

  afterEach(() => vi.restoreAllMocks());

  it('counts a pixel as painted from half covered up, and gives the painted part of the mask as its share', async () => {
    // Four of the eight pixels reach 128; the faint edge pixels (127 and below) do not.
    const { mask } = maskOf(4, 2, [255, 128, 127, 0, 200, 1, 129, 64]);
    const upload = await maskUpload(mask);
    expect(upload?.share).toBe(0.5);
  });

  it('writes the file with painted pixels fully opaque and the rest transparent, at the mask\'s size', async () => {
    const { mask, put, crisp } = maskOf(4, 2, [255, 128, 127, 0, 200, 1, 129, 64]);
    const upload = await maskUpload(mask);
    expect([...put.mock.calls[0][0].data.filter((_: number, index: number) => index % 4 === 3)]).toEqual([255, 255, 0, 0, 255, 0, 255, 0]);
    expect([crisp.width, crisp.height]).toEqual([4, 2]);
    expect([upload?.file.name, upload?.file.type]).toEqual(['mask.png', 'image/png']);
  });

  it('has no upload when nothing reaches half covered', async () => {
    const { mask } = maskOf(2, 2, [127, 0, 10, 99]);
    expect(await maskUpload(mask)).toBeNull();
  });

  it('has no upload where the canvas cannot be read', async () => {
    const mask = { width: 2, height: 2, getContext: () => null } as unknown as HTMLCanvasElement;
    expect(await maskUpload(mask)).toBeNull();
  });
});
