import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { bakeMascotMarkup, canvasSupported, denoiseSteps, drawDenoiseIn, drawStream, streamCells, type MascotSprite, type Placement } from './mascotCanvas';

/** A recording stand-in for a 2D context. */
function fakeContext() {
  return { canvas: { width: 100, height: 100 }, globalAlpha: 1, fillStyle: '', imageSmoothingQuality: 'low', clearRect: vi.fn(), fillRect: vi.fn(), drawImage: vi.fn() };
}

type Fake = ReturnType<typeof fakeContext>;
const asContext = (ctx: Fake): CanvasRenderingContext2D => ctx as unknown as CanvasRenderingContext2D;

/** A sprite whose every pixel is opaque, so every grid cell draws. Sampling reads from a stubbed canvas. */
function stubSampling(): void {
  const ctx = {
    ...fakeContext(),
    getImageData: (_x: number, _y: number, w: number, h: number) => ({ data: new Uint8ClampedArray(w * h * 4).fill(200) }),
  };
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(ctx as unknown as CanvasRenderingContext2D);
}

function sprite(): MascotSprite {
  return { width: 100, height: 100, canvas: document.createElement('canvas') };
}

function placement(x: number, y: number, size: number): Placement {
  return { sprite: sprite(), x, y, width: size, height: size };
}

describe('the canvas effects', () => {
  beforeEach(stubSampling);
  afterEach(() => vi.restoreAllMocks());

  it('is only available where there is a 2D canvas to draw on', () => {
    vi.restoreAllMocks();
    expect(canvasSupported()).toBe(false);
  });

  it('samples him in as noise first and ends on the crisp image', () => {
    const at = placement(0, 0, 100);
    const steps = denoiseSteps(at.sprite);

    const start = fakeContext();
    drawDenoiseIn(asContext(start), at, steps, 0);
    expect(start.fillRect).toHaveBeenCalled();
    expect(start.drawImage).not.toHaveBeenCalled();

    const end = fakeContext();
    drawDenoiseIn(asContext(end), at, steps, 60_000);
    expect(end.drawImage).toHaveBeenCalledWith(at.sprite.canvas, 0, 0, 100, 100);
  });

  it('streams from the old spot to the new one: he starts as the old image and lands as the new', () => {
    const from = placement(0, 100, 60);
    const to = placement(300, 20, 120);
    const cells = streamCells(to.sprite);

    const start = fakeContext();
    drawStream(asContext(start), from, to, cells, 0, 40);
    expect(start.drawImage).toHaveBeenCalledWith(from.sprite.canvas, 0, 100, 60, 60);
    expect(start.drawImage).toHaveBeenCalledTimes(1);

    const middle = fakeContext();
    drawStream(asContext(middle), from, to, cells, 0.5, 40);
    expect(middle.fillRect).toHaveBeenCalled();

    const end = fakeContext();
    drawStream(asContext(end), from, to, cells, 1, 40);
    expect(end.drawImage).toHaveBeenCalledTimes(1);
    expect(end.drawImage).toHaveBeenCalledWith(to.sprite.canvas, 300, 20, 120, 120);
  });

  /** Report whether the first and last cells of the top row have left their place early in the flight. */
  function movedColumns(from: Placement, to: Placement): { first: boolean; last: boolean } {
    const ctx = fakeContext();
    drawStream(asContext(ctx), from, to, streamCells(to.sprite), 0.25, 0);
    // Every cell is drawn, row by row, so the first and sixteenth calls are the top row's two ends.
    const moved = (call: number, column: number): boolean => Math.abs((ctx.fillRect.mock.calls[call]![0] as number) - (from.x + (column * from.width) / 16)) > 0.5;
    return { first: moved(0, 0), last: moved(15, 15) };
  }

  it('lets the side facing the destination leave first, whichever way he goes', () => {
    const left = movedColumns(placement(400, 100, 80), placement(0, 100, 80));
    expect(left).toEqual({ first: true, last: false });

    const right = movedColumns(placement(0, 100, 80), placement(400, 100, 80));
    expect(right).toEqual({ first: false, last: true });
  });
});

describe('bakeMascotMarkup', () => {
  function mascotSvg(): SVGSVGElement {
    const holder = document.createElement('div');
    holder.innerHTML = `<svg class="mascot" role="img" aria-label="x"><g class="fx"><rect class="spark"/></g><path class="hidden" style="opacity: 0"/><path class="shown" style="opacity: 0.5"/><path id="plain"/></svg>`;
    return holder.querySelector('svg')!;
  }

  it('leaves out the mood effects and hidden parts', () => {
    const markup = bakeMascotMarkup(mascotSvg(), 80, 80);

    expect(markup).not.toContain('spark');
    expect(markup).not.toContain('hidden');
    expect(markup).toContain('id="plain"');
  });

  it('writes the pose each part has now into it', () => {
    expect(bakeMascotMarkup(mascotSvg(), 80, 64)).toContain('opacity: 0.5');
  });
});
