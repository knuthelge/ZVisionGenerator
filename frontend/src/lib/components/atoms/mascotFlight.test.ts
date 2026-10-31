import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { arrive, depart } from './mascotFlight';

function spot(key: string, parent: HTMLElement = document.body): HTMLElement {
  const node = document.createElement('div');
  node.dataset.mascotSpot = key;
  node.innerHTML = '<svg class="mascot" width="64" height="64"><path class="body"/></svg>';
  parent.append(node);
  return node;
}

/** Give every element a size, as a browser would. */
function stubLayout(width = 64): void {
  vi.spyOn(Element.prototype, 'getBoundingClientRect').mockReturnValue({
    left: 10, top: 20, right: 10 + width, bottom: 20 + width, width, height: width, x: 10, y: 20, toJSON: () => ({}),
  });
}

function reduceMotion(reduce: boolean): void {
  vi.stubGlobal('matchMedia', (query: string) => ({ matches: reduce && query.includes('reduce') }));
}

function canAnimate(can: boolean): void {
  if (can) (Element.prototype as { animate?: unknown }).animate = vi.fn();
  else delete (Element.prototype as { animate?: unknown }).animate;
}

const canvases = (): NodeListOf<HTMLCanvasElement> => document.querySelectorAll('canvas');

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  canAnimate(false);
  document.body.innerHTML = '';
  document.documentElement.style.removeProperty('--z-mascot-effect');
});

describe('arriving where no canvas effect is drawn', () => {
  beforeEach(() => stubLayout());

  it('is instant where the browser cannot animate', () => {
    expect(arrive(spot('a1'), 'a1')()).toMatchObject({ duration: 0 });
  });

  it('is instant under reduced motion, with no canvas', () => {
    canAnimate(true);
    reduceMotion(true);
    const node = spot('a2');

    expect(arrive(node, 'a2')()).toMatchObject({ duration: 0 });
    expect(canvases().length).toBe(0);
    expect(node.style.visibility).toBe('');
  });

  it('pops in where the browser cannot draw the canvas effect', () => {
    canAnimate(true);
    reduceMotion(false);
    const node = spot('a3');

    expect(arrive(node, 'a3')().duration).toBeGreaterThan(0);
    expect(canvases().length).toBe(0);
    expect(node.style.visibility).toBe('');
  });

  it('swaps instantly after another spot under reduced motion', () => {
    canAnimate(true);
    reduceMotion(true);
    depart(spot('a4'), 'a4');

    expect(arrive(spot('a4'), 'a4')()).toMatchObject({ duration: 0 });
    expect(canvases().length).toBe(0);
  });

  it('lets a spot nobody takes over simply disappear', async () => {
    canAnimate(true);
    reduceMotion(false);
    depart(spot('a5'), 'a5');

    await new Promise((resolve) => setTimeout(resolve, 200));
    expect(canvases().length).toBe(0);
  });
});

describe('the canvas effects', () => {
  let frames: FrameRequestCallback[];

  /** A canvas API just capable enough to run an effect. */
  function stubCanvas(): void {
    const ctx = {
      globalAlpha: 1,
      imageSmoothingQuality: 'high',
      fillStyle: '',
      setTransform: vi.fn(),
      clearRect: vi.fn(),
      fillRect: vi.fn(),
      drawImage: vi.fn(),
      getImageData: (_x: number, _y: number, w: number, h: number) => ({ data: new Uint8ClampedArray(w * h * 4).fill(220) }),
    };
    vi.stubGlobal('CanvasRenderingContext2D', class {});
    vi.stubGlobal('Image', class {
      src = '';
      decode(): Promise<void> { return Promise.resolve(); }
    });
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(ctx as unknown as CanvasRenderingContext2D);
    frames = [];
    vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) => frames.push(callback));
  }

  /** Run the queued animation frames, `ms` after the effect started. */
  function frameAt(ms: number): void {
    frames.splice(0).forEach((callback) => callback(ms));
  }

  beforeEach(() => {
    stubLayout();
    reduceMotion(false);
    canAnimate(true);
    stubCanvas();
    vi.spyOn(performance, 'now').mockReturnValue(0);
    document.documentElement.style.setProperty('--z-mascot-effect', '90');
  });

  it('samples the mascot in over a hidden spot, then hands over and removes the canvas', async () => {
    const node = spot('c1');

    arrive(node, 'c1')();
    expect(node.style.visibility).toBe('hidden');
    await vi.waitFor(() => expect(canvases().length).toBe(1));
    expect(canvases()[0]!.style.pointerEvents).toBe('none');

    frameAt(16);
    expect(canvases().length).toBe(1);
    frameAt(60_000);
    expect(canvases().length).toBe(0);
    expect(node.style.visibility).toBe('');
  });

  it('streams from the previous spot to the new one, then hands over and removes the canvas', async () => {
    depart(spot('c2'), 'c2');
    const node = spot('c2');

    arrive(node, 'c2')();
    expect(node.style.visibility).toBe('hidden');
    await vi.waitFor(() => expect(canvases().length).toBe(1));

    frameAt(100);
    expect(canvases().length).toBe(1);
    frameAt(60_000);
    expect(canvases().length).toBe(0);
    expect(node.style.visibility).toBe('');
  });

  it('shows the spot again when it is gone before the effect is ready', async () => {
    const node = spot('c3');
    arrive(node, 'c3')();
    node.remove();

    await vi.waitFor(() => expect(node.style.visibility).toBe(''));
    expect(canvases().length).toBe(0);
  });

  it('keeps the effect under the dialog his spot is in, as under any layer above the page', async () => {
    const dialog = document.createElement('div');
    dialog.style.cssText = 'position: fixed; z-index: 150;';
    document.body.append(dialog);

    arrive(spot('c4', dialog), 'c4')();
    await vi.waitFor(() => expect(canvases().length).toBe(1));

    // Over the dialog that holds him, so the layers above the dialog still cover it.
    expect(Number(canvases()[0]!.style.zIndex)).toBeGreaterThan(150);
  });
});
