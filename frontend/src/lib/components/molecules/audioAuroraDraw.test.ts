import { describe, expect, it } from 'vitest';
import type { AudioWorkflow } from '$lib/types';
import { type AuroraColors, type AuroraStage } from './audioAurora';
import { createAuroraPainter, type PaintInput } from './audioAuroraDraw';

const COLORS: AuroraColors = {
  text: [240, 240, 240],
  primary: [20, 184, 166],
  hover: [45, 212, 191],
  subtle: [10, 60, 55],
  coral: [250, 110, 90],
  amber: [250, 190, 60],
  wave: [120, 120, 130],
  surface: [20, 20, 24],
  haze: [90, 90, 100],
};

/** A canvas context that records every call and every assignment, so two frames can be compared. */
function recorder(): { ctx: CanvasRenderingContext2D; log: string[] } {
  const log: string[] = [];
  const gradient = { addColorStop: (...args: unknown[]) => void log.push(`stop ${JSON.stringify(args)}`) };
  const ctx = new Proxy({} as Record<string, unknown>, {
    get: (_target, name: string) => (...args: unknown[]) => {
      log.push(`${name} ${JSON.stringify(args)}`);
      return name === 'createLinearGradient' ? gradient : undefined;
    },
    set: (_target, name: string, value: unknown) => {
      log.push(`set ${name} ${typeof value === 'string' ? value : ''}`);
      return true;
    },
  });
  return { ctx: ctx as unknown as CanvasRenderingContext2D, log };
}

const input = (stage: AuroraStage, overrides: Partial<PaintInput> = {}, kind: AudioWorkflow = 'music'): PaintInput => ({
  kind,
  colors: COLORS,
  time: 5000,
  dt: 0,
  head: 0.4,
  level: 0.4,
  stage,
  plan: 0.5,
  part: 1,
  parts: 1,
  boost: 1,
  snap: false,
  px: 1,
  phone: false,
  land: null,
  peaks: [],
  ...overrides,
});

/** Paint *first*, then *second*, and return the calls of the second frame. */
function secondFrame(first: PaintInput, second: PaintInput): string[] {
  const painter = createAuroraPainter();
  painter.paint(recorder().ctx, 632, 158, first);
  const { ctx, log } = recorder();
  painter.paint(ctx, 632, 158, second);
  return log;
}

describe('the painter across stages', () => {
  it('shows a stage at once when snapping, as a painter that began on it', () => {
    const fresh = secondFrame(input('decode'), input('decode', { snap: true }));
    const snapped = secondFrame(input('wait'), input('decode', { snap: true }));

    expect(snapped).toEqual(fresh);
  });

  it('cross-fades from the old stage by real time when not snapping', () => {
    const fresh = secondFrame(input('decode'), input('decode', { dt: 1 / 30 }));
    const fading = secondFrame(input('wait'), input('decode', { dt: 1 / 30 }));

    expect(fading).not.toEqual(fresh);
  });

  it('has finished the cross-fade after its time, whatever the frame rate', () => {
    const fresh = secondFrame(input('decode'), input('decode', { dt: 0 }));
    const painter = createAuroraPainter();
    painter.paint(recorder().ctx, 632, 158, input('wait'));
    for (let i = 0; i < 30; i += 1) painter.paint(recorder().ctx, 632, 158, input('decode', { dt: 0.05 }));
    const { ctx, log } = recorder();
    painter.paint(ctx, 632, 158, input('decode', { dt: 0.05 }));

    expect(log).toEqual(fresh);
  });
});
