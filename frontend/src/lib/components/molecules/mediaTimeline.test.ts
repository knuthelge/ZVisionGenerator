import { describe, expect, it, vi } from 'vitest';

import { bloomBarCount, bloomBars, drawVideoTimeline, drawWaveform, easeBloom, secondTicks, wavePlayedBars, waveBarCount, waveValue, type TimelineColors } from './mediaTimeline';

const COLORS: TimelineColors = { rest: 'rest', played: 'played', tick: 'tick', shade: 'shade', playhead: 'playhead', bloomRest: 'bloomRest', waveRest: 'waveRest' };

/** A canvas context that records the fill style of every rectangle it fills. */
function fakeContext() {
  const fills: { style: string; x: number; y: number; width: number; height: number }[] = [];
  const ctx = {
    fillStyle: '',
    shadowColor: '',
    shadowBlur: 0,
    clearRect: vi.fn(),
    beginPath: vi.fn(),
    save: vi.fn(),
    restore: vi.fn(),
    fillRect(x: number, y: number, width: number, height: number) { fills.push({ style: this.fillStyle, x, y, width, height }); },
    roundRect(x: number, y: number, width: number, height: number) { this.pending = { x, y, width, height }; },
    fill() { if (this.pending) fills.push({ style: this.fillStyle, ...this.pending }); },
    pending: null as null | { x: number; y: number; width: number; height: number },
  };
  return { ctx: ctx as unknown as CanvasRenderingContext2D, fills };
}

describe('bloom', () => {
  it('has a bar for every frame of a short clip and thins out on a long one', () => {
    expect(bloomBarCount(10, 800)).toBe(10);
    expect(bloomBarCount(500, 800)).toBe(160);
    expect(bloomBarCount(500, 60)).toBe(24);
    expect(bloomBarCount(0, 800)).toBe(160);
  });

  it('rises tallest at the focus and falls away from it', () => {
    const bars = bloomBars({ width: 600, frames: 120, fraction: 0.5, focus: 300, bloom: 1 });
    const heights = bars.map((bar) => bar.height);
    const tallest = Math.max(...heights);

    expect(tallest).toBeLessThanOrEqual(30);
    expect(bars.find((bar) => bar.height === tallest)!.x).toBeCloseTo(300, -1);
    expect(heights[0]).toBeLessThan(tallest);
  });

  it('colours the bars up to the playhead as played', () => {
    const bars = bloomBars({ width: 600, frames: 120, fraction: 0.5, focus: 300, bloom: 1 });

    expect(bars.every((bar) => bar.played === bar.x <= 300)).toBe(true);
  });

  it('draws nothing at rest', () => {
    expect(bloomBars({ width: 600, frames: 120, fraction: 0.5, focus: 300, bloom: 0 })).toEqual([]);
  });

  it('eases a seventh of the gap each frame, snaps near the target, and jumps under reduced motion', () => {
    expect(easeBloom(0, 1, false)).toBeCloseTo(0.14);
    expect(easeBloom(0.999, 1, false)).toBe(1);
    expect(easeBloom(0, 1, true)).toBe(1);
    expect(easeBloom(1, 0, true)).toBe(0);
  });
});

describe('secondTicks', () => {
  it('has one tick for each whole second inside the clip', () => {
    expect(secondTicks(4, 400)).toEqual([100, 200, 300]);
    expect(secondTicks(0.5, 400)).toEqual([]);
    expect(secondTicks(4, 0)).toEqual([]);
  });
});

describe('drawVideoTimeline', () => {
  const view = { width: 400, height: 56, fraction: 0.25, duration: 4, frames: 100, scrubbing: false, bloom: 0, focus: 0 };

  it('draws the rest line, the played line and a playhead', () => {
    const { ctx, fills } = fakeContext();
    drawVideoTimeline(ctx, view, COLORS);

    expect(fills[0]).toMatchObject({ style: 'shade', x: 0, width: 400, height: 1 });
    expect(fills[1]).toMatchObject({ style: 'rest', x: 0, width: 400, height: 2 });
    expect(fills[2]).toMatchObject({ style: 'played', x: 0, width: 100, height: 2 });
    expect(fills.at(-1)).toMatchObject({ style: 'playhead', width: 3, height: 16 });
  });

  it('makes the playhead taller while scrubbing', () => {
    const { ctx, fills } = fakeContext();
    drawVideoTimeline(ctx, { ...view, scrubbing: true }, COLORS);

    expect(fills.at(-1)).toMatchObject({ style: 'playhead', height: 26 });
  });

  it('keeps the playhead inside the canvas at both ends', () => {
    const start = fakeContext();
    drawVideoTimeline(start.ctx, { ...view, fraction: 0 }, COLORS);
    const end = fakeContext();
    drawVideoTimeline(end.ctx, { ...view, fraction: 1 }, COLORS);

    expect(start.fills.at(-1)!.x).toBeGreaterThanOrEqual(0);
    expect(end.fills.at(-1)!.x + 3).toBeLessThanOrEqual(400);
  });

  it('draws a dark line just under the hairline, so it reads on a bright frame', () => {
    const { ctx, fills } = fakeContext();
    drawVideoTimeline(ctx, view, COLORS);

    // The hairline sits 12 px up from the bottom and is 2 px thick; the dark line is the pixel below it.
    expect(fills[0].y).toBe(fills[1].y + fills[1].height);
  });

  it('draws only the rest line and its dark line when the video cannot play', () => {
    const { ctx, fills } = fakeContext();
    drawVideoTimeline(ctx, view, COLORS, false);

    expect(fills.map((fill) => fill.style)).toEqual(['shade', 'rest']);
  });
});

describe('waveform', () => {
  it('fits 4 px bars with a 3 px gap', () => {
    expect(waveBarCount(7 * 10 - 3)).toBe(10);
    expect(waveBarCount(1)).toBe(1);
  });

  it('counts the bars whose middle the playhead has passed', () => {
    // Ten bars across 67 px: bar i spans 7i to 7i + 4, so its middle is at 7i + 2.
    expect(wavePlayedBars(67, 0)).toBe(0);
    expect(wavePlayedBars(67, 1.9 / 67)).toBe(0);
    expect(wavePlayedBars(67, 2 / 67)).toBe(1);
    expect(wavePlayedBars(67, 9 / 67)).toBe(2);
    expect(wavePlayedBars(67, 1)).toBe(10);
  });

  it('takes the tallest peak of the slice each bar covers, and a low flat line without peaks', () => {
    const peaks = [10, 200, 30, 40];

    expect(waveValue(peaks, 0, 2)).toBe(200);
    expect(waveValue(peaks, 1, 2)).toBe(40);
    expect(waveValue([], 3, 10)).toBe(16);
  });

  it('colours the bars before the playhead as played and the rest as unplayed', () => {
    const { ctx, fills } = fakeContext();
    drawWaveform(ctx, { width: 70 - 3, height: 100, fraction: 0.5 }, [255, 255, 255], COLORS);
    const styles = fills.map((fill) => fill.style);

    expect(styles).toContain('played');
    expect(styles).toContain('waveRest');
    expect(styles.indexOf('played')).toBeLessThan(styles.indexOf('waveRest'));
  });

  it('keeps a silent bar at 5% of the height, at least 2 px', () => {
    const tall = fakeContext();
    drawWaveform(tall.ctx, { width: 4, height: 200, fraction: 0 }, [0], COLORS);
    const short = fakeContext();
    drawWaveform(short.ctx, { width: 4, height: 10, fraction: 0 }, [0], COLORS);

    expect(tall.fills[0].height).toBe(10);
    expect(short.fills[0].height).toBe(2);
  });
});
