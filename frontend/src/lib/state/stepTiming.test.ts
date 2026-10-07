import { describe, expect, it } from 'vitest';

import { trackStepTiming, type StepTiming } from './stepTiming';

function run(samples: Array<[step: number, now: number, paused?: boolean, scope?: string]>): StepTiming | null {
  let timing: StepTiming | null = null;
  for (const [step, now, paused = false, scope = 'job:denoise:20'] of samples) {
    timing = trackStepTiming(timing, { scope, step, paused, now });
  }
  return timing;
}

describe('trackStepTiming', () => {
  it('has no estimate until one step has been seen', () => {
    expect(run([[3, 1000]])?.stepMs).toBeNull();
  });

  it('averages every step since the stage started', () => {
    expect(run([[3, 1000], [4, 2500]])?.stepMs).toBe(1500);
    // A step reported right after the one before (as after an mflux preview) does not collapse the estimate.
    expect(run([[1, 0], [2, 2000], [3, 4000], [4, 4010], [5, 8000]])?.stepMs).toBe(2000);
  });

  it('ignores repeated samples for the same step', () => {
    expect(run([[3, 1000], [3, 1800], [4, 2500]])?.stepMs).toBe(1500);
  });

  it('keeps the estimate through a pause and measures afresh after it', () => {
    const resumed = run([[3, 1000], [4, 2000], [4, 2500, true], [4, 8000], [5, 8500]]);
    expect(resumed?.stepMs).toBe(1000);
    expect(trackStepTiming(resumed, { scope: 'job:denoise:20', step: 6, paused: false, now: 9100 }).stepMs).toBe(600);
  });

  it('starts over when the stage changes', () => {
    expect(run([[3, 1000], [4, 2000], [0, 2100, false, 'job:refine:3']])?.stepMs).toBeNull();
  });
});
