import { describe, expect, it } from 'vitest';

import { alignFrameCount, durationSpec, framesToSeconds, secondsToFrames } from './videoDuration';

describe('video duration conversion', () => {
  it('aligns frame counts to the nearest 8k+1 like the server', () => {
    expect(alignFrameCount(50, 8)).toBe(49);
    expect(alignFrameCount(121, 8)).toBe(121);
    expect(alignFrameCount(1, 8)).toBe(9);
    expect(alignFrameCount(50, 1)).toBe(50);
    // Halfway counts round to even steps, as Python's round() does.
    expect(alignFrameCount(53, 8)).toBe(49);
    expect(alignFrameCount(61, 8)).toBe(65);
  });

  it('snaps seconds to the nearest 8k+1 frame count for LTX', () => {
    expect(secondsToFrames(5, 24, 8)).toBe(121);
    expect(secondsToFrames(2, 24, 8)).toBe(49);
    expect(secondsToFrames(2.1, 24, 8)).toBe(49);
    expect(secondsToFrames(2.2, 24, 8)).toBe(57);
  });

  it('allows frame counts beyond the old 256 ceiling', () => {
    expect(secondsToFrames(20, 24, 8)).toBe(481);
  });

  it('never goes below one alignment step', () => {
    expect(secondsToFrames(0, 24, 8)).toBe(9);
    expect(secondsToFrames(0.01, 24, 1)).toBe(2);
  });

  it('converts frame counts back to seconds', () => {
    expect(framesToSeconds(121, 24)).toBe(5);
    expect(framesToSeconds(49, 24)).toBe(2);
    expect(framesToSeconds(1, 24)).toBe(0);
  });

  it('round-trips every valid frame count', () => {
    for (const frames of [9, 17, 25, 33, 49, 97, 121, 257, 481]) {
      expect(secondsToFrames(framesToSeconds(frames, 24), 24, 8)).toBe(frames);
    }
  });

  it('steps by one alignment step with no upper bound', () => {
    expect(durationSpec(24, 8)).toEqual({ step: 8 / 24, min: 8 / 24 });
    expect(durationSpec(24, 8).max).toBeUndefined();
  });
});
