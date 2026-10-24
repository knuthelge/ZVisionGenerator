import { describe, expect, it } from 'vitest';

import type { FrameRule } from '$lib/types';
import {
  alignFrames,
  defaultExtendFrames,
  durationSpec,
  extendFramesFromSeconds,
  extendTotal,
  framesToSeconds,
  secondsToFrames,
  snapExtendFrames,
} from './videoDuration';

const LTX: FrameRule = { alignment: 8, offset: 1, round: 'nearest' };
const ANY: FrameRule = { alignment: 1, offset: 1, round: 'nearest' };
const H3: FrameRule = { alignment: 17, offset: 5, round: 'up' };

describe('video duration conversion', () => {
  it('aligns frame counts to the nearest 8k+1 like the server', () => {
    expect(alignFrames(50, LTX)).toBe(49);
    expect(alignFrames(121, LTX)).toBe(121);
    expect(alignFrames(1, LTX)).toBe(9);
    expect(alignFrames(50, ANY)).toBe(50);
    // Halfway counts round to even steps, as Python's round() does.
    expect(alignFrames(53, LTX)).toBe(49);
    expect(alignFrames(61, LTX)).toBe(65);
  });

  it('snaps seconds to the nearest 8k+1 frame count for LTX', () => {
    expect(secondsToFrames(5, 24, LTX)).toBe(121);
    expect(secondsToFrames(2, 24, LTX)).toBe(49);
    expect(secondsToFrames(2.1, 24, LTX)).toBe(49);
    expect(secondsToFrames(2.2, 24, LTX)).toBe(57);
  });

  it('allows frame counts beyond the old 256 ceiling', () => {
    expect(secondsToFrames(20, 24, LTX)).toBe(481);
  });

  it('never goes below one alignment step', () => {
    expect(secondsToFrames(0, 24, LTX)).toBe(9);
    expect(secondsToFrames(0.01, 24, ANY)).toBe(2);
  });

  it('converts frame counts back to seconds', () => {
    expect(framesToSeconds(121, 24)).toBe(5);
    expect(framesToSeconds(49, 24)).toBe(2);
    expect(framesToSeconds(1, 24)).toBe(0);
  });

  it('round-trips every valid frame count', () => {
    for (const frames of [9, 17, 25, 33, 49, 97, 121, 257, 481]) {
      expect(secondsToFrames(framesToSeconds(frames, 24), 24, LTX)).toBe(frames);
    }
  });

  it('steps by one alignment step with no upper bound', () => {
    expect(durationSpec(24, LTX)).toEqual({ step: 8 / 24, min: 8 / 24 });
    expect(durationSpec(24, LTX).max).toBeUndefined();
  });
});

describe('H3 frame rule (17k + 5, rounding up)', () => {
  it('moves a count to the next valid one', () => {
    expect(alignFrames(22, H3)).toBe(22);
    expect(alignFrames(23, H3)).toBe(39);
    expect(alignFrames(1, H3)).toBe(22);
    expect(alignFrames(124, H3)).toBe(124);
    expect(alignFrames(125, H3)).toBe(141);
  });

  it('converts seconds by counting whole frames before the first one', () => {
    expect(secondsToFrames(5, 24, H3)).toBe(124);
    expect(secondsToFrames(5.13, 24, H3)).toBe(124);
    expect(secondsToFrames(5.83, 24, H3)).toBe(141);
    expect(secondsToFrames(0.5, 24, H3)).toBe(22);
    expect(secondsToFrames(20, 24, H3)).toBe(481);
  });

  it('reads 124 frames as 5.125 s', () => {
    expect(framesToSeconds(124, 24)).toBe(5.125);
  });

  it('steps by one alignment step', () => {
    expect(durationSpec(24, H3)).toEqual({ step: 17 / 24, min: 17 / 24 });
  });
});

describe('extend length', () => {
  it('snaps frame counts to multiples of 8, never to 8k+1', () => {
    expect(snapExtendFrames(50)).toBe(48);
    expect(snapExtendFrames(49)).toBe(48);
    expect(snapExtendFrames(48)).toBe(48);
    expect(snapExtendFrames(56)).toBe(56);
    expect(snapExtendFrames(Number.NaN)).toBe(8);
  });

  it('snaps to the step of the model that has one', () => {
    expect(snapExtendFrames(30, 17)).toBe(34);
    expect(snapExtendFrames(8, 17)).toBe(17);
    expect(snapExtendFrames(Number.NaN, 17)).toBe(17);
    expect(extendFramesFromSeconds(2, 24, 17)).toBe(51);
    expect(defaultExtendFrames(24, 17)).toBe(51);
  });

  it('defaults to two seconds at the source frame rate', () => {
    expect(defaultExtendFrames(24)).toBe(48);
    expect(defaultExtendFrames(30000 / 1001)).toBe(56);
  });

  it('adds the new frames to the source for the total', () => {
    const total = extendTotal(97, 48, 24);
    expect(total.frames).toBe(145);
    expect(total.seconds).toBeCloseTo(6.0417, 3);
  });
});
