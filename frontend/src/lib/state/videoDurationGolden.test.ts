import { describe, expect, it } from 'vitest';

import type { FrameRule } from '$lib/types';
import { alignFrames, defaultExtendFrames, durationSpec, extendFramesFromSeconds, framesToSeconds, secondsToFrames, snapExtendFrames } from './videoDuration';

// G6: the LTX outputs of the duration helpers, recorded from the implementation before frame rules and extend steps
// became parameters. Each row is the inputs followed by the output; the rows must not change when the helpers generalise.
// The rule of an LTX model (`8k + 1`) and of a model that takes any count.
const LTX: FrameRule = { alignment: 8, offset: 1, round: 'nearest' };
const ANY: FrameRule = { alignment: 1, offset: 1, round: 'nearest' };

const ALIGN_8: [number, number][] = [[-3, 9], [0, 9], [1, 9], [2, 9], [5, 9], [8, 9], [9, 9], [12, 9], [13, 17], [17, 17], [20, 17], [25, 25], [33, 33], [49, 49], [50, 49], [53, 49], [57, 57], [61, 65], [100, 97], [120, 121], [121, 121], [122, 121], [125, 129], [256, 257], [257, 257], [481, 481], [500, 497]];
const ALIGN_1: [number, number][] = [[-3, 2], [0, 2], [1, 2], [2, 2], [5, 5], [8, 8], [9, 9], [12, 12], [13, 13], [17, 17], [20, 20], [25, 25], [33, 33], [49, 49], [50, 50], [53, 53], [57, 57], [61, 61], [100, 100], [120, 120], [121, 121], [122, 122], [125, 125], [256, 256], [257, 257], [481, 481], [500, 500]];
const SECONDS_TO_FRAMES_8: [number, number, number][] = [[0, 24, 9], [0.01, 24, 9], [0.5, 24, 17], [1, 24, 25], [2, 24, 49], [2.1, 24, 49], [2.2, 24, 57], [3, 24, 73], [5, 24, 121], [5.13, 24, 121], [5.83, 24, 137], [6, 24, 145], [7.5, 24, 177], [10, 24, 241], [15, 24, 361], [20, 24, 481], [0, 25, 9], [0.01, 25, 9], [0.5, 25, 17], [1, 25, 25], [2, 25, 49], [2.1, 25, 57], [2.2, 25, 57], [3, 25, 73], [5, 25, 129], [5.13, 25, 129], [5.83, 25, 145], [6, 25, 153], [7.5, 25, 185], [10, 25, 249], [15, 25, 377], [20, 25, 497], [0, 30, 9], [0.01, 30, 9], [0.5, 30, 17], [1, 30, 33], [2, 30, 65], [2.1, 30, 65], [2.2, 30, 65], [3, 30, 89], [5, 30, 153], [5.13, 30, 153], [5.83, 30, 177], [6, 30, 177], [7.5, 30, 225], [10, 30, 305], [15, 30, 449], [20, 30, 601], [0, 29.97, 9], [0.01, 29.97, 9], [0.5, 29.97, 17], [1, 29.97, 33], [2, 29.97, 57], [2.1, 29.97, 65], [2.2, 29.97, 65], [3, 29.97, 89], [5, 29.97, 153], [5.13, 29.97, 153], [5.83, 29.97, 177], [6, 29.97, 177], [7.5, 29.97, 225], [10, 29.97, 297], [15, 29.97, 449], [20, 29.97, 601]];
const SECONDS_TO_FRAMES_1: [number, number, number][] = [[0, 24, 2], [0.01, 24, 2], [0.5, 24, 13], [1, 24, 25], [2, 24, 49], [2.1, 24, 51], [2.2, 24, 54], [3, 24, 73], [5, 24, 121], [5.13, 24, 124], [5.83, 24, 141], [6, 24, 145], [7.5, 24, 181], [10, 24, 241], [15, 24, 361], [20, 24, 481]];
const FRAMES_TO_SECONDS: [number, number, number][] = [[0, 24, 0], [1, 24, 0], [9, 24, 0.3333333333333333], [49, 24, 2], [121, 24, 5], [481, 24, 20], [0, 25, 0], [1, 25, 0], [9, 25, 0.32], [49, 25, 1.92], [121, 25, 4.8], [481, 25, 19.2], [0, 30, 0], [1, 30, 0], [9, 30, 0.26666666666666666], [49, 30, 1.6], [121, 30, 4], [481, 30, 16], [0, 29.97, 0], [1, 29.97, 0], [9, 29.97, 0.26693360026693363], [49, 29.97, 1.6016016016016017], [121, 29.97, 4.004004004004004], [481, 29.97, 16.016016016016017]];
const DURATION_SPEC: [number, { step: number; min: number }][] = [[24, {"step": 0.3333333333333333, "min": 0.3333333333333333}], [25, {"step": 0.32, "min": 0.32}], [30, {"step": 0.26666666666666666, "min": 0.26666666666666666}], [29.97, {"step": 0.26693360026693363, "min": 0.26693360026693363}]];
const SNAP_EXTEND: [number, number][] = [[0, 8], [7, 8], [12, 16], [50, 48], [100, 96]];
const EXTEND_FROM_SECONDS: [number, number, number][] = [[0.1, 24, 8], [1, 24, 24], [2, 24, 48], [2.5, 24, 64], [5, 24, 120], [0.1, 25, 8], [1, 25, 24], [2, 25, 48], [2.5, 25, 64], [5, 25, 128], [0.1, 30, 8], [1, 30, 32], [2, 30, 64], [2.5, 30, 72], [5, 30, 152], [0.1, 29.97, 8], [1, 29.97, 32], [2, 29.97, 56], [2.5, 29.97, 72], [5, 29.97, 152]];
const DEFAULT_EXTEND: [number, number][] = [[24, 48], [25, 48], [30, 64], [29.97, 56]];

describe('G6: LTX duration helpers keep their outputs', () => {
  it.each(ALIGN_8)('alignFrames(%d, LTX) is %d', (frames, expected) => {
    expect(alignFrames(frames, LTX)).toBe(expected);
  });

  it.each(ALIGN_1)('alignFrames(%d, ANY) is %d', (frames, expected) => {
    expect(alignFrames(frames, ANY)).toBe(expected);
  });

  it.each(SECONDS_TO_FRAMES_8)('secondsToFrames(%d s, %d fps, LTX) is %d', (seconds, fps, expected) => {
    expect(secondsToFrames(seconds, fps, LTX)).toBe(expected);
  });

  it.each(SECONDS_TO_FRAMES_1.map(([seconds, , expected]) => [seconds, expected]))('secondsToFrames(%d s, 24 fps, ANY) is %d', (seconds, expected) => {
    expect(secondsToFrames(seconds, 24, ANY)).toBe(expected);
  });

  it.each(FRAMES_TO_SECONDS)('framesToSeconds(%d, %d fps) is %d', (frames, fps, expected) => {
    expect(framesToSeconds(frames, fps)).toBe(expected);
  });

  it.each(DURATION_SPEC)('durationSpec(%d fps, LTX) steps by one alignment', (fps, expected) => {
    expect(durationSpec(fps, LTX)).toEqual(expected);
  });

  it('durationSpec(24 fps, ANY) steps by one frame', () => {
    expect(durationSpec(24, ANY)).toEqual({ step: 1 / 24, min: 1 / 24 });
  });

  it.each(SNAP_EXTEND)('snapExtendFrames(%d) is %d', (frames, expected) => {
    expect(snapExtendFrames(frames)).toBe(expected);
  });

  it.each(EXTEND_FROM_SECONDS)('extendFramesFromSeconds(%d s, %d fps) is %d', (seconds, fps, expected) => {
    expect(extendFramesFromSeconds(seconds, fps)).toBe(expected);
  });

  it.each(DEFAULT_EXTEND)('defaultExtendFrames(%d fps) is %d', (fps, expected) => {
    expect(defaultExtendFrames(fps)).toBe(expected);
  });
});
