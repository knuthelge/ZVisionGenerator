import { expect, it } from 'vitest';

import type { FrameRule } from '$lib/types';
import { secondsToFrames } from './videoDuration';

// The rule of an LTX model (`8k + 1`) and recorded frame counts at the common rates other than 24 fps
// (seconds, fps, frames).
const LTX: FrameRule = { alignment: 8, offset: 1, round: 'nearest' };
const NON_24_FPS: [number, number, number][] = [[0, 25, 9], [0.01, 25, 9], [0.5, 25, 17], [1, 25, 25], [2, 25, 49], [2.1, 25, 57], [2.2, 25, 57], [3, 25, 73], [5, 25, 129], [5.13, 25, 129], [5.83, 25, 145], [6, 25, 153], [7.5, 25, 185], [10, 25, 249], [15, 25, 377], [20, 25, 497], [0, 30, 9], [0.01, 30, 9], [0.5, 30, 17], [1, 30, 33], [2, 30, 65], [2.1, 30, 65], [2.2, 30, 65], [3, 30, 89], [5, 30, 153], [5.13, 30, 153], [5.83, 30, 177], [6, 30, 177], [7.5, 30, 225], [10, 30, 305], [15, 30, 449], [20, 30, 601], [0, 29.97, 9], [0.01, 29.97, 9], [0.5, 29.97, 17], [1, 29.97, 33], [2, 29.97, 57], [2.1, 29.97, 65], [2.2, 29.97, 65], [3, 29.97, 89], [5, 29.97, 153], [5.13, 29.97, 153], [5.83, 29.97, 177], [6, 29.97, 177], [7.5, 29.97, 225], [10, 29.97, 297], [15, 29.97, 449], [20, 29.97, 601]];

it('rounds seconds to the nearest LTX frame count at 25, 30 and 29.97 fps', () => {
  for (const [seconds, fps, expected] of NON_24_FPS) {
    expect(secondsToFrames(seconds, fps, LTX), `${seconds} s at ${fps} fps`).toBe(expected);
  }
});
