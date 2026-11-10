import { describe, expect, it } from 'vitest';

import { readsAsClock, secondsLabel, timeLabel } from './mediaTime';

describe('media time labels', () => {
  it('reads seconds with one decimal', () => {
    expect(secondsLabel(3.2)).toBe('3.2 s');
    expect(secondsLabel(0)).toBe('0.0 s');
    expect(secondsLabel(Number.NaN)).toBe('0.0 s');
    expect(secondsLabel(-1)).toBe('0.0 s');
  });

  it('reads a take of a minute or more as a clock, and a video always in seconds', () => {
    expect(readsAsClock('audio', 59.9)).toBe(false);
    expect(readsAsClock('audio', 60)).toBe(true);
    expect(readsAsClock('video', 600)).toBe(false);
    expect(timeLabel(65, true)).toBe('1:05');
    expect(timeLabel(65, false)).toBe('65.0 s');
  });
});
