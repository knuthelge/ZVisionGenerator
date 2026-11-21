import { describe, expect, it } from 'vitest';

import { clipTime } from './clipTime';

describe('clipTime', () => {
  it('says the estimated minutes, rounded, at least one', () => {
    expect(clipTime({ expected_seconds: 720 })).toBe('About 12 min (estimated).');
    expect(clipTime({ expected_seconds: 20 })).toBe('About 1 min (estimated).');
    expect(clipTime({ expected_seconds: 89 })).toBe('About 1 min (estimated).');
  });

  it('says nothing where the server has no estimate', () => {
    expect(clipTime({ expected_seconds: null })).toBeNull();
    expect(clipTime({ expected_seconds: 0 })).toBeNull();
    expect(clipTime(null)).toBeNull();
  });
});
