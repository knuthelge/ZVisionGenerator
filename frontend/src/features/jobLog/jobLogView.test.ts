import { describe, expect, it } from 'vitest';

import type { JobLogEntry } from '$lib/types';

import { hasErrorDetail, jobLogDuration, jobLogTime, jobLogTone } from './jobLogView';

describe('jobLogTone', () => {
  it.each([
    ['completed', 0, 'completed'],
    ['completed', 2, 'partial'],
    ['cancelled', 0, 'cancelled'],
    ['cancelled', 1, 'partial'],
    ['failed', 0, 'failed'],
    ['failed', 3, 'failed'],
  ] as const)('reads a %s job with %i failed runs as %s', (status, failed_runs, tone) => {
    expect(jobLogTone({ status, failed_runs })).toBe(tone);
  });
});

describe('hasErrorDetail', () => {
  it('is true with an error text or failed runs, false otherwise', () => {
    expect(hasErrorDetail({ error: 'out of memory', failed_runs: 0 })).toBe(true);
    expect(hasErrorDetail({ error: null, failed_runs: 2 })).toBe(true);
    expect(hasErrorDetail({ error: '  ', failed_runs: 0 })).toBe(false);
    expect(hasErrorDetail({ error: null, failed_runs: 0 })).toBe(false);
  });
});

describe('jobLogDuration', () => {
  it('formats a recorded duration and leaves none blank', () => {
    expect(jobLogDuration({ duration_secs: 12.34 })).toBe('12.3 s');
    expect(jobLogDuration({ duration_secs: 125 })).toBe('2m 05s');
    expect(jobLogDuration({ duration_secs: 0 })).toBe('');
  });
});

describe('jobLogTime', () => {
  it('dates an entry by completion, falling back to creation', () => {
    const base: Pick<JobLogEntry, 'completed_at' | 'created_at'> = { completed_at: 20, created_at: 10 };
    expect(jobLogTime(base)).toBe(20);
    expect(jobLogTime({ ...base, completed_at: null })).toBe(10);
    expect(jobLogTime({ completed_at: null, created_at: null })).toBeNull();
  });
});
