import { describe, expect, it } from 'vitest';

import type { JobLogEntry } from '$lib/types';

import { hasErrorDetail, jobLogDuration, jobLogTime, jobLogTone, promptFirstLine, timeAgo } from './jobLogView';

const NOW_MS = Date.UTC(2026, 5, 15, 12, 0, 0);
const NOW_S = NOW_MS / 1000;

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

describe('promptFirstLine', () => {
  it('returns the first non-blank line, trimmed', () => {
    expect(promptFirstLine('\n  a red fox  \nsecond line')).toBe('a red fox');
    expect(promptFirstLine('one\r\ntwo')).toBe('one');
  });

  it('returns an empty string for a blank prompt', () => {
    expect(promptFirstLine('  \n ')).toBe('');
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

describe('timeAgo', () => {
  it('says just now under a minute and for a time in the future', () => {
    expect(timeAgo(NOW_S - 30, NOW_MS)).toBe('just now');
    expect(timeAgo(NOW_S + 120, NOW_MS)).toBe('just now');
  });

  it('counts minutes, hours and days', () => {
    expect(timeAgo(NOW_S - 60, NOW_MS)).toBe('1 min ago');
    expect(timeAgo(NOW_S - 59 * 60, NOW_MS)).toBe('59 min ago');
    expect(timeAgo(NOW_S - 3600, NOW_MS)).toBe('1 h ago');
    expect(timeAgo(NOW_S - 23 * 3600, NOW_MS)).toBe('23 h ago');
    expect(timeAgo(NOW_S - 86400, NOW_MS)).toBe('1 d ago');
    expect(timeAgo(NOW_S - 7 * 86400, NOW_MS)).toBe('7 d ago');
  });

  it('shows the local date after a week', () => {
    const old = NOW_S - 8 * 86400;
    expect(timeAgo(old, NOW_MS)).toBe(new Date(old * 1000).toLocaleDateString());
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
