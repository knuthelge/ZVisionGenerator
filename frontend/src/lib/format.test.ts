import { describe, expect, it } from 'vitest';

import { promptFirstLine, timeAgo } from './format';

const NOW_MS = Date.UTC(2026, 5, 15, 12, 0, 0);
const NOW_S = NOW_MS / 1000;

describe('promptFirstLine', () => {
  it('returns the first non-blank line, trimmed', () => {
    expect(promptFirstLine('\n  a red fox  \nsecond line')).toBe('a red fox');
    expect(promptFirstLine('one\r\ntwo')).toBe('one');
  });

  it('returns an empty string for a blank prompt', () => {
    expect(promptFirstLine('  \n ')).toBe('');
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
