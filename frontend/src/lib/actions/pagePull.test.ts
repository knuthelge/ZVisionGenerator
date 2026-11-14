import { describe, expect, it } from 'vitest';
import { PAGE_PULL_PX, pagePullCommits, pagePullReady, pullDistance, pullProgress } from './pagePull';

describe('pullDistance', () => {
  it('counts a drag in the pull direction and ignores one the other way', () => {
    expect(pullDistance(-50, 'up')).toBe(50);
    expect(pullDistance(50, 'up')).toBe(0);
    expect(pullDistance(50, 'down')).toBe(50);
    expect(pullDistance(-50, 'down')).toBe(0);
  });
});

describe('pullProgress', () => {
  it('grows with the pull up to the threshold and stays there', () => {
    expect(pullProgress(PAGE_PULL_PX / 2, 'down')).toBeCloseTo(0.5);
    expect(pullProgress(PAGE_PULL_PX * 3, 'down')).toBe(1);
    expect(pullProgress(-20, 'down')).toBe(0);
  });
});

describe('pagePullReady', () => {
  it('is ready once the pull passes the threshold, in either direction', () => {
    expect(pagePullReady(40, 'down')).toBe(false);
    expect(pagePullReady(90, 'down')).toBe(true);
    expect(pagePullReady(-40, 'up')).toBe(false);
    expect(pagePullReady(-90, 'up')).toBe(true);
  });

  it('is not ready for a pull the other way', () => {
    expect(pagePullReady(-200, 'down')).toBe(false);
    expect(pagePullReady(200, 'up')).toBe(false);
  });
});

describe('pagePullCommits', () => {
  it('commits a long pull, down or up', () => {
    expect(pagePullCommits({ offset: 100, velocity: 0.1 }, 'down')).toBe(true);
    expect(pagePullCommits({ offset: -100, velocity: -0.1 }, 'up')).toBe(true);
  });

  it('springs back from a short, slow pull', () => {
    expect(pagePullCommits({ offset: 40, velocity: 0.1 }, 'down')).toBe(false);
    expect(pagePullCommits({ offset: -40, velocity: -0.1 }, 'up')).toBe(false);
  });

  it('commits a short flick in the pull direction', () => {
    expect(pagePullCommits({ offset: 30, velocity: 1 }, 'down')).toBe(true);
    expect(pagePullCommits({ offset: -30, velocity: -1 }, 'up')).toBe(true);
  });

  it('does not commit a flick that is too short to be a pull, or too slow', () => {
    expect(pagePullCommits({ offset: 8, velocity: 2 }, 'down')).toBe(false);
    expect(pagePullCommits({ offset: 30, velocity: 0.3 }, 'down')).toBe(false);
  });

  it('stays for a drag the other way, however far or fast', () => {
    expect(pagePullCommits({ offset: -200, velocity: -1 }, 'down')).toBe(false);
    expect(pagePullCommits({ offset: 200, velocity: 1 }, 'up')).toBe(false);
  });

  it('stays when the finger ends moving back against the pull', () => {
    expect(pagePullCommits({ offset: 30, velocity: -1 }, 'down')).toBe(false);
  });
});
