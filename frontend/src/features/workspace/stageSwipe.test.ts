import { describe, expect, it } from 'vitest';
import { galleryPullReady, stageSwipeOutcome, wheelStep } from './stageSwipe';

const all = { older: true, newer: true, gallery: true };

describe('stageSwipeOutcome', () => {
  it('steps older on a swipe left and newer on a swipe right', () => {
    expect(stageSwipeOutcome({ axis: 'x', offset: -80, velocity: -0.1 }, all)).toBe('older');
    expect(stageSwipeOutcome({ axis: 'x', offset: 80, velocity: 0.1 }, all)).toBe('newer');
  });

  it('steps on a short flick, and springs back from a short slow drag', () => {
    expect(stageSwipeOutcome({ axis: 'x', offset: -24, velocity: -0.8 }, all)).toBe('older');
    expect(stageSwipeOutcome({ axis: 'x', offset: -24, velocity: -0.05 }, all)).toBe('stay');
  });

  it('springs back past either end of the history', () => {
    expect(stageSwipeOutcome({ axis: 'x', offset: 120, velocity: 0.5 }, { ...all, newer: false })).toBe('stay');
    expect(stageSwipeOutcome({ axis: 'x', offset: -120, velocity: -0.5 }, { ...all, older: false })).toBe('stay');
  });

  it('opens the Gallery on a long pull up or a flick up, where the Gallery is a swipe away', () => {
    expect(stageSwipeOutcome({ axis: 'y', offset: -100, velocity: -0.1 }, all)).toBe('gallery');
    expect(stageSwipeOutcome({ axis: 'y', offset: -30, velocity: -1 }, all)).toBe('gallery');
    expect(stageSwipeOutcome({ axis: 'y', offset: -40, velocity: -0.1 }, all)).toBe('stay');
    expect(stageSwipeOutcome({ axis: 'y', offset: -200, velocity: -1 }, { ...all, gallery: false })).toBe('stay');
  });
});

describe('galleryPullReady', () => {
  it('is ready once the pull passes the threshold', () => {
    expect(galleryPullReady(-40)).toBe(false);
    expect(galleryPullReady(-90)).toBe(true);
  });
});

describe('wheelStep', () => {
  it('steps once a sideways trackpad gesture has moved far enough, older to the left', () => {
    expect(wheelStep(20)).toBe(0);
    expect(wheelStep(70)).toBe(1);
    expect(wheelStep(-70)).toBe(-1);
  });
});
