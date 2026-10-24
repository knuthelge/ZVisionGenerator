import { describe, expect, it } from 'vitest';

import { pagerOutcome, pullBackdropOpacity, pullOutcome, pullTransform } from './viewerPager';

const both = { prev: true, next: true };

describe('viewer pager', () => {
  it('pages when the drag went a quarter of the pane, or was a flick, that way', () => {
    expect(pagerOutcome(-100, 0, 390, both)).toBe('next');
    expect(pagerOutcome(100, 0, 390, both)).toBe('prev');
    expect(pagerOutcome(-40, 0, 390, both)).toBe('stay');
    expect(pagerOutcome(-40, -0.5, 390, both)).toBe('next');
    // A flick against the drag's direction does not count.
    expect(pagerOutcome(-40, 0.5, 390, both)).toBe('stay');
  });

  it('stays at the ends of the set', () => {
    expect(pagerOutcome(-200, 0, 390, { prev: true, next: false })).toBe('stay');
    expect(pagerOutcome(200, 0, 390, { prev: false, next: true })).toBe('stay');
  });

  it('closes on a pull down, opens the details on a pull up, and stays otherwise', () => {
    expect(pullOutcome(90, 0, true)).toBe('close');
    expect(pullOutcome(30, 0.6, true)).toBe('close');
    expect(pullOutcome(-70, 0, true)).toBe('details');
    expect(pullOutcome(-70, 0, false)).toBe('stay');
    expect(pullOutcome(30, 0, true)).toBe('stay');
  });

  it('shrinks the pulled image a little and clears the backdrop as it goes', () => {
    expect(pullTransform(0)).toBe('translate(0px, 0px) scale(1.000)');
    expect(pullTransform(200)).toBe('translate(0px, 200px) scale(0.750)');
    expect(pullBackdropOpacity(0)).toBe(1);
    expect(pullBackdropOpacity(200)).toBe(0.5);
    expect(pullBackdropOpacity(900)).toBe(0);
  });
});
