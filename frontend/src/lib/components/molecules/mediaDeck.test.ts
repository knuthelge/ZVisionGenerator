import { describe, expect, it } from 'vitest';

import { COMPACT_PICTURE, TIGHT_PICTURE, deckFor, deckMetrics } from './mediaDeck';

describe('deckMetrics', () => {
  it('gives a finger bigger targets than a mouse at every size', () => {
    for (const size of ['normal', 'compact', 'tight'] as const) {
      expect(deckMetrics(true, size).control).toBeGreaterThan(deckMetrics(false, size).control);
    }
  });

  it('gets tighter as the picture gets shorter', () => {
    for (const coarse of [false, true]) {
      const normal = deckMetrics(coarse, 'normal');
      const compact = deckMetrics(coarse, 'compact');
      const tight = deckMetrics(coarse, 'tight');

      expect(compact.track).toBeLessThanOrEqual(normal.track);
      expect(tight.track).toBeLessThanOrEqual(compact.track);
      expect(tight.control).toBeLessThanOrEqual(compact.control);
    }
  });

  it('puts the compact picture above the tight one', () => {
    expect(TIGHT_PICTURE).toBeLessThan(COMPACT_PICTURE);
  });
});

describe('deckFor', () => {
  const metrics = deckMetrics(false, 'normal');

  it('lies on the picture\'s bottom edge: the deck\'s bottom is the gap under the picture', () => {
    expect(deckFor(metrics, 0).bottom).toBe(0);
    expect(deckFor(metrics, 137).bottom).toBe(137);
  });

  it('puts the row at the bottom of the deck and the timeline over it', () => {
    const deck = deckFor(metrics, 0);

    expect(deck.rowBottom).toBeLessThan(deck.laneBottom);
    expect(deck.laneBottom).toBeLessThan(deck.height);
    expect(deck.reach).toBeLessThanOrEqual(deck.height);
  });

  it('is as tall as the row and the timeline together, with the timeline drawn over the row\'s top', () => {
    const deck = deckFor(metrics, 0);

    expect(deck.height).toBe(deck.laneBottom + metrics.track);
    expect(deck.laneBottom).toBeLessThan(deck.rowBottom + metrics.control);
  });

  it('draws the timeline closer to the row in a tight deck, so a short picture keeps its middle', () => {
    const tight = deckMetrics(false, 'tight');
    const normal = deckFor(metrics, 0, 'normal');
    const short = deckFor(tight, 0, 'tight');

    expect(short.height).toBeLessThan(normal.height);
    expect(short.laneBottom - short.rowBottom).toBeLessThan(normal.laneBottom - normal.rowBottom);
  });

  it('is taller where a finger needs the bigger targets', () => {
    expect(deckFor(deckMetrics(true, 'normal'), 0).height).toBeGreaterThan(deckFor(metrics, 0).height);
  });
});
