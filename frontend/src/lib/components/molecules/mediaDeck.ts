/** Where a video player's controls lie on its picture: the deck is a box over the picture's bottom, as wide as the picture. */

/** The heights the deck is built from, in CSS pixels: the timeline's lane and a control. A short picture gets smaller ones. */
export interface DeckMetrics {
  track: number;
  control: number;
}

/** How tight the deck is: it follows the picture's height. */
export type DeckSize = 'normal' | 'compact' | 'tight';

/** Every offset is in CSS pixels from the deck's own bottom edge, except `bottom`, which is from the player's. */
export interface Deck {
  /** The deck's bottom edge, from the player's bottom edge. */
  bottom: number;
  height: number;
  /** The timeline's lane, from the deck's bottom edge. */
  laneBottom: number;
  /** The row of the time and the controls, from the deck's bottom edge. */
  rowBottom: number;
  /** The top of what the deck takes, from the deck's bottom edge: the big Play stays above it. */
  reach: number;
}

const PAD = 8;
/** The hairline's middle above the bottom of the timeline's lane, which the timeline draws 12 px over its bottom. */
const LINE = 11;

/** A picture shorter than this gets the compact deck. */
export const COMPACT_PICTURE = 320;
/** A picture shorter than this gets the tight deck, with the line closer to the row. */
export const TIGHT_PICTURE = 180;

/**
 * Return where a deck's parts lie over a picture whose bottom edge is *gap* pixels above the player's.
 * The row is at the bottom and the timeline just over it, so the controls sit on the picture's bottom edge.
 */
export function deckFor(m: DeckMetrics, gap: number, size: DeckSize = 'normal'): Deck {
  const rowBottom = PAD;
  const laneBottom = rowBottom + m.control - (size === 'tight' ? 12 : 4);
  const lineBottom = laneBottom + LINE;
  return { bottom: gap, height: laneBottom + m.track, laneBottom, rowBottom, reach: lineBottom + 8 };
}

/** Return the heights for a deck: touch gets bigger targets, and a shorter picture gets a tighter deck. */
export function deckMetrics(coarse: boolean, size: DeckSize): DeckMetrics {
  if (coarse) return size === 'tight' ? { track: 36, control: 44 } : size === 'compact' ? { track: 44, control: 44 } : { track: 54, control: 48 };
  return size === 'tight' ? { track: 34, control: 30 } : size === 'compact' ? { track: 42, control: 32 } : { track: 44, control: 36 };
}
