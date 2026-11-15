/**
 * The job ring's drawing rules, as pure numbers.
 *
 * One rule lays out the arcs at every count: the gap between two arcs is a quarter of each arc's slot (the
 * circumference divided by the count), never more than the ring's own gap, and arcs keep round caps. So two to
 * twenty-four arcs look as they always did, and as the count grows the arcs shorten into dots that touch until a big
 * batch reads as one ring. There are no count thresholds.
 */

export type RingSize = 'L' | 'M' | 'S' | 'XS';

/** What one ring size is made of, in px. `blob` is Blob's box (0 where there is none) and `mark` the outcome mark's diameter. */
export interface RingSpec {
  box: number;
  stroke: number;
  blob: number;
  mark: number;
  /** The mark's icon. */
  icon: number;
}

export const RING_SPECS: Readonly<Record<RingSize, RingSpec>> = {
  L: { box: 144, stroke: 4, blob: 112, mark: 22, icon: 12 },
  M: { box: 128, stroke: 4, blob: 100, mark: 22, icon: 12 },
  S: { box: 70, stroke: 3, blob: 56, mark: 18, icon: 10 },
  XS: { box: 14, stroke: 2.5, blob: 0, mark: 0, icon: 0 },
};

/** An arc under this many px cannot be seen filling, so its step ghost moves to the stamp. */
export const MIN_GHOST_PX = 4;

/** The ring's own gap between arcs, by box size: 144 and 128 give 5 px, 70 gives 3 px, 14 gives 1 px. */
export function baseGap(box: number): number {
  return box >= 100 ? 5 : box >= 50 ? 3 : 1;
}

/** Where the arcs of a ring of *count* slots lie along its circumference. */
export interface ArcLayout {
  /** Length of one arc's slot (circumference / count). */
  pitch: number;
  /** The visible gap between two arcs. */
  gap: number;
  /** Dash length of a full arc, excluding its round caps (0 once the arcs have become dots). */
  dash: number;
  /** Where arc 0's dash starts: half a gap and one cap in. */
  offset: number;
}

export function arcLayout(circumference: number, count: number, stroke: number, box: number): ArcLayout {
  if (count <= 1) return { pitch: circumference, gap: 0, dash: circumference, offset: 0 };
  const pitch = circumference / count;
  const gap = Math.min(baseGap(box), pitch / 4);
  return { pitch, gap, dash: Math.max(0, pitch - gap - stroke), offset: (gap + stroke) / 2 };
}

/** The radius the arcs are drawn at inside *box*. */
export function ringRadius(box: number, stroke: number): number {
  return box / 2 - stroke / 2 - 1;
}

/** How far down (px) the outcome mark must move from the bottom edge of the box to sit centred on the ring's 6 o'clock. */
export function markDrop(box: number, stroke: number, mark: number): number {
  return box / 2 + ringRadius(box, stroke) - (box - mark / 2);
}

/** Where a step ghost shows: on the arc, or (when one step is too short to see on it) as the stamp's foot line. */
export type GhostPlace = 'ring' | 'stamp';

/**
 * Decide where the step ghost shows.
 *
 * A step is *share* of one arc, and an arc of *count* has a visible length at this ring size; under `MIN_GHOST_PX` the
 * ghost moves to the stamp, which only exists while a percentage shows.
 */
export function ghostPlace(size: RingSize, count: number, share: number, hasStamp: boolean): GhostPlace | null {
  const { box, stroke } = RING_SPECS[size];
  const circumference = 2 * Math.PI * ringRadius(box, stroke);
  const stepPx = arcLayout(circumference, Math.max(1, count), stroke, box).dash * share;
  if (stepPx >= MIN_GHOST_PX) return 'ring';
  return hasStamp ? 'stamp' : null;
}

/** Where the working head is on the ring, in degrees clockwise from 12 o'clock: `from` at the end of the head arc's solid fill, `to` at the ghost's front (equal to `from` without a ghost). */
export interface HeadAngles {
  from: number;
  to: number;
}

/**
 * Locate the working head of a ring of *box* px and *stroke* px.
 *
 * The spark is drawn at `from`, and rides to `to` over one step while a ghost shows on the ring. The same angles say
 * where Blob looks. Null when no arc is the head.
 */
export function headAngles(
  box: number,
  stroke: number,
  arcs: readonly { fill: number }[],
  head: number | null,
  ghost: { arc: number; from: number; share: number } | null,
): HeadAngles | null {
  const headArc = head !== null ? arcs[head] : undefined;
  if (!headArc || head === null) return null;
  const circumference = 2 * Math.PI * ringRadius(box, stroke);
  const layout = arcLayout(circumference, Math.max(1, arcs.length), stroke, box);
  const degrees = (arc: number, along: number): number => (((arc * layout.pitch + layout.offset + layout.dash * along) / circumference) * 360);
  const from = degrees(head, headArc.fill);
  const to = ghost && arcs[ghost.arc] ? degrees(ghost.arc, ghost.from + ghost.share) : from;
  return { from, to };
}
