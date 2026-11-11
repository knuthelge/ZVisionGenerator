import type { Extension } from './editOps';
import { extendedSize } from './editOps';

/** What the Image to edit section and Generate need to know about the edit. */
export interface EditFacts {
  /** An image to edit is chosen. */
  hasImage: boolean;
  /** The image is chosen but cannot be used: an upload that did not come back, or a file that moved. What to tell the user, or null. */
  lost: string | null;
  /** The share of the picture that is painted, 0 to 1. */
  painted: number;
  extension: Extension | null;
  /** The paint has to be painted again: a reused edit, or a mask that did not come back from a reload. */
  repaint: boolean;
  /** The picture's size, when known. */
  width: number | null;
  height: number | null;
  /** The picture has transparent pixels; null until it has been looked at. */
  cutout: boolean | null;
  /** The model can change the whole picture from the prompt alone. */
  wholePicture: boolean;
}

/** Whether anything marks a region: paint, or an extension (the new area counts as painted). */
export function hasRegion(facts: Pick<EditFacts, 'painted' | 'extension'>): boolean {
  return facts.painted > 0 || facts.extension !== null;
}

/**
 * Why Generate is off, or null when it can run. A cut-out whose transparency is not known yet counts as opaque: the server refuses
 * a whole-picture edit of one, and says so.
 */
export function editReason(facts: EditFacts): string | null {
  if (!facts.hasImage) return 'Choose an image to edit';
  if (facts.lost) return facts.lost;
  // A reused edit must not silently become a whole-picture edit, whichever model it runs on, nor lose its paint behind an extension.
  if (facts.repaint && facts.painted === 0) return 'Paint the area again';
  if (hasRegion(facts)) return null;
  if (!facts.wholePicture || facts.cutout === true) return 'Paint the area to change';
  return null;
}

/** The prompt box's hint: what goes in the region, or what changes in the whole picture. */
export function editPlaceholder(facts: Pick<EditFacts, 'painted' | 'extension'>): string {
  return hasRegion(facts) ? 'What should be here?' : 'What should change?';
}

/** "18% of the image", with the smallest share worded as "under 1%". */
function share(fraction: number): string {
  const percent = Math.round(fraction * 100);
  return percent < 1 ? 'under 1%' : `${percent}%`;
}

/** The painted share, or null when nothing is painted. */
export function paintSummary(facts: Pick<EditFacts, 'painted'>): string | null {
  return facts.painted > 0 ? `Painted area: ${share(facts.painted)} of the image` : null;
}

/** The extension and the new size, or null when the picture is not extended. */
export function extendSummary(facts: Pick<EditFacts, 'extension' | 'width' | 'height'>): string | null {
  const { extension, width, height } = facts;
  if (!extension) return null;
  const size = width !== null && height !== null ? extendedSize(width, height, extension) : null;
  return `Extends ${extension.side} by ${extension.amount} px${size ? `, ${size.width} × ${size.height}` : ''}`;
}

/** The phone card's one-line summary: the paint and the extension, short, or the picture's size while there is neither. */
export function cardSummary(facts: Pick<EditFacts, 'painted' | 'extension' | 'width' | 'height'>): string {
  const parts = [
    facts.painted > 0 ? `${share(facts.painted)} painted` : null,
    facts.extension ? `extends ${facts.extension.side} by ${facts.extension.amount} px` : null,
  ].filter((part): part is string => part !== null);
  if (parts.length > 0) return `${parts.join(' · ').replace(/^./, (letter) => letter.toUpperCase())}`;
  return facts.width !== null && facts.height !== null ? `${facts.width} × ${facts.height}` : 'No paint yet';
}

/** The painter's size line: `832 × 1216` or `832 × 1216 → 832 × 1472`. */
export function sizeLine(width: number, height: number, extension: Extension | null): string {
  if (!extension) return `${width} × ${height}`;
  const size = extendedSize(width, height, extension);
  return `${width} × ${height} → ${size.width} × ${size.height}`;
}

/** The section's summary line in its header: the file name, or None. */
export function sectionSummary(name: string | null): string {
  return name ?? 'None';
}
