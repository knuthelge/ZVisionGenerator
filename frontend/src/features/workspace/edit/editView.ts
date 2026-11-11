import type { EditMode } from '$lib/types';
import type { Extension } from './editOps';
import { extendedSize } from './editOps';

/** What the Image to edit section and Generate need to know about the edit. */
export interface EditFacts {
  /** What the edit marks: the paint (or nothing, for a whole-picture edit) or an extension. Only the mode's own work is sent. */
  mode: EditMode;
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
}

/**
 * Why Generate is off, or null when it can run. Extend needs a side. Paint needs paint, or nothing painted on an opaque picture:
 * that whole-picture edit is what an empty Paint sends. A cut-out whose transparency is not known yet counts as opaque: the server
 * refuses a whole-picture edit of one, and says so.
 */
export function editReason(facts: EditFacts): string | null {
  if (!facts.hasImage) return 'Choose an image to edit';
  if (facts.lost) return facts.lost;
  if (facts.mode === 'extend') return facts.extension ? null : 'Choose a side to extend';
  // A reused edit must not silently become a whole-picture edit.
  if (facts.repaint && facts.painted === 0) return 'Paint the area again';
  if (facts.painted > 0) return null;
  if (facts.cutout === true) return 'Paint the area to change';
  return null;
}

/** The prompt box's hint: what goes in the marked area (the painted part, or the new area), or what changes in the whole picture. */
export function editPlaceholder(facts: Pick<EditFacts, 'mode' | 'painted'>): string {
  return facts.mode === 'extend' || facts.painted > 0 ? 'What should be here?' : 'What should change?';
}

/** "18% of the picture", with the smallest share worded as "under 1%". */
function share(fraction: number): string {
  const percent = Math.round(fraction * 100);
  return percent < 1 ? 'under 1%' : `${percent}%`;
}

/** What the edit will send, as the mode line of the Image to edit section and the phone card shows it: the mode and its state. */
export interface ModeSummary {
  mode: EditMode;
  /** The badge: the mode's name. */
  label: string;
  /** What the mode holds, short enough for a phone's card: the painted share, the side and the new size, or why there is nothing yet. */
  detail: string;
  /** The same in full, for a screen reader: a whole-picture edit says what an empty Paint sends. Absent when `detail` says it all. */
  spoken?: string;
  /** `muted` while the mode holds nothing and Generate waits for it; `warning` for paint that has to be painted again. */
  tone: 'normal' | 'muted' | 'warning';
}

/** Whether the edit sends no region at all: Paint with nothing painted, on a picture that can be edited whole from the prompt. */
export function wholePictureEdit(facts: Pick<EditFacts, 'mode' | 'painted' | 'repaint' | 'cutout'>): boolean {
  return facts.mode === 'paint' && facts.painted === 0 && !facts.repaint && facts.cutout !== true;
}

/** The mode line: always the one mode that will be sent, with what it holds. */
export function modeSummary(facts: Pick<EditFacts, 'mode' | 'painted' | 'extension' | 'repaint' | 'width' | 'height' | 'cutout'>): ModeSummary {
  if (facts.mode === 'extend') {
    const { extension, width, height } = facts;
    if (!extension) return { mode: 'extend', label: 'Extend', detail: 'No side chosen yet', tone: 'muted' };
    const size = width !== null && height !== null ? extendedSize(width, height, extension) : null;
    const side = extension.side.replace(/^./, (letter) => letter.toUpperCase());
    return { mode: 'extend', label: 'Extend', detail: `${side}, ${extension.amount} px${size ? ` · ${size.width} × ${size.height}` : ''}`, tone: 'normal' };
  }
  if (facts.painted > 0) return { mode: 'paint', label: 'Paint', detail: `${share(facts.painted)} of the picture`, tone: 'normal' };
  if (facts.repaint) return { mode: 'paint', label: 'Paint', detail: 'Paint the area again', tone: 'warning' };
  if (wholePictureEdit(facts)) return { mode: 'paint', label: 'Paint', detail: 'Whole picture, from the prompt', spoken: 'Nothing painted, so the prompt changes the whole picture', tone: 'normal' };
  return { mode: 'paint', label: 'Paint', detail: 'Nothing painted yet', tone: 'muted' };
}

/** The painter's size line: `832 × 1216` or `832 × 1216 → 832 × 1472`. */
export function sizeLine(width: number, height: number, extension: Extension | null): string {
  if (!extension) return `${width} × ${height}`;
  const size = extendedSize(width, height, extension);
  return `${width} × ${height} → ${size.width} × ${size.height}`;
}
