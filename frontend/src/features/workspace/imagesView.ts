import { fileName } from '$lib/components/molecules/assetDetails';
import { draft } from '$lib/state/draft.svelte';
import { heldUploads } from '$lib/state/heldUploads.svelte';
import { hasPath, slotSet, type ImageTab, type SlotState } from '$lib/state/imageSlots';
import { referenceRows, type ReferenceRow } from '$lib/state/references';
import { workspaceTab } from '$lib/state/workspaceTab';
import type { WorkspaceContext } from '$lib/types';
import { workspaceCapabilities, type WorkspaceCapabilities } from './capabilities';
import { currentEditReason } from './edit/currentEdit';

/** What one image slot (the start image, the first frame or the last frame) holds and what the run does with it. */
export interface SlotView {
  /** The slot's host path, or null. */
  path: string | null;
  /** The file held for the slot, or null. */
  file: File | null;
  /** The slot holds an image, as a path or as an upload marker (`slotSet`). */
  set: boolean;
  /** The run sends the slot's image when it holds one: always for the start image, only with a first frame for the last frame. */
  used: boolean;
  /** The image is used and gone: a moved path or an upload that did not come back. */
  gone: boolean;
  /** What to tell the user when the image is gone, or null. */
  error: string | null;
  /** The name to show: the held file's, the path's file name, else the name an upload was stored under (before its file comes back). */
  name: string;
}

/** One reference row, with whether its image is gone (a moved path or an upload that did not come back). */
export interface ReferenceView extends ReferenceRow {
  gone: boolean;
  /** What to tell the user when the image is gone, or null. */
  error: string | null;
  /** The name to show: the held file's or the path's file name; empty for an upload whose file did not come back. */
  name: string;
}

/**
 * The one answer to which images a Generate uses. It is derived from the draft, the model's capabilities and the held
 * uploads; the Images section and its phone popover render it, the summary and the chip format it, Generate reads `block`
 * and the submit reads `sent`.
 */
export interface ImagesView {
  tab: ImageTab;
  /** The model takes a start image or references, so the Images section is offered. */
  visible: boolean;
  /** The tab's start image (the first frame in Video); null when the model takes none. */
  start: SlotView | null;
  /** The last frame (Video, on a model that takes one); `used` is false without a first frame, when it is not sent and Generate is blocked. */
  last: SlotView | null;
  /** The Strength row: the start image is set and an image-to-image run uses it. */
  strength: boolean;
  /** The reference list (Image, on a model that takes references); null otherwise. */
  references: {
    /** The rows Generate sends and the list shows: the first `max` of the list. */
    rows: ReferenceView[];
    max: number;
    hasRoom: boolean;
  } | null;
  /** Generate's reason when an image the run uses is gone (the start image or the frames before a reference) or a last frame has no first frame; null before the held uploads are read back. */
  block: string | null;
  /** The images a Generate sends besides the references: an upload wins over a path. A kept last frame is not sent without a first frame. */
  sent: { image: string | File | null; strength: number | null; end: string | File | null };
}

/** What `imagesView` reads from the draft and the screen. */
export interface ImagesViewInput {
  state: SlotState & { referenceImagePaths: readonly string[] };
  caps: Pick<WorkspaceCapabilities, 'showRefImage' | 'takesLastFrame' | 'maxReferences' | 'showI2IStrength'>;
  /** The held uploads (`heldUploads`): the start image of Image, the first frame and the last frame of Video, and one entry per reference row. */
  files: { startImage: File | null; firstFrame: File | null; lastFrame: File | null; references: readonly (File | null)[] };
  /** The held uploads have been read back, so an upload without a file is known to be lost. */
  restored: boolean;
  /** Edit, where the image to edit stands in for the images: its reason (no image, a lost upload, paint to add) is Generate's. Absent outside Edit. */
  edit?: { block: string | null };
}

// An image is gone when its path is flagged missing, or when it is an upload (a marker) whose file did not come back from the
// restore. Before the restore has settled an upload is not yet known to be lost.
function _missing(path: string | null, marker: string | null, file: File | null, flagged: readonly string[], restored: boolean): boolean {
  if (file !== null) return false;
  if (hasPath(path)) return flagged.includes(path);
  return marker !== null && restored;
}

// What to tell the user about a gone image: an upload lost to a reload is picked again, a moved path may also be removed.
function _missingReason(upload: boolean): string {
  return upload ? 'Choose this image again.' : 'This image has moved or been deleted. Choose it again or remove it.';
}

function _slot(path: string | null, marker: string | null, file: File | null, used: boolean, state: SlotState, restored: boolean): SlotView {
  const set = slotSet(path, marker);
  const name = file?.name ?? (hasPath(path) ? fileName(path) : (marker ?? ''));
  const gone = used && set && _missing(path, marker, file, state.missingReferencePaths, restored);
  return { path, file, set, used, gone, error: gone ? _missingReason(!hasPath(path)) : null, name };
}

function _reference(row: ReferenceRow, state: SlotState, restored: boolean): ReferenceView {
  // An uploaded row has an empty path, which stands for its marker.
  const upload = row.path === '';
  const gone = _missing(upload ? null : row.path, upload ? '' : null, row.file, state.missingReferencePaths, restored);
  return { ...row, gone, error: gone ? _missingReason(upload) : null, name: row.file?.name ?? (upload ? '' : fileName(row.path)) };
}

/** Generate's reason when a last frame is set and the first frame is not. */
const LAST_NEEDS_FIRST = 'Add a first frame. A last frame needs one.';

function _block(view: Pick<ImagesView, 'tab' | 'start' | 'last' | 'references'>, restored: boolean): string | null {
  if (!restored) return null;
  if (view.start?.gone) return view.tab === 'video' ? 'The first frame is missing.' : 'The start image is missing.';
  if (view.last?.gone) return 'The last frame is missing.';
  if (view.last?.set && view.start !== null && !view.start.set) return LAST_NEEDS_FIRST;
  const first = view.references?.rows.findIndex((row) => row.gone) ?? -1;
  return first === -1 ? null : `Reference ${first + 1} is missing.`;
}

function _sent(view: Pick<ImagesView, 'tab' | 'start' | 'last'>, strength: number): ImagesView['sent'] {
  const { start, last } = view;
  if (start === null || !start.set) return { image: null, strength: null, end: null };
  const image = start.file ?? (hasPath(start.path) ? start.path : null);
  const end = image !== null && last?.used === true ? (last.file ?? (hasPath(last.path) ? last.path : null)) : null;
  return { image, strength: view.tab === 'image' ? strength : null, end };
}

/**
 * Derive which images the open tab uses: the start image (the first frame in Video), the last frame, the reference rows,
 * whether Strength is offered, the reason Generate is blocked and what a submit sends. A slot the model cannot use is null.
 */
export function imagesView({ state, caps, files, restored, edit }: ImagesViewInput): ImagesView {
  const tab: ImageTab = workspaceTab(state.workflow) === 'video' ? 'video' : 'image';
  const video = tab === 'video';
  const start = caps.showRefImage
    ? _slot(video ? state.firstFramePath : state.referenceImagePath, video ? state.firstFrameUpload : state.referenceImageUpload, video ? files.firstFrame : files.startImage, true, state, restored)
    : null;
  const last = caps.takesLastFrame
    ? _slot(state.endImagePath, state.endImageUpload, files.lastFrame, start?.set === true, state, restored)
    : null;
  const max = caps.maxReferences;
  const all = referenceRows(state.referenceImagePaths, files.references).map((row) => _reference(row, state, restored));
  const references = max > 0 ? { rows: all.slice(0, max), max, hasRoom: all.length < max } : null;
  const partial = { tab, start, last, references };
  return {
    ...partial,
    visible: start !== null || references !== null,
    strength: start?.set === true && caps.showI2IStrength,
    block: edit ? edit.block : _block(partial, restored),
    sent: _sent(partial, state.referenceImageStrength),
  };
}

/** The view of the open tab, read from the draft and the held uploads. */
export function currentImagesView(context: WorkspaceContext): ImagesView {
  const caps = workspaceCapabilities(context, draft.state);
  return imagesView({ state: draft.state, caps, files: heldUploads, restored: heldUploads.restored, edit: caps.isEdit ? { block: currentEditReason() } : undefined });
}
