import type { DraftState, WorkspaceContext } from '$lib/types';

/** The tabs that hold images: Image keeps a start image and references, Video a first frame and a last frame. */
export type ImageTab = 'image' | 'video';

/** What a slot holds: a path on the host, a file uploaded from this device, or nothing. */
export type SlotValue = { path: string } | { file: File } | null;

type StartFields = Pick<DraftState, 'referenceImagePath' | 'referenceImageUpload' | 'firstFramePath' | 'firstFrameUpload'>;
export type SlotState = StartFields & Pick<DraftState, 'workflow' | 'endImagePath' | 'endImageUpload' | 'referenceImageStrength' | 'missingReferencePaths'>;

/** Whether a path is set: not null and not empty. */
export function hasPath(path: string | null): path is string {
  return path !== null && path !== '';
}

/**
 * Whether a slot is set: it has a path or an upload marker. The workflow and what a Generate sends both follow this one
 * rule, so a file held without a marker is not an image (the draft is what a run reads).
 */
export function slotSet(path: string | null, marker: string | null): boolean {
  return hasPath(path) || marker !== null;
}

/** Whether the tab's start image (the first frame in Video) is set, as a path or as an upload. */
export function startImageSet(state: StartFields, tab: ImageTab): boolean {
  return tab === 'image'
    ? slotSet(state.referenceImagePath, state.referenceImageUpload)
    : slotSet(state.firstFramePath, state.firstFrameUpload);
}

/** Return the draft fields that make a value the tab's start image: the path or the upload's file name, never both. */
export function startSlotPatch(tab: ImageTab, value: SlotValue): Partial<DraftState> {
  const path = value !== null && 'path' in value ? value.path : null;
  const upload = value !== null && 'file' in value ? value.file.name : null;
  return tab === 'image'
    ? { referenceImagePath: path, referenceImageUpload: upload }
    : { firstFramePath: path, firstFrameUpload: upload };
}

/** Return the draft fields that make a value the last frame: the path or the upload's file name, never both. */
export function lastSlotPatch(value: SlotValue): Partial<DraftState> {
  return {
    endImagePath: value !== null && 'path' in value ? value.path : null,
    endImageUpload: value !== null && 'file' in value ? value.file.name : null,
  };
}

/** What a model takes for the images of a tab. */
export interface ImageSupport {
  /** The start image: an image model with image-to-image, or a video model that starts from an image. */
  start: boolean;
  /** The last frame: a video model that takes a start image and does not hide the last-frame control. */
  last: boolean;
  /** How many references the model takes; always 0 in Video. */
  references: number;
}

/**
 * Return what a model takes for the images of a tab. This is the one place that reads a model's image capability flags: the
 * workflow, the Images section, Generate and the Use as menu all follow it. Without a context nothing is known, so the start
 * image is assumed usable, and so is the last frame in Video.
 */
export function imageSupport(ctx: WorkspaceContext | null, model: string, tab: ImageTab): ImageSupport {
  if (ctx === null) return { start: true, last: tab === 'video', references: 0 };
  if (tab === 'image') {
    const defaults = ctx.image_model_defaults?.[model] ?? ctx.defaults;
    return { start: defaults?.supports_img2img !== false, last: false, references: Math.max(0, defaults?.max_reference_images ?? 0) };
  }
  const defaults = ctx.video_model_defaults?.[model] ?? ctx.video_defaults;
  const start = defaults?.supports_i2v !== false;
  return { start, last: start && !(defaults?.hidden_controls ?? []).includes('end_image_path'), references: 0 };
}
