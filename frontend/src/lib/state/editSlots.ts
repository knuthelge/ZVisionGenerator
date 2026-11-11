import { slotSet, type SlotValue } from './imageSlots';
import type { DraftState, ExtendSide } from '$lib/types';

/** The sides a picture can grow on. */
export const EXTEND_SIDES: readonly ExtendSide[] = ['top', 'right', 'bottom', 'left'];

/** How far an extension reaches when none has been chosen yet, in pixels. */
const EDIT_EXTEND_DEFAULT = 256;

/** The image to edit: a path on the host or an upload's marker. */
type EditSourceFields = Pick<DraftState, 'editSourcePath' | 'editSourceUpload'>;

/** Whether the draft holds an image to edit, as a path or as an upload. */
export function editSourceSet(state: EditSourceFields): boolean {
  return slotSet(state.editSourcePath, state.editSourceUpload);
}

/** Return the draft fields that make a value the image to edit: the path or the upload's file name, never both. */
export function editSourcePatch(value: SlotValue): Pick<DraftState, 'editSourcePath' | 'editSourceUpload'> {
  return {
    editSourcePath: value !== null && 'path' in value ? value.path : null,
    editSourceUpload: value !== null && 'file' in value ? value.file.name : null,
  };
}

/** The draft's paint fields with Paint chosen, nothing painted, no extension and no repaint: what any change of the image to edit leaves. */
export const NO_EDIT_PAINT: Pick<DraftState, 'editMode' | 'editMaskUpload' | 'editPaintedShare' | 'editExtendSide' | 'editExtendAmount' | 'editRepaint'> = {
  editMode: 'paint',
  editMaskUpload: null,
  editPaintedShare: 0,
  editExtendSide: null,
  editExtendAmount: EDIT_EXTEND_DEFAULT,
  editRepaint: false,
};

/** The draft's edit fields with no image and no paint. */
export const NO_EDIT: Pick<DraftState, keyof EditSourceFields | keyof typeof NO_EDIT_PAINT> = {
  editSourcePath: null,
  editSourceUpload: null,
  ...NO_EDIT_PAINT,
};
