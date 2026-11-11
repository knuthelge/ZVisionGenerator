import { draft } from '$lib/state/draft.svelte';
import { editSourceSet } from '$lib/state/editSlots';
import { heldUploads } from '$lib/state/heldUploads.svelte';
import { imageEdit } from './editStore.svelte';
import { editReason, type EditFacts } from './editView';

/** What is wrong with the draft's image to edit, in words: an upload that did not come back, or a file that moved. Null when it is usable. */
function lostSource(restored: boolean): string | null {
  const { editSourcePath: path, editSourceUpload: marker, missingReferencePaths } = draft.state;
  if (path) return missingReferencePaths.includes(path) ? 'This image has moved or been deleted. Choose it again or remove it.' : null;
  return marker !== null && restored && heldUploads.editSource === null ? 'Choose this image again.' : null;
}

/**
 * What the open Edit is made of, read from the draft (what Generate sends) and the model, with the size and transparency the
 * store has learnt about the picture.
 */
export function currentEditFacts(): EditFacts {
  const s = draft.state;
  const source = imageEdit.source;
  // A mask whose file did not come back from a reload is no paint: it has to be painted again.
  const maskLost = s.editMaskUpload !== null && heldUploads.restored && heldUploads.editMask === null;
  return {
    mode: s.editMode,
    hasImage: editSourceSet(s),
    lost: lostSource(heldUploads.restored),
    painted: s.editMaskUpload !== null && !maskLost ? s.editPaintedShare : 0,
    extension: s.editExtendSide ? { side: s.editExtendSide, amount: s.editExtendAmount } : null,
    repaint: s.editRepaint || maskLost,
    width: source?.width ?? null,
    height: source?.height ?? null,
    cutout: source?.cutout ?? null,
  };
}

/** Why Generate is off while editing, or null. */
export function currentEditReason(): string | null {
  return editReason(currentEditFacts());
}
