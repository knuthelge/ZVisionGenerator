import { draft } from '$lib/state/draft.svelte';
import { heldUploads } from '$lib/state/heldUploads.svelte';
import { imageEdit } from './editStore.svelte';
import { readPicture, sourceOfFile, sourceOfPath, type EditSource } from './editSource';

/**
 * Show a source as the image to edit and learn what it does not tell yet, in the background, from one load of the picture: its size
 * and whether it is a cut-out, which matter even when the painter is never opened (the card shows the size, and a cut-out cannot be
 * extended). An answer that never comes leaves the fact unknown, and Generate stays on: the server refuses what it cannot run.
 */
function showSource(source: EditSource): void {
  imageEdit.setSource(source);
  if (source.width === null || source.height === null || source.cutout === null) {
    void readPicture(source).then((picture) => {
      if (picture) imageEdit.learn(source.id, picture);
    });
  }
}

/**
 * Make a source the image to edit. The draft takes it (a path, or an upload held for Generate) and any paint, extension and
 * history of the image before it go.
 */
export function chooseSource(source: EditSource): void {
  draft.setEditSource(source.file !== null ? { file: source.file } : source.path !== null ? { path: source.path } : null);
  showSource(source);
}

/** Forget the image to edit and everything done to it. */
export function clearSource(): void {
  draft.setEditSource(null);
  imageEdit.setSource(null);
}

/**
 * Bring the shown image in line with the draft's: after a reload, a Reuse, an Edit from the Gallery, a Reset, a delete or a move
 * the draft holds the truth and the store only draws it. Nothing changes while they agree.
 */
export function followDraft(): void {
  const { editSourcePath: path, editSourceUpload: marker } = draft.state;
  const file = heldUploads.editSource;
  const shown = imageEdit.source;
  if (path) {
    if (shown?.path !== path) showSource(sourceOfPath(path));
  } else if (marker !== null && file !== null) {
    if (shown?.file !== file) showSource(sourceOfFile(file));
  } else if (shown !== null) {
    imageEdit.setSource(null);
  }
}
