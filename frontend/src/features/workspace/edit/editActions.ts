import { draft } from '$lib/state/draft.svelte';
import { heldUploads } from '$lib/state/heldUploads.svelte';
import { imageEdit } from './editStore.svelte';
import { readCutout, readSize, sourceOfFile, sourceOfPath, type EditSource } from './editSource';

/**
 * Show a source as the image to edit and learn what it does not tell yet, in the background: its size, and whether it is a cut-out,
 * which matters even when the painter is never opened (Klein with nothing painted is refused on one). An answer that never comes
 * leaves the fact unknown, and Generate stays on: the server refuses what it cannot run.
 */
function showSource(source: EditSource): void {
  imageEdit.setSource(source);
  if (source.width === null || source.height === null) {
    void readSize(source).then((size) => {
      if (size) imageEdit.learn(source.id, size);
    });
  }
  if (source.cutout === null) {
    void readCutout(source).then((cutout) => {
      if (cutout !== null) imageEdit.learn(source.id, { cutout });
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
