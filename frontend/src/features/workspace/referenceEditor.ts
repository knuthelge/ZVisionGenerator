import { draft } from '$lib/state/draft.svelte';
import { heldUploads } from '$lib/state/heldUploads.svelte';
import { referencePathsKey, referenceRows, skippedMessage, splitImageFiles, type ReferenceList, type ReferenceRow } from '$lib/state/references';
import { addToast } from '$lib/state/toasts.svelte';

/** How a reference list is worded: the row label from its number, and the hint on how to name the images in the prompt. */
export interface ReferenceWording {
  /** The row's label: the number it has in the prompt. */
  label: (number: number) => string;
  /** The hint under the list. */
  hint: string;
}

/**
 * Return the wording of a list: the Image tab's rows read "Reference N" and are named "image N" in the prompt; the Video tab's
 * and Extend kind's rows read `<Picture N>`, the tag the video model reads, and the hint shows how to write it.
 */
export function referenceWording(list: ReferenceList): ReferenceWording {
  if (list === 'image') return { label: (number) => `Reference ${number}`, hint: 'Refer to them in the prompt as image 1, image 2…' };
  return {
    label: (number) => `<Picture ${number}>`,
    hint: 'Refer to them in your prompt as <Picture 1>, <Picture 2>, e.g. "The man from <Picture 1> walks into the kitchen."',
  };
}

/**
 * Return the edits of the reference list of the Image tab, the Video tab's New kind or the Extend kind (*list*), each one changing the draft's paths and the held uploads together (the rows are the
 * draft's paths paired with `heldUploads`, so every screen that edits the list builds it the same way). *max* is how many
 * references the model takes:
 * `append` adds rows after the last one (as many as fit, one toast telling how many did not and why; *unsupported* counts files
 * already turned away), `appendFiles` adds the PNG, JPEG and WebP files among some files as uploaded rows,
 * `replaceWithFile` and `replaceWithPath` replace row *index* (a null file does nothing), and `remove` drops it.
 */
export function referenceEditor(max: () => number, list: () => ReferenceList) {
  /** Take new rows as the list; the path that was replaced or removed stops being flagged missing. */
  function commit(next: ReferenceRow[], freed?: string): void {
    const paths = next.map((row) => row.path);
    draft.patch({
      [referencePathsKey(list())]: paths,
      missingReferencePaths: draft.state.missingReferencePaths.filter((path) => path !== freed),
    });
    heldUploads.setReferencesOf(list(), next.map((row) => row.file));
  }

  /** The rows as they are now: the draft's paths with the held uploads. */
  function current(): ReferenceRow[] {
    return referenceRows(draft.state[referencePathsKey(list())], heldUploads.referencesOf(list()));
  }

  function replace(index: number, row: ReferenceRow): void {
    const rows = current();
    commit(rows.map((existing, i) => (i === index ? row : existing)), rows[index]?.path);
  }

  function remove(index: number): void {
    const rows = current();
    commit(rows.filter((_, i) => i !== index), rows[index]?.path);
  }

  function append(additions: ReferenceRow[], unsupported = 0): void {
    if (additions.length === 0 && unsupported === 0) return;
    const rows = current();
    const cap = max();
    const taken = additions.slice(0, Math.max(0, cap - rows.length));
    if (taken.length > 0) commit([...rows, ...taken]);
    const note = skippedMessage(taken.length, additions.length - taken.length, cap, unsupported);
    if (note) addToast(note, 'info');
  }

  return {
    append,
    appendFiles(files: Iterable<File>): void {
      const { supported, unsupported } = splitImageFiles(files);
      append(supported.map((file) => ({ path: '', file })), unsupported);
    },
    replaceWithFile(index: number, file: File | null): void {
      if (file !== null) replace(index, { path: '', file });
    },
    replaceWithPath(index: number, path: string): void {
      replace(index, { path, file: null });
    },
    remove,
  };
}
