import { draft } from '$lib/state/draft.svelte';
import { heldUploads } from '$lib/state/heldUploads.svelte';
import { referenceRows, skippedMessage, splitImageFiles, type ReferenceRow } from '$lib/state/references';
import { addToast } from '$lib/state/toasts.svelte';

/**
 * Return the edits of the reference list, each one changing the draft's paths and the held uploads together (the rows are the
 * draft's paths paired with `heldUploads`, so every screen that edits the list builds it the same way). *max* is how many
 * references the model takes:
 * `append` adds rows after the last one (as many as fit, one toast telling how many did not and why; *unsupported* counts files
 * already turned away), `appendFiles` adds the PNG, JPEG and WebP files among some files as uploaded rows,
 * `replaceWithFile` and `replaceWithPath` replace row *index* (a null file does nothing), and `remove` drops it.
 */
export function referenceEditor(max: () => number) {
  /** Take new rows as the list; the path that was replaced or removed stops being flagged missing. */
  function commit(next: ReferenceRow[], freed?: string): void {
    draft.patch({
      referenceImagePaths: next.map((row) => row.path),
      missingReferencePaths: draft.state.missingReferencePaths.filter((path) => path !== freed),
    });
    heldUploads.setReferences(next.map((row) => row.file));
  }

  /** The rows as they are now: the draft's paths with the held uploads. */
  function current(): ReferenceRow[] {
    return referenceRows(draft.state.referenceImagePaths, heldUploads.references);
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
