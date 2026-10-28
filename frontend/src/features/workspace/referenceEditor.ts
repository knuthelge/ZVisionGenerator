import { draft } from '$lib/state/draft.svelte';
import { skippedMessage, splitImageFiles, type ReferenceRow } from '$lib/state/references';
import { addToast } from '$lib/state/toasts.svelte';

/** What an editor of the reference list needs from the screen that hosts it. */
export interface ReferenceEditorOptions {
  /** The rows as they are now (the draft's paths with the pending uploads). */
  rows: () => ReferenceRow[];
  /** How many references the model takes. */
  max: () => number;
  /** The Workspace keeps the uploaded files; this hands it the new list, one entry per row. */
  onfileschange: (files: (File | null)[]) => void;
}

/**
 * Return the edits of the reference list, each one changing the draft's paths and the pending uploads together:
 * `append` adds rows after the last one (as many as fit, one toast telling how many did not and why; *unsupported* counts files
 * already turned away), `appendFiles` adds the PNG, JPEG and WebP files among some files as uploaded rows,
 * `replaceWithFile` and `replaceWithPath` replace row *index* (a null file does nothing), and `remove` drops it.
 */
export function referenceEditor(options: ReferenceEditorOptions) {
  /** Take new rows as the list; the path that was replaced or removed stops being flagged missing. */
  function commit(next: ReferenceRow[], freed?: string): void {
    draft.patch({
      referenceImagePaths: next.map((row) => row.path),
      missingReferencePaths: draft.state.missingReferencePaths.filter((path) => path !== freed),
    });
    options.onfileschange(next.map((row) => row.file));
  }

  function replace(index: number, row: ReferenceRow): void {
    const rows = options.rows();
    commit(rows.map((existing, i) => (i === index ? row : existing)), rows[index]?.path);
  }

  function remove(index: number): void {
    const rows = options.rows();
    commit(rows.filter((_, i) => i !== index), rows[index]?.path);
  }

  function append(additions: ReferenceRow[], unsupported = 0): void {
    if (additions.length === 0 && unsupported === 0) return;
    const rows = options.rows();
    const max = options.max();
    const taken = additions.slice(0, Math.max(0, max - rows.length));
    if (taken.length > 0) commit([...rows, ...taken]);
    const note = skippedMessage(taken.length, additions.length - taken.length, max, unsupported);
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
