import { errorText } from '$lib/api/client';
import { moveAssets } from '$lib/api/gallery';
import { draft } from '$lib/state/draft.svelte';
import { extendInspection } from '$lib/state/extendInspection.svelte';
import { galleryStore } from '$lib/state/gallery.svelte';
import { jobStore } from '$lib/state/job.svelte';
import { reportMove } from '$lib/state/moveReport';
import { addToast } from '$lib/state/toasts.svelte';
import type { MoveResult, WorkspaceContext } from '$lib/types';

/** The move asked for last; the next one starts when it settles. */
let _lastMove: Promise<void> = Promise.resolve();

/**
 * Move files to a project, or out of every project with null, and bring the shared state in line with the answer: the
 * loaded Gallery lists, the job's outputs and the Workspace slots, which follow a moved file to its new path. A toast
 * says how it went. The caller handles what is its own, such as the selection and a viewer.
 *
 * Moves run one after another in the order they were asked, whichever page asked: a move started while another runs
 * waits for its turn, so a file is never moved from two places at once and a quick run of moves all land.
 *
 * @param ids The IDs of the files to move, loaded or not. The server's answer carries each file's old path, which is what the slots hold.
 * @param project The destination project's name, or null for No project.
 * @param context Tells the draft whether the model takes a start image; the extend source is inspected again with it.
 * @returns The server's answer, or null when the request itself failed (a toast says so and nothing changed).
 */
export function moveAndReconcile(ids: readonly string[], project: string | null, context: WorkspaceContext | null): Promise<MoveResult | null> {
  const run = _lastMove.then(() => moveNow(ids, project, context));
  _lastMove = run.then(() => undefined, () => undefined);
  return run;
}

async function moveNow(ids: readonly string[], project: string | null, context: WorkspaceContext | null): Promise<MoveResult | null> {
  let result: MoveResult;
  try {
    result = await moveAssets([...ids], project);
  } catch (err) {
    addToast(errorText(err, 'Move failed'), 'error');
    return null;
  }
  galleryStore.applyMoves(result.moved);
  jobStore.replaceOutputs(result.moved);
  const movedPaths = new Map<string, string>();
  for (const { from_path, asset } of result.moved) {
    if (from_path && asset.file_path) movedPaths.set(from_path, asset.file_path);
  }
  // The extend source reports the folder it saves beside, which the move changed.
  if (draft.retargetPaths(movedPaths)) void extendInspection.inspect(context);
  reportMove(result, project);
  return result;
}
