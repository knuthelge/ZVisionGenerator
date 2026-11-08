import { addToast } from '$lib/state/toasts.svelte';
import type { ToastType } from '$lib/state/toastQueue';
import type { MoveResult } from '$lib/types';

/** What a move tells the user: the toast's words and tone. */
export interface MoveOutcome {
  message: string;
  tone: ToastType;
}

/** The server's reason for the first file that failed, without its closing full stop. */
function firstReason(result: MoveResult): string {
  return result.failed[0]?.detail.replace(/\.$/, '') ?? '';
}

/**
 * Describe a move's result: success names the destination, a partial one says how many failed, that they stay selected and
 * why (the first reason), a move where every file failed is an error with its reason, and a move that found every file
 * already in the destination is a quiet note, not an error.
 *
 * @param result What the server moved and refused.
 * @param project Where the files were sent, or null for No project.
 */
export function moveOutcome(result: MoveResult, project: string | null): MoveOutcome {
  const moved = result.moved.length;
  const failed = result.failed.length;
  const name = project ?? 'No project';
  if (failed > 0 && moved === 0) return { message: `Move failed: ${firstReason(result)}`, tone: 'error' };
  if (failed > 0) return { message: `Moved ${moved}; ${failed} failed and stays selected: ${firstReason(result)}`, tone: 'warning' };
  if (moved === 0) return { message: `Already in ${name}.`, tone: 'info' };
  return { message: moved === 1 ? `Moved to ${name}.` : `Moved ${moved} files to ${name}.`, tone: 'success' };
}

/** Show a move's result as a toast. */
export function reportMove(result: MoveResult, project: string | null): void {
  const { message, tone } = moveOutcome(result, project);
  addToast(message, tone);
}
