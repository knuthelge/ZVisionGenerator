import type { GalleryAsset } from '$lib/types';
import { inProject } from '$lib/state/projectPick.svelte';
import { addToast } from '$lib/state/toasts.svelte';

/**
 * Tell the user a job finished. A result that landed outside the picked project names where it was saved, so a file that is not
 * on the page they are on is not mistaken for a lost one.
 *
 * @param outputs What the job saved.
 * @param pick The project this browser works in, or null for All files.
 */
export function completionToast(outputs: readonly Pick<GalleryAsset, 'project'>[], pick: string | null): void {
  const elsewhere = outputs.find((output) => !inProject(output, pick));
  addToast(elsewhere ? `Generation complete. Saved to ${elsewhere.project ?? 'No project'}.` : 'Generation complete', 'success');
}
