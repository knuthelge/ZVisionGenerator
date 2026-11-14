import { errorText } from '$lib/api/client';
import { setStarred } from '$lib/api/gallery';
import { galleryStore } from '$lib/state/gallery.svelte';
import { jobStore } from '$lib/state/job.svelte';
import { addToast } from '$lib/state/toasts.svelte';
import type { GalleryAsset } from '$lib/types';

/** IDs whose star request is in flight, so a double click or a repeated key sends one request, not two. */
const pending = new Set<string>();

/**
 * Star or unstar a file, and bring the shared state in line with the answer: the loaded Gallery lists and the job's
 * outputs, which the Workspace viewer reads. A toast says so when the request fails, and nothing changes then. The
 * caller handles what is its own, such as a Gallery viewer stepping past a file that left a starred list. A call for a
 * file whose request is still in flight does nothing and says nothing.
 *
 * @param asset The file to star or unstar.
 * @param starred The state it should have.
 * @returns Whether the star was saved; false when the request failed or one for the file was already pending.
 */
export async function starAndReconcile(asset: Pick<GalleryAsset, 'id'>, starred: boolean): Promise<boolean> {
  if (pending.has(asset.id)) return false;
  pending.add(asset.id);
  try {
    await setStarred(asset.id, starred);
  } catch (err) {
    addToast(errorText(err, 'Could not star the file'), 'error');
    return false;
  } finally {
    pending.delete(asset.id);
  }
  galleryStore.setStarred(asset.id, starred);
  jobStore.setStarred(asset.id, starred);
  return true;
}
