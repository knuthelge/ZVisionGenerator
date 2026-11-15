import { saveAdjusted, type AdjustBody } from '$lib/api/adjust';
import { errorText } from '$lib/api/client';
import { draft } from '$lib/state/draft.svelte';
import { galleryStore } from '$lib/state/gallery.svelte';
import { jobStore } from '$lib/state/job.svelte';
import { addToast } from '$lib/state/toasts.svelte';
import type { GalleryAsset } from '$lib/types';

/** IDs whose adjust request is in flight, so a double click or a repeated key sends one request, not two. */
const pending = new Set<string>();

/** Where this browser remembers that the Replace confirmation was answered once. */
const REPLACE_ASKED_KEY = 'ziv-adjust-replace-asked-v1';

/** The section of the side panel that is open: at most one, or none. */
export type AdjustSection = 'crop' | 'light' | 'mixer' | 'none';

const SECTIONS: readonly AdjustSection[] = ['crop', 'light', 'mixer', 'none'];

/** Where this browser remembers which section of the side panel was open last. */
const SECTION_KEY = 'ziv-adjust-section-v1';

/** The section to open with: the one left open last time, or Light & colour for a first visit and for a value that is not one of the four. */
export function loadAdjustSection(): AdjustSection {
  try {
    const stored = localStorage.getItem(SECTION_KEY);
    return SECTIONS.find((section) => section === stored) ?? 'light';
  } catch {
    return 'light';
  }
}

/** Remember the open section (or none) for the next time the editor opens. */
export function saveAdjustSection(section: AdjustSection): void {
  try {
    localStorage.setItem(SECTION_KEY, section);
  } catch {
    // Storage is unavailable (private mode, quota): it opens on Light & colour again after a reload.
  }
}

/** Whether the first Replace still has to be confirmed: it asks once, and never again on this browser after that. */
export function replaceNeedsAsking(): boolean {
  try {
    return localStorage.getItem(REPLACE_ASKED_KEY) !== '1';
  } catch {
    return true;
  }
}

/** Remember that Replace was confirmed, so the next one does not ask. */
export function rememberReplaceAsked(): void {
  try {
    localStorage.setItem(REPLACE_ASKED_KEY, '1');
  } catch {
    // Storage is unavailable (private mode, quota): it asks again after a reload.
  }
}

/**
 * Save an adjustment and bring the shared state in line with the answer. A copy joins the loaded Gallery lists beside its
 * source. A replace swaps the file in the lists and the job's outputs, and refreshes what holds its path: preview URLs
 * and an Edit source (see `draft.refreshSlotsHolding`). A failure shows a toast without an action and changes
 * nothing, so the caller keeps its settings and the user saves again. A call for a file whose request is
 * still in flight does nothing and says nothing.
 *
 * @param asset The file being adjusted.
 * @param body The request, from `adjustBody`; `body.save` says whether it is a copy or a replace.
 * @returns The saved asset: the new copy, or the replaced file as it now is. Null on a failure or while a request for the file is pending.
 */
export async function adjustAndReconcile(asset: GalleryAsset, body: AdjustBody): Promise<GalleryAsset | null> {
  if (pending.has(asset.id)) return null;
  pending.add(asset.id);
  let saved: GalleryAsset;
  try {
    saved = await saveAdjusted(body);
  } catch (err) {
    addToast(`Adjust failed: ${errorText(err, 'Please try again.')}`, 'error');
    return null;
  } finally {
    pending.delete(asset.id);
  }
  if (body.save === 'copy') {
    galleryStore.addCopy(saved, asset.id);
  } else {
    galleryStore.replaceAsset(saved);
    jobStore.replaceOutput(saved);
    draft.refreshSlotsHolding(asset.file_path);
  }
  return saved;
}
