import { draft } from '$lib/state/draft.svelte';
import { extendInspection } from '$lib/state/extendInspection.svelte';
import type { WorkspaceContext } from '$lib/types';

/**
 * Empty the draft's single-file slots (start image, first frame, last frame, the Video and Extend audio tracks, the extend
 * source and the voice recording) that hold one of these files, because the files were deleted. A slot pointing at a
 * missing file would fail the next run; the workflow follows from what is left. Clearing the extend source also forgets its
 * inspection. Reference rows stay, so the prompt's "image 2" keeps meaning the same row. A moved file is not gone: the
 * slots follow it (`draft.retargetPaths`).
 *
 * @param paths Absolute paths of the files that are gone.
 * @param context Tells whether the model takes a start image; without it one is assumed.
 * @returns Whether a reference row holds one of the paths, so the caller can re-check its images and flag the row missing.
 */
export function emptySlotsHolding(paths: Iterable<string | null | undefined>, context: WorkspaceContext | null): boolean {
  const gone = new Set<string>();
  for (const path of paths) if (path) gone.add(path);
  if (gone.size === 0) return false;
  const { referenceImagePath, firstFramePath, endImagePath, referenceImagePaths, audioTrackPath, extendAudioTrackPath, extendSourcePath, voicePath } = draft.state;
  if (referenceImagePath && gone.has(referenceImagePath)) draft.setStartImage('image', null, context);
  if (firstFramePath && gone.has(firstFramePath)) draft.setStartImage('video', null, context);
  if (endImagePath && gone.has(endImagePath)) draft.setLastFrame(null);
  if (audioTrackPath && gone.has(audioTrackPath)) draft.setAudioTrack('video', null);
  if (extendAudioTrackPath && gone.has(extendAudioTrackPath)) draft.setAudioTrack('extend', null);
  if (voicePath && gone.has(voicePath)) draft.setVoice(null);
  if (extendSourcePath && gone.has(extendSourcePath)) {
    draft.setExtendSourcePath(null);
    void extendInspection.inspect(context);
  }
  return referenceImagePaths.some((path) => gone.has(path));
}
