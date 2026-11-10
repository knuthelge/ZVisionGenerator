import { errorText } from '$lib/api/client';
import { grabFrame, type SavedFrame } from '$lib/api/videoFrame';
import type { FrameTarget } from '$lib/state/assetActions';
import type { FramePoint } from '$lib/state/framePoint';
import { addToast } from '$lib/state/toasts.svelte';
import type { GalleryAsset } from '$lib/types';

/** What the toast says when the server could not read the frame and gave no reason. */
export const FRAME_ERROR = 'Could not read that frame.';

// The videos with an extraction under way, and the target each one is for. The time chip, the frame menu's rows and a
// closed stage ⋯ read it: every frame row of that video is disabled while it runs, and a second pick for it does nothing.
let running = $state.raw<Readonly<Record<string, FrameTarget>>>({});

/** Return the target of the extraction running for the video with this id, or null when none is. */
export function busyFor(id: string): FrameTarget | null {
  return running[id] ?? null;
}

/**
 * Extract the frame at *point* of a video and hand the saved PNG to *apply*.
 *
 * It does nothing while the same video already has an extraction running. A failure shows the server's reason (else a
 * plain one) as an error toast and leaves the draft untouched; the busy state clears either way. *apply* runs once the
 * extraction is over, even when the viewer that asked has since moved on.
 */
export async function runFrameGrab(asset: GalleryAsset, point: FramePoint, target: FrameTarget, apply: (saved: SavedFrame) => void | Promise<void>): Promise<void> {
  if (!asset.file_path || busyFor(asset.id) !== null) return;
  running = { ...running, [asset.id]: target };
  let saved: SavedFrame;
  try {
    saved = await grabFrame(asset.file_path, point.time);
  } catch (error) {
    addToast(errorText(error, FRAME_ERROR), 'error');
    return;
  } finally {
    const { [asset.id]: _done, ...rest } = running;
    running = rest;
  }
  await apply(saved);
}
