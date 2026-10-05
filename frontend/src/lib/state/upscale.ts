import { ApiError } from '$lib/api/client';
import { submitUpscale } from '$lib/api/gallery';
import { jobStore } from '$lib/state/job.svelte';
import { addToast } from '$lib/state/toasts.svelte';
import type { GalleryAsset, JobContext, UpscaleFactor } from '$lib/types';

/**
 * Submit an upscale of *asset*: it starts at once, or joins the queue behind the running job.
 * Reports the outcome in a toast; resolves to the job, or null when the server refused it.
 */
export async function startUpscale(asset: GalleryAsset, factor: UpscaleFactor): Promise<JobContext | null> {
  let job: JobContext;
  try {
    job = await submitUpscale(asset.id, factor);
  } catch (error) {
    addToast(`Upscale failed: ${upscaleErrorText(error)}`, 'error', 8000);
    return null;
  }
  jobStore.jobSubmitted(job);
  if (job.queue_position) addToast(`Upscale of ${asset.filename} added to the queue as #${job.queue_position}.`, 'info');
  else addToast(`Upscaling ${asset.filename} ${factor}×`, 'success');
  return job;
}

/** Return the server's reason for a refused upscale, e.g. a size limit. */
export function upscaleErrorText(error: unknown): string {
  if (error instanceof ApiError && error.detail) return error.detail;
  return error instanceof Error ? error.message : 'Please try again.';
}
