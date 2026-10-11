import { errorText } from '$lib/api/client';
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
    addToast(`Upscale failed: ${errorText(error, 'Please try again.')}`, 'error');
    return null;
  }
  jobStore.jobSubmitted(job);
  if (job.queue_position) addToast(`Upscale of ${asset.filename} added to the queue as #${job.queue_position}.`, 'info');
  else addToast(`Upscaling ${asset.filename} ${factor}×`, 'success');
  return job;
}
