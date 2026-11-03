import { errorText } from '$lib/api/client';
import { planUpscale, submitUpscale } from '$lib/api/gallery';
import { confirmDownloads } from '$lib/state/confirmDownloads';
import { jobStore } from '$lib/state/job.svelte';
import { addToast } from '$lib/state/toasts.svelte';
import type { GalleryAsset, JobContext, UpscaleFactor, UpscaleMethod } from '$lib/types';

/** The files being upscaled: asking about downloads and submitting hold the file, so a second pick of it waits for the first. */
const busy = new Set<string>();

/**
 * Submit an upscale of *asset*: it asks first when the upscale downloads a model, then starts at once or joins the
 * queue behind the running job. Reports the outcome in a toast; resolves to the job, or null when the server
 * refused it, the user declined the download, or the file is already being upscaled.
 */
export async function startUpscale(asset: GalleryAsset, factor: UpscaleFactor, method: UpscaleMethod): Promise<JobContext | null> {
  if (busy.has(asset.id)) return null;
  busy.add(asset.id);
  let job: JobContext;
  try {
    if (!(await confirmDownloads(await planUpscale(asset.id, factor, method)))) return null;
    job = await submitUpscale(asset.id, factor, method);
  } catch (error) {
    addToast(`Upscale failed: ${errorText(error, 'Please try again.')}`, 'error');
    return null;
  } finally {
    busy.delete(asset.id);
  }
  jobStore.jobSubmitted(job);
  if (job.queue_position) addToast(`Upscale of ${asset.filename} added to the queue as #${job.queue_position}.`, 'info');
  else addToast(`Upscaling ${asset.filename} ${factor}×${method === 'creative' ? ' creative' : ''}`, 'success');
  return job;
}
