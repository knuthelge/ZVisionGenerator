<script lang="ts">
  import { Badge, Icon } from '$lib/components/atoms';
  import { isPaused, isVideo, itemActivity, jobPhase, stepFraction } from '$lib/state/jobView';
  import type { ActiveJobState } from '$lib/types';

  interface Props {
    job: ActiveJobState;
    class?: string;
  }

  let { job, class: extraClass = '' }: Props = $props();

  /** Fraction of the steps at which the first live preview arrives (the server renders them at 25, 50 and 75%). */
  const FIRST_PREVIEW_AT = 0.25;

  const phase = $derived(jobPhase(job));
  const activity = $derived(itemActivity(job));
  // The slot keeps the output's shape from the start, read from the job's meta ("2:3 · m · 8 steps"), so it never jumps.
  const aspect = $derived.by(() => {
    const match = /^(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)/.exec(job.meta ?? '');
    return match ? Number(match[1]) / Number(match[2]) : 1;
  });
  // Paused between images: the image that just finished, sharp. Otherwise the transient live preview.
  const pausedBetween = $derived(isPaused(job) && phase === 'generate' && !!job.itemFinished);
  const lastOutput = $derived(job.outputs[job.outputs.length - 1]);
  let failedUrl = $state<string | null>(null);
  const src = $derived(pausedBetween ? (lastOutput?.thumbnail_url ?? lastOutput?.url ?? null) : job.previewUrl);
  const shown = $derived(src !== null && failedUrl !== src);
  const tag = $derived(pausedBetween ? 'Last finished image' : `Live preview · ${Math.round(stepFraction(job) * 100)}%`);
  // What fills the slot until a preview arrives, by phase.
  const note = $derived.by(() => {
    if (phase === 'queue') return 'Waiting in the queue';
    if (phase === 'enhance') return 'Enhancing the prompt';
    if (phase === 'load') return 'Loading the model';
    if (isVideo(job)) return 'No live preview for video';
    if (activity === 'stepping' && stepFraction(job) > FIRST_PREVIEW_AT + 0.15) return 'No live preview for this model';
    return `First preview at ${FIRST_PREVIEW_AT * 100}%`;
  });
</script>

<!-- The forming image, as large as the stage allows; the job's facts sit in the card beside it. -->
<figure class="job-preview {extraClass}" class:empty={!shown} style:aspect-ratio={aspect} data-testid="job-preview">
  {#if shown && src}
    <img
      {src}
      alt={pausedBetween ? 'The last finished image' : 'Live preview of the generation in progress'}
      onerror={(event) => { failedUrl = event.currentTarget.getAttribute('src'); }}
    >
    <figcaption class="job-preview-tag">{tag}</figcaption>
  {:else}
    <figcaption class="job-preview-note">
      <Badge>{#if isVideo(job)}<Icon name="film" size={12} />{/if}{note}</Badge>
    </figcaption>
  {/if}
</figure>

<style>
  .job-preview { position: relative; margin: 0; max-width: 100%; max-height: 100%; height: 100%; overflow: hidden; border-radius: var(--radius-xs); background: var(--color-bg-surface); }
  .job-preview img { display: block; width: 100%; height: 100%; object-fit: cover; }
  .job-preview.empty { display: grid; place-items: center; box-shadow: inset 0 0 0 1px var(--color-border-subtle); }
  .job-preview-tag { position: absolute; top: 8px; left: 8px; height: 20px; padding: 0 6px; border-radius: var(--radius-xs); background: color-mix(in srgb, var(--color-bg-overlay) 70%, transparent); font-family: var(--font-mono); font-size: 10px; line-height: 20px; color: var(--color-text-primary); }
  .job-preview-note { padding: 8px; text-align: center; }
</style>
