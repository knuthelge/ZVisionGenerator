<script lang="ts">
  import { Icon, Waveform } from '$lib/components/atoms';
  import { isAudioJob, isPaused, isVideo, jobPhase, livePreviewNote, outputAspect, stepFraction } from '$lib/state/jobView';
  import type { ActiveJobState } from '$lib/types';

  interface Props {
    job: ActiveJobState;
    class?: string;
  }

  let { job, class: extraClass = '' }: Props = $props();

  const AUDIO_SLOT_ASPECT = 4;

  const phase = $derived(jobPhase(job));
  // The slot keeps the output's shape from the start, so it never jumps.
  // A take has no picture: its slot is a 4:1 strip.
  const audio = $derived(isAudioJob(job));
  const aspect = $derived(audio ? AUDIO_SLOT_ASPECT : outputAspect(job));
  // Paused between images: the image that just finished, sharp. Otherwise the transient live preview.
  const pausedBetween = $derived(isPaused(job) && phase === 'generate' && !!job.itemFinished);
  const lastOutput = $derived(job.outputs[job.outputs.length - 1]);
  // A finished cut-out sits on a checkerboard; the live preview is the opaque render and never does.
  const transparent = $derived(pausedBetween && lastOutput?.transparent === true);
  let failedUrl = $state<string | null>(null);
  // Paused between takes the slot shows the waveform of the take that just finished; otherwise there is nothing to preview.
  const lastPeaks = $derived(audio && pausedBetween ? (lastOutput?.peaks ?? []) : []);
  const src = $derived(pausedBetween ? (lastOutput?.thumbnail_url ?? lastOutput?.url ?? null) : job.previewUrl);
  const shown = $derived(!audio && src !== null && failedUrl !== src);
  const wave = $derived(lastPeaks.length > 0);
  const tag = $derived(pausedBetween ? 'Last finished image' : `Live preview · ${Math.round(stepFraction(job) * 100)}%`);
  // What fills the slot until a preview arrives, by phase.
  const note = $derived.by(() => {
    if (phase === 'queue') return 'Waiting in the queue';
    if (phase === 'enhance') return 'Enhancing the prompt';
    if (phase === 'load') return 'Loading the model';
    if (audio) return 'No live preview for audio';
    if (isVideo(job)) return 'No live preview for video';
    return livePreviewNote(job);
  });
</script>

<!-- The forming image, as large as the stage allows; the job's facts sit in the card beside it. -->
<figure class="job-preview {extraClass}" class:empty={!shown && !wave} style:aspect-ratio={aspect} style:--ar={aspect} data-testid="job-preview">
  {#if shown && src}
    <img
      {src}
      alt={pausedBetween ? 'The last finished image' : 'Live preview of the generation in progress'}
      class:ui-checkerboard={transparent}
      data-transparent={transparent}
      onerror={(event) => { failedUrl = event.currentTarget.getAttribute('src'); }}
    >
    <figcaption class="media-tag job-preview-tag">{tag}</figcaption>
  {:else if wave}
    <span class="job-preview-wave"><Waveform peaks={lastPeaks} /></span>
    <figcaption class="media-tag job-preview-tag">Last finished take</figcaption>
  {:else}
    <figcaption class="job-preview-note">{#if audio}<Icon name="waveform" size={14} />{:else if isVideo(job)}<Icon name="film" size={14} />{/if}{note}</figcaption>
  {/if}
</figure>

<style>
  /* As large as the stage allows at the output's shape: the width that fits both ways in the size container around it
     (a portrait phone fits a landscape job by width, a desktop stage a portrait one by height). */
  .job-preview { position: relative; flex: none; margin: 0; width: min(100cqw, 100cqh * var(--ar, 1)); max-width: 100%; max-height: 100%; overflow: hidden; border-radius: var(--radius-xs); background: var(--color-bg-surface); }
  .job-preview img { display: block; width: 100%; height: 100%; object-fit: cover; }
  /* Until a preview arrives, a slow sheen crosses the output's shape, so the slot reads as working rather than empty. */
  .job-preview.empty { display: grid; place-items: center; background: linear-gradient(105deg, transparent 30%, color-mix(in srgb, var(--color-text-primary) 5%, transparent) 50%, transparent 70%) var(--color-bg-surface) no-repeat; background-size: 250% 100%; animation: job-preview-sheen 2.4s ease-in-out infinite; }
  @keyframes job-preview-sheen { from { background-position: 100% 0; } to { background-position: -150% 0; } }
  @media (prefers-reduced-motion: reduce) { .job-preview.empty { animation: none; } }
  .job-preview-tag { position: absolute; top: 8px; left: 8px; }
  .job-preview-wave { position: absolute; inset: 0; display: block; padding: 14% 6%; }
  .job-preview-note { display: inline-flex; align-items: center; gap: 6px; padding: 8px 16px; text-align: center; font-size: var(--text-content); font-weight: 600; color: var(--color-text-secondary); }
</style>
