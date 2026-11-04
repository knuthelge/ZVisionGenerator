<script lang="ts">
  import { Icon, Waveform } from '$lib/components/atoms';
  import { activeDownload, isActive, isAudioJob, isPaused, isVideo, jobPhase, livePreviewNote, outputAspect, stepFraction } from '$lib/state/jobView';
  import type { ActiveJobState, AudioWorkflow } from '$lib/types';
  import AudioAurora from './AudioAurora.svelte';
  import { lastTakeCaption } from './audioAurora';
  import { claimPlayback, releasePlayback } from '$lib/state/nowPlaying';

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
  // From a take's first progress event the slot shows the Aurora animation, which lands on the take's waveform when it is ready.
  // Before that, while the job loads or enhances (and before its first take reports), the same animation waits at zero progress.
  const waiting = $derived(audio && phase !== 'queue' && isActive(job) && !job.audio && job.outputs.length === 0);
  // A job that failed or was stopped keeps no live Aurora: the ribbons would hold their last frame for good. A finished job still lets its last take land.
  const stopped = $derived(!isActive(job) && job.status !== 'completed');
  const live = $derived(audio && !stopped && ((phase === 'generate' && !!job.audio) || waiting));
  // The load step is a download first on a model's first use: say so, as the job card does.
  const loadNote = $derived(activeDownload(job) ? 'Downloading the model' : 'Loading the model');
  const waitingNote = $derived(phase === 'enhance' ? 'Enhancing the prompt' : phase === 'load' ? loadNote : 'Starting');
  // The outputs the job had when the current take began: a take that is skipped or fails adds none, and has no waveform to land on.
  let outputsBefore = $state(0);
  $effect.pre(() => {
    if (!job.itemFinished) outputsBefore = job.outputs.length;
  });
  const readyOutput = $derived(live && job.itemFinished && job.outputs.length > outputsBefore ? lastOutput : undefined);
  // Between takes, paused after one, and once the job failed or was stopped, the slot keeps the waveform of the last take saved; otherwise there is nothing to preview.
  const holdsLast = $derived(audio && phase === 'generate' && (pausedBetween || stopped || (!job.audio && job.outputs.length > 0)));
  const lastPeaks = $derived(holdsLast ? (lastOutput?.peaks ?? []) : []);
  const src = $derived(pausedBetween ? (lastOutput?.thumbnail_url ?? lastOutput?.url ?? null) : job.previewUrl);
  const shown = $derived(!audio && src !== null && failedUrl !== src);
  const wave = $derived(lastPeaks.length > 0);
  // The held take can be played from the slot: a job that waits or ended leaves it no other player on a phone.
  // It is the waveform the slot shows, whether the Aurora has landed on it (a take just finished, or the job is paused) or the slot holds it still.
  const heldUrl = $derived(wave ? (lastOutput?.url ?? null) : readyOutput && (readyOutput.peaks?.length ?? 0) > 0 ? readyOutput.url : null);
  let held = $state<HTMLAudioElement | null>(null);
  let heldPlaying = $state(false);
  $effect(() => {
    const element = held;
    void heldUrl;
    heldPlaying = false;
    return () => {
      if (!element) return;
      element.pause();
      releasePlayback(element);
    };
  });
  function toggleHeld(): void {
    if (!held) return;
    if (held.paused) void held.play().catch(() => { heldPlaying = false; });
    else held.pause();
  }
  const tag = $derived(pausedBetween ? 'Last finished image' : `Live preview · ${Math.round(stepFraction(job) * 100)}%`);
  // What fills the slot until a preview arrives, by phase.
  const note = $derived.by(() => {
    if (phase === 'queue') return 'Waiting in the queue';
    if (phase === 'enhance') return 'Enhancing the prompt';
    if (phase === 'load') return loadNote;
    if (audio) return stopped ? 'No take saved' : 'Waiting for the take';
    if (isVideo(job)) return 'No live preview for video';
    return livePreviewNote(job);
  });
</script>

<!-- The held take's player: the audio element and the button that plays it. -->
{#snippet heldPlayer()}
  {#if heldUrl}
    <!-- svelte-ignore a11y_media_has_caption because a take has no captions to show -->
    <audio
      bind:this={held}
      src={heldUrl}
      preload="none"
      onplay={() => { heldPlaying = true; if (held) claimPlayback(held); }}
      onpause={() => { heldPlaying = false; }}
      onended={() => { heldPlaying = false; }}
    ></audio>
    <button type="button" class="job-preview-play" aria-label={heldPlaying ? 'Pause the last take' : 'Play the last take'} aria-pressed={heldPlaying} onclick={toggleHeld}>
      <Icon name={heldPlaying ? 'pause' : 'play'} size={16} />
    </button>
  {/if}
{/snippet}

<!-- The forming image, as large as the stage allows; the job's facts sit in the card beside it. -->
<figure class="job-preview {extraClass}" class:empty={!shown && !wave && !live} style:aspect-ratio={aspect} style:--ar={aspect} data-testid="job-preview">
  {#if shown && src}
    <img
      {src}
      alt={pausedBetween ? 'The last finished image' : 'Live preview of the generation in progress'}
      class:ui-checkerboard={transparent}
      data-transparent={transparent}
      onerror={(event) => { failedUrl = event.currentTarget.getAttribute('src'); }}
    >
    <figcaption class="media-tag job-preview-tag">{tag}</figcaption>
  {:else if live}
    <AudioAurora
      kind={job.workflow as AudioWorkflow}
      progress={job.audio ?? null}
      {waitingNote}
      peaks={readyOutput ? (readyOutput.peaks ?? []) : null}
      seconds={readyOutput?.duration ?? null}
      idle={isPaused(job) || !isActive(job) || (!!job.itemFinished && !readyOutput)}
    />
    {@render heldPlayer()}
  {:else if wave}
    <span class="job-preview-wave"><Waveform peaks={lastPeaks} /></span>
    {@render heldPlayer()}
    <figcaption class="media-tag job-preview-tag">{lastTakeCaption(lastOutput?.duration)}</figcaption>
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
  .job-preview-wave { position: absolute; inset: 14% 6%; display: block; }
  .job-preview-play { position: absolute; right: 8px; bottom: 8px; display: grid; place-items: center; width: 36px; height: 36px; padding: 0; border: 0; border-radius: 50%; cursor: pointer; color: var(--color-text-primary); background: color-mix(in srgb, var(--color-bg-surface) 80%, transparent); box-shadow: 0 0 0 1px var(--color-border-subtle); }
  @media (hover: hover) { .job-preview-play:hover { background: var(--color-bg-surface); } }
  .job-preview-note { display: inline-flex; align-items: center; gap: 6px; padding: 8px 16px; text-align: center; font-size: var(--text-content); font-weight: 600; color: var(--color-text-secondary); }
</style>
