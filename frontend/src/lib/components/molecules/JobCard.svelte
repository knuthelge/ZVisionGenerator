<script lang="ts">
  import type { ActiveJobState } from '$lib/types';
  import ProgressBar from './ProgressBar.svelte';

  interface Props {
    job: ActiveJobState;
    oncancel?: (jobId: string) => void;
    onpause?: (jobId: string) => void;
    onresume?: (jobId: string) => void;
    onnext?: (jobId: string) => void;
    onrepeat?: (jobId: string) => void;
  }

  let {
    job,
    oncancel,
    onpause,
    onresume,
    onnext,
    onrepeat
  }: Props = $props();

  const stagePct = $derived(job.totalSteps > 0 ? 0 : 0); // Stage-level not tracked separately; use step for both
  const stepPct  = $derived(job.totalSteps > 0 ? job.currentStep / job.totalSteps : 0);

  function formatElapsed(secs: number): string {
    const m = Math.floor(secs / 60);
    const s = Math.floor(secs % 60);
    return `${m}:${String(s).padStart(2, '0')}`;
  }

  function formatDuration(secs: number): string {
    if (!Number.isFinite(secs) || secs < 0) return '--:--';
    const rounded = Math.round(secs);
    const h = Math.floor(rounded / 3600);
    const m = Math.floor((rounded % 3600) / 60);
    const s = rounded % 60;
    if (h > 0) return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
    return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
  }

  const elapsedStr = $derived(formatElapsed(job.elapsed));
  const remainingStr = $derived(job.remaining > 0 ? formatDuration(job.remaining) : '--:--');
  const stepLabel  = $derived(`${job.currentStep} / ${job.totalSteps}`);
  const batchMeta  = $derived(`${job.batchIndex + 1} / ${job.runs}`);
  const stepPhase  = $derived(
    job.stageName
      ? `Stage: ${job.stageName}`
      : 'Waiting for first denoiser step...'
  );

  const progressState = $derived(
    job.status === 'completed' ? 'completed' :
    job.status === 'failed'    ? 'failed'    :
                                 'running'
  );

  const jobTypeLabel = $derived(job.workflow.toUpperCase());
  const metaLine = $derived(`${job.model} · ${job.runs} run${job.runs !== 1 ? 's' : ''}`);
</script>

<article class="rounded-xl border border-zinc-800 bg-zinc-950/80 p-4 shadow-lg">
  <!-- Header -->
  <div class="flex items-start justify-between gap-4">
    <div class="min-w-0">
      <p class="text-xs font-semibold uppercase tracking-[0.24em] text-teal-300/80">{jobTypeLabel}</p>
      <p class="mt-2 text-sm text-zinc-400 line-clamp-2">{job.prompt}</p>
      <p class="mt-1 text-xs font-mono text-zinc-500">{metaLine}</p>
    </div>
    <!-- Cancel button -->
    {#if job.status === 'running' || job.status === 'pending'}
      <button
        type="button"
        onclick={() => oncancel?.(job.job_id)}
        class="shrink-0 rounded-md border border-zinc-700 px-2.5 py-1.5 text-xs font-medium text-zinc-400 hover:border-red-500 hover:text-red-400 transition-colors focus-visible:focus-ring"
        aria-label="Cancel job"
      >
        Cancel
      </button>
    {/if}
  </div>

  <!-- Progress bars -->
  <div class="mt-4">
    <ProgressBar
      stagePct={stepPct}
      stageLabel={job.stageName || 'Workflow'}
      stepPct={stepPct}
      {stepPhase}
      {stepLabel}
      batchLabel={job.batchLabel || 'Waiting for batch context...'}
      {batchMeta}
      elapsed={elapsedStr}
      remaining={remainingStr}
      state={progressState}
    />
  </div>

  <!-- Job controls (pause/resume/next/repeat) -->
  {#if job.status === 'running'}
    <div class="mt-3 flex flex-wrap gap-2">
      {#if job.paused}
        <button
          type="button"
          onclick={() => onresume?.(job.job_id)}
          class="rounded-md border border-zinc-700 px-3 py-1.5 text-xs font-medium text-zinc-300 transition hover:border-teal-500 hover:text-zinc-100 focus-visible:focus-ring"
        >
          Resume
        </button>
      {:else}
        <button
          type="button"
          onclick={() => onpause?.(job.job_id)}
          class="rounded-md border border-zinc-700 px-3 py-1.5 text-xs font-medium text-zinc-300 transition hover:border-teal-500 hover:text-zinc-100 focus-visible:focus-ring"
        >
          Pause
        </button>
      {/if}
      <button
        type="button"
        onclick={() => onnext?.(job.job_id)}
        class="rounded-md border border-zinc-700 px-3 py-1.5 text-xs font-medium text-zinc-300 transition hover:border-teal-500 hover:text-zinc-100 focus-visible:focus-ring"
      >
        Next
      </button>
      {#if !job.supported_controls || job.supported_controls.includes('repeat')}
        <button
          type="button"
          onclick={() => onrepeat?.(job.job_id)}
          class="rounded-md border border-zinc-700 px-3 py-1.5 text-xs font-medium text-zinc-300 transition hover:border-teal-500 hover:text-zinc-100 focus-visible:focus-ring"
        >
          Repeat
        </button>
      {/if}
    </div>
  {/if}

  <!-- Status line -->
  <div class="mt-3 flex items-center justify-between gap-4 text-sm">
    <span class="text-zinc-200 capitalize">{job.status}</span>
    <span class="font-mono text-xs text-zinc-500">{job.job_id}</span>
  </div>
  {#if job.message}
    <p class="mt-2 text-sm text-zinc-400">{job.message}</p>
  {/if}

  <!-- Output previews (on completion) -->
  {#if job.outputs.length > 0}
    <div class="mt-3 grid grid-cols-3 gap-2">
      {#each job.outputs as output (output.path)}
        <a href={output.url} target="_blank" rel="noopener noreferrer" class="block">
          {#if output.media_type === 'video'}
            <video
              src={output.thumbnail_url || output.url}
              class="w-full aspect-square object-cover rounded-md border border-zinc-800"
              muted
              preload="none"
            ></video>
          {:else}
            <img
              src={output.thumbnail_url || output.url}
              alt={output.filename}
              class="w-full aspect-square object-cover rounded-md border border-zinc-800"
              loading="lazy"
            />
          {/if}
        </a>
      {/each}
    </div>
  {/if}
</article>
