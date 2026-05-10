<script lang="ts">
  import type { ActiveJobState } from '$lib/types';

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

  const stepPct  = $derived(job.totalSteps > 0 ? job.currentStep / job.totalSteps : 0);
  const supportedControls = $derived(new Set(job.supported_controls ?? []));
  const active = $derived(job.status === 'queued' || job.status === 'running' || job.status === 'paused');
  const canCancel = $derived(active && (supportedControls.has('quit') || supportedControls.has('cancel')));
  const canPause = $derived(job.status === 'running' && supportedControls.has('pause'));
  const canResume = $derived((job.status === 'paused' || job.paused) && supportedControls.has('resume'));
  const canNext = $derived(job.status === 'running' && supportedControls.has('next'));
  const canRepeat = $derived(job.status === 'running' && supportedControls.has('repeat'));
  const hasInlineControls = $derived(canPause || canResume || canNext || canRepeat);

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
  const stepWidth = $derived(`${Math.min(100, Math.max(0, stepPct * 100))}%`);

  const progressState = $derived(
    job.status === 'completed' ? 'completed' :
    job.status === 'failed' || job.status === 'cancelled' ? 'failed' :
                                 'running'
  );
  const progressFill = $derived(
    progressState === 'completed' ? 'bg-emerald-400' :
    progressState === 'failed' ? 'bg-red-400' :
    'bg-teal-400'
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
    {#if canCancel}
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

  <!-- Progress -->
  <div class="mt-4">
    <div>
      <div class="flex items-center justify-between gap-3 text-[11px] uppercase tracking-[0.18em] text-zinc-500">
        <span>{job.batchLabel || 'Waiting for batch context...'}</span>
        <span class="font-mono">{batchMeta}</span>
      </div>

      <div class="mt-3 h-1.5 overflow-hidden rounded-full bg-zinc-900">
        <div
          class="h-full rounded-full {progressFill} transition-all duration-300"
          style="width: {stepWidth}"
        ></div>
      </div>
      <div class="mt-2 flex items-center justify-between gap-4 text-[11px] text-zinc-500">
        <span>{stepPhase}</span>
        <span class="font-mono">{stepLabel}</span>
      </div>

      <div class="mt-3 grid grid-cols-2 gap-3 text-[11px] text-zinc-500">
        <div class="rounded-md border border-zinc-800 bg-zinc-900/70 px-3 py-2">
          <span class="block uppercase tracking-[0.18em] text-zinc-600">Elapsed</span>
          <span class="mt-1 block font-mono text-zinc-300">{elapsedStr}</span>
        </div>
        <div class="rounded-md border border-zinc-800 bg-zinc-900/70 px-3 py-2">
          <span class="block uppercase tracking-[0.18em] text-zinc-600">Remaining</span>
          <span class="mt-1 block font-mono text-zinc-300">{remainingStr}</span>
        </div>
      </div>
    </div>
  </div>

  <!-- Job controls (pause/resume/next/repeat) -->
  {#if hasInlineControls}
    <div class="mt-3 flex flex-wrap gap-2">
      {#if canResume}
        <button
          type="button"
          onclick={() => onresume?.(job.job_id)}
          class="rounded-md border border-zinc-700 px-3 py-1.5 text-xs font-medium text-zinc-300 transition hover:border-teal-500 hover:text-zinc-100 focus-visible:focus-ring"
        >
          Resume
        </button>
      {/if}
      {#if canPause}
        <button
          type="button"
          onclick={() => onpause?.(job.job_id)}
          class="rounded-md border border-zinc-700 px-3 py-1.5 text-xs font-medium text-zinc-300 transition hover:border-teal-500 hover:text-zinc-100 focus-visible:focus-ring"
        >
          Pause
        </button>
      {/if}
      {#if canNext}
        <button
          type="button"
          onclick={() => onnext?.(job.job_id)}
          class="rounded-md border border-zinc-700 px-3 py-1.5 text-xs font-medium text-zinc-300 transition hover:border-teal-500 hover:text-zinc-100 focus-visible:focus-ring"
        >
          Next
        </button>
      {/if}
      {#if canRepeat}
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
      {#each job.outputs as output (output.id)}
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
