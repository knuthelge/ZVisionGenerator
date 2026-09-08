<script lang="ts">
  import type { ActiveJobState } from '$lib/types';

  interface Props {
    job: ActiveJobState;
    oncancel?: (jobId: string) => void | Promise<unknown>;
    onpause?: (jobId: string) => void | Promise<unknown>;
    onresume?: (jobId: string) => void | Promise<unknown>;
    onnext?: (jobId: string) => void | Promise<unknown>;
    onrepeat?: (jobId: string) => void | Promise<unknown>;
  }

  let {
    job,
    oncancel,
    onpause,
    onresume,
    onnext,
    onrepeat
  }: Props = $props();

  type Control = 'Pause' | 'Resume' | 'Next' | 'Repeat' | 'Cancel';
  let feedback = $state<{ jobId: string; action: Control; state: 'pending' | 'accepted' | 'failed'; message: string } | null>(null);
  const currentFeedback = $derived(feedback?.jobId === job.job_id ? feedback : null);
  const pending = $derived(currentFeedback?.state === 'pending');

  async function sendControl(action: Control, callback: Props['oncancel']): Promise<void> {
    if (pending || !callback) return;
    const jobId = job.job_id;
    feedback = { jobId, action, state: 'pending', message: `Sending ${action.toLowerCase()} request…` };
    try {
      const result = callback(jobId);
      if (result) await result;
      feedback = { jobId, action, state: 'accepted', message: `${action} request accepted. Changes may take a moment.` };
    } catch (error) {
      feedback = { jobId, action, state: 'failed', message: `${action} failed. ${error instanceof Error ? error.message : 'Please try again.'}` };
    }
  }

  const readableStage = $derived(job.stageName.replaceAll('_', ' '));
  const displayMessage = $derived(job.message === `Running ${readableStage}.` ? '' : job.message);

  const hasProgress = $derived(Number.isFinite(job.totalSteps) && job.totalSteps > 0 && Number.isFinite(job.currentStep));
  const stepPct = $derived(hasProgress ? Math.min(100, Math.max(0, job.currentStep / job.totalSteps * 100)) : 0);
  const supportedControls = $derived(new Set(job.supported_controls ?? []));
  const active = $derived(job.status === 'queued' || job.status === 'running' || job.status === 'paused');
  const canCancel = $derived(active && (supportedControls.has('quit') || supportedControls.has('cancel')));
  const canPause = $derived(job.status === 'running' && !job.paused && supportedControls.has('pause'));
  const canResume = $derived(active && (job.status === 'paused' || job.paused) && supportedControls.has('resume'));
  const canNext = $derived(job.status === 'running' && supportedControls.has('next'));
  const canRepeat = $derived(job.status === 'running' && supportedControls.has('repeat'));
  const hasInlineControls = $derived(canPause || canResume || canNext || canRepeat);

  function formatElapsed(secs: number): string {
    if (!Number.isFinite(secs) || secs < 0) return '--:--';
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
  const runCount = $derived(Number.isFinite(job.runs) ? Math.max(0, Math.floor(job.runs)) : 0);
  const currentRun = $derived(Math.min(Math.max(0, Number.isFinite(job.batchIndex) ? Math.floor(job.batchIndex) : 0), Math.max(0, runCount - 1)));
  const batchStart = $derived(Math.floor(currentRun / 16) * 16);
  const visibleRuns = $derived(Array.from({ length: Math.min(16, Math.max(0, runCount - batchStart)) }, (_, index) => batchStart + index));

  function runState(index: number): string {
    if (job.status === 'completed') return 'Completed';
    if (job.status === 'queued' || job.status === 'pending') return 'Waiting';
    // Advancing can also mean skipping a run; do not imply successful output.
    if (index < currentRun) return 'Previous';
    if (index > currentRun) return 'Waiting';
    if (job.status === 'failed') return 'Failed';
    if (job.status === 'cancelled') return 'Stopped';
    return job.status === 'paused' || job.paused ? 'Paused' : 'Current';
  }
  const stepPhase  = $derived(
    job.stageName
      ? readableStage.charAt(0).toUpperCase() + readableStage.slice(1)
      : job.status === 'queued' || job.status === 'pending' ? 'Waiting to start' : active ? 'Preparing generation' : 'Generation finished'
  );
  const stepWidth = $derived(`${stepPct}%`);

  const progressState = $derived(
    job.status === 'completed' ? 'completed' :
    job.status === 'failed' ? 'failed' :
    job.status === 'cancelled' ? 'cancelled' :
                                 'running'
  );
  const progressFill = $derived(
    progressState === 'completed' ? 'bg-emerald-400' :
    progressState === 'failed' ? 'bg-red-400' :
    progressState === 'cancelled' ? 'bg-zinc-500' : 'bg-primary-main'
  );

  const workflowLabels = { txt2img: 'Text to image', img2img: 'Image to image', txt2vid: 'Text to video', img2vid: 'Image to video' };
  const jobTypeLabel = $derived(workflowLabels[job.workflow] ?? job.workflow);
  const statusLabel = $derived(active && job.paused ? 'paused' : job.status);
</script>

<article class="job-card">
  <!-- Header -->
  <div class="job-header">
    <h3>{jobTypeLabel}</h3>
    <span class="job-status" data-status={statusLabel}><span class="status-dot"></span>{statusLabel}</span>
  </div>
  <div class="job-body">
  <p class="text-sm text-text-primary line-clamp-2 break-words" title={job.prompt}>{job.prompt || 'No prompt supplied'}</p>
  <p class="mt-1 text-xs text-text-muted truncate" title={job.model}>{job.model}</p>

  {#if runCount > 1}
    <div class="batch-info">
      <div class="batch-heading"><span>Batch</span><span class="font-mono">{job.status === 'completed' ? `${runCount} / ${runCount}` : batchMeta}</span></div>
      <ol class="batch-runs" aria-label="Batch runs">
        {#each visibleRuns as index (index)}
          {@const state = runState(index)}
          <li data-state={state} aria-current={index === currentRun && active ? 'step' : undefined} aria-label="Run {index + 1}: {state}" title="Run {index + 1}: {state}">
            <span aria-hidden="true">{state === 'Completed' ? '✓' : index + 1}</span>
          </li>
        {/each}
      </ol>
      <div class="batch-caption">
        <span>{job.status === 'completed' ? 'Batch complete' : `Run ${currentRun + 1} · ${runState(currentRun)}`}</span>
        {#if runCount > 16}<span>Showing {batchStart + 1}–{batchStart + visibleRuns.length}</span>{/if}
      </div>
    </div>
  {/if}

  <!-- Progress -->
  <div class="mt-4">
      <div class="flex items-center justify-between gap-3 text-xs">
        <span class="text-text-secondary break-words min-w-0">{stepPhase}</span>
        <span class="font-mono text-text-primary shrink-0">{hasProgress ? `${Math.round(stepPct)}%` : '—'}</span>
      </div>

      <div
        class="progress-track mt-2 h-1.5 overflow-hidden rounded-full bg-zinc-800"
        class:step-pulse={job.status === 'running' && !job.paused && hasProgress && job.currentStep < job.totalSteps}
        role="progressbar"
        aria-label="Generation stage progress"
        aria-valuemin="0"
        aria-valuemax="100"
        aria-valuenow={hasProgress ? Math.round(stepPct) : undefined}
        aria-valuetext={hasProgress ? `${stepLabel} steps` : stepPhase}
      >
        <div
          class="h-full rounded-full {progressFill} transition-all duration-300"
          style="width: {stepWidth}"
        ></div>
      </div>
      <div class="mt-2 flex items-center justify-between gap-3 text-[11px] text-text-muted">
        <span class="min-w-0 break-words">{job.batchLabel || (runCount > 1 ? 'Current run' : 'Single run')}</span>
        <span class="font-mono shrink-0">{hasProgress ? `${stepLabel} steps` : 'Awaiting steps'}</span>
      </div>

      <dl class="job-timing">
        <div><dt>Elapsed</dt><dd>{elapsedStr}</dd></div>
        <div><dt>Remaining</dt><dd>{remainingStr}</dd></div>
      </dl>
  </div>
  {#if displayMessage}
    <p class="job-message" class:failed={job.status === 'failed'} role="status">{displayMessage}</p>
  {/if}

  <!-- Job controls (pause/resume/next/repeat) -->
  {#if hasInlineControls || canCancel}
    <div class="job-actions">
      {#if canResume}
        <button
          type="button"
          onclick={() => sendControl('Resume', onresume)}
          disabled={pending || !onresume}
          class="rounded-md border border-zinc-700 px-3 py-1.5 text-xs font-medium text-zinc-300 transition hover:border-teal-500 hover:text-zinc-100 focus-visible:focus-ring"
        >
          {pending && currentFeedback?.action === 'Resume' ? 'Sending…' : 'Resume'}
        </button>
      {/if}
      {#if canPause}
        <button
          type="button"
          onclick={() => sendControl('Pause', onpause)}
          disabled={pending || !onpause}
          class="rounded-md border border-zinc-700 px-3 py-1.5 text-xs font-medium text-zinc-300 transition hover:border-teal-500 hover:text-zinc-100 focus-visible:focus-ring"
        >
          {pending && currentFeedback?.action === 'Pause' ? 'Sending…' : 'Pause'}
        </button>
      {/if}
      {#if canNext}
        <button
          type="button"
          onclick={() => sendControl('Next', onnext)}
          disabled={pending || !onnext}
          class="rounded-md border border-zinc-700 px-3 py-1.5 text-xs font-medium text-zinc-300 transition hover:border-teal-500 hover:text-zinc-100 focus-visible:focus-ring"
        >
          {pending && currentFeedback?.action === 'Next' ? 'Sending…' : 'Next'}
        </button>
      {/if}
      {#if canRepeat}
        <button
          type="button"
          onclick={() => sendControl('Repeat', onrepeat)}
          disabled={pending || !onrepeat}
          class="rounded-md border border-zinc-700 px-3 py-1.5 text-xs font-medium text-zinc-300 transition hover:border-teal-500 hover:text-zinc-100 focus-visible:focus-ring"
        >
          {pending && currentFeedback?.action === 'Repeat' ? 'Sending…' : 'Repeat'}
        </button>
      {/if}
      {#if canCancel}
        <button type="button" onclick={() => sendControl('Cancel', oncancel)} disabled={pending || !oncancel} class="cancel-button" aria-label="Cancel job">{pending && currentFeedback?.action === 'Cancel' ? 'Sending…' : 'Cancel'}</button>
      {/if}
    </div>
  {/if}

  <div role="status" aria-live="polite" aria-atomic="true">
    {#if currentFeedback}
      <p class="control-feedback" class:failed={currentFeedback.state === 'failed'}>{currentFeedback.message}</p>
    {/if}
  </div>

  <!-- Output previews (on completion) -->
  {#if job.outputs.length > 0}
    <div class="mt-3 grid grid-cols-3 gap-2">
      {#each job.outputs as output (output.id)}
        <a href={output.url} target="_blank" rel="noopener noreferrer" class="block" aria-label="Open {output.filename}">
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
  </div>
  <footer class="job-footer"><span>Job</span><span class="truncate font-mono" title={job.job_id}>{job.job_id}</span></footer>
</article>

<style>
  .progress-track { position: relative; isolation: isolate; }
  .step-pulse::after {
    content: '';
    position: absolute;
    top: 0;
    bottom: 0;
    left: 0;
    width: 35%;
    pointer-events: none;
    background: linear-gradient(90deg, transparent, var(--color-primary-main) 50%, transparent);
    opacity: 0.3;
    transform: translateX(386%);
    animation: step-sweep 1.6s linear infinite;
  }
  @keyframes step-sweep {
    0% { transform: translateX(386%); }
    100% { transform: translateX(-100%); }
  }
  @media (prefers-reduced-motion: reduce) {
    .step-pulse::after { animation: none; display: none; }
  }
  .job-card { width: 100%; overflow: hidden; border: 1px solid var(--color-border-strong); border-radius: 8px; background: var(--color-bg-surface); }
  .job-header { display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 10px 14px; border-bottom: 1px solid var(--color-border-subtle); }
  h3 { font-size: 12px; font-weight: 600; }
  .job-status { display: flex; align-items: center; gap: 6px; font-size: 11px; text-transform: capitalize; color: var(--color-text-secondary); }
  .status-dot { width: 6px; height: 6px; border-radius: 50%; background: currentColor; }
  .job-status[data-status='running'], .job-status[data-status='completed'] { color: var(--color-primary-main); }
  .job-status[data-status='paused'] { color: var(--color-warning); }
  .job-status[data-status='failed'], .job-message.failed { color: var(--color-error); }
  .job-body { padding: 12px 14px; }
  .batch-info { margin-top: 14px; }
  .batch-heading, .batch-caption { display: flex; justify-content: space-between; gap: 8px; font-size: 11px; color: var(--color-text-muted); }
  .batch-runs { display: flex; gap: 3px; margin: 6px 0; padding: 0; list-style: none; }
  .batch-runs li { flex: 1; min-width: 0; height: 22px; display: flex; align-items: center; justify-content: center; border-radius: 3px; border: 1px solid var(--color-border-strong); background: var(--color-bg-base); color: var(--color-text-muted); font-size: 9px; font-variant-numeric: tabular-nums; }
  .batch-runs li[data-state='Previous'], .batch-runs li[data-state='Completed'] { background: var(--color-primary-subtle); border-color: var(--color-primary-subtle); color: var(--color-text-primary); }
  .batch-runs li[data-state='Current'] { border-color: var(--color-primary-main); color: var(--color-primary-main); box-shadow: inset 0 -2px var(--color-primary-main); font-weight: 600; }
  .batch-runs li[data-state='Paused'] { border-color: var(--color-warning); color: var(--color-warning); }
  .batch-runs li[data-state='Failed'] { border-color: var(--color-error); color: var(--color-error); }
  .batch-runs li[data-state='Stopped'] { border-style: dashed; }
  .job-timing { display: flex; flex-wrap: wrap; justify-content: space-between; gap: 8px 16px; margin-top: 14px; font-size: 11px; }
  .job-timing div { display: flex; align-items: baseline; gap: 8px; }
  dt { color: var(--color-text-muted); }
  dd { font-family: var(--font-mono); font-variant-numeric: tabular-nums; color: var(--color-text-primary); }
  .job-message { margin-top: 12px; padding: 8px 10px; background: var(--color-bg-base); border-radius: 4px; font-size: 12px; color: var(--color-text-secondary); overflow-wrap: anywhere; }
  .job-actions { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 12px; padding-top: 12px; border-top: 1px solid var(--color-border-subtle); }
  .job-actions button { padding: 5px 9px; font-size: 11px; border: 1px solid var(--color-border-strong); border-radius: 4px; color: var(--color-text-secondary); }
  .job-actions button:hover { background: var(--color-bg-surface-hover); color: var(--color-text-primary); }
  .job-actions .cancel-button { margin-left: auto; }
  .job-actions button:disabled { opacity: 0.5; cursor: not-allowed; }
  .control-feedback { margin-top: 8px; font-size: 12px; color: var(--color-primary-main); overflow-wrap: anywhere; }
  .control-feedback.failed { color: var(--color-error); }
  .job-actions .cancel-button:hover { color: var(--color-error); border-color: var(--color-error); }
  .job-footer { display: flex; gap: 8px; padding: 7px 14px; font-size: 10px; color: var(--color-text-muted); background: var(--color-bg-base); border-top: 1px solid var(--color-border-subtle); }
</style>
