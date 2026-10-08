<script lang="ts">
  import { untrack } from 'svelte';
  import { acceptsPageShortcut } from '$lib/keyboard';
  import { workflowLabel } from '$lib/state/assetActions';
  import { trackStepTiming, type StepTiming } from '$lib/state/stepTiming';
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
  const displayMessage = $derived(
    job.message === `Running ${readableStage}.` || job.message === 'Preparing generation.' || job.message.startsWith('Batch completed') ? '' : job.message
  );

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

  // The control each key sends right now, or null when the job does not offer it.
  function controlForKey(key: string): [Control, Props['oncancel']] | null {
    switch (key.toLowerCase()) {
      case 'escape': return canCancel ? ['Cancel', oncancel] : null;
      case 'p': return canResume ? ['Resume', onresume] : canPause ? ['Pause', onpause] : null;
      case 'n': return canNext ? ['Next', onnext] : null;
      case 'r': return canRepeat ? ['Repeat', onrepeat] : null;
      default: return null;
    }
  }

  // Esc stops, P pauses or resumes, N skips to the next and R repeats, unless a field, popover or dialog has the key.
  $effect(() => {
    if (!active) return;
    function handleKeydown(event: KeyboardEvent): void {
      if (!acceptsPageShortcut(event)) return;
      const control = controlForKey(event.key);
      if (!control?.[1]) return;
      event.preventDefault();
      void sendControl(...control);
    }
    document.addEventListener('keydown', handleKeydown);
    return () => document.removeEventListener('keydown', handleKeydown);
  });

  // Live previews are transient: shown only while the current generation runs, never stored.
  let failedPreviewUrl = $state<string | null>(null);
  const showPreview = $derived(active && !!job.previewUrl && failedPreviewUrl !== job.previewUrl);

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

  // Count elapsed up every second between server updates, never backwards; it holds still while paused.
  let elapsedBase = $state({ jobId: '', elapsed: 0, at: 0 });
  let clock = $state(0);
  $effect(() => {
    const { job_id: jobId, elapsed } = job;
    const now = performance.now();
    const previous = untrack(() => elapsedBase);
    const counted = previous.jobId === jobId ? previous.elapsed + (now - previous.at) / 1000 : 0;
    elapsedBase = { jobId, elapsed: Math.max(elapsed, counted), at: now };
    clock = now;
  });
  const ticking = $derived(active && !job.paused);
  $effect(() => {
    if (!ticking) return;
    const timer = setInterval(() => { clock = performance.now(); }, 1000);
    return () => clearInterval(timer);
  });
  const liveElapsed = $derived(
    !active ? job.elapsed : elapsedBase.elapsed + (ticking ? Math.max(0, clock - elapsedBase.at) / 1000 : 0)
  );
  const elapsedStr = $derived(formatElapsed(liveElapsed));
  const remainingStr = $derived(job.remaining > 0 ? formatDuration(job.remaining) : '--:--');
  const stepLabel  = $derived(`${job.currentStep} / ${job.totalSteps}`);
  const hasPromptProgress = $derived(Number.isInteger(job.promptNumber) && Number.isInteger(job.promptCount)
    && (job.promptNumber ?? 0) > 0 && (job.promptCount ?? 0) > 1 && job.promptNumber! <= job.promptCount!);
  const runCount = $derived(Number.isFinite(job.runs) ? Math.max(0, Math.floor(job.runs)) : 0);
  const currentRun = $derived(Math.min(Math.max(0, Number.isFinite(job.batchIndex) ? Math.floor(job.batchIndex) : 0), Math.max(0, runCount - 1)));
  const promptsPerRun = $derived(hasPromptProgress ? job.promptCount! : 1);
  const sequenceCount = $derived(Math.max(1, runCount) * promptsPerRun);
  const sequenceIndex = $derived(currentRun * promptsPerRun + (hasPromptProgress ? job.promptNumber! - 1 : 0));
  const sequenceStart = $derived(Math.max(0, Math.min(sequenceIndex - 11, sequenceCount - 24)));
  const sequenceItems = $derived(Array.from({ length: Math.min(24, sequenceCount) }, (_, index) => sequenceStart + index));

  function sequenceState(index: number): string {
    if (job.status === 'queued' || job.status === 'pending') return 'waiting';
    if (job.status === 'completed' || index < sequenceIndex) return 'previous';
    if (index > sequenceIndex) return 'waiting';
    if (job.paused || job.status === 'paused') return 'paused';
    if (job.status === 'failed') return 'failed';
    if (job.status === 'cancelled') return 'stopped';
    if (job.status === 'unknown') return 'unknown';
    return 'current';
  }

  function sequenceLabel(index: number): string {
    const run = Math.floor(index / promptsPerRun) + 1;
    return `Run ${run}${hasPromptProgress ? ` · Prompt ${(index % promptsPerRun) + 1}` : ''}: ${sequenceState(index)}`;
  }
  const stepPhase  = $derived(
    job.stageName
      ? readableStage.charAt(0).toUpperCase() + readableStage.slice(1)
      : job.status === 'queued' || job.status === 'pending' ? 'Waiting to start' : active ? 'Preparing generation' : 'Generation finished'
  );
  const stepWidth = $derived(`${stepPct}%`);
  // current_step counts finished steps, so the step being worked on is the next one.
  const showCurrentStep = $derived(active && hasProgress && job.currentStep >= 0 && job.currentStep < job.totalSteps);
  const currentStepWidth = $derived(`${100 / job.totalSteps}%`);
  // Time each step so the current one can fill in over about as long as the last one took.
  let stepTiming = $state<StepTiming | null>(null);
  $effect(() => {
    const sample = { scope: `${job.job_id}:${job.stageName}:${job.totalSteps}`, step: job.currentStep, paused: job.paused, now: performance.now() };
    stepTiming = trackStepTiming(untrack(() => stepTiming), sample);
  });
  const stepMs = $derived(stepTiming?.stepMs ?? null);

  const progressState = $derived(
    job.status === 'completed' ? 'completed' :
    job.status === 'failed' ? 'failed' :
    job.status === 'cancelled' || job.status === 'unknown' ? 'cancelled' :
                                 'running'
  );
  const progressFill = $derived(
    progressState === 'completed' ? 'bg-emerald-400' :
    progressState === 'failed' ? 'bg-red-400' :
    progressState === 'cancelled' ? 'bg-zinc-500' : 'bg-primary-main'
  );

  const jobTypeLabel = $derived(workflowLabel(job.workflow));
  const statusLabel = $derived(active && job.paused ? 'paused' : job.status);
</script>

<article class="job-card">
  <!-- Header -->
  <div class="job-header">
    <h3>{jobTypeLabel}</h3>
    <span class="job-status" data-status={statusLabel}><span class="status-dot"></span>{statusLabel}</span>
  </div>
  <div class="job-body">
  <div class="job-main">
  {#if runCount > 1 || hasPromptProgress}
    <p class="mb-2 flex flex-wrap items-center gap-x-2 text-ui font-medium text-primary-main" role="status" aria-live="polite" aria-atomic="true">
      <span>Run {currentRun + 1} of {runCount || 1}</span>
      {#if hasPromptProgress}<span class="text-text-secondary">· Prompt {job.promptNumber} of {job.promptCount}</span>{/if}
    </p>
    <div class="sequence-strip" role="list" aria-label="Generation sequence">
      {#if sequenceStart > 0}<span class="sequence-more" aria-hidden="true">…</span>{/if}
      {#each sequenceItems as index (index)}
        <div
          role="listitem"
          class="sequence-segment"
          class:run-boundary={hasPromptProgress && index > sequenceStart && index % promptsPerRun === 0}
          data-state={sequenceState(index)}
          aria-label={sequenceLabel(index)}
          aria-current={index === sequenceIndex && active ? 'step' : undefined}
          title={sequenceLabel(index)}
        ></div>
      {/each}
      {#if sequenceStart + sequenceItems.length < sequenceCount}<span class="sequence-more" aria-hidden="true">…</span>{/if}
    </div>
  {/if}
  <p class="text-content text-text-primary line-clamp-2 break-words" title={job.prompt}>{job.prompt || 'No prompt supplied'}</p>
  {#if job.enhancedPrompt}
    <p class="job-enhanced line-clamp-3 break-words" title={job.enhancedPrompt}><span class="job-enhanced-label">✨ Enhanced</span> {job.enhancedPrompt}</p>
  {:else if job.enhanceStatus === 'failed'}
    <p class="job-enhanced break-words" data-enhance-status="failed"><span class="job-enhanced-label">Not enhanced:</span> the prompt enhancer failed, so the original prompt is used.</p>
  {:else if job.enhanceStatus === 'skipped'}
    <p class="job-enhanced break-words" data-enhance-status="skipped"><span class="job-enhanced-label">Not enhanced:</span> skipped, so the original prompt is used.</p>
  {/if}
  <p class="mt-1 text-ui text-text-muted truncate" title={job.model}>{job.model}</p>
  {#if job.notices?.length}
    <p class="job-notices" data-testid="job-notices">{job.notices.join(' · ')}</p>
  {/if}

  <!-- Progress -->
  <div class="mt-4">
      <div class="flex items-center justify-between gap-3 text-ui">
        <span class="text-text-secondary break-words min-w-0">{stepPhase}</span>
        <span class="font-mono text-text-primary shrink-0">{hasProgress ? `${Math.round(stepPct)}%` : '—'}</span>
      </div>

      <div
        class="progress-track mt-2 h-1.5 overflow-hidden rounded-full bg-border-subtle"
        class:step-pulse={job.status === 'running' && !job.paused && hasProgress && job.currentStep < job.totalSteps}
        role="progressbar"
        aria-label="Generation stage progress"
        aria-valuemin="0"
        aria-valuemax="100"
        aria-valuenow={hasProgress ? Math.round(stepPct) : undefined}
        aria-valuetext={hasProgress ? `${stepLabel} steps` : stepPhase}
      >
        <div
          class="h-full {progressFill} transition-all duration-300"
          style="width: {stepWidth}"
        ></div>
        {#if showCurrentStep && stepMs}
          {#key job.currentStep}
            <div
              class="current-step"
              data-state={job.paused ? 'paused' : 'running'}
              style="left: {stepWidth}; width: {currentStepWidth}; --step-ms: {Math.round(stepMs)}ms"
            ></div>
          {/key}
        {/if}
      </div>
      <div class="mt-2 text-right text-meta text-text-muted">
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
          title="Resume (P)"
          disabled={pending || !onresume}
          class="ui-btn ui-btn-sm"
        >
          {pending && currentFeedback?.action === 'Resume' ? 'Sending…' : 'Resume'}
        </button>
      {/if}
      {#if canPause}
        <button
          type="button"
          onclick={() => sendControl('Pause', onpause)}
          title="Pause (P)"
          disabled={pending || !onpause}
          class="ui-btn ui-btn-sm"
        >
          {pending && currentFeedback?.action === 'Pause' ? 'Sending…' : 'Pause'}
        </button>
      {/if}
      {#if canNext}
        <button
          type="button"
          onclick={() => sendControl('Next', onnext)}
          title="Next (N)"
          disabled={pending || !onnext}
          class="ui-btn ui-btn-sm"
        >
          {pending && currentFeedback?.action === 'Next' ? 'Sending…' : 'Next'}
        </button>
      {/if}
      {#if canRepeat}
        <button
          type="button"
          onclick={() => sendControl('Repeat', onrepeat)}
          title="Repeat (R)"
          disabled={pending || !onrepeat}
          class="ui-btn ui-btn-sm"
        >
          {pending && currentFeedback?.action === 'Repeat' ? 'Sending…' : 'Repeat'}
        </button>
      {/if}
      {#if canCancel}
        <button type="button" onclick={() => sendControl('Cancel', oncancel)} disabled={pending || !oncancel} class="ui-btn ui-btn-sm ui-btn-danger cancel-button" aria-label="Cancel job" title="Cancel (Esc)">{pending && currentFeedback?.action === 'Cancel' ? 'Sending…' : 'Cancel'}</button>
      {/if}
    </div>
  {/if}

  <div role="status" aria-live="polite" aria-atomic="true">
    {#if currentFeedback}
      <p class="control-feedback" class:failed={currentFeedback.state === 'failed'}>{currentFeedback.message}</p>
    {/if}
  </div>

  </div>

  <!-- Right column: the live preview. -->
  {#if showPreview}
    <div class="job-media">
      <span class="media-label" id="{job.job_id}-preview-label">Live preview</span>
      <figure class="live-preview" aria-labelledby="{job.job_id}-preview-label">
        <img src={job.previewUrl} alt="Live preview of the generation in progress" onerror={(event) => { failedPreviewUrl = event.currentTarget.getAttribute('src'); }} />
      </figure>
    </div>
  {/if}
  </div>
  <footer class="job-footer"><span>Job</span><span class="truncate font-mono" title={job.job_id}>{job.job_id}</span></footer>
</article>

<style>
  .sequence-strip { display: flex; align-items: center; gap: 4px; margin: 4px 0 14px; }
  .sequence-segment { flex: 1; min-width: 0; height: 8px; border-radius: 3px; background: var(--color-bg-base); border: 1px solid var(--color-border-strong); }
  .sequence-segment.run-boundary { margin-left: 5px; }
  .sequence-segment[data-state='previous'] { background: var(--color-primary-subtle); border-color: var(--color-primary-main); opacity: 0.5; }
  .sequence-segment[data-state='current'] { height: 12px; background: var(--color-primary-main); border-color: var(--color-primary-main); box-shadow: 0 0 8px var(--color-primary-subtle); }
  .sequence-segment[data-state='paused'] { height: 12px; background: var(--color-warning); border-color: var(--color-warning); }
  .sequence-segment[data-state='failed'] { height: 12px; background: var(--color-error); border-color: var(--color-error); }
  .sequence-segment[data-state='stopped'],
  .sequence-segment[data-state='unknown'] { height: 12px; border-style: dashed; }
  .sequence-more { color: var(--color-text-muted); font-size: var(--text-meta); line-height: 1; }

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
  /* A ghost fill that grows across the current step over the last step's duration. */
  .current-step {
    position: absolute;
    top: 0;
    bottom: 0;
    background: var(--color-primary-main);
    opacity: 0.4;
    transform-origin: left;
    animation: current-step-fill var(--step-ms) linear forwards;
  }
  .current-step[data-state='paused'] { animation-play-state: paused; }
  @keyframes current-step-fill {
    from { transform: scaleX(0); }
    to { transform: scaleX(1); }
  }
  @media (prefers-reduced-motion: reduce) {
    .step-pulse::after { animation: none; display: none; }
    .current-step { animation: none; }
  }
  .job-card { width: 100%; overflow: hidden; border: 1px solid var(--color-border-strong); border-radius: var(--radius-md); background: var(--color-bg-surface); }
  .job-header { display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 10px 14px; border-bottom: 1px solid var(--color-border-subtle); }
  h3 { font-size: var(--text-ui); font-weight: 600; }
  .job-status { display: flex; align-items: center; gap: 6px; font-size: var(--text-meta); text-transform: capitalize; color: var(--color-text-secondary); }
  .status-dot { width: 6px; height: 6px; border-radius: 50%; background: currentColor; }
  .job-status[data-status='running'], .job-status[data-status='completed'] { color: var(--color-primary-main); }
  .job-status[data-status='paused'] { color: var(--color-warning); }
  .job-status[data-status='failed'], .job-message.failed { color: var(--color-error); }
  .job-body { display: flex; flex-wrap: wrap; gap: 16px 20px; padding: 12px 14px; }
  .job-main { flex: 1 1 280px; min-width: 0; }
  .job-media { flex: 1 1 300px; min-width: 0; display: flex; flex-direction: column; gap: 6px; }
  .media-label { font-size: var(--text-ui); color: var(--color-text-secondary); }
  .job-timing { display: flex; flex-wrap: wrap; justify-content: space-between; gap: 8px 16px; margin-top: 14px; font-size: var(--text-meta); }
  .job-timing div { display: flex; align-items: baseline; gap: 8px; }
  dt { color: var(--color-text-muted); }
  dd { font-family: var(--font-mono); font-variant-numeric: tabular-nums; color: var(--color-text-primary); }
  .job-enhanced { margin-top: 6px; font-size: var(--text-ui); color: var(--color-text-secondary); }
  .job-notices { margin-top: 4px; font-size: var(--text-ui); color: var(--color-warning); }
  .job-enhanced-label { color: var(--color-primary-main); font-weight: 500; }
  .job-message { margin-top: 12px; padding: 8px 10px; background: var(--color-bg-base); border-radius: 4px; font-size: var(--text-ui); color: var(--color-text-secondary); overflow-wrap: anywhere; }
  .job-actions { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 12px; padding-top: 12px; border-top: 1px solid var(--color-border-subtle); }

  .job-actions .cancel-button { margin-left: auto; }
  .control-feedback { margin-top: 8px; font-size: var(--text-ui); color: var(--color-primary-main); overflow-wrap: anywhere; }
  .control-feedback.failed { color: var(--color-error); }
  .live-preview { margin: 0; overflow: hidden; border: 1px solid var(--color-border-subtle); border-radius: 6px; background: var(--color-bg-base); }
  .live-preview img { display: block; width: 100%; max-height: min(40vh, 18rem); object-fit: contain; }
  .job-footer { display: flex; gap: 8px; padding: 7px 14px; font-size: var(--text-meta); color: var(--color-text-muted); background: var(--color-bg-base); border-top: 1px solid var(--color-border-subtle); }
</style>
