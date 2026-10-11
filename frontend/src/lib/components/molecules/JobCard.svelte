<script lang="ts">
  import { untrack } from 'svelte';
  import { Badge, Icon, StepProgress } from '$lib/components/atoms';
  import type { IconName } from '$lib/components/atoms/Icon.svelte';
  import { acceptsPageShortcut } from '$lib/keyboard';
  import { workflowLabel } from '$lib/state/assetActions';
  import {
    etaLabel, formatClock, headline, isActive, isPaused, itemActivity, itemNoun, iterations, jobBar, jobControls, jobPhase, jobTone,
    stageBars, type JobTone
  } from '$lib/state/jobView';
  import { trackStepTiming, type StepTiming } from '$lib/state/stepTiming';
  import type { ActiveJobState, JobControlAction } from '$lib/types';
  import Alert from './Alert.svelte';
  import JobControls from './JobControls.svelte';
  import JobPhaseLog from './JobPhaseLog.svelte';

  type ControlCallback = (jobId: string) => void | Promise<unknown>;

  interface Props {
    job: ActiveJobState;
    oncancel?: ControlCallback;
    onpause?: ControlCallback;
    onresume?: ControlCallback;
    onnext?: ControlCallback;
    onrepeat?: ControlCallback;
  }

  let { job, oncancel, onpause, onresume, onnext, onrepeat }: Props = $props();

  /** How long the first Esc keeps Cancel armed. */
  const ESC_ARM_MS = 3000;
  /** How long an accepted control counts as pending before the server's control_queued event confirms it. */
  const REQUEST_GRACE_MS = 3000;

  const TONE_COLORS: Record<JobTone, string> = {
    queued: 'var(--color-text-muted)', preparing: 'var(--color-info)', running: 'var(--color-primary-main)',
    paused: 'var(--color-warning)', stale: 'var(--color-warning)', completed: 'var(--color-success)',
    failed: 'var(--color-error)', stopped: 'var(--color-text-muted)', unknown: 'var(--color-text-muted)'
  };

  /** performance.now() of the last tick; the clock, ETA and armed Esc follow it. */
  let clock = $state(performance.now());

  const tone = $derived(jobTone(job));
  const active = $derived(isActive(job));
  const offline = $derived(job.connection === 'reconnecting');
  const controls = $derived(jobControls(job));
  const phase = $derived(jobPhase(job));
  const noun = $derived(itemNoun(job));
  const bar = $derived(jobBar(job));
  const progress = $derived(iterations(job));
  const total = $derived(progress.total);
  const done = $derived(progress.done);

  // ── Controls ─────────────────────────────────────────

  const callbacks = $derived<Record<JobControlAction, ControlCallback | undefined>>({
    pause: onpause, resume: onresume, next: onnext, repeat: onrepeat, quit: oncancel
  });
  const names = $derived<Record<JobControlAction, string>>({
    pause: 'Pause',
    resume: 'Resume',
    next: phase === 'enhance' ? 'Skip this rewrite'
      : job.stageName === 'upscale' && stageBars(job).length ? `Skip this ${noun} (the base ${noun} is discarded too)` : `Skip this ${noun}`,
    repeat: `Repeat this ${noun}`,
    quit: 'Cancel job'
  });

  let sending = $state<{ jobId: string; action: JobControlAction } | null>(null);
  let requested = $state<{ jobId: string; action: JobControlAction; at: number } | null>(null);
  let rejected = $state<{ jobId: string; action: JobControlAction; reason: string } | null>(null);
  let escArmedAt = $state<{ jobId: string; at: number } | null>(null);

  const pending = $derived.by<JobControlAction | null>(() => {
    if (sending?.jobId === job.job_id) return sending.action;
    if (job.controlQueued) return job.controlQueued;
    return requested?.jobId === job.job_id && clock - requested.at < REQUEST_GRACE_MS ? requested.action : null;
  });
  const cancelArmed = $derived(escArmedAt?.jobId === job.job_id && clock - escArmedAt.at < ESC_ARM_MS);
  const available = $derived(new Set([...controls.job, ...controls.item]));

  async function send(action: JobControlAction): Promise<void> {
    const callback = callbacks[action];
    if (!callback || !available.has(action) || offline) return;
    if (pending && pending !== action && action !== 'quit') return;
    if (sending?.jobId === job.job_id) return;
    const jobId = job.job_id;
    escArmedAt = null;
    rejected = null;
    sending = { jobId, action };
    try {
      const result = callback(jobId);
      if (result) await result;
      // Resume takes effect at once; the others wait for the server to act on them.
      if (action !== 'resume') requested = { jobId, action, at: performance.now() };
    } catch (error) {
      rejected = { jobId, action, reason: error instanceof Error ? error.message : '' };
    } finally {
      sending = null;
    }
  }

  // Esc arms Cancel and a second Esc sends it; P pauses or resumes, N skips and R repeats, unless a field, popover or dialog has the key.
  $effect(() => {
    if (!active) return;
    function handleKeydown(event: KeyboardEvent): void {
      if (!acceptsPageShortcut(event)) return;
      const key = event.key.toLowerCase();
      let action: JobControlAction | null = null;
      if (key === 'escape') action = available.has('quit') ? 'quit' : null;
      else if (key === 'p') action = available.has('resume') ? 'resume' : available.has('pause') ? 'pause' : null;
      else if (key === 'n') action = available.has('next') ? 'next' : null;
      else if (key === 'r') action = available.has('repeat') ? 'repeat' : null;
      if (!action || !callbacks[action]) return;
      event.preventDefault();
      // Checked against the time now: the display clock only ticks once a second.
      const armed = escArmedAt?.jobId === job.job_id && performance.now() - escArmedAt.at < ESC_ARM_MS;
      if (action === 'quit' && !armed) {
        escArmedAt = { jobId: job.job_id, at: performance.now() };
        clock = escArmedAt.at;
        return;
      }
      void send(action);
    }
    document.addEventListener('keydown', handleKeydown);
    return () => document.removeEventListener('keydown', handleKeydown);
  });

  // ── Clock ────────────────────────────────────────────

  // Count elapsed up every second between server updates, never backwards; it holds still while paused.
  let elapsedBase = $state({ jobId: '', elapsed: 0, at: 0 });
  $effect(() => {
    const { job_id: jobId, elapsed } = job;
    const now = performance.now();
    const previous = untrack(() => elapsedBase);
    const counted = previous.jobId === jobId ? previous.elapsed + (now - previous.at) / 1000 : 0;
    elapsedBase = { jobId, elapsed: Math.max(elapsed, counted), at: now };
    clock = now;
  });
  const ticking = $derived(active && !isPaused(job));
  $effect(() => {
    if (!active && !cancelArmed) return;
    const timer = setInterval(() => { clock = performance.now(); }, 1000);
    return () => clearInterval(timer);
  });
  const liveElapsed = $derived(
    !active ? job.elapsed : elapsedBase.elapsed + (ticking && !offline ? Math.max(0, clock - elapsedBase.at) / 1000 : 0)
  );
  // Server time now (epoch seconds), for the running phase's duration. `clock` makes it tick.
  const serverNow = $derived.by(() => {
    void clock;
    return Date.now() / 1000 + (job.serverOffset ?? 0);
  });
  const sinceUpdate = $derived.by(() => {
    void clock;
    return job.lastEventAt ? Math.max(0, Math.round((Date.now() - job.lastEventAt) / 1000)) : 0;
  });
  const eta = $derived(etaLabel(job));

  // Time each step so the step in progress can fill in over about as long as the last one took.
  let stepTiming = $state<StepTiming | null>(null);
  $effect(() => {
    const sample = { scope: `${job.job_id}:${job.stageName}:${job.totalSteps}`, step: job.currentStep, paused: isPaused(job), now: performance.now() };
    stepTiming = trackStepTiming(untrack(() => stepTiming), sample);
  });
  const stepMs = $derived(stepTiming?.stepMs ?? null);
  // A single-stage, single-item job: the job bar is the step bar, so it carries the live step cues.
  const headerSteps = $derived(total === 1 && active && phase === 'generate' && !stageBars(job).length);
  const activity = $derived(itemActivity(job));
  const headerLive = $derived(ticking && !offline);

  // ── Notice ───────────────────────────────────────────

  const notice = $derived.by<{ text: string; variant: 'info' | 'warning' | 'error'; icon: IconName } | null>(() => {
    if (cancelArmed) return { text: 'Press Esc again to cancel', variant: 'error', icon: 'alert' };
    if (rejected?.jobId === job.job_id) {
      const name = names[rejected.action].split(' ')[0];
      return { text: `${name} didn’t go through.${rejected.reason ? ` ${rejected.reason}` : ' Try again.'}`, variant: 'error', icon: 'alert' };
    }
    if (pending === 'pause') {
      const text = phase === 'load' ? 'Pauses once the model is loaded' : phase === 'enhance' ? 'Pausing after this rewrite' : `Pausing after this ${noun}`;
      return { text, variant: 'warning', icon: 'clock' };
    }
    if (pending === 'next') return { text: phase === 'enhance' ? 'Skipping this rewrite' : 'Skipping after this step', variant: 'info', icon: 'clock' };
    if (pending === 'repeat') return { text: `Repeating after this ${noun}`, variant: 'info', icon: 'clock' };
    if (pending === 'quit') return { text: 'Stopping at the next step', variant: 'warning', icon: 'clock' };
    return null;
  });

  const statusIcon = $derived<IconName | null>(tone === 'stale' ? 'offline' : tone === 'paused' ? 'pause' : null);
</script>

<article
  class="ui-panel job-card"
  data-tone={tone}
  style:--job-tone={TONE_COLORS[tone]}
  style:--job-fill={tone === 'stale' ? TONE_COLORS.running : tone === 'completed' ? TONE_COLORS.running : TONE_COLORS[tone]}
  aria-label="Active job"
>
  <header class="job-head">
    <div class="job-row">
      <span class="job-status">
        {#if statusIcon}<Icon name={statusIcon} size={14} />{:else}<span class="job-dot"></span>{/if}
        <span class="ui-area-label">{workflowLabel(job.workflow)}</span>
      </span>
      <span class="job-notice" role="status" aria-live="polite">
        {#if notice}<Badge variant={notice.variant} class="job-notice-badge"><Icon name={notice.icon} size={12} />{notice.text}</Badge>{/if}
      </span>
      {#if active}
        {#if controls.job.length}
          <JobControls controls={controls.job} label="Job controls" {names} {pending} disabled={offline} {cancelArmed} onsend={(action) => void send(action)} />
        {:else if !controls.item.length && phase === 'generate'}
          <span class="ui-help">No controls for {noun} jobs yet</span>
        {/if}
      {/if}
    </div>
    <div class="job-row job-title-row">
      <h3 class="job-title">{headline(job)}</h3>
      <span class="job-time">
        {#if !active}
          <span class="num">{formatClock(liveElapsed)}</span> total
        {:else if offline}
          Last update {sinceUpdate} s ago
        {:else}
          {#if eta}<b>{eta}</b>{' · '}{/if}<span class="num">{formatClock(liveElapsed)}</span> elapsed
        {/if}
      </span>
    </div>
    {#if headerSteps}
      <!-- Between steps (getting ready, or saving after the last one) the bar sweeps so the job never looks stuck. -->
      <StepProgress
        class="job-bar"
        value={activity === 'stepping' ? job.currentStep : activity === 'finishing' ? 1 : 0}
        max={activity === 'stepping' ? job.totalSteps : 1}
        label="Job progress"
        live={headerLive && activity === 'stepping'}
        busy={headerLive && activity !== 'stepping'}
        {stepMs}
        valuetext={activity === 'stepping' ? `Step ${job.currentStep} of ${job.totalSteps}` : activity === 'finishing' ? 'Finishing up' : 'Starting'}
      />
    {:else if bar.kind === 'bar'}
      <div class="ui-progress job-bar" role="progressbar" aria-label="Job progress" aria-valuemin="0" aria-valuemax={total} aria-valuenow={done} aria-valuetext="{done} of {total} {noun}s done">
        <span class="ui-progress-fill" style:width="{bar.fraction * 100}%"></span>
      </div>
    {:else}
      <div class="job-bar job-segments" role="progressbar" aria-label="Job progress" aria-valuemin="0" aria-valuemax={total} aria-valuenow={done} aria-valuetext="{done} of {total} {noun}s done">
        {#each bar.segments as segment, index (index)}
          <i class="job-segment" class:run-break={segment.runBreak} data-state={segment.state}>
            {#if segment.state === 'now'}<b style:width="{segment.fill * 100}%"></b>{/if}
          </i>
        {/each}
      </div>
    {/if}
  </header>

  {#if offline}
    <div class="job-alert"><Alert tone="warning" live>Lost the live connection. Controls come back when it reconnects.</Alert></div>
  {/if}

  {#snippet itemControls()}
    {#if controls.item.length}
      <JobControls controls={controls.item} label={phase === 'enhance' ? 'Current rewrite' : `Current ${noun}`} {names} {pending} disabled={offline} onsend={(action) => void send(action)} />
    {/if}
  {/snippet}
  <JobPhaseLog {job} now={serverNow} {stepMs} itemControls={controls.item.length ? itemControls : undefined} />
</article>

<style>
  .job-card {
    --job-gutter: 12px;
    /* 16px glyph + 12px gap: labels and detail share this edge. */
    --job-indent: 28px;
    width: 100%;
    overflow: hidden;
    font-size: var(--text-content);
    line-height: 20px;
    container-type: inline-size;
  }
  .job-head { display: flex; flex-direction: column; gap: 2px; padding: 10px var(--job-gutter) 14px; border-bottom: 1px solid var(--color-border-subtle); }
  .job-row { display: flex; align-items: center; gap: 12px; min-width: 0; height: var(--spacing-control-sm); }
  .job-title-row { align-items: baseline; justify-content: space-between; gap: 16px; height: 24px; }
  .job-status { display: flex; flex: none; align-items: center; gap: 8px; color: var(--job-tone); }
  .job-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--job-tone); }
  /* The feedback slot sits against the buttons it answers and never moves the layout. */
  .job-notice { display: flex; flex: 1; justify-content: flex-end; min-width: 0; overflow: hidden; }
  .job-notice :global(.job-notice-badge) { height: var(--spacing-control-sm); max-width: 100%; padding: 0 8px; gap: 6px; overflow: hidden; border-radius: var(--radius-sm); font-size: var(--text-ui); text-overflow: ellipsis; }
  .job-head :global(.job-controls) { flex: none; margin-right: -7px; }
  .job-title { flex: 1; min-width: 0; overflow: hidden; font-family: var(--font-heading); font-size: var(--text-title); font-weight: 800; line-height: 24px; white-space: nowrap; text-overflow: ellipsis; }
  .job-time { flex: none; font-size: var(--text-ui); line-height: 16px; white-space: nowrap; color: var(--color-text-muted); }
  .job-time b { font-weight: 500; color: var(--color-text-primary); }
  .job-time .num { font-variant-numeric: tabular-nums; }

  /* The job bar: 6px, one segment per item up to 48, a wider gap at each run boundary. */
  .job-head :global(.job-bar) { --progress-color: var(--job-fill); height: 6px; margin-top: 10px; }
  .job-segments { display: flex; gap: 2px; }
  .job-segment { position: relative; flex: 1; min-width: 0; overflow: hidden; border-radius: 1px; background: var(--color-zinc-700); }
  .job-segment.run-break { margin-left: 4px; }
  .job-segment[data-state='done'] { background: var(--job-fill); }
  .job-segment[data-state='failed'] { background: var(--color-error); }
  .job-segment > b { position: absolute; inset: 0 auto 0 0; background: var(--job-fill); transition: width 0.3s ease; }
  /* A failed job's saved items stay neutral; only the item that failed is red. */
  [data-tone='failed'] .job-segment[data-state='done'] { background: var(--color-text-muted); }
  [data-tone='stale'] .job-head :global(.job-bar) { opacity: 0.4; filter: saturate(0); }

  .job-alert { padding: var(--job-gutter) var(--job-gutter) 4px; }

  /* Phone width: the time drops under the headline instead of cutting it short. */
  @container (max-width: 440px) {
    .job-title-row { flex-wrap: wrap; height: auto; row-gap: 0; }
    .job-title { flex-basis: 100%; }
  }
</style>
