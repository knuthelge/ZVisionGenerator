<script lang="ts">
  import { untrack } from 'svelte';
  import { errorText } from '$lib/api/client';
  import { Icon } from '$lib/components/atoms';
  import type { IconName } from '$lib/components/atoms/Icon.svelte';
  import { acceptsPageShortcut } from '$lib/keyboard';
  import { workflowLabel } from '$lib/state/assetActions';
  import { formatClock } from '$lib/state/clock';
  import { caption, cardHeadline, pendingNotice, splitModelName, timeParts } from '$lib/state/jobCaption';
  import { ringView, stepKey, type RingView } from '$lib/state/jobRing';
  import { etaLabel, GENERIC_FAILURE_MESSAGE, isActive, isPaused, isVoiceDesignJob, itemNoun, jobControls, jobPhase, jobTone, outcome, stageBars, type JobTone } from '$lib/state/jobView';
  import { layout } from '$lib/state/layoutMode.svelte';
  import { useStepTiming } from '$lib/state/stepClock.svelte';
  import type { ActiveJobState, JobControlAction } from '$lib/types';
  import Alert from './Alert.svelte';
  import JobControlsRow from './JobControls.svelte';
  import JobPhaseLog from './JobPhaseLog.svelte';
  import RingHero, { type RingMascot } from './RingHero.svelte';
  import SaveProject from './SaveProject.svelte';
  import { requestConfirm } from './confirm.svelte';

  type ControlCallback = (jobId: string) => void | Promise<unknown>;

  interface Props {
    job: ActiveJobState;
    oncancel?: ControlCallback;
    onpause?: ControlCallback;
    onresume?: ControlCallback;
    onnext?: ControlCallback;
    onrepeat?: ControlCallback;
    /** Blob, drawn inside the ring at the size the snippet receives; left out while he is elsewhere (the review overlay). */
    mascot?: RingMascot;
    /** `panel` is a surface of its own; `sheet` sits straight on the Job sheet's surface, which is the card. */
    variant?: 'panel' | 'sheet';
  }

  let { job, oncancel, onpause, onresume, onnext, onrepeat, mascot, variant = 'panel' }: Props = $props();

  /** How long the first Esc keeps Cancel armed. */
  const ESC_ARM_MS = 3000;
  /** How long an accepted control counts as pending before the server's control_queued event confirms it. */
  const REQUEST_GRACE_MS = 3000;

  /** The workflow kicker's colour: teal while work runs (preparing included), amber when it waits on something, muted at rest and once ended. */
  const KICKER_COLORS: Partial<Record<JobTone, string>> = {
    preparing: 'var(--color-primary-main)', running: 'var(--color-primary-main)', paused: 'var(--color-warning)', stale: 'var(--color-warning)',
  };

  /** performance.now() of the last tick; the clock, ETA and armed Esc follow it. */
  let clock = $state(performance.now());

  const tone = $derived(jobTone(job));
  const active = $derived(isActive(job));
  const offline = $derived(job.connection === 'reconnecting');
  const controls = $derived(jobControls(job));
  const phase = $derived(jobPhase(job));
  const noun = $derived(itemNoun(job));
  // A finger (or a phone-sized window): icon controls, a smaller ring and a dialog before Cancel.
  const touch = $derived(layout.coarse || layout.compact);

  // ── Controls ─────────────────────────────────────────

  const callbacks = $derived<Partial<Record<JobControlAction, ControlCallback>>>({
    pause: onpause, resume: onresume, next: onnext, repeat: onrepeat, quit: oncancel
  });
  const names = $derived<Record<JobControlAction, string>>({
    pause: 'Pause',
    resume: 'Resume',
    next: phase === 'enhance' ? 'Skip this rewrite'
      : job.stageName === 'upscale' && stageBars(job).length ? `Skip this ${noun} (the base ${noun} is discarded too)` : `Skip this ${noun}`,
    repeat: `Repeat this ${noun}`,
    approve: 'Approve and upscale',
    discard: `Discard this ${noun}`,
    quit: 'Cancel job'
  });

  let sending = $state<{ jobId: string; action: JobControlAction } | null>(null);
  let requested = $state<{ jobId: string; action: JobControlAction; at: number; wallAt: number } | null>(null);
  let rejected = $state<{ jobId: string; action: JobControlAction; reason: string } | null>(null);
  let escArmedAt = $state<{ jobId: string; at: number } | null>(null);

  // A control waits for the job's next step; once the job has ended there is none to wait for.
  const pending = $derived.by<JobControlAction | null>(() => {
    if (!active) return null;
    if (sending?.jobId === job.job_id) return sending.action;
    if (job.controlQueued) return job.controlQueued;
    if (requested?.jobId !== job.job_id || clock - requested.at >= REQUEST_GRACE_MS) return null;
    // The event that settled the control ends the wait at once; the grace only bridges the gap before control_queued arrives.
    return job.controlSettledAt !== undefined && job.controlSettledAt >= requested.wallAt ? null : requested.action;
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
      if (action !== 'resume') requested = { jobId, action, at: performance.now(), wallAt: Date.now() };
    } catch (error) {
      rejected = { jobId, action, reason: errorText(error, '') };
    } finally {
      sending = null;
    }
  }

  /** Ask before Cancel on a finger, so nothing grows on the card and one slip cannot end the job. */
  async function confirmCancel(): Promise<void> {
    const approved = await requestConfirm({
      question: 'Cancel this job?',
      info: `The ${noun}s already ${isVoiceDesignJob(job) ? 'made' : 'saved'} are kept.`,
      confirmLabel: 'Cancel job',
      cancelLabel: 'Keep running',
    });
    if (approved) await send('quit');
  }

  /** Send a control the user clicked. A mouse cancels directly (Esc arms it twice); a finger asks first. */
  async function tapControl(action: JobControlAction): Promise<void> {
    if (action === 'quit' && touch) return confirmCancel();
    await send(action);
  }

  // Esc arms Cancel and a second Esc sends it (on a finger, the first Esc opens the dialog); P pauses or resumes, N skips and R repeats, unless a field, popover or dialog has the key.
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
      if (action === 'quit' && touch) {
        void confirmCancel();
        return;
      }
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

  // ── The hero ─────────────────────────────────────────

  // Time each step so the step in progress can fill in over about as long as the last one took.
  const stepTiming = useStepTiming(() => job);
  const ring = $derived<RingView>(ringView(job, stepTiming.ms));
  const headlineText = $derived(cardHeadline(job));
  const headline = $derived(splitModelName(headlineText, job.model_label || job.model));
  const now = $derived(caption(job));
  const ended = $derived(outcome(job));
  const times = $derived(timeParts({ active, offline, eta: etaLabel(job), elapsed: formatClock(liveElapsed), sinceUpdate }));
  /** What the job saved, "3 of 4 saved to the Gallery · 1 not run", as parts that break only between each other. */
  const savedParts = $derived(ended?.saved.split(' · ') ?? []);

  // ── Notice ───────────────────────────────────────────

  const notice = $derived.by<{ text: string; tone: 'info' | 'warning' | 'error'; icon: IconName } | null>(() => {
    if (tone === 'failed') return { text: ended?.text || GENERIC_FAILURE_MESSAGE, tone: 'error', icon: 'alert' };
    if (!active) return null;
    if (offline) return { text: 'Lost the live connection. Controls come back when it reconnects.', tone: 'warning', icon: 'offline' };
    if (cancelArmed && !touch) return { text: 'Press Esc again to cancel', tone: 'error', icon: 'alert' };
    if (rejected?.jobId === job.job_id) {
      const name = names[rejected.action].split(' ')[0];
      return { text: `${name} didn’t go through.${rejected.reason ? ` ${rejected.reason}` : ' Try again.'}`, tone: 'error', icon: 'alert' };
    }
    const waiting = pendingNotice(job, pending);
    return waiting ? { ...waiting, icon: 'clock' } : null;
  });
</script>

{#snippet dotted(parts: string[])}{#each parts as part, index (index)}<span class="part" class:after={index > 0}>{part}</span>{/each}{/snippet}

<article class="job-card {variant}" class:ui-panel={variant === 'panel'} class:touch data-tone={tone} aria-label="Active job">
  <div class="kicker-bar" data-testid="job-kicker">
    <span class="ui-area-label kicker" data-testid="job-workflow" style:color={KICKER_COLORS[tone] ?? 'var(--color-text-muted)'}>{workflowLabel(job.workflow)}</span>
    {#if job.save_project}<SaveProject name={job.save_project} testid="job-project" />{/if}
  </div>

  <div class="hero">
    <RingHero size={touch ? 'M' : 'L'} view={ring} {mascot} stepMs={stepTiming.ms} stepKey={stepKey(job)} />
    <div class="text">
      <div class="top">
        <h3 class="headline">{#key headlineText}<span class="swap">{#each headline as part, index (index)}{#if part.model}<span class="model" data-testid="headline-model">{part.text}</span>{:else}{part.text}{/if}{/each}</span>{/key}</h3>
        <p class="time dotline" data-testid="job-time">
          {#each times as part, index (index)}<span class="part" class:after={index > 0}>{#if part.kind === 'eta'}<b>{part.text}</b>{:else if part.kind === 'estimate'}<span class="estimate">{part.text}</span>{:else}{part.text}{/if}</span>{/each}
        </p>
      </div>
      <div class="bottom">
        {#if now}
          <div class="caption" class:wrap={now.wrap} data-testid="job-caption">
            <p class="lead dotline" data-testid="caption-lead">{#if now.wrap}{now.lead.join(' ')}{:else}{@render dotted(now.lead)}{/if}</p>
            {#if now.line.length}<p class="raw dotline" data-testid="caption-line">{@render dotted(now.line)}</p>{/if}
          </div>
        {/if}
        <div class="foot">
          {#if active}
            {#if controls.job.length || controls.item.length}
              <JobControlsRow {controls} itemLabel={phase === 'enhance' ? 'rewrite' : noun} {names} {pending} disabled={offline} {cancelArmed} {touch} onsend={(action) => void tapControl(action)} />
            {:else if phase === 'generate'}
              <p class="saved">No controls for {noun} jobs yet</p>
            {/if}
          {:else if ended}
            <!-- Once the job ends, the controls give way to what it saved, in the same place. -->
            <p class="saved" data-testid="job-saved"><Icon name={job.outputs.length ? 'grid' : 'minus'} size={12} /><span class="dotline">{@render dotted(savedParts)}</span></p>
          {/if}
        </div>
      </div>
    </div>
  </div>

  <!-- Two regions, always mounted so a screen reader hears a notice when it appears, with no height of their own while empty.
       Each notice is announced through one of them: errors assertively, everything else politely. -->
  <div role="status" aria-live="polite" data-testid="job-notice-region">
    {#if notice && notice.tone !== 'error'}
      <div class="notice" data-testid="job-notice"><Alert tone={notice.tone} icon={notice.icon}>{notice.text}</Alert></div>
    {/if}
  </div>
  <div role="alert" data-testid="job-alert-region">
    {#if notice && notice.tone === 'error'}
      <div class="notice" data-testid="job-notice"><Alert tone="error" icon={notice.icon}>{notice.text}</Alert></div>
    {/if}
  </div>

  <JobPhaseLog {job} now={serverNow} {touch} />
</article>

<style>
  .job-card { container-type: inline-size; min-width: 0; flex: none; font-size: var(--text-content); line-height: 20px; }
  /* In the Job sheet the sheet is the card: no second container of the same shade. */
  .job-card.sheet { background: transparent; }

  /* The kicker bar runs across the card's top like a panel head: the workflow at the left, the project at the right. */
  .kicker-bar { display: flex; flex-wrap: wrap; align-items: center; column-gap: 10px; row-gap: 2px; min-width: 0; min-height: 16px; padding: 12px 14px 0; }
  .touch .kicker-bar { padding: 10px 12px 0; }
  .kicker { flex: none; line-height: 16px; white-space: nowrap; transition: color 600ms var(--ease-standard); }
  .kicker-bar > :global(.save-project) { flex: 0 1 auto; margin-left: auto; }

  /* The hero: the ring on the left, one block centred on it. */
  .hero { display: flex; gap: 14px; padding: 8px 14px 14px 12px; }
  .touch .hero { gap: 12px; padding: 6px 12px 8px; }
  .text { display: flex; min-width: 0; flex: 1; flex-direction: column; justify-content: center; }
  .top, .bottom { display: flex; min-width: 0; flex-direction: column; }
  .bottom { margin-top: 12px; gap: 10px; }

  .headline { font-family: var(--font-heading); font-size: var(--text-title); font-weight: 800; line-height: 20px; color: var(--color-text-primary); text-wrap: balance; }
  .swap { display: inline; animation: headline-in 220ms var(--ease-soft-out); }
  @keyframes headline-in { from { opacity: 0; translate: 0 6px; } }
  /* The model's name never breaks inside itself ("Downloading / Z-Image Turbo"). */
  .model { white-space: nowrap; }
  .time { margin-top: 2px; font-size: var(--text-ui); line-height: 16px; color: var(--color-text-muted); font-variant-numeric: tabular-nums; }
  .time b { font-weight: 600; color: var(--color-text-primary); }
  .estimate { font-weight: 400; color: var(--color-text-muted); }

  /* "·"-joined lines break only between parts. The "·" hangs in the gap before its part; a part that starts a new line
     has its dot in the clipped margin, so no line starts with "·". */
  .part { display: inline-block; white-space: nowrap; }
  .part:not(:last-child) { margin-right: 1em; }
  .dotline { margin-left: -1em; padding-left: 1em; clip-path: inset(0 0 0 1em); }
  .after::before { content: '·'; display: inline-block; width: 1em; margin-left: -1em; text-align: center; color: var(--color-border-strong); }
  .raw .after::before { color: inherit; }

  .caption { min-width: 0; }
  .lead { overflow: hidden; font-size: var(--text-ui); font-weight: 600; line-height: 16px; color: var(--color-text-secondary); }
  .caption.wrap .lead { display: -webkit-box; font-weight: 400; -webkit-line-clamp: 2; line-clamp: 2; -webkit-box-orient: vertical; }
  .raw { margin-top: 2px; font-family: var(--font-mono); font-size: var(--text-meta); line-height: 15px; color: var(--color-text-muted); }
  .foot { display: flex; align-items: center; min-height: var(--spacing-control-sm); }
  .saved { display: flex; align-items: flex-start; gap: 6px; font-size: var(--text-ui); line-height: 16px; color: var(--color-text-secondary); }
  .saved :global(svg) { flex: none; margin-top: 2px; color: var(--color-text-muted); }

  /* The notice sits under the hero and never moves it. */
  .notice { padding: 0 12px 12px; }
</style>
