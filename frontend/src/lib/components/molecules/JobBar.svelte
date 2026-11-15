<script lang="ts">
  import { errorText } from '$lib/api/client';
  import { Badge, Icon } from '$lib/components/atoms';
  import { formatClock } from '$lib/state/clock';
  import { barSubline, pendingNotice, promptText, shortHeadline } from '$lib/state/jobCaption';
  import { ringView, stepKey } from '$lib/state/jobRing';
  import { etaLabel, headline, isActive, isPaused, isVoiceDesignJob, jobControls, jobTone, outcome } from '$lib/state/jobView';
  import { useStepTiming } from '$lib/state/stepClock.svelte';
  import { addToast } from '$lib/state/toasts.svelte';
  import type { ActiveJobState, JobControlAction } from '$lib/types';
  import RingHero, { type RingMascot } from './RingHero.svelte';
  import SaveProject from './SaveProject.svelte';

  type ControlCallback = (jobId: string) => void | Promise<unknown>;

  interface Props {
    job: ActiveJobState;
    /** Jobs waiting behind this one. */
    queuedCount?: number;
    onpause?: ControlCallback;
    onresume?: ControlCallback;
    /** Blob, drawn inside the ring at the size the snippet receives. */
    mascot?: RingMascot;
    /** Open the full card: phases, timings, Skip, Repeat, Cancel and the queue. */
    ondetails: () => void;
  }

  let { job, queuedCount = 0, onpause, onresume, mascot, ondetails }: Props = $props();

  const active = $derived(isActive(job));
  const paused = $derived(isPaused(job));
  const offline = $derived(job.connection === 'reconnecting');
  const controls = $derived(jobControls(job));
  const available = $derived(new Set([...controls.job, ...controls.item]));
  const pending = $derived(active ? (job.controlQueued ?? null) : null);
  const stepTiming = useStepTiming(() => job);
  const ring = $derived(ringView(job, stepTiming.ms));
  const sub = $derived(barSubline(job, etaLabel(job), formatClock(job.elapsed)));
  const ended = $derived(outcome(job));
  const notice = $derived(pendingNotice(job, pending));

  let sending = $state(false);

  async function send(action: JobControlAction, callback: ControlCallback | undefined): Promise<void> {
    if (!callback || !available.has(action) || offline || sending) return;
    sending = true;
    try {
      await callback(job.job_id);
    } catch (error) {
      addToast(errorText(error, `${action} didn’t go through.`), 'error');
    } finally {
      sending = false;
    }
  }
</script>

<!-- The phone's job bar: the ring with Blob on its top-left corner, the headline over "ETA · step", Pause or Resume and ⋯
     with the queue count, then the prompt (or what the job saved) and the project. ⋯ opens the Job sheet, which holds Skip,
     Repeat and Cancel. A design job has no Pause. An ended job keeps its ring, mark and colours, and nothing pulses. -->
<div class="job-bar ui-overlay" data-tone={jobTone(job)} data-testid="job-bar" role="group" aria-label={active ? 'Running job' : 'Last job'}>
  <div class="ring-slot"><RingHero size="S" view={ring} {mascot} host="var(--color-bg-overlay)" stepMs={stepTiming.ms} stepKey={stepKey(job)} /></div>
  <div class="row">
    <div class="text">
      <p class="headline" data-testid="job-bar-headline">{shortHeadline(headline(job))}</p>
      <p class="sub" data-testid="job-bar-sub">{#if sub.eta}<b>{sub.eta}</b>{' · '}{/if}{sub.text}</p>
    </div>
    {#if active}
      {#if paused}
        <button type="button" class="ui-btn ui-btn-primary ui-btn-icon" aria-label="Resume" title="Resume" disabled={!available.has('resume') || offline || sending} onclick={() => void send('resume', onresume)}><Icon name="play" size={18} /></button>
      {:else if !isVoiceDesignJob(job)}
        <button type="button" class="ui-btn ui-btn-icon" class:ui-btn-on={pending === 'pause'} aria-label={pending === 'pause' ? 'Pausing' : 'Pause'} aria-busy={pending === 'pause' ? 'true' : undefined} title="Pause" disabled={!available.has('pause') || offline || sending || !!pending} onclick={() => void send('pause', onpause)}><Icon name="pause" size={18} /></button>
      {/if}
    {/if}
    <button type="button" class="ui-btn ui-btn-icon details" aria-haspopup="dialog" aria-label={queuedCount > 0 ? `Job details and queue, ${queuedCount} queued` : 'Job details'} title="Job details" onclick={ondetails}>
      <Icon name="more" size={18} />
      {#if queuedCount > 0}<span class="count" data-testid="queue-count" aria-hidden="true">{queuedCount}</span>{/if}
    </button>
  </div>
  <p class="foot">
    {#if notice}<Badge variant={notice.tone}><Icon name="clock" size={12} />{notice.text}</Badge>{/if}
    {#if !active && ended}<span class="saved">{ended.saved}</span>{:else}<span class="prompt" title={promptText(job)}>{promptText(job)}</span>{/if}
    {#if job.save_project}<SaveProject name={job.save_project} testid="job-bar-project" />{/if}
  </p>
</div>

<style>
  .job-bar { position: relative; display: flex; flex-direction: column; gap: 6px; padding: 10px 10px 10px 12px; font-size: var(--text-ui); }
  /* The ring sits on the bar's top-left corner, 20 px above it; its foot lines up with the foot of the headline and step line (the first row's 40 px), clear of the prompt line below. */
  .ring-slot { position: absolute; left: 8px; top: -20px; pointer-events: none; }
  .row { display: flex; align-items: center; gap: 6px; min-height: 40px; padding-left: 80px; }
  .text { min-width: 0; flex: 1; }
  .headline { overflow: hidden; font-family: var(--font-heading); font-size: 14px; font-weight: 800; line-height: 18px; color: var(--color-text-primary); white-space: nowrap; text-overflow: ellipsis; }
  .sub { overflow: hidden; margin-top: 1px; font-size: var(--text-ui); line-height: 16px; color: var(--color-text-muted); white-space: nowrap; text-overflow: ellipsis; font-variant-numeric: tabular-nums; }
  .sub b { font-weight: 600; color: var(--color-text-primary); }
  .details { position: relative; }
  /* The queue count: teal on the overlay, over the ⋯ that opens the queue. */
  .count { position: absolute; top: -5px; right: -5px; min-width: 18px; height: 18px; padding: 0 5px; border-radius: 9999px; background: color-mix(in srgb, var(--color-primary-main) 22%, var(--color-bg-overlay)); box-shadow: 0 0 0 2px var(--color-bg-overlay); font-family: var(--font-mono); font-size: var(--text-meta); font-weight: 600; line-height: 18px; color: var(--color-primary-main); }
  .foot { display: flex; align-items: center; gap: 6px; min-width: 0; height: 18px; font-size: var(--text-ui); line-height: 16px; }
  .foot > :global(.ui-badge) { flex: none; }
  .prompt, .saved { min-width: 0; overflow: hidden; color: var(--color-text-secondary); white-space: nowrap; text-overflow: ellipsis; }
  /* The project ends the row at the bar's right edge; the prompt is cut first. */
  .foot > :global(.save-project) { flex: none; max-width: 55%; margin-left: auto; padding-left: 4px; }
</style>
