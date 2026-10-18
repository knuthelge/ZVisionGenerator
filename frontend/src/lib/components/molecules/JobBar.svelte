<script lang="ts">
  import { errorText } from '$lib/api/client';
  import { Badge, Icon } from '$lib/components/atoms';
  import { workflowLabel } from '$lib/state/assetActions';
  import { etaLabel, headline, isActive, isPaused, itemNoun, iterations, jobBar, jobControls, jobPhase, stageBars } from '$lib/state/jobView';
  import { addToast } from '$lib/state/toasts.svelte';
  import type { ActiveJobState, JobControlAction } from '$lib/types';
  import ActionMenu, { type ActionMenuEntry } from './ActionMenu.svelte';
  import { requestConfirm } from './confirm.svelte';

  type ControlCallback = (jobId: string) => void | Promise<unknown>;

  interface Props {
    job: ActiveJobState;
    /** Jobs waiting behind this one. */
    queuedCount?: number;
    oncancel?: ControlCallback;
    onpause?: ControlCallback;
    onresume?: ControlCallback;
    onnext?: ControlCallback;
    onrepeat?: ControlCallback;
    /** Open the full card (phases, timings, the queue). */
    ondetails: () => void;
  }

  let { job, queuedCount = 0, oncancel, onpause, onresume, onnext, onrepeat, ondetails }: Props = $props();

  const active = $derived(isActive(job));
  const paused = $derived(isPaused(job));
  const offline = $derived(job.connection === 'reconnecting');
  const controls = $derived(jobControls(job));
  const available = $derived(new Set([...controls.job, ...controls.item]));
  const noun = $derived(itemNoun(job));
  const phase = $derived(jobPhase(job));
  const bar = $derived(jobBar(job));
  const bars = $derived(stageBars(job));
  const items = $derived(iterations(job));
  const eta = $derived(etaLabel(job));
  const pending = $derived(job.controlQueued ?? null);
  const shownPrompt = $derived(job.enhancedPrompt || job.prompt || 'No prompt supplied');
  const notice = $derived.by(() => {
    if (pending === 'pause') return phase === 'enhance' ? 'Pausing after this rewrite' : `Pausing after this ${noun}`;
    if (pending === 'next') return phase === 'enhance' ? 'Skipping this rewrite' : 'Skipping after this step';
    if (pending === 'repeat') return `Repeating after this ${noun}`;
    if (pending === 'quit') return 'Stopping at the next step';
    if (offline) return 'Reconnecting…';
    return null;
  });

  let sending = $state(false);
  let moreOpen = $state(false);
  let moreButton = $state<HTMLButtonElement | null>(null);

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

  async function cancel(): Promise<void> {
    const approved = await requestConfirm({
      question: 'Cancel this job?',
      info: `The ${noun}s already saved are kept.`,
      confirmLabel: 'Cancel job',
      cancelLabel: 'Keep running',
    });
    if (approved) await send('quit', oncancel);
  }

  const moreItems = $derived<ActionMenuEntry[]>([
    { kind: 'item', id: 'details', label: 'Job details', onselect: ondetails },
    ...(available.has('next') ? [{ kind: 'item' as const, id: 'next', label: phase === 'enhance' ? 'Skip this rewrite' : `Skip to the next ${noun}`, onselect: () => void send('next', onnext) }] : []),
    ...(available.has('repeat') ? [{ kind: 'item' as const, id: 'repeat', label: `Repeat this ${noun}`, onselect: () => void send('repeat', onrepeat) }] : []),
    ...(queuedCount > 0 ? [{ kind: 'item' as const, id: 'queue', label: `Queue · ${queuedCount} waiting`, onselect: ondetails }] : []),
    ...(available.has('quit') ? [{ kind: 'separator' as const }, { kind: 'item' as const, id: 'quit', label: 'Cancel job', danger: true, onselect: () => void cancel() }] : []),
  ]);
</script>

<!-- The compact job card: one row of facts, the progress, the prompt. The full card is a tap away under ⋯. -->
<div class="job-bar ui-overlay" data-tone={paused ? 'paused' : 'running'} data-testid="job-bar" aria-label="Running job">
  <div class="jb-row">
    <span class="jb-status"><i class="jb-dot"></i><span class="ui-area-label jb-workflow">{workflowLabel(job.workflow)}</span></span>
    <span class="jb-headline">{headline(job)}</span>
    {#if eta}<span class="jb-eta">{eta}</span>{/if}
    <span class="jb-controls">
      {#if active}
        {#if paused}
          <button type="button" class="ui-btn ui-btn-sm ui-btn-primary" disabled={!available.has('resume') || offline || sending} onclick={() => void send('resume', onresume)}><Icon name="play" size={12} />Resume</button>
        {:else}
          <button type="button" class="ui-btn ui-btn-sm" disabled={!available.has('pause') || offline || sending || pending === 'pause'} onclick={() => void send('pause', onpause)}><Icon name="pause" size={12} />{pending === 'pause' ? 'Pausing' : 'Pause'}</button>
        {/if}
      {/if}
      <button type="button" bind:this={moreButton} class="ui-btn ui-btn-sm ui-btn-icon" aria-label="More job controls" aria-haspopup="menu" aria-expanded={moreOpen} onclick={() => { moreOpen = !moreOpen; }}><Icon name="more" size={14} /></button>
      <ActionMenu open={moreOpen} anchor={moreButton} items={moreItems} label="Job" onclose={() => { moreOpen = false; }} />
    </span>
  </div>
  {#if bar.kind === 'segments'}
    <div class="jb-segments" role="progressbar" aria-label="Job progress" aria-valuemin="0" aria-valuemax={items.total} aria-valuenow={items.done}>
      {#each bar.segments as segment, index (index)}
        <i class="jb-segment" class:run-break={segment.runBreak} data-state={segment.state}>{#if segment.state === 'now'}<b style:width="{segment.fill * 100}%"></b>{/if}</i>
      {/each}
    </div>
  {:else if bars.length > 1}
    <!-- An item with two passes (generate, then upscale): both bars side by side. -->
    <div class="jb-stages">
      {#each bars as stage (stage.name)}
        <div class="jb-stage" data-state={stage.state}>
          <span class="jb-stage-label">{stage.label}<small>{stage.state === 'current' ? `${stage.current}/${stage.total}` : stage.state === 'done' ? 'done' : job.review ? 'after review' : 'next'}</small></span>
          <div class="ui-progress"><span class="ui-progress-fill" style:width="{stage.state === 'done' ? 100 : stage.state === 'current' && stage.total > 0 ? (stage.current / stage.total) * 100 : 0}%"></span></div>
        </div>
      {/each}
    </div>
  {:else}
    <div class="ui-progress" role="progressbar" aria-label="Job progress" aria-valuemin="0" aria-valuemax="1" aria-valuenow={bar.fraction}><span class="ui-progress-fill" style:width="{bar.fraction * 100}%"></span></div>
  {/if}
  <p class="jb-prompt">
    {#if notice}<Badge variant={pending === 'quit' || offline ? 'warning' : 'info'}>{notice}</Badge>{/if}
    {#if job.enhanceStatus === 'enhanced'}<Badge variant="info"><Icon name="sparkle" size={10} />Enhanced</Badge>{/if}
    {#if queuedCount > 0}<button type="button" class="jb-queue" onclick={ondetails}>{queuedCount} queued</button>{/if}
    <span class="jb-prompt-text" title={shownPrompt}>{shownPrompt}</span>
  </p>
</div>

<style>
  .job-bar { display: flex; flex-direction: column; gap: 8px; padding: 10px 12px; font-size: var(--text-ui); --jb-tone: var(--color-primary-main); }
  .job-bar[data-tone='paused'] { --jb-tone: var(--color-warning); }
  .jb-row { display: flex; align-items: center; gap: 8px; min-height: 28px; }
  .jb-status { display: flex; flex: none; align-items: center; gap: 6px; color: var(--jb-tone); }
  .jb-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--jb-tone); animation: jb-pulse 1.2s infinite; }
  .job-bar[data-tone='paused'] .jb-dot { animation: none; }
  @keyframes jb-pulse { 50% { opacity: 0.3; } }
  .jb-headline { min-width: 0; overflow: hidden; font-weight: 600; white-space: nowrap; text-overflow: ellipsis; color: var(--color-text-primary); }
  .jb-eta { flex: none; color: var(--color-text-muted); font-variant-numeric: tabular-nums; }
  .jb-controls { display: flex; flex: none; align-items: center; gap: 4px; margin-left: auto; }
  .jb-segments { display: flex; gap: 2px; height: 6px; }
  .jb-segment { position: relative; flex: 1; min-width: 0; overflow: hidden; border-radius: 1px; background: var(--color-zinc-700); }
  .jb-segment.run-break { margin-left: 4px; }
  .jb-segment[data-state='done'] { background: var(--jb-tone); }
  .jb-segment[data-state='failed'] { background: var(--color-error); }
  .jb-segment > b { position: absolute; inset: 0 auto 0 0; background: var(--jb-tone); transition: width 0.3s ease; }
  .jb-stages { display: flex; gap: 8px; }
  .jb-stage { display: flex; flex: 1; min-width: 0; flex-direction: column; gap: 4px; }
  .jb-stage-label { display: flex; justify-content: space-between; gap: 6px; overflow: hidden; font-size: var(--text-meta); font-weight: 600; white-space: nowrap; color: var(--color-text-secondary); }
  .jb-stage-label small { font-family: var(--font-mono); font-weight: 400; color: var(--color-text-muted); }
  .jb-stage[data-state='current'] .jb-stage-label { color: var(--color-text-primary); }
  .jb-stage[data-state='done'] .ui-progress-fill { background: color-mix(in srgb, var(--color-primary-main) 55%, var(--color-zinc-700)); }
  .job-bar .ui-progress { height: 6px; }
  .jb-prompt { display: flex; align-items: center; gap: 6px; min-width: 0; overflow: hidden; color: var(--color-text-muted); }
  .jb-prompt > :global(.ui-badge) { flex: none; }
  .jb-queue { flex: none; font-size: var(--text-ui); color: var(--color-text-secondary); text-decoration: underline dotted; text-underline-offset: 3px; }
  .jb-prompt-text { min-width: 0; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; }
  @media (max-width: 400px) { .jb-workflow { display: none; } }
  /* A finger gets the touch layer's full control size, not the small one. */
  @media (pointer: coarse) {
    .jb-controls .ui-btn { height: var(--spacing-control); }
    .jb-controls .ui-btn-icon { width: var(--spacing-control); }
  }
</style>
