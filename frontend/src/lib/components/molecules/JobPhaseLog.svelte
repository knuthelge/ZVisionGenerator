<script lang="ts">
  import { Badge, Icon } from '$lib/components/atoms';
  import type { IconName } from '$lib/components/atoms/Icon.svelte';
  import { formatClock } from '$lib/state/clock';
  import { promptText } from '$lib/state/jobCaption';
  import { activeDownload, downloadRowKey, downloadView, isActive, isVideo, jobPhase, phaseRows, positionLabel, readySummary, type PhaseRow, type PhaseRowStatus } from '$lib/state/jobView';
  import type { ActiveJobState } from '$lib/types';
  import JobInputs from './JobInputs.svelte';

  interface Props {
    job: ActiveJobState;
    /** Server time now, in epoch seconds, for the running phase's duration. */
    now: number;
    /** A finger: rows and prompt are taller, and the prompt opens in full on a tap. */
    touch?: boolean;
  }

  let { job, now, touch = false }: Props = $props();

  const STATUS_ICONS: Record<PhaseRowStatus, IconName> = {
    done: 'check', current: 'circle-dot', paused: 'pause', pending: 'circle', skipped: 'minus', failed: 'circle-x', stopped: 'stop', unknown: 'minus',
  };

  const active = $derived(isActive(job));
  const generating = $derived(jobPhase(job) === 'generate');
  const rows = $derived(phaseRows(job));
  const generateRow = $derived(rows[rows.length - 1]);
  const setupRows = $derived(rows.slice(0, -1));
  const ready = $derived(readySummary(job));
  const download = $derived(activeDownload(job));
  // An upscale leads with its source file; its recorded prompt, when it has one, is labelled as the source's.
  const upscale = $derived(job.workflow === 'upscale');
  const shownPrompt = $derived(upscale ? (job.enhancedPrompt || job.prompt) : promptText(job));
  let promptOpen = $state(false);

  function duration(row: PhaseRow): string {
    if (row.secs !== undefined) return formatClock(row.secs);
    // Only once an event has given the server clock, so a skewed client clock never shows a wild duration.
    if (row.startedAt !== undefined && active && job.serverOffset !== undefined) return formatClock(now - row.startedAt);
    return '';
  }
</script>

{#snippet row(item: PhaseRow, warn = false)}
  <div class="job-item" data-s={item.status} data-testid="phase-row" data-phase={item.key} class:warned={warn}>
    <span class="ico"><Icon name={warn ? 'alert' : STATUS_ICONS[item.status]} size={14} /></span>
    <span class="label"><b>{item.label}</b>{#if item.detail && item.key !== 'queue'}<span class="sep">{item.detail}</span>{/if}</span>
    <span class="dur">{duration(item)}</span>
  </div>
{/snippet}

<div class="job-log" class:touch data-testid="job-log">
  {#if generating}
    <!-- One line that scrolls sideways. A warning goes first, right after "Ready", and turns the row's icon amber, so it
         can never be the part that scrolls out of view. -->
    <div class="job-item" data-s="done" data-testid="phase-row" data-phase="ready" class:warned={ready.warn}>
      <span class="ico"><Icon name={ready.warn ? 'alert' : 'check'} size={14} /></span>
      <span class="label scroll" data-scrolls><b>Ready</b>{#if ready.warn}<span class="sep warn">enhancer failed, original prompt used</span>{/if}{#each ready.parts as part (part)}<span class="sep">{part}</span>{/each}</span>
      <span class="dur">{ready.secs !== undefined ? formatClock(ready.secs) : ''}</span>
    </div>
  {:else}
    {#each setupRows as item (item.key)}
      {@render row(item)}
      {#if download && (item.status === 'current' || item.status === 'paused') && downloadRowKey(download) === item.key}
        <p class="help" data-testid="download-help">{downloadView(download).help}</p>
      {/if}
    {/each}
  {/if}
  {@render row(generateRow)}
  <div class="body">
    {#if upscale}<JobInputs inputs={job.inputs ?? []} />{/if}
    {#if shownPrompt}
      <p class="ui-help" class:after-input={upscale} data-testid="job-position">{positionLabel(job)}</p>
      {#if touch}
        <button type="button" class="prompt prompt-btn" data-testid="job-prompt" class:open={promptOpen} aria-expanded={promptOpen} title={shownPrompt} onclick={() => { promptOpen = !promptOpen; }}>{shownPrompt}</button>
      {:else}
        <p class="prompt" data-testid="job-prompt" title={shownPrompt}>{shownPrompt}</p>
      {/if}
    {/if}
    {#if !upscale}<JobInputs inputs={job.inputs ?? []} />{/if}
    {#if job.enhanceStatus === 'enhanced' || job.enhanceStatus === 'skipped' || job.notices?.length || (isVideo(job) && active)}
      <div class="badges">
        {#if job.enhanceStatus === 'enhanced'}<Badge variant="info"><Icon name="sparkle" size={12} />Enhanced</Badge>{/if}
        {#if job.enhanceStatus === 'skipped'}<Badge>Not enhanced</Badge>{/if}
        {#each job.notices ?? [] as notice (notice)}<Badge variant="warning"><Icon name="alert" size={12} />{notice}</Badge>{/each}
        {#if isVideo(job) && active}<Badge><Icon name="film" size={12} />No live preview for video</Badge>{/if}
      </div>
    {/if}
  </div>
</div>

<style>
  .job-log { padding: 0 12px 6px 8px; border-top: 1px solid var(--color-border-subtle); }
  /* In the Job sheet the sheet is the card, so the log has the sheet's own insets. */
  :global(.job-card.sheet) .job-log { padding: 0 16px 6px 12px; }
  /* The status icons keep their 14 px size with little room either side: 8 px from the card, an 8 px gap. */
  .job-item { display: grid; grid-template-columns: 14px 1fr auto; align-items: start; column-gap: 8px; min-height: 30px; border-bottom: 1px solid var(--color-border-subtle); }
  .job-item:has(+ .body), .job-item:has(+ .help) { border-bottom: 0; }
  .touch .job-item { min-height: 34px; }
  .ico { display: grid; place-items: center; height: 30px; color: var(--color-text-muted); }
  .touch .ico { height: 34px; }
  .job-item[data-s='done'] .ico, .job-item[data-s='current'] .ico { color: var(--color-primary-main); }
  /* Once the job ended, its finished steps are history, not work in progress. */
  :global(.job-card:not([data-tone='running'], [data-tone='preparing'], [data-tone='paused'], [data-tone='stale'], [data-tone='queued'])) .job-item[data-s='done'] .ico { color: var(--color-text-muted); }
  .job-item[data-s='paused'] .ico { color: var(--color-warning); }
  .job-item[data-s='failed'] .ico { color: var(--color-error); }
  .job-item.warned .ico { color: var(--color-warning) !important; }
  /* Parts wrap between parts, never under the duration; a part that starts a new line has its dot in the clipped margin. */
  .label { min-width: 0; margin-left: -12px; padding: 5px 0 5px 12px; clip-path: inset(0 0 0 12px); line-height: 20px; color: var(--color-text-primary); }
  .label > :global(*) { display: inline-block; white-space: nowrap; }
  .label > :global(*:not(:last-child)) { margin-right: 12px; }
  /* The Ready row: one line that scrolls sideways, fading out at the right edge when there is more. */
  .label.scroll { clip-path: none; margin-left: 0; padding: 5px 24px 5px 0; overflow-x: auto; white-space: nowrap; scrollbar-width: none; mask-image: linear-gradient(90deg, black calc(100% - 28px), transparent); }
  .label.scroll::-webkit-scrollbar { display: none; }
  .label b { font-weight: 500; }
  .job-item[data-s='pending'] .label, .job-item[data-s='skipped'] .label { color: var(--color-text-muted); }
  .job-item[data-s='pending'] .label b { font-weight: 400; }
  .sep { font-size: var(--text-ui); color: var(--color-text-muted); }
  .sep::before { content: '·'; display: inline-block; width: 12px; margin-left: -12px; text-align: center; color: var(--color-border-strong); }
  .warn { color: var(--color-warning); }
  .dur { padding-top: 7px; font-size: var(--text-ui); line-height: 16px; color: var(--color-text-muted); font-variant-numeric: tabular-nums; }
  .job-item[data-s='current'] .dur, .job-item[data-s='paused'] .dur { color: var(--color-text-primary); }
  .help { padding: 0 0 10px 22px; font-size: var(--text-meta); line-height: 1.45; color: var(--color-text-muted); border-bottom: 1px solid var(--color-border-subtle); }

  .body { padding: 0 0 8px 22px; }
  .after-input { margin-top: 6px; }
  .prompt { display: -webkit-box; margin-top: 1px; overflow: hidden; font-size: var(--text-content); line-height: 19px; color: var(--color-text-secondary); text-align: left; -webkit-line-clamp: 3; line-clamp: 3; -webkit-box-orient: vertical; overflow-wrap: anywhere; }
  .touch .prompt { -webkit-line-clamp: 2; line-clamp: 2; }
  .prompt-btn { width: 100%; padding: 0; }
  .prompt-btn.open { -webkit-line-clamp: unset; line-clamp: unset; }
  .badges { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 6px; }
</style>
