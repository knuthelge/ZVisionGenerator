<script lang="ts">
  import { tick } from 'svelte';
  import { Icon } from '$lib/components/atoms';
  import { copyText, selectElementText } from '$lib/clipboard';
  import { workflowLabel } from '$lib/state/assetActions';
  import { addToast } from '$lib/state/toasts.svelte';
  import { hasJobSettings } from '$lib/state/jobSettings';
  import type { JobLogEntry } from '$lib/types';
  import { promptFirstLine, timeAgo } from '$lib/format';
  import { hasErrorDetail, jobLogDuration, jobLogTime, jobLogTone, jobLogToneLabel, type JobLogTone } from './jobLogView';

  interface Props {
    entry: JobLogEntry;
    /** The moment the relative time is counted from, in epoch milliseconds. */
    now: number;
    /** Refill the Workspace form from this entry's settings. */
    onloadsettings: (entry: JobLogEntry) => void;
    /** Remove this entry from the log; the row stays disabled until it settles. */
    ondelete: (entry: JobLogEntry) => Promise<void>;
  }

  let { entry, now, onloadsettings, ondelete }: Props = $props();

  const uid = $props.id();

  const ICONS = { completed: 'check', failed: 'close', partial: 'alert', cancelled: 'stop' } as const;
  const TONE_COLORS: Record<JobLogTone, string> = {
    completed: 'var(--color-success)',
    failed: 'var(--color-error)',
    partial: 'var(--color-warning)',
    cancelled: 'var(--color-text-muted)'
  };

  let expanded = $state(false);
  let deleting = $state(false);
  let errorElement = $state<HTMLElement | null>(null);

  const tone = $derived(jobLogTone(entry));
  const prompt = $derived(promptFirstLine(entry.prompt));
  const duration = $derived(jobLogDuration(entry));
  const finishedAt = $derived(jobLogTime(entry));
  const detailed = $derived(hasErrorDetail(entry));
  const errorText = $derived(entry.error?.trim() ?? '');
  const loadable = $derived(hasJobSettings(entry.settings));

  /** Copy the error; without a clipboard, select it so the keyboard shortcut can. */
  async function copyError(): Promise<void> {
    if (await copyText(errorText)) {
      addToast('Error copied', 'success');
      return;
    }
    expanded = true;
    await tick();
    if (errorElement) selectElementText(errorElement);
    addToast('The error is selected. Press Ctrl+C or \u2318C to copy it.', 'info');
  }

  async function remove(): Promise<void> {
    deleting = true;
    try {
      await ondelete(entry);
    } finally {
      deleting = false;
    }
  }
</script>

<li class="job-log-row" data-testid="job-log-row" data-tone={tone} style:--row-tone={TONE_COLORS[tone]}>
  <div class="row-main">
    <span class="row-icon" title={jobLogToneLabel(tone)}>
      <Icon name={ICONS[tone]} size={14} />
      <span class="sr-only">{jobLogToneLabel(tone)}</span>
    </span>
    <div class="row-body">
      <div class="row-title">
        <span class="ui-area-label">{workflowLabel(entry.workflow)}</span>
        {#if entry.model}<span class="row-model">{entry.model}</span>{/if}
      </div>
      <p class="row-prompt" class:row-prompt-empty={!prompt}>{prompt || 'No prompt'}</p>
      <p class="row-meta">
        {#if duration}<span>{duration}</span>{/if}
        {#if finishedAt !== null}<span title={new Date(finishedAt * 1000).toLocaleString()}>{timeAgo(finishedAt, now)}</span>{/if}
      </p>
    </div>
    {#if detailed}
      <button
        type="button"
        class="ui-btn ui-btn-quiet ui-btn-icon ui-btn-sm"
        aria-label={expanded ? 'Hide details' : 'Show details'}
        aria-expanded={expanded}
        aria-controls="{uid}-detail"
        onclick={() => { expanded = !expanded; }}
      >
        <Icon name="chevdown" size={14} class={expanded ? 'rotate-180' : ''} />
      </button>
    {/if}
  </div>
  <div class="row-actions" role="group" aria-label="Job actions">
    {#if loadable}
      <button type="button" class="ui-btn ui-btn-sm" title="Copy this job's settings into the form" onclick={() => onloadsettings(entry)}>
        <Icon name="reuse" size={14} />Load settings
      </button>
    {/if}
    {#if errorText}
      <button type="button" class="ui-btn ui-btn-sm" onclick={copyError}>
        <Icon name="copy" size={14} />Copy error
      </button>
    {/if}
    <button type="button" class="ui-btn ui-btn-sm ui-btn-danger" disabled={deleting} title="Remove this job from the log" onclick={remove}>
      <Icon name="trash" size={14} />Delete
    </button>
  </div>
  {#if detailed && expanded}
    <div class="row-detail" id="{uid}-detail" data-testid="job-log-detail">
      {#if errorText}<p class="row-error" bind:this={errorElement}>{errorText}</p>{/if}
      {#if entry.failed_runs > 0}
        <p class="row-runs">{entry.failed_runs} {entry.failed_runs === 1 ? 'run' : 'runs'} failed</p>
      {/if}
    </div>
  {/if}
</li>

<style>
  .job-log-row { padding: 10px 16px; border-bottom: 1px solid var(--color-border-subtle); }
  .row-main { display: flex; align-items: flex-start; gap: 10px; }
  .row-icon { display: grid; flex: none; place-items: center; width: 22px; height: 22px; border-radius: 50%; color: var(--row-tone); background: color-mix(in srgb, var(--row-tone) 14%, transparent); }
  .row-body { min-width: 0; flex: 1; }
  .row-title { display: flex; align-items: baseline; gap: 8px; min-width: 0; }
  .row-model { min-width: 0; overflow: hidden; font-size: var(--text-meta); color: var(--color-text-muted); white-space: nowrap; text-overflow: ellipsis; }
  .row-prompt { margin-top: 2px; overflow: hidden; font-size: var(--text-content); color: var(--color-text-primary); white-space: nowrap; text-overflow: ellipsis; }
  .row-prompt-empty { color: var(--color-text-muted); }
  .row-meta { display: flex; gap: 8px; margin-top: 2px; font-size: var(--text-meta); color: var(--color-text-muted); }
  .row-meta > span + span::before { content: '\00b7'; margin-right: 8px; }
  .row-actions { display: flex; flex-wrap: wrap; gap: 4px; margin: 6px 0 0 32px; }
  .row-detail { margin: 8px 0 0 32px; padding: 8px 10px; max-height: 12rem; overflow-y: auto; border-radius: var(--radius-sm); background: var(--color-bg-base); font-size: var(--text-ui); }
  .row-error { color: var(--color-text-secondary); font-family: var(--font-mono); white-space: pre-wrap; overflow-wrap: anywhere; user-select: text; }
  .row-runs { color: var(--row-tone); }
  .row-error + .row-runs { margin-top: 6px; }
</style>
