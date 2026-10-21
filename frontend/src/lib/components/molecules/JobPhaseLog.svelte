<script lang="ts">
  import type { Snippet } from 'svelte';
  import { Badge, Icon, StepProgress } from '$lib/components/atoms';
  import type { IconName } from '$lib/components/atoms/Icon.svelte';
  import Alert from './Alert.svelte';
  import {
    activeDownload, downloadRowKey, downloadView, enhanceProgress, finishingLabel, finishingText, formatClock, isActive, isPaused, isVideo, itemActivity, iterations, jobPhase, loadBody, outcome, phaseRows,
    positionLabel, readySummary, stageBars, stepFraction, type PhaseRow, type PhaseRowStatus
  } from '$lib/state/jobView';
  import type { ActiveJobState, JobDownload } from '$lib/types';

  interface Props {
    job: ActiveJobState;
    /** Server time now, in epoch seconds, for the running phase's duration. */
    now: number;
    /** How long the last step took, for the live step bar. */
    stepMs?: number | null;
    /** Item controls (Skip, Repeat) shown on the current task's line, before its duration. */
    itemControls?: Snippet;
    /** Show the live preview beside the text; off when the stage shows it large (JobPreview). */
    preview?: boolean;
  }

  let { job, now, stepMs = null, itemControls, preview = true }: Props = $props();

  const STATUS_ICONS: Record<PhaseRowStatus, IconName> = {
    done: 'check', current: 'circle-dot', paused: 'pause', pending: 'circle', skipped: 'minus', failed: 'circle-x', stopped: 'stop', unknown: 'info'
  };
  /** The live preview fits this box: landscape images get the width, portrait ones the height. */
  const PREVIEW_MAX_WIDTH = 280;
  const PREVIEW_MAX_HEIGHT = 200;
  /** Fraction of the steps at which the first live preview arrives (the server renders them at 25, 50 and 75%). */
  const FIRST_PREVIEW_AT = 0.25;

  const active = $derived(isActive(job));
  const generating = $derived(jobPhase(job) === 'generate');
  const rows = $derived(phaseRows(job));
  const generateRow = $derived(rows[rows.length - 1]);
  const ready = $derived(readySummary(job));
  const ended = $derived(outcome(job));
  const total = $derived(iterations(job).total);
  const bars = $derived(stageBars(job));
  const rewrite = $derived(enhanceProgress(job));
  const pausedBetween = $derived(isPaused(job) && generating && !!job.itemFinished);
  const running = $derived(active && !isPaused(job) && job.connection !== 'reconnecting');
  const shownPrompt = $derived(job.enhancedPrompt || job.prompt || 'No prompt supplied');
  const stepPct = $derived(Math.round(stepFraction(job) * 100));
  const activity = $derived(itemActivity(job));
  const download = $derived(activeDownload(job));

  // Reserve the preview slot from the start, sized from the job's aspect ratio ("2:3 · m · 8 steps"), so text never re-wraps.
  const aspect = $derived.by(() => {
    const match = /^(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)/.exec(job.meta ?? '');
    return match ? Number(match[1]) / Number(match[2]) : 1;
  });
  const previewWidth = $derived(Math.round(Math.min(PREVIEW_MAX_WIDTH, PREVIEW_MAX_HEIGHT * aspect)));
  // Paused between images: show the image that just finished, sharp. Otherwise the transient live preview.
  const lastOutput = $derived(job.outputs[job.outputs.length - 1]);
  let failedPreviewUrl = $state<string | null>(null);
  const previewSrc = $derived(pausedBetween ? (lastOutput?.thumbnail_url ?? null) : job.previewUrl);
  const showMedia = $derived(preview && active && generating && !isVideo(job) && !job.review);
  // Some models render no live preview; once well past the first one, say so instead of promising it.
  const previewNote = $derived(activity === 'stepping' && stepFraction(job) > FIRST_PREVIEW_AT + 0.15
    ? 'No live preview for this model' : `First preview at ${FIRST_PREVIEW_AT * 100}%`);

  function duration(row: PhaseRow): string {
    if (row.secs !== undefined) return formatClock(row.secs);
    // Only once an event has given the server clock, so a skewed client clock never shows a wild duration.
    if (row.startedAt !== undefined && active && job.serverOffset !== undefined) return formatClock(now - row.startedAt);
    return '';
  }
</script>

{#snippet line(status: PhaseRowStatus, warn: boolean, label: Snippet, dur: string, controls?: Snippet)}
  <div class="job-line" class:has-controls={!!controls}>
    <Icon name={warn ? 'alert' : STATUS_ICONS[status]} size={16} />
    <span class="job-label">{@render label()}</span>
    {#if controls}<span class="job-line-controls">{@render controls()}</span>{/if}
    <span class="job-dur">{dur}</span>
  </div>
{/snippet}

{#snippet downloadBlock(current: JobDownload)}
  {@const view = downloadView(current)}
  <div class="job-step">
    <div class="job-step-head">
      <span class="dl-line">{#each view.segments as segment, index (index)}{#if index > 0}{' · '}{/if}<span class="dl-seg">{segment}</span>{/each}</span>
      {#if view.percent !== null}<span class="num dl-pct">{view.percent}%</span>{:else}<span class="dl-pct">Size unknown</span>{/if}
    </div>
    {#if view.percent !== null}
      <StepProgress value={current.bytes_done} max={current.bytes_total ?? 1} label={view.label} live={running} valuetext={view.valuetext} />
    {:else}
      <StepProgress value={0} max={1} label={view.label} busy={running} indeterminate valuetext={view.valuetext} />
    {/if}
  </div>
  <p class="ui-help">{view.help}</p>
{/snippet}

{#snippet stepBlock()}
  {#if pausedBetween}
    <p class="ui-help">Resumes with the next {isVideo(job) ? 'video' : 'image'}.</p>
  {:else if bars.length}
    <div class="job-stages">
      {#each bars as bar, index (bar.name)}
        {@const finishing = activity === 'finishing' && index === bars.length - 1}
        <div class="job-step">
          <div class="job-step-head">
            {#if bar.state === 'current' && bar.total > 0}
              <span>{bar.label} · step {bar.current} of {bar.total}</span><span class="num">{Math.round(bar.current / bar.total * 100)}%</span>
            {:else if bar.state === 'current'}
              <span>{bar.label}</span><span>Starting…</span>
            {:else if bar.state === 'done'}
              <span>{bar.label} · {bar.total} steps</span>
              {#if finishing}<span>{finishingLabel(job)}</span>{:else}<span class="job-step-done"><Icon name="check" size={12} />Done</span>{/if}
            {:else}
              <span>{bar.label}</span><span>Next</span>
            {/if}
          </div>
          <StepProgress
            class={bar.state === 'done' && !finishing ? 'job-stage-done' : ''}
            value={bar.state === 'done' ? 1 : bar.current}
            max={bar.state === 'done' ? 1 : Math.max(1, bar.total)}
            label="{bar.label} stage"
            live={running && bar.state === 'current'}
            busy={running && (finishing || (bar.state === 'current' && bar.total === 0))}
            {stepMs}
          />
        </div>
      {/each}
    </div>
  {:else}
    <div class="job-step">
      <div class="job-step-head">
        {#if activity === 'finishing'}
          <span>{finishingLabel(job)}</span><span class="num">100%</span>
        {:else if activity === 'starting'}
          <span>Starting…</span>
        {:else if activity === 'reviewing'}
          <span>Waiting for your review</span>
        {:else}
          <span>Step {job.currentStep} of {job.totalSteps}</span><span class="num">{stepPct}%</span>
        {/if}
      </div>
      <!-- A single-item job shows its steps as the job bar in the header. -->
      {#if total > 1}
        <StepProgress
          value={activity === 'stepping' ? job.currentStep : activity === 'finishing' ? 1 : 0}
          max={activity === 'stepping' ? job.totalSteps : 1}
          label="Current {isVideo(job) ? 'video' : 'image'}"
          live={running && activity === 'stepping'}
          busy={running && activity !== 'stepping'}
          {stepMs}
          valuetext={activity === 'stepping' ? `Step ${job.currentStep} of ${job.totalSteps}` : activity === 'finishing' ? finishingText(job) : 'Starting'}
        />
      {/if}
    </div>
  {/if}
{/snippet}

<div class="job-log">
  {#if generating}
    <div class="job-item" data-s="done">
      {#snippet readyLabel()}
        <b>Ready</b>
        {#each ready.parts as part (part)}<span class="job-sep">{part}</span>{/each}
        {#if ready.warn}<span class="job-sep job-warn">enhancer failed, original prompt used</span>{/if}
      {/snippet}
      {@render line('done', ready.warn, readyLabel, ready.secs !== undefined ? formatClock(ready.secs) : '')}
    </div>

    <div class="job-item" data-s={generateRow.status}>
      {#snippet generateLabel()}<b>{generateRow.label}</b>{#if generateRow.detail}<span>{generateRow.detail}</span>{/if}{/snippet}
      {@render line(generateRow.status, false, generateLabel, duration(generateRow), active ? itemControls : undefined)}
      <div class="job-detail">
        {#if ended}
          {#if ended.error}
            <Alert tone="error" live>{ended.text}</Alert>
          {:else}
            <p class="job-prompt">{ended.text}</p>
          {/if}
          <p class="ui-help">{ended.saved}</p>
        {:else}
          <div class="job-split" class:no-media={!showMedia}>
            <div class="job-text">
              <div>
                <p class="ui-help">{positionLabel(job)}</p>
                <p class="job-prompt" title={shownPrompt}>{shownPrompt}</p>
              </div>
              {#if job.enhanceStatus === 'enhanced' || job.enhanceStatus === 'skipped' || job.notices?.length || isVideo(job)}
                <div class="job-badges">
                  {#if job.enhanceStatus === 'enhanced'}<Badge variant="info"><Icon name="sparkle" size={12} />Enhanced</Badge>{/if}
                  {#if job.enhanceStatus === 'skipped'}<Badge>Not enhanced</Badge>{/if}
                  {#each job.notices ?? [] as notice (notice)}<Badge variant="warning"><Icon name="alert" size={12} />{notice}</Badge>{/each}
                  {#if isVideo(job)}<Badge><Icon name="film" size={12} />No live preview for video</Badge>{/if}
                </div>
              {/if}
              {@render stepBlock()}
            </div>
            {#if showMedia}
              <figure class="job-media" class:empty={!previewSrc || failedPreviewUrl === previewSrc} style:width="{previewWidth}px" style:aspect-ratio={aspect}>
                {#if previewSrc && failedPreviewUrl !== previewSrc}
                  <img
                    src={previewSrc}
                    alt={pausedBetween ? 'The last finished image' : 'Live preview of the generation in progress'}
                    onerror={(event) => { failedPreviewUrl = event.currentTarget.getAttribute('src'); }}
                  />
                {:else}
                  <span class="ui-help">{previewNote}</span>
                {/if}
              </figure>
            {/if}
          </div>
        {/if}
      </div>
    </div>
  {:else}
    {#each rows as row (row.key)}
      <div class="job-item" data-s={row.status}>
        {#snippet rowLabel()}<b>{row.label}</b>{#if row.detail && row.key !== 'queue'}<span>{row.detail}</span>{/if}{/snippet}
        {@render line(row.status, false, rowLabel, duration(row), row.key === 'enhance' && row.status === 'current' ? itemControls : undefined)}
        {#if row.status === 'failed' || row.status === 'stopped' || row.status === 'unknown'}
          <div class="job-detail">
            {#if ended?.error}<Alert tone="error" live>{ended.text}</Alert>{:else if ended}<p class="job-prompt">{ended.text}</p>{/if}
          </div>
        {:else if row.status === 'current' || row.status === 'paused'}
          {#if row.key === 'queue'}
            <div class="job-detail"><p class="ui-help">The worker picks it up as soon as it’s free.</p></div>
          {:else if row.key === 'enhance'}
            <div class="job-detail">
              {#if download && downloadRowKey(download) === 'enhance'}
                {@render downloadBlock(download)}
              {:else if rewrite}
                <div class="job-step">
                  <div class="job-step-head"><span>Rewriting prompt {rewrite.current} of {rewrite.total}</span></div>
                  <StepProgress value={rewrite.done} max={rewrite.total} label="Prompt enhancement" live={running} {stepMs} valuetext="{rewrite.done} of {rewrite.total} prompts rewritten" />
                </div>
              {:else}
                <p class="ui-help">{job.message || 'Loading the prompt enhancer.'}</p>
              {/if}
            </div>
          {:else if row.key === 'load'}
            <div class="job-detail">
              {#if download && downloadRowKey(download) === 'load'}
                {@render downloadBlock(download)}
              {:else}
                <p class="ui-help">{loadBody(job)}</p>
              {/if}
            </div>
          {/if}
        {/if}
      </div>
    {/each}
  {/if}
</div>

<style>
  .job-log { padding: 0 var(--job-gutter) 4px; container-type: inline-size; }
  .job-item + .job-item { border-top: 1px solid var(--color-border-subtle); }
  .job-line { display: grid; grid-template-columns: 16px minmax(0, 1fr) auto; column-gap: 12px; align-items: center; min-height: var(--spacing-row); }
  .job-line.has-controls { grid-template-columns: 16px minmax(0, 1fr) auto auto; }
  /* Icon glyphs sit ~2px inside their box; pull them onto the gutter. */
  .job-line > :global(svg) { margin-left: -2px; color: var(--color-text-muted); }
  .job-item[data-s='done'] .job-line > :global(svg) { color: var(--color-primary-main); }
  .job-item[data-s='current'] .job-line > :global(svg) { color: var(--job-tone); }
  .job-item[data-s='paused'] .job-line > :global(svg) { color: var(--color-warning); }
  .job-item[data-s='failed'] .job-line > :global(svg) { color: var(--color-error); }
  .job-item .job-line:has(:global(.job-warn)) > :global(svg) { color: var(--color-warning); }
  .job-label { display: flex; align-items: baseline; gap: 8px; min-width: 0; white-space: nowrap; overflow: hidden; }
  .job-label :global(b) { flex: none; font-weight: 500; }
  .job-label :global(span) { flex: none; font-size: var(--text-ui); color: var(--color-text-muted); }
  /* Only the last part gives way when the line runs out of room. */
  .job-label :global(span:last-child) { flex: 0 1 auto; min-width: 0; overflow: hidden; text-overflow: ellipsis; }
  .job-label :global(.job-sep)::before { content: '·'; margin-right: 8px; color: var(--color-border-strong); }
  .job-label :global(.job-warn) { color: var(--color-warning); }
  .job-item[data-s='pending'] .job-label :global(b), .job-item[data-s='skipped'] .job-label :global(b) { font-weight: 400; color: var(--color-text-muted); }
  .job-line-controls { display: flex; margin-right: -3px; }
  .job-dur { font-size: var(--text-ui); line-height: 16px; font-variant-numeric: tabular-nums; color: var(--color-text-muted); }
  .job-item[data-s='current'] .job-dur, .job-item[data-s='paused'] .job-dur { color: var(--color-text-primary); }

  .job-detail { display: flex; flex-direction: column; gap: 12px; padding: 0 0 12px var(--job-indent); }
  .job-split { display: grid; grid-template-columns: minmax(0, 1fr) auto; column-gap: 16px; align-items: stretch; }
  .job-split.no-media { grid-template-columns: minmax(0, 1fr); }
  .job-text { display: flex; flex-direction: column; justify-content: space-between; gap: 12px; min-width: 0; }
  .job-prompt { color: var(--color-text-secondary); display: -webkit-box; -webkit-line-clamp: 4; line-clamp: 4; -webkit-box-orient: vertical; overflow: hidden; overflow-wrap: anywhere; }
  .job-badges { display: flex; flex-wrap: wrap; gap: 6px; }
  .job-stages { display: flex; flex-direction: column; gap: 12px; }
  .job-step { display: flex; flex-direction: column; gap: 8px; }
  /* As thick as the job bar: while an image runs, its steps matter as much. */
  .job-step :global(.ui-progress) { --progress-color: var(--job-fill); height: 6px; }
  .job-step-head { display: flex; justify-content: space-between; gap: 12px; font-size: var(--text-ui); line-height: 16px; color: var(--color-text-secondary); }
  .job-step-head > span + span { color: var(--color-text-muted); }
  .job-step-head .num { font-variant-numeric: tabular-nums; color: var(--color-text-primary); }
  .job-step-head .dl-line { min-width: 0; font-variant-numeric: tabular-nums; }
  .job-step-head .dl-pct { flex: none; }
  .dl-seg { white-space: nowrap; }
  .job-step-done { display: inline-flex; align-items: center; gap: 4px; }
  .job-step :global(.job-stage-done) { --progress-color: color-mix(in srgb, var(--color-primary-main) 45%, var(--color-zinc-700)); }
  /* Never more than about half the detail, so a wide preview leaves the text a usable column on a narrow stage. */
  .job-media { margin: 0; flex: none; max-width: 45cqi; align-self: start; overflow: hidden; border-radius: var(--radius-sm); background: var(--color-bg-overlay); }
  .job-media.empty { display: grid; place-items: center; padding: 8px; text-align: center; box-shadow: inset 0 0 0 1px var(--color-border-subtle); }
  .job-media img { display: block; width: 100%; height: 100%; object-fit: cover; }
  /* Reconnecting: the text keeps full contrast; only the stale bars and preview fade. */
  :global([data-tone='stale']) .job-log :global(.ui-progress), :global([data-tone='stale']) .job-media { opacity: 0.4; filter: saturate(0); }

  /* Too narrow for text beside the preview (phone width): stack them, preview first, and let log lines wrap. */
  @container (max-width: 440px) {
    .job-label { flex-wrap: wrap; row-gap: 0; padding: 5px 0; white-space: normal; }
    .job-split { grid-template-columns: minmax(0, 1fr); row-gap: 12px; }
    .job-media { order: -1; max-width: 100%; }
  }
</style>
