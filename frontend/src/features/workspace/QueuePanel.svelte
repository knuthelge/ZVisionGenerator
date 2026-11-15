<script lang="ts">
  import { Icon } from '$lib/components/atoms';
  import { SaveProject } from '$lib/components/molecules';
  import { promptText } from '$lib/state/jobCaption';
  import { hasJobSettings } from '$lib/state/jobSettings';
  import { inputSummary } from '$lib/state/jobInputs';
  import { layout } from '$lib/state/layoutMode.svelte';
  import type { JobSnapshot } from '$lib/types';
  import { queueMetaParts } from './queueView';

  interface Props {
    /** Queued jobs, oldest first. */
    jobs: JobSnapshot[];
    onremove: (job: JobSnapshot) => void;
    onload: (job: JobSnapshot) => void;
    onclear: () => void;
    /** Only the first line shows. The user's own toggle wins for the rest of the job. */
    folded?: boolean;
    /** The user pressed the head: they decide, from now on, whether the queue is folded. */
    ontoggle?: () => void;
    /** `panel` is a surface of its own; `sheet` is a band on the Job sheet's surface. */
    variant?: 'panel' | 'sheet';
  }

  let { jobs, onremove, onload, onclear, folded = $bindable(false), ontoggle, variant = 'panel' }: Props = $props();

  const uid = $props.id();
  const touch = $derived(layout.coarse || layout.compact);
  const toggleLabel = $derived(`Up next · ${jobs.length}`);
</script>

<!-- A position as a solid 22 px circle holding its number, with no outline. The next job's circle is teal-tinted. -->
{#snippet position(n: number, next: boolean)}
  <span class="pos" class:next role="img" aria-label={next ? `Position ${n}, starts next` : `Position ${n}`} title={next ? 'Starts next' : undefined}>{n}</span>
{/snippet}

{#if jobs.length > 0}
  <section class="queue-panel {variant}" class:folded class:touch aria-labelledby="{uid}-title" data-testid="queue-panel">
    <div class="head">
      <!-- The heading holds the button, as an accordion does; the button is named by the title alone, not by the first prompt beside it. -->
      <h3 class="head-title">
      <button type="button" class="toggle" aria-expanded={!folded} aria-controls={folded ? undefined : `${uid}-list`} aria-labelledby="{uid}-title" onclick={() => { folded = !folded; ontoggle?.(); }}>
        <Icon name={folded ? 'chevright' : 'chevdown'} size={14} />
        <span id="{uid}-title" class="ui-area-label title">{toggleLabel}</span>
        {#if folded}
          <span class="peek">
            <span class="stack" aria-hidden="true">{#each jobs.slice(0, 3) as job, index (job.job_id ?? job.id)}{@render position(job.queue_position ?? index + 1, index === 0)}{/each}</span>
            <span class="peek-text">{promptText(jobs[0])}</span>
          </span>
        {/if}
      </button>
      </h3>
      {#if !folded}<button type="button" class="ui-btn ui-btn-sm ui-btn-danger" onclick={onclear}>Clear queue</button>{/if}
    </div>
    {#if !folded}
      <ol class="list" id="{uid}-list">
        {#each jobs as job, index (job.job_id ?? job.id)}
          {@const upscale = job.workflow === 'upscale' && (job.inputs?.length ?? 0) > 0}
          {@const files = upscale ? null : inputSummary(job.inputs ?? [])}
          <li class="row queue-item" data-job-id={job.job_id ?? job.id}>
            {@render position(job.queue_position ?? index + 1, index === 0)}
            <div class="text">
              <p class="prompt" data-testid="queue-prompt" title={upscale ? job.prompt : promptText(job)}>{promptText(job)}</p>
              <p class="meta" data-testid="queue-meta">{#each queueMetaParts(job) as part, k (k)}<span class:kind={k === 0}>{part}</span>{/each}</p>
              {#if files || job.save_project}
                <p class="files" data-testid="queue-files">
                  {#if files}
                    <span class="file" title={files.title} data-testid="queue-input">
                      <Icon name="file" size={12} />
                      <span class="file-name">{files.name}</span>{#if files.more > 0}<span class="more">+{files.more}</span>{/if}
                    </span>
                  {/if}
                  {#if job.save_project}<SaveProject name={job.save_project} testid="queue-project" />{/if}
                </p>
              {/if}
            </div>
            <div class="actions">
              {#if hasJobSettings(job.settings)}
                <button type="button" class="ui-btn ui-btn-row queue-action" aria-label="Load settings" title="Load settings: copy this job's settings into the form" onclick={() => onload(job)}><Icon name="reuse" size={14} /></button>
              {/if}
              <button type="button" class="ui-btn ui-btn-row ui-btn-danger queue-remove" aria-label="Remove from queue" title="Remove from queue" onclick={() => onremove(job)}><Icon name="close" size={14} /></button>
            </div>
          </li>
        {/each}
      </ol>
    {/if}
  </section>
{/if}

<style>
  .queue-panel { flex: none; min-width: 0; border-radius: var(--radius-md); background: var(--color-bg-surface); }
  /* In the Job sheet the queue is a band split from the card by a line. */
  .queue-panel.sheet { border-radius: 0; background: transparent; border-top: 1px solid var(--color-border-subtle); }
  .head { display: flex; align-items: center; gap: 8px; min-height: 36px; padding: 4px 8px 4px 6px; }
  .touch .head { min-height: 44px; }
  .queue-panel:not(.folded) .head { border-bottom: 1px solid var(--color-border-subtle); }
  .toggle { display: flex; flex: 1; align-items: center; gap: 6px; min-width: 0; min-height: 28px; padding: 0 6px; border-radius: var(--radius-sm); color: var(--color-text-muted); text-align: left; }
  .touch .toggle { min-height: 40px; }
  @media (hover: hover) { .toggle:hover { color: var(--color-text-secondary); } }
  .head-title { display: flex; flex: 1; min-width: 0; margin: 0; font: inherit; }
  .title { flex: none; white-space: nowrap; }
  .peek { display: flex; align-items: center; gap: 10px; min-width: 0; margin-left: 6px; }
  .stack { display: flex; flex: none; }
  .stack > :global(* + *) { margin-left: -6px; }
  /* Overlapping markers are cut from each other by the panel's own shade, not by a line. */
  .stack .pos { box-shadow: 0 0 0 2px var(--color-bg-surface); }
  .peek-text { min-width: 0; overflow: hidden; font-size: var(--text-ui); color: var(--color-text-secondary); white-space: nowrap; text-overflow: ellipsis; }

  .list { margin: 0; padding: 0 8px 0 12px; list-style: none; }
  .sheet .list { padding: 0 12px 0 16px; }
  .row { display: flex; align-items: flex-start; gap: 10px; padding: 9px 0; }
  .row + .row { border-top: 1px solid var(--color-border-subtle); }
  .pos { display: grid; flex: none; place-items: center; width: 22px; height: 22px; margin-top: 1px; border-radius: 50%; background: var(--color-bg-raised); font-family: var(--font-mono); font-size: var(--text-meta); line-height: 1; color: var(--color-text-secondary); }
  .pos.next { background: var(--color-primary-subtle); color: var(--color-primary-hover); }
  .text { min-width: 0; flex: 1; }
  .prompt { overflow: hidden; font-size: var(--text-content); line-height: 18px; color: var(--color-text-primary); white-space: nowrap; text-overflow: ellipsis; }
  .meta { margin-top: 2px; overflow: hidden; text-overflow: ellipsis; font-size: var(--text-meta); line-height: 15px; color: var(--color-text-muted); white-space: nowrap; }
  .meta > span + span::before { content: '·'; margin: 0 5px; }
  .kind { color: var(--color-text-secondary); }
  .files { display: flex; align-items: center; gap: 12px; min-width: 0; margin-top: 2px; font-size: var(--text-meta); line-height: 16px; color: var(--color-text-muted); }
  .file { display: inline-flex; align-items: center; gap: 4px; min-width: 0; overflow: hidden; white-space: nowrap; }
  .file :global(svg) { flex: none; }
  .file-name { min-width: 0; overflow: hidden; text-overflow: ellipsis; }
  .more { flex: none; font-family: var(--font-mono); }
  .actions { display: flex; flex: none; gap: 2px; margin-top: -1px; }
  .touch .actions { margin-top: -6px; }
</style>
