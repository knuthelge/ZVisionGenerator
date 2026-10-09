<script lang="ts">
  import { Icon } from '$lib/components/atoms';
  import type { JobSnapshot } from '$lib/types';

  interface Props {
    /** Queued jobs, oldest first. */
    jobs: JobSnapshot[];
    onremove: (job: JobSnapshot) => void;
    onload: (job: JobSnapshot) => void;
    onclear: () => void;
  }

  let { jobs, onremove, onload, onclear }: Props = $props();

  /** Whether the job was submitted from the form, so its settings can be loaded back (an upscale was not). */
  function hasSettings(job: JobSnapshot): boolean {
    return Object.keys(job.settings ?? {}).length > 0;
  }

  function details(job: JobSnapshot): string {
    const runs = `${job.runs} ${job.runs === 1 ? 'run' : 'runs'}`;
    return [job.model, job.workflow, runs, job.meta].filter(Boolean).join(' · ');
  }
</script>

{#if jobs.length > 0}
  <section class="queue-panel" aria-labelledby="ws-queue-title" data-testid="queue-panel">
    <div class="queue-head">
      <h3 id="ws-queue-title" class="ui-area-label queue-title"><Icon name="list" size={14} />Up next · {jobs.length}</h3>
      <button type="button" class="ui-btn ui-btn-sm ui-btn-danger" onclick={onclear}>Clear queue</button>
    </div>
    <ol class="queue-list">
      {#each jobs as job, index (job.job_id ?? job.id)}
        <li class="queue-item" data-job-id={job.job_id ?? job.id}>
          <span class="queue-pos" class:next={index === 0} aria-label="Position {job.queue_position ?? index + 1}">{job.queue_position ?? index + 1}</span>
          <div class="queue-text">
            <span class="queue-prompt" title={job.prompt}>{job.prompt || (job.workflow === 'upscale' ? 'No prompt recorded' : 'Prompt file')}</span>
            <span class="queue-meta">{details(job)}</span>
          </div>
          {#if index === 0}<span class="ui-badge ui-badge-info queue-tag">Starts next</span>{/if}
          {#if hasSettings(job)}
            <button type="button" class="ui-btn ui-btn-sm queue-action" title="Copy this job's settings into the form" onclick={() => onload(job)}>
              <Icon name="reuse" size={14} />Load settings
            </button>
          {/if}
          <button type="button" class="ui-btn ui-btn-row ui-btn-danger queue-remove" aria-label="Remove from queue" title="Remove from queue" onclick={() => onremove(job)}>
            <Icon name="close" size={14} />
          </button>
        </li>
      {/each}
    </ol>
  </section>
{/if}

<style>
  .queue-panel { display: flex; flex-direction: column; gap: 8px; margin-top: 16px; }
  .queue-head { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
  .queue-title { display: flex; align-items: center; gap: 6px; }
  .queue-list { display: flex; flex-direction: column; gap: 6px; margin: 0; padding: 0; list-style: none; }
  .queue-item { display: flex; align-items: center; gap: 10px; padding: 8px 8px 8px 10px; border-radius: var(--radius-md); background: var(--color-bg-surface); }
  .queue-pos { flex: none; display: grid; place-items: center; width: 22px; height: 22px; border-radius: var(--radius-xs); font-family: var(--font-mono); font-size: var(--text-meta); background: var(--color-bg-raised); color: var(--color-text-secondary); }
  .queue-pos.next { background: var(--color-primary-subtle); color: var(--color-primary-hover); }
  .queue-text { display: flex; flex: 1; min-width: 0; flex-direction: column; gap: 2px; }
  .queue-prompt { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: var(--text-content); color: var(--color-text-primary); }
  .queue-meta { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-family: var(--font-mono); font-size: var(--text-meta); color: var(--color-text-muted); }
  @media (max-width: 639px) {
    .queue-tag { display: none; }
    .queue-action { font-size: 0; gap: 0; }
  }
</style>
