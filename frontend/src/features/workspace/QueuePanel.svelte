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
      <h3 id="ws-queue-title" class="field-label queue-title"><Icon name="list" size={14} />Up next · {jobs.length}</h3>
      <button type="button" class="queue-clear" onclick={onclear}>Clear queue</button>
    </div>
    <ol class="queue-list">
      {#each jobs as job, index (job.job_id ?? job.id)}
        <li class="queue-item" data-job-id={job.job_id ?? job.id}>
          <span class="queue-pos" class:next={index === 0} aria-label="Position {job.queue_position ?? index + 1}">{job.queue_position ?? index + 1}</span>
          <div class="queue-text">
            <span class="queue-prompt" title={job.prompt}>{job.prompt || (job.workflow === 'upscale' ? 'No prompt recorded' : 'Prompt file')}</span>
            <span class="queue-meta">{details(job)}</span>
          </div>
          {#if index === 0}<span class="queue-tag">Starts next</span>{/if}
          {#if hasSettings(job)}
            <button type="button" class="queue-action" title="Copy this job's settings into the form" onclick={() => onload(job)}>
              <Icon name="reuse" size={14} />Load settings
            </button>
          {/if}
          <button type="button" class="queue-remove" aria-label="Remove from queue" title="Remove from queue" onclick={() => onremove(job)}>
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
  .queue-clear { padding: 4px 8px; border-radius: 6px; font-size: 12px; color: var(--color-text-secondary); }
  .queue-clear:hover { background: var(--color-bg-surface-hover); color: var(--color-text-primary); }
  .queue-list { display: flex; flex-direction: column; gap: 6px; margin: 0; padding: 0; list-style: none; }
  .queue-item { display: flex; align-items: center; gap: 10px; padding: 8px 8px 8px 10px; border: 1px solid var(--color-border-subtle); border-radius: 8px; background: var(--color-bg-surface); }
  .queue-pos { flex: none; display: grid; place-items: center; width: 22px; height: 22px; border-radius: 999px; font-family: var(--font-mono); font-size: 11px; background: var(--color-zinc-800); color: var(--color-text-secondary); }
  .queue-pos.next { background: var(--color-primary-subtle); color: var(--color-primary-hover); }
  .queue-text { display: flex; flex: 1; min-width: 0; flex-direction: column; gap: 2px; }
  .queue-prompt { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 13px; color: var(--color-text-primary); }
  .queue-meta { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-family: var(--font-mono); font-size: 11px; color: var(--color-text-muted); }
  .queue-tag { flex: none; padding: 1px 6px; border-radius: 999px; font-size: 11px; white-space: nowrap; background: var(--color-primary-subtle); color: var(--color-primary-hover); }
  .queue-action { flex: none; display: inline-flex; align-items: center; gap: 6px; padding: 5px 8px; border-radius: 6px; font-size: 12px; color: var(--color-text-secondary); }
  .queue-action:hover, .queue-remove:hover { background: var(--color-bg-surface-hover); color: var(--color-text-primary); }
  .queue-remove { flex: none; display: grid; place-items: center; width: 32px; height: 32px; border-radius: 6px; color: var(--color-text-secondary); }
  @media (max-width: 639px) {
    .queue-tag { display: none; }
    .queue-action { font-size: 0; gap: 0; }
  }
</style>
