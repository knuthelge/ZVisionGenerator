<script lang="ts">
  import { untrack } from 'svelte';
  import { Button, Spinner } from '$lib/components/atoms';
  import Alert from '$lib/components/molecules/Alert.svelte';
  import EmptyState from '$lib/components/molecules/EmptyState.svelte';
  import Modal from '$lib/components/molecules/Modal.svelte';
  import Segmented from '$lib/components/molecules/Segmented.svelte';
  import { requestConfirm } from '$lib/components/molecules/confirm.svelte';
  import { ApiError, errorText, isRetryable } from '$lib/api/client';
  import { clearJobLog, deleteJobLogEntry, getJobLog } from '$lib/api/jobLog';
  import { addToast } from '$lib/state/toasts.svelte';
  import type { JobLogEntry, JobLogFilter, JobLogPage } from '$lib/types';
  import JobLogRow from './JobLogRow.svelte';

  interface Props {
    open?: boolean;
    /** Refill the Workspace form from an entry's settings; the owner closes the drawer. */
    onloadsettings?: (entry: JobLogEntry) => void;
  }

  let { open = $bindable(false), onloadsettings }: Props = $props();

  const FILTERS: { value: JobLogFilter; label: string }[] = [
    { value: 'all', label: 'All' },
    { value: 'failed', label: 'Failed' },
    { value: 'completed', label: 'Completed' }
  ];
  const EMPTY_TITLES: Record<JobLogFilter, string> = {
    all: 'No finished jobs yet',
    failed: 'No failed jobs',
    completed: 'No completed jobs'
  };

  let filter = $state<JobLogFilter>('all');
  let page = $state(1);
  let data = $state<JobLogPage | null>(null);
  let error = $state<unknown>(null);
  let loading = $state(false);
  // The moment the rows' relative times are counted from: when their page arrived.
  let now = $state(Date.now());
  // Only the newest request may fill the list, so a slow earlier page cannot overwrite a later one.
  let latestRequest = 0;

  async function load(): Promise<void> {
    const request = ++latestRequest;
    loading = true;
    error = null;
    try {
      const result = await getJobLog(page, filter);
      if (request !== latestRequest) return;
      data = result;
      page = result.page;
      now = Date.now();
    } catch (err) {
      if (request !== latestRequest) return;
      data = null;
      error = err;
    } finally {
      if (request === latestRequest) loading = false;
    }
  }

  // Each opening starts at the newest entries of every outcome.
  $effect(() => {
    if (!open) {
      latestRequest++;
      return;
    }
    untrack(() => {
      filter = 'all';
      page = 1;
      data = null;
      void load();
    });
  });

  function setFilter(next: JobLogFilter): void {
    filter = next;
    page = 1;
    void load();
  }

  function goToPage(next: number): void {
    page = next;
    void load();
  }

  async function deleteEntry(entry: JobLogEntry): Promise<void> {
    try {
      await deleteJobLogEntry(entry.job_id);
    } catch (err) {
      addToast(errorText(err, 'Could not delete the job from the log.'), 'error');
      // Gone already (another tab): the list is stale, so show the real one.
      if (err instanceof ApiError && err.status === 404) await load();
      return;
    }
    // The server clamps a page that deletes emptied.
    await load();
  }

  async function clearLog(): Promise<void> {
    const count = data?.total_count ?? 0;
    const approved = await requestConfirm({
      question: 'Clear the whole job log?',
      info: `All ${count} ${count === 1 ? 'entry is' : 'entries are'} removed, not just the ones shown.`,
      confirmLabel: 'Clear log',
      cancelLabel: 'Keep',
      danger: true
    });
    if (!approved) return;
    try {
      await clearJobLog();
    } catch (err) {
      addToast(errorText(err, 'Could not clear the job log.'), 'error');
      return;
    }
    page = 1;
    await load();
    addToast('Job log cleared', 'success');
  }
</script>

<Modal bind:open title="Job log" placement="side" size="md" fill>
  {#snippet footer()}
    <Button
      size="sm"
      variant="danger"
      disabled={filter !== 'all' || !data || data.total_count === 0}
      title={filter === 'all' ? 'Remove every entry from the job log' : 'Switch to All to clear the whole log'}
      onclick={clearLog}
    >Clear log</Button>
  {/snippet}
  <div class="flex shrink-0 items-center border-b border-border-subtle px-4 py-2">
    <Segmented options={FILTERS} value={filter} label="Show jobs" size="sm" testId="job-log-filter" onchange={setFilter} />
  </div>
  <div class="min-h-0 flex-1 overflow-y-auto" data-testid="job-log-list">
    {#if error}
      <div class="flex flex-col items-start gap-2 p-4">
        <Alert tone="error" live>{errorText(error, 'Could not load the job log.')}</Alert>
        {#if isRetryable(error)}<Button size="sm" onclick={load}>Retry</Button>{/if}
      </div>
    {:else if !data}
      <div class="flex justify-center p-6"><Spinner size="lg" label="Loading the job log" /></div>
    {:else if data.entries.length === 0}
      <EmptyState title={EMPTY_TITLES[filter]}>
        {#if filter === 'all'}Jobs that finish, fail or are cancelled are recorded here.{/if}
      </EmptyState>
    {:else}
      <ul aria-label="Finished jobs" aria-busy={loading}>
        {#each data.entries as entry (entry.job_id)}
          <JobLogRow {entry} {now} onloadsettings={(picked) => onloadsettings?.(picked)} ondelete={deleteEntry} />
        {/each}
      </ul>
    {/if}
  </div>
  {#if data && data.total_pages > 1}
    <nav class="flex shrink-0 items-center justify-between gap-2 border-t border-border-subtle px-4 py-2" aria-label="Job log pages">
      <Button size="sm" disabled={loading || data.page <= 1} onclick={() => goToPage(data!.page - 1)}>Newer</Button>
      <span class="text-meta text-text-muted">Page {data.page} of {data.total_pages}</span>
      <Button size="sm" disabled={loading || data.page >= data.total_pages} onclick={() => goToPage(data!.page + 1)}>Older</Button>
    </nav>
  {/if}
</Modal>
