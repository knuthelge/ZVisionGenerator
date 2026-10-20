<script lang="ts">
  import { untrack } from 'svelte';
  import { Button, Spinner } from '$lib/components/atoms';
  import Alert from '$lib/components/molecules/Alert.svelte';
  import EmptyState from '$lib/components/molecules/EmptyState.svelte';
  import Modal from '$lib/components/molecules/Modal.svelte';
  import Segmented from '$lib/components/molecules/Segmented.svelte';
  import { errorText, isRetryable } from '$lib/api/client';
  import { getJobLog } from '$lib/api/jobLog';
  import type { JobLogFilter, JobLogPage } from '$lib/types';
  import JobLogRow from './JobLogRow.svelte';

  interface Props {
    open?: boolean;
  }

  let { open = $bindable(false) }: Props = $props();

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
</script>

<Modal bind:open title="Job log" placement="side" size="md" fill>
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
          <JobLogRow {entry} {now} />
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
