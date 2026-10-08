<script lang="ts">
  import type { Snippet } from 'svelte';
  import { Spinner } from '$lib/components/atoms';
  import { Alert, PageHeader } from '$lib/components/molecules';

  interface Props {
    title: string;
    description?: string;
    loading?: boolean;
    error?: string | null;
    /** Page-level controls in the page bar, e.g. Save. */
    actions?: Snippet;
    class?: string;
    children?: Snippet;
  }

  let {
    title,
    description,
    loading = false,
    error = null,
    actions,
    class: extraClass = '',
    children
  }: Props = $props();
</script>

<main class="flex min-h-0 flex-1 flex-col bg-bg-base {extraClass}">
  <PageHeader {title} {description} actions={loading || error ? undefined : actions} />
  <div class="custom-scrollbar min-h-0 flex-1 overflow-y-auto">
    <div class="p-4">
      {#if loading}
        <p class="flex items-center justify-center gap-2 py-24 text-ui text-text-muted" role="status"><Spinner />Loading…</p>
      {:else if error}
        <Alert tone="error" live>{error}</Alert>
      {:else}
        {@render children?.()}
      {/if}
    </div>
  </div>
</main>
