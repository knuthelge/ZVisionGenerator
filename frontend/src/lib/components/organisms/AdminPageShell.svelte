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

{#snippet shellActions()}
  <!-- The actions are in the frame from the first paint (the bar keeps its height) and wake once the page has loaded. -->
  <fieldset class="contents" disabled={loading || error !== null} data-testid="page-actions">{@render actions?.()}</fieldset>
{/snippet}

<main class="flex min-h-0 flex-1 flex-col bg-bg-base {extraClass}">
  <PageHeader {title} {description} contained actions={actions ? shellActions : undefined} />
  <!-- The padding sits outside the centred block, so content lines up with the page bar's title. -->
  <div class="min-h-0 flex-1 overflow-y-auto p-4">
    <div class="mx-auto max-w-page">
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
