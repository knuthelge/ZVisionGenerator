<script lang="ts">
  import type { Snippet } from 'svelte';
  import Label from '../atoms/Label.svelte';

  interface Props {
    label?: string;
    for?: string;
    required?: boolean;
    helper?: string;
    error?: string | null;
    status?: string | null;
    statusTone?: 'muted' | 'success' | 'warning' | 'error';
    feedbackId?: string;
    announceFeedback?: boolean;
    class?: string;
    children?: Snippet;
  }

  let {
    label,
    for: htmlFor,
    required = false,
    helper,
    error = null,
    status = null,
    statusTone = 'muted',
    feedbackId,
    announceFeedback = false,
    class: extraClass = '',
    children
  }: Props = $props();

  const statusClass = $derived(
    statusTone === 'success'
      ? 'ui-help-success'
      : statusTone === 'warning'
        ? 'ui-help-warning'
        : statusTone === 'error'
          ? 'ui-help-error'
          : ''
  );
  const feedbackText = $derived(error || status || helper || null);
  const feedbackClass = $derived(error ? 'ui-help-error' : status ? statusClass : '');
</script>

<div class="flex min-w-0 flex-col gap-1.5 {extraClass}">
  {#if label}
    <Label for={htmlFor} {required}>{label}</Label>
  {/if}
  {@render children?.()}
  {#if feedbackText}
    <p
      id={feedbackId}
      class="ui-help {feedbackClass}"
      role={announceFeedback ? (error ? 'alert' : 'status') : undefined}
      aria-live={announceFeedback ? (error ? 'assertive' : 'polite') : undefined}
      aria-atomic={announceFeedback ? 'true' : undefined}
    >{feedbackText}</p>
  {/if}
</div>
