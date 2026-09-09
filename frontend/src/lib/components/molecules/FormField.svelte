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
      ? 'text-emerald-400'
      : statusTone === 'warning'
        ? 'text-amber-400'
        : statusTone === 'error'
          ? 'text-red-400'
          : 'text-zinc-500'
  );
  const feedbackText = $derived(error || status || helper || null);
  const feedbackClass = $derived(error ? 'text-red-400' : status ? statusClass : 'text-zinc-500');
</script>

<div class="flex flex-col gap-1.5 {extraClass}">
  {#if label}
    <Label for={htmlFor} {required}>{label}</Label>
  {/if}
  {@render children?.()}
  {#if feedbackText}
    <p
      id={feedbackId}
      class="text-xs {feedbackClass}"
      role={announceFeedback ? (error ? 'alert' : 'status') : undefined}
      aria-live={announceFeedback ? (error ? 'assertive' : 'polite') : undefined}
      aria-atomic={announceFeedback ? 'true' : undefined}
    >{feedbackText}</p>
  {/if}
</div>
