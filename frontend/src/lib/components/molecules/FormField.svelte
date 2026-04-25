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
</script>

<div class="flex flex-col gap-1.5 {extraClass}">
  {#if label}
    <Label for={htmlFor} {required}>{label}</Label>
  {/if}
  {@render children?.()}
  {#if error}
    <p class="text-xs text-red-400">{error}</p>
  {:else if status}
    <p class="text-xs {statusClass}">{status}</p>
  {:else if helper}
    <p class="text-xs text-zinc-500">{helper}</p>
  {/if}
</div>
