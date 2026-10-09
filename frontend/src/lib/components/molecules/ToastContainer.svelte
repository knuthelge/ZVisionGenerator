<script lang="ts">
  import { fly } from 'svelte/transition';
  import Toast from './Toast.svelte';
  import { toasts, dismissToast } from '$lib/state/toasts.svelte';

  const current = $derived(toasts[0]);
</script>

<!-- One toast at a time, bottom centre of the page content; pages with a fixed left sidebar set --toast-inset-left. -->
<div class="toast-zone pointer-events-none fixed z-[150] grid justify-items-center px-4" role="region" aria-label="Notifications" aria-live="polite">
  {#if current}
    {#key current.id}
      <div class="toast-slot pointer-events-none flex max-w-full justify-center" transition:fly={{ y: 8, duration: 150 }}>
        <Toast toast={current} pending={toasts.length - 1} ondismiss={dismissToast} />
      </div>
    {/key}
  {/if}
</div>

<style>
  .toast-zone { left: var(--toast-inset-left, 0px); right: 0; bottom: 16px; }
  /* The outgoing and incoming toast share one cell, so a swap cross-fades in place instead of sitting side by side. */
  .toast-slot { grid-area: 1 / 1; }
</style>
