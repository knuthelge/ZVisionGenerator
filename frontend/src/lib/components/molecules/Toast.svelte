<script lang="ts">
  import { onMount } from 'svelte';

  interface Props {
    id: string;
    type?: 'info' | 'success' | 'error' | 'warning';
    message: string;
    timeout?: number;
    ondismiss?: (id: string) => void;
  }

  let {
    id,
    type = 'info',
    message,
    timeout = 5000,
    ondismiss
  }: Props = $props();

  const styles: Record<string, string> = {
    info:    'bg-zinc-800 border-zinc-700 text-zinc-100',
    success: 'bg-emerald-900/40 border-emerald-700/50 text-emerald-100',
    error:   'bg-red-900/40 border-red-700/50 text-red-100',
    warning: 'bg-amber-900/40 border-amber-700/50 text-amber-100'
  };

  const iconColor: Record<string, string> = {
    info:    'text-zinc-400',
    success: 'text-emerald-400',
    error:   'text-red-400',
    warning: 'text-amber-400'
  };

  let visible = $state(true);
  let timer: ReturnType<typeof setTimeout> | null = null;

  function dismiss(): void {
    visible = false;
    ondismiss?.(id);
  }

  onMount(() => {
    if (timeout > 0) {
      timer = setTimeout(dismiss, timeout);
    }
    return () => {
      if (timer) clearTimeout(timer);
    };
  });
</script>

{#if visible}
  <div
    role="alert"
    aria-live="polite"
    class="pointer-events-auto flex items-start gap-3 rounded-lg border px-4 py-3 shadow-lg max-w-sm {styles[type]}"
  >
    <!-- Icon -->
    <div class="mt-0.5 shrink-0 {iconColor[type]}">
      {#if type === 'success'}
        <svg class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2" aria-hidden="true">
          <path stroke-linecap="round" stroke-linejoin="round" d="M5 13l4 4L19 7" />
        </svg>
      {:else if type === 'error'}
        <svg class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2" aria-hidden="true">
          <path stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12" />
        </svg>
      {:else if type === 'warning'}
        <svg class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2" aria-hidden="true">
          <path stroke-linecap="round" stroke-linejoin="round" d="M12 9v4m0 4h.01M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" />
        </svg>
      {:else}
        <svg class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2" aria-hidden="true">
          <path stroke-linecap="round" stroke-linejoin="round" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
      {/if}
    </div>
    <!-- Message -->
    <p class="flex-1 text-sm">{message}</p>
    <!-- Dismiss -->
    <button
      type="button"
      onclick={dismiss}
      class="shrink-0 rounded p-0.5 opacity-60 hover:opacity-100 transition-opacity focus-visible:focus-ring"
      aria-label="Dismiss notification"
    >
      <svg class="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2" aria-hidden="true">
        <path stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12" />
      </svg>
    </button>
  </div>
{/if}
