<script lang="ts">
  import { onMount } from 'svelte';
  import { ERROR_YIELD_MS, type ToastItem } from '$lib/state/toastQueue';

  interface Props {
    toast: ToastItem;
    /** Toasts waiting behind this one, shown as "+N". */
    pending?: number;
    ondismiss?: (id: string) => void;
  }

  let { toast, pending = 0, ondismiss }: Props = $props();

  let el: HTMLDivElement;
  let paused = $state(false);
  // A toast that appears under a resting pointer gets no mouseenter, so check on mount.
  onMount(() => { if (el.matches(':hover')) paused = true; });

  function dismiss(): void {
    ondismiss?.(toast.id);
  }

  function runAction(): void {
    toast.action?.run();
    dismiss();
  }

  // A toast that stays until dismissed still gives way after a while once others are waiting behind it.
  const timeout = $derived(toast.timeout > 0 ? toast.timeout : pending > 0 ? ERROR_YIELD_MS : 0);

  // Count down while the toast is on screen; a merged repeat restarts it, and hover or focus pauses it.
  $effect(() => {
    void toast.count;
    if (timeout <= 0 || paused) return;
    const timer = setTimeout(dismiss, timeout);
    return () => clearTimeout(timer);
  });
</script>

<!-- The container is the polite live region; an error is also an alert so it is announced at once. -->
<div
  bind:this={el}
  role={toast.type === 'error' ? 'alert' : undefined}
  data-tone={toast.type}
  class="toast ui-overlay pointer-events-auto flex items-center gap-2.5 py-1.5 pr-1.5 pl-3"
  onmouseenter={() => (paused = true)}
  onmouseleave={() => (paused = false)}
  onfocusin={() => (paused = true)}
  onfocusout={() => (paused = false)}
>
  <div class="toast-icon shrink-0">
    {#if toast.type === 'success'}
      <svg class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2" aria-hidden="true">
        <path stroke-linecap="round" stroke-linejoin="round" d="M5 13l4 4L19 7" />
      </svg>
    {:else if toast.type === 'error'}
      <svg class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2" aria-hidden="true">
        <path stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12" />
      </svg>
    {:else if toast.type === 'warning'}
      <svg class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2" aria-hidden="true">
        <path stroke-linecap="round" stroke-linejoin="round" d="M12 9v4m0 4h.01M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" />
      </svg>
    {:else}
      <svg class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2" aria-hidden="true">
        <path stroke-linecap="round" stroke-linejoin="round" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
      </svg>
    {/if}
  </div>
  <p class="flex-1 py-1 text-ui leading-snug">
    {toast.message}
    {#if toast.count > 1}<span class="ml-1 font-mono text-meta text-text-muted" data-testid="toast-count">×{toast.count}</span>{/if}
  </p>
  {#if pending > 0}
    <span class="toast-pending shrink-0 font-mono text-meta text-text-muted" data-testid="toast-pending" title="{pending} more waiting">+{pending}</span>
  {/if}
  {#if toast.action}
    <button type="button" class="toast-action shrink-0" onclick={runAction}>{toast.action.label}</button>
  {/if}
  <button
    type="button"
    onclick={dismiss}
    class="toast-dismiss shrink-0 text-text-muted"
    aria-label="Dismiss notification"
  >
    <svg class="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2" aria-hidden="true">
      <path stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12" />
    </svg>
  </button>
</div>

<style>
  /* The tone tints the surface and its ring, so success and error read apart at a glance. Info stays teal. */
  .toast { --toast-tone: var(--color-primary-main); min-height: var(--spacing-bar); max-width: min(520px, calc(100vw - 32px)); background: color-mix(in srgb, var(--toast-tone) 12%, var(--color-bg-overlay)); box-shadow: 0 0 0 1px color-mix(in srgb, var(--toast-tone) 35%, transparent); }
  .toast[data-tone='success'] { --toast-tone: var(--color-success); }
  .toast[data-tone='warning'] { --toast-tone: var(--color-warning); }
  .toast[data-tone='error'] { --toast-tone: var(--color-error); }
  .toast-icon { color: var(--toast-tone); }
  .toast-pending { padding: 1px 6px; border-radius: 999px; background: var(--color-bg-surface-hover); }
  .toast-action { height: var(--spacing-control-sm); padding: 0 8px; border-radius: var(--radius-sm); font-size: var(--text-ui); font-weight: 700; color: var(--toast-tone); }
  .toast-action:hover { background: color-mix(in srgb, var(--toast-tone) 18%, transparent); }
  .toast-dismiss { display: grid; place-items: center; width: var(--spacing-control-sm); height: var(--spacing-control-sm); border-radius: var(--radius-sm); transition: color 0.12s ease; }
  .toast-dismiss:hover { color: var(--color-text-primary); }
</style>
