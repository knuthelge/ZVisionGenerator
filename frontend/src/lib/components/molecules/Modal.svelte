<script lang="ts">
  import type { Snippet } from 'svelte';
  import { onMount, onDestroy } from 'svelte';

  interface Props {
    open?: boolean;
    title?: string;
    onclose?: () => void;
    children?: Snippet;
    footer?: Snippet;
  }

  let {
    open = $bindable(false),
    title,
    onclose,
    children,
    footer
  }: Props = $props();

  let dialogEl = $state<HTMLDivElement | null>(null);
  let previouslyFocused: Element | null = null;

  function close(): void {
    open = false;
    onclose?.();
  }

  function handleKeydown(e: KeyboardEvent): void {
    if (e.key === 'Escape') {
      e.preventDefault();
      close();
    }
  }

  $effect(() => {
    if (open) {
      previouslyFocused = document.activeElement;
      dialogEl?.focus();
    } else {
      (previouslyFocused as HTMLElement | null)?.focus();
    }
  });

  onMount(() => {
    document.addEventListener('keydown', handleKeydown);
    return () => document.removeEventListener('keydown', handleKeydown);
  });
</script>

{#if open}
  <!-- Outer container: stacking context -->
  <div class="fixed inset-0 z-50">
    <!-- Backdrop (native button so click-to-dismiss requires no role suppression) -->
    <button
      type="button"
      class="absolute inset-0 bg-black/70 backdrop-blur-sm"
      onclick={close}
      aria-label="Close dialog"
      tabindex="-1"
    ></button>
    <!-- Flex centering layer -->
    <div class="flex items-center justify-center h-full p-4 pointer-events-none">
      <!-- Dialog -->
      <div
        bind:this={dialogEl}
        role="dialog"
        aria-modal="true"
        aria-label={title}
        tabindex="-1"
        class="relative w-full max-w-lg rounded-xl border border-zinc-700 bg-zinc-900 shadow-2xl focus:outline-none pointer-events-auto"
      >
      {#if title}
        <div class="flex items-center justify-between px-6 py-4 border-b border-zinc-800">
          <h2 class="text-base font-semibold text-zinc-100">{title}</h2>
          <button
            type="button"
            onclick={close}
            class="rounded-md p-1 text-zinc-500 hover:text-zinc-100 hover:bg-zinc-800 transition-colors focus-visible:focus-ring"
            aria-label="Close dialog"
          >
            <svg class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2" aria-hidden="true">
              <path stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
      {/if}
      <div class="px-6 py-4">
        {@render children?.()}
      </div>
      {#if footer}
        <div class="flex items-center justify-end gap-3 px-6 py-4 border-t border-zinc-800">
          {@render footer()}
        </div>
      {/if}
      </div>
    </div>
  </div>
{/if}
