<script lang="ts">
  import type { Snippet } from 'svelte';

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

  const FOCUSABLE = [
    'button:not([disabled])',
    '[href]',
    'input:not([disabled])',
    'select:not([disabled])',
    'textarea:not([disabled])',
    '[tabindex]:not([tabindex="-1"])',
  ].join(', ');

  $effect(() => {
    if (open) {
      previouslyFocused = document.activeElement;
      dialogEl?.focus();
    } else {
      (previouslyFocused as HTMLElement | null)?.focus();
    }
  });

  // Registers keyboard and focus-escape listeners whenever the modal is open.
  // Uses $effect so that `el` is captured from the live $state value at effect
  // run time, avoiding any stale-closure issue that arises when onMount captures
  // $state signals before the dialog element exists in the DOM.
  $effect(() => {
    if (!open) return;

    // Capture the current element reference for this effect run.
    const el = dialogEl;

    function handleKeydown(e: KeyboardEvent): void {
      if (e.key === 'Escape') {
        e.preventDefault();
        close();
        return;
      }

      if (e.key === 'Tab' && el) {
        const focusable = Array.from(el.querySelectorAll<HTMLElement>(FOCUSABLE));
        if (focusable.length === 0) {
          e.preventDefault();
          return;
        }
        const first = focusable[0];
        const last = focusable[focusable.length - 1];
        if (e.shiftKey) {
          if (document.activeElement === first || document.activeElement === el) {
            e.preventDefault();
            last.focus();
          }
        } else {
          if (document.activeElement === last || document.activeElement === el) {
            e.preventDefault();
            first.focus();
          }
        }
      }
    }

    // Safety net: if focus escapes the dialog in a real browser (e.g. when the
    // browser's native tab order bypasses the keydown boundary check), the
    // capture-phase focusin listener detects the leak and immediately returns
    // focus to the first focusable element inside the dialog.
    function handleFocusIn(e: FocusEvent): void {
      if (!el || el.contains(e.target as Node)) return;
      const focusable = Array.from(el.querySelectorAll<HTMLElement>(FOCUSABLE));
      (focusable[0] ?? el).focus();
    }

    document.addEventListener('keydown', handleKeydown);
    document.addEventListener('focusin', handleFocusIn, true);
    return () => {
      document.removeEventListener('keydown', handleKeydown);
      document.removeEventListener('focusin', handleFocusIn, true);
    };
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
