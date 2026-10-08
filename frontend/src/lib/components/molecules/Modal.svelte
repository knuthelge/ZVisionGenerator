<script lang="ts">
  import type { Snippet } from 'svelte';
  import Icon from '../atoms/Icon.svelte';

  interface Props {
    open?: boolean;
    title?: string;
    /** `lg` widens the dialog for lists and editors; `sm` fits a short question. */
    size?: 'sm' | 'md' | 'lg';
    /** `alertdialog` for a dialog that asks the user to approve an action. */
    role?: 'dialog' | 'alertdialog';
    /** Id of the element that names the dialog, used instead of `title`. */
    labelledby?: string;
    /** Id of the element that describes the dialog. */
    describedby?: string;
    /** Stack above full-screen overlays such as the asset viewer. */
    elevated?: boolean;
    onclose?: () => void;
    children?: Snippet;
    footer?: Snippet;
  }

  let {
    open = $bindable(false),
    title,
    size = 'md',
    role = 'dialog',
    labelledby,
    describedby,
    elevated = false,
    onclose,
    children,
    footer
  }: Props = $props();

  const SIZE_CLASSES = { sm: 'max-w-md', md: 'max-w-lg', lg: 'max-w-3xl' } as const;

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
  <div class="fixed inset-0 {elevated ? 'z-[200]' : 'z-50'}">
    <!-- Backdrop (native button so click-to-dismiss requires no role suppression) -->
    <button
      type="button"
      class="absolute inset-0 bg-scrim"
      onclick={close}
      aria-label="Close dialog"
      tabindex="-1"
    ></button>
    <!-- Flex centering layer -->
    <div class="flex items-center justify-center h-full p-4 pointer-events-none">
      <!-- Dialog -->
      <div
        bind:this={dialogEl}
        {role}
        aria-modal="true"
        aria-label={labelledby ? undefined : title}
        aria-labelledby={labelledby}
        aria-describedby={describedby}
        tabindex="-1"
        class="ui-overlay relative flex max-h-full w-full {SIZE_CLASSES[size]} flex-col focus:outline-none pointer-events-auto"
      >
      {#if title}
        <div class="flex items-center justify-between gap-3 border-b border-border-subtle px-4 py-2.5">
          <h2 class="font-heading text-content font-extrabold text-text-primary">{title}</h2>
          <button
            type="button"
            onclick={close}
            class="ui-btn ui-btn-quiet ui-btn-icon ui-btn-sm"
            aria-label="Close dialog"
          >
            <Icon name="close" size={14} />
          </button>
        </div>
      {/if}
      <div class="min-h-0 overflow-y-auto px-4 py-3 text-content text-text-secondary">
        {@render children?.()}
      </div>
      {#if footer}
        <div class="flex flex-wrap items-center justify-end gap-2 border-t border-border-subtle px-4 py-2.5">
          {@render footer()}
        </div>
      {/if}
      </div>
    </div>
  </div>
{/if}
