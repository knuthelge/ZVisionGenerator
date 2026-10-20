<script lang="ts" module>
  // Open modals, innermost last: only the innermost one answers Escape, Tab and focus leaks,
  // so a dialog opened from another dialog (a path browser over New file) closes alone.
  const openStack: symbol[] = [];
</script>

<script lang="ts">
  import type { Snippet } from 'svelte';
  import Icon from '../atoms/Icon.svelte';
  import { portal } from '$lib/actions/portal';
  import { pushEscapeLayer } from '$lib/actions/escapeStack';
  import { layout } from '$lib/state/layoutMode.svelte';
  import Sheet from './Sheet.svelte';

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
    /** Fill the screen below 640 px and lay the body out edge to edge, for browsers and editors that scroll inside. */
    fill?: boolean;
    /** `side` docks the dialog to the right edge at full height, as a drawer; a phone still gets its bottom sheet. */
    placement?: 'center' | 'side';
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
    fill = false,
    placement = 'center',
    onclose,
    children,
    footer
  }: Props = $props();

  const SIZE_CLASSES = { sm: 'max-w-md', md: 'max-w-lg', lg: 'max-w-3xl' } as const;

  const side = $derived(placement === 'side');

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

  // On a phone the dialog is a Sheet, which moves focus and answers Escape itself.
  $effect(() => {
    if (layout.compact) return;
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
    if (!open || layout.compact) return;

    // Capture the current element reference for this effect run.
    const el = dialogEl;
    const token = Symbol('modal');
    openStack.push(token);
    const innermost = () => openStack[openStack.length - 1] === token;

    // Escape goes to the innermost overlay of any kind (a sheet or popover may sit above or below this dialog).
    const releaseEscape = pushEscapeLayer(close);

    function handleKeydown(e: KeyboardEvent): void {
      if (!innermost()) return;

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
      if (!innermost() || !el || el.contains(e.target as Node)) return;
      const focusable = Array.from(el.querySelectorAll<HTMLElement>(FOCUSABLE));
      (focusable[0] ?? el).focus();
    }

    document.addEventListener('keydown', handleKeydown);
    document.addEventListener('focusin', handleFocusIn, true);
    return () => {
      openStack.splice(openStack.indexOf(token), 1);
      releaseEscape();
      document.removeEventListener('keydown', handleKeydown);
      document.removeEventListener('focusin', handleFocusIn, true);
    };
  });
</script>

{#if layout.compact}
  <!-- On a phone every dialog is a sheet from the bottom edge: it pulls down to close, like every sheet there. -->
  <Sheet bind:open title={title ?? ''} {labelledby} {describedby} {role} layer={elevated ? 'modal-top' : 'modal'} {fill} bare={fill} onclose={() => onclose?.()} {footer}>
    {#if fill}
      <div class="flex min-h-0 flex-1 flex-col overflow-hidden text-content text-text-secondary">{@render children?.()}</div>
    {:else}
      <div class="px-4 py-3 text-content text-text-secondary">{@render children?.()}</div>
    {/if}
  </Sheet>
{:else if open}
  <!-- Outer container: its own stacking context, moved to <body> so the opener's cannot trap it. -->
  <div use:portal class="modal-root ui-viewport" class:elevated>
    <!-- Backdrop (native button so click-to-dismiss requires no role suppression) -->
    <button
      type="button"
      class="absolute inset-0 bg-scrim"
      onclick={close}
      aria-label="Close dialog"
      tabindex="-1"
    ></button>
    <!-- Flex layer: centres the dialog, or docks it to the right edge for the side placement. -->
    <div class="flex h-full pointer-events-none {side ? 'items-stretch justify-end p-0' : `items-center justify-center p-4 ${fill ? 'max-sm:p-0' : ''}`}">
      <!-- Dialog -->
      <div
        bind:this={dialogEl}
        {role}
        aria-modal="true"
        aria-label={labelledby ? undefined : title}
        aria-labelledby={labelledby}
        aria-describedby={describedby}
        tabindex="-1"
        data-placement={placement}
        class="modal-dialog ui-overlay relative flex max-h-full w-full flex-col focus:outline-none pointer-events-auto {SIZE_CLASSES[size]} {side ? 'h-full rounded-none border-l border-border-subtle' : fill ? 'max-sm:h-full max-sm:max-w-none max-sm:rounded-none' : ''}"
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
      <div class="min-h-0 text-content text-text-secondary {fill ? 'flex flex-1 flex-col overflow-hidden' : 'overflow-y-auto px-4 py-3'}">
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

<style>
  .modal-root { z-index: var(--z-modal); }
  .modal-root.elevated { z-index: var(--z-modal-top); }
  /* The dialog settles in rather than appearing at once, as a phone's sheet slides up. */
  .modal-root { animation: modal-fade 140ms ease-out; }
  .modal-dialog { animation: modal-rise 180ms cubic-bezier(0.32, 0.72, 0, 1); }
  .modal-dialog[data-placement='side'] { animation: modal-slide 200ms cubic-bezier(0.32, 0.72, 0, 1); }
  @keyframes modal-fade { from { opacity: 0; } }
  @keyframes modal-rise { from { opacity: 0; translate: 0 8px; scale: 0.98; } }
  @keyframes modal-slide { from { translate: 100% 0; } }
</style>
