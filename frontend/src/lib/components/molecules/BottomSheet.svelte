<script lang="ts">
  import type { Snippet } from 'svelte';
  import Icon from '../atoms/Icon.svelte';
  import { pushEscapeLayer } from '$lib/actions/escapeStack';
  import { sheetDrag } from '$lib/actions/sheetDrag';

  interface Props {
    open?: boolean;
    title: string;
    /** Short status after the title, e.g. "3 changed". */
    meta?: string;
    /** Extra controls in the head, before Close. */
    headActions?: Snippet;
    onclose?: () => void;
    /**
     * Keep the content in the page while closed (hidden), so form fields inside it still submit and keep their state.
     * Off, the content mounts only while the sheet is open.
     */
    keepMounted?: boolean;
    /** May grow to the top strip; otherwise sized to its content, up to 86% of the screen. Never taller than its content. */
    tall?: boolean;
    children: Snippet;
    footer?: Snippet;
  }

  let { open = $bindable(false), title, meta, headActions, onclose, keepMounted = false, tall = false, children, footer }: Props = $props();

  let panel = $state<HTMLDivElement | null>(null);

  function close(): void {
    open = false;
    onclose?.();
  }

  // Escape closes the sheet when it is the innermost overlay; focus moves into the sheet and back out when it closes.
  $effect(() => {
    if (!open) return;
    const previous = document.activeElement as HTMLElement | null;
    panel?.focus();
    const release = pushEscapeLayer(close);
    return () => {
      release();
      previous?.focus?.();
    };
  });
</script>

{#if open || keepMounted}
  <div class="sheet-root" class:open hidden={!open} data-testid="bottom-sheet">
    <button type="button" class="sheet-scrim" aria-label="Close sheet" tabindex="-1" onclick={close}></button>
    <div bind:this={panel} use:sheetDrag={{ onclose: close }} class="sheet ui-sheet" class:tall role="dialog" aria-modal={open ? 'true' : undefined} aria-label={title} tabindex="-1">
      <div class="ui-sheet-grab" aria-hidden="true"></div>
      <div class="ui-sheet-head sheet-head">
        <h2 class="ui-area-label">{title}</h2>
        {#if meta}<span class="sheet-meta">{meta}</span>{/if}
        <span class="ui-sheet-end">
          {@render headActions?.()}
          <button type="button" class="ui-btn ui-btn-icon" aria-label="Close" onclick={close}><Icon name="close" size={16} /></button>
        </span>
      </div>
      <div class="sheet-body">
        {@render children()}
      </div>
      {#if footer}
        <div class="sheet-foot">{@render footer()}</div>
      {/if}
    </div>
  </div>
{/if}

<style>
  .sheet-root { position: fixed; inset: 0; z-index: var(--z-sheet); }
  .sheet-scrim { position: absolute; inset: 0; background: var(--color-scrim); }
  .sheet { position: absolute; left: 0; right: 0; bottom: 0; display: flex; max-height: min(86%, 86dvh); flex-direction: column; outline: none; }
  .sheet.tall { max-height: calc(100% - 44px - env(safe-area-inset-top, 0px) - 8px); }
  .sheet-head { border-bottom: 1px solid var(--color-border-subtle); }
  .sheet-meta { font-family: var(--font-mono); font-size: var(--text-meta); color: var(--color-primary-main); }
  .sheet-body { display: flex; flex: 1; min-height: 0; flex-direction: column; overflow-y: auto; overscroll-behavior: contain; padding-bottom: env(safe-area-inset-bottom, 0px); }
  .sheet-foot { flex: none; padding: 10px max(12px, env(safe-area-inset-right, 0px)) calc(10px + env(safe-area-inset-bottom, 0px)) max(12px, env(safe-area-inset-left, 0px)); border-top: 1px solid var(--color-border-subtle); }
</style>
