<script lang="ts">
  import type { Snippet } from 'svelte';
  import Icon from '../atoms/Icon.svelte';

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
    /** Takes the full height below the top strip; otherwise sized to its content, up to 86% of the screen. */
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

  // Escape closes; focus moves into the sheet and back out when it closes.
  $effect(() => {
    if (!open) return;
    const previous = document.activeElement as HTMLElement | null;
    panel?.focus();
    function handleKeydown(event: KeyboardEvent): void {
      if (event.key !== 'Escape' || event.defaultPrevented) return;
      event.preventDefault();
      close();
    }
    document.addEventListener('keydown', handleKeydown);
    return () => {
      document.removeEventListener('keydown', handleKeydown);
      previous?.focus?.();
    };
  });
</script>

{#if open || keepMounted}
  <div class="sheet-root" class:open hidden={!open} data-testid="bottom-sheet">
    <button type="button" class="sheet-scrim" aria-label="Close" tabindex="-1" onclick={close}></button>
    <div bind:this={panel} class="sheet ui-pane" class:tall role="dialog" aria-modal={open ? 'true' : undefined} aria-label={title} tabindex="-1">
      <div class="sheet-grab" aria-hidden="true"><i></i></div>
      <div class="sheet-head">
        <h2 class="ui-area-label sheet-title">{title}</h2>
        {#if meta}<span class="sheet-meta">{meta}</span>{/if}
        <span class="sheet-head-end">
          {@render headActions?.()}
          <button type="button" class="ui-btn ui-btn-sm ui-btn-icon" aria-label="Close" onclick={close}><Icon name="close" size={14} /></button>
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
  .sheet-root { position: fixed; inset: 0; z-index: 130; }
  .sheet-scrim { position: absolute; inset: 0; background: var(--color-scrim); }
  .sheet { position: absolute; left: 0; right: 0; bottom: 0; display: flex; max-height: min(86%, 86dvh); flex-direction: column; border-radius: 14px 14px 0 0; background: var(--color-bg-surface); box-shadow: 0 -1px 0 color-mix(in srgb, var(--color-text-primary) 8%, transparent); outline: none; }
  .sheet.tall { top: calc(44px + env(safe-area-inset-top, 0px) + 8px); max-height: none; }
  .sheet-grab { display: flex; flex: none; justify-content: center; padding: 8px 0 0; }
  .sheet-grab i { width: 36px; height: 4px; border-radius: 2px; background: var(--color-border-strong); }
  .sheet-head { display: flex; flex: none; align-items: center; gap: 8px; padding: 6px 12px 8px max(16px, env(safe-area-inset-left, 0px)); border-bottom: 1px solid var(--color-border-subtle); }
  .sheet-title { font-size: var(--text-ui); }
  .sheet-meta { font-family: var(--font-mono); font-size: var(--text-meta); color: var(--color-primary-main); }
  .sheet-head-end { display: flex; align-items: center; gap: 6px; margin-left: auto; }
  .sheet-body { display: flex; flex: 1; min-height: 0; flex-direction: column; overflow-y: auto; overscroll-behavior: contain; padding-bottom: env(safe-area-inset-bottom, 0px); }
  .sheet-foot { flex: none; padding: 10px max(12px, env(safe-area-inset-right, 0px)) calc(10px + env(safe-area-inset-bottom, 0px)) max(12px, env(safe-area-inset-left, 0px)); border-top: 1px solid var(--color-border-subtle); }
  /* On a wide screen a sheet is a centred panel instead of a full-width drawer. */
  @media (min-width: 640px) {
    .sheet { left: 50%; width: min(520px, calc(100vw - 32px)); transform: translateX(-50%); }
  }
</style>
