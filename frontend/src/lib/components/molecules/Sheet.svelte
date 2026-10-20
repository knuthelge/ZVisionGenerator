<script lang="ts">
  import type { Snippet } from 'svelte';
  import Icon from '../atoms/Icon.svelte';
  import { portal as portalAction } from '$lib/actions/portal';
  import { pushEscapeLayer } from '$lib/actions/escapeStack';
  import { SHEET_SETTLE_MS, sheetDrag } from '$lib/actions/sheetDrag';

  interface Props {
    open?: boolean;
    /** Names the sheet; shown in its head unless `labelledby` points at another name. */
    title: string;
    /** Id of an element inside the sheet that names it instead of `title`. */
    labelledby?: string;
    describedby?: string;
    /** Short status after the title, e.g. "3 changed". */
    meta?: string;
    /** Extra controls in the head, before Close. */
    headActions?: Snippet;
    /** `menu` for an action menu, `alertdialog` for a question to approve. */
    role?: 'dialog' | 'alertdialog' | 'menu';
    /** Which rung of the overlay ladder the sheet sits on (`global.css`). */
    layer?: 'sheet' | 'modal' | 'menu' | 'modal-top';
    onclose?: () => void;
    /**
     * Keep the content in the page while closed (hidden), so form fields inside it still submit and keep their state.
     * Off, the content mounts only while the sheet is open (and while it slides away).
     */
    keepMounted?: boolean;
    /** Always the full height, for browsers and editors that scroll inside; otherwise the height of the content. */
    fill?: boolean;
    /** Lay the content out edge to edge with no padding or scrolling of its own (it scrolls itself). */
    bare?: boolean;
    /** The sheet's panel, for a caller that moves focus or handles keys inside it. */
    panel?: HTMLDivElement | null;
    onkeydown?: (event: KeyboardEvent) => void;
    children: Snippet;
    footer?: Snippet;
  }

  let {
    open = $bindable(false),
    title,
    labelledby,
    describedby,
    meta,
    headActions,
    role = 'dialog',
    layer = 'sheet',
    onclose,
    keepMounted = false,
    fill = false,
    bare = false,
    panel = $bindable(null),
    onkeydown,
    children,
    footer,
  }: Props = $props();

  // In the page (`present`) and slid in (`shown`) are apart for the length of the slide, so the sheet animates both ways.
  let present = $state(open);
  let shown = $state(false);

  $effect(() => {
    if (open) {
      present = true;
      // Two frames: the first lays the sheet out below the screen, the second slides it up.
      let second = 0;
      const first = requestAnimationFrame(() => { second = requestAnimationFrame(() => { shown = true; }); });
      return () => { cancelAnimationFrame(first); cancelAnimationFrame(second); };
    }
    shown = false;
    const timer = setTimeout(() => { present = false; }, SHEET_SETTLE_MS);
    return () => clearTimeout(timer);
  });

  function close(): void {
    if (!open) return;
    open = false;
    onclose?.();
  }

  // Escape closes the sheet when it is the innermost overlay; focus moves into the sheet and back out when it closes.
  $effect(() => {
    if (!open) return;
    const previous = document.activeElement as HTMLElement | null;
    panel?.focus({ preventScroll: true });
    const release = pushEscapeLayer(close);
    return () => {
      release();
      previous?.focus?.({ preventScroll: true });
    };
  });

  /** Portal unless the content must stay inside its form. */
  function place(node: HTMLElement): { destroy: () => void } | void {
    if (!keepMounted) return portalAction(node);
  }
</script>

{#if present || keepMounted}
  <div use:place class="sheet-root ui-viewport" data-layer={layer} data-shown={shown} hidden={!present} inert={!open} data-testid="sheet">
    <button type="button" class="sheet-scrim" aria-label="Close" tabindex="-1" onclick={close}></button>
    <div
      bind:this={panel}
      use:sheetDrag={{ onclose: close }}
      class="sheet ui-sheet"
      class:fill
      {role}
      aria-modal={open && role !== 'menu' ? 'true' : undefined}
      aria-label={labelledby ? undefined : title}
      aria-labelledby={labelledby}
      aria-describedby={describedby}
      tabindex="-1"
      {onkeydown}
    >
      <div class="ui-sheet-grab" data-sheet-handle aria-hidden="true"></div>
      <div class="ui-sheet-head sheet-head" data-sheet-handle role="presentation">
        {#if title}<h2 class="ui-area-label sheet-title">{title}</h2>{/if}
        {#if meta}<span class="sheet-meta">{meta}</span>{/if}
        <span class="ui-sheet-end">
          {@render headActions?.()}
          <button type="button" class="ui-btn ui-btn-icon" aria-label="Close" onclick={close}><Icon name="close" size={16} /></button>
        </span>
      </div>
      <div class="sheet-body" class:bare>
        {@render children()}
      </div>
      {#if footer}
        <div class="sheet-foot">{@render footer()}</div>
      {/if}
    </div>
  </div>
{/if}

<style>
  .sheet-root { z-index: var(--z-sheet); }
  .sheet-root[data-layer='modal'] { z-index: var(--z-modal); }
  .sheet-root[data-layer='menu'] { z-index: var(--z-menu); }
  .sheet-root[data-layer='modal-top'] { z-index: var(--z-modal-top); }
  .sheet-scrim { position: absolute; inset: 0; background: var(--color-scrim); opacity: 0; transition: opacity 220ms ease; }
  .sheet-root[data-shown='true'] .sheet-scrim { opacity: 1; }
  /* As tall as its content, up to the screen under the top strip; a full sheet always takes that height. */
  .sheet {
    position: absolute; left: 0; right: 0; bottom: 0; display: flex; max-height: calc(100% - 44px - env(safe-area-inset-top, 0px) - 8px); flex-direction: column; outline: none;
    translate: 0 100%; transition: translate 280ms cubic-bezier(0.32, 0.72, 0, 1);
  }
  .sheet-root[data-shown='true'] .sheet { translate: 0 0; }
  .sheet.fill { height: calc(100% - 44px - env(safe-area-inset-top, 0px) - 8px); }
  .sheet-head { border-bottom: 1px solid var(--color-border-subtle); }
  .sheet-title { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .sheet-meta { flex: none; font-family: var(--font-mono); font-size: var(--text-meta); color: var(--color-primary-main); }
  .sheet-body { display: flex; flex: 1 1 auto; min-height: 0; flex-direction: column; overflow-y: auto; overscroll-behavior: contain; padding-bottom: env(safe-area-inset-bottom, 0px); }
  .sheet-body.bare { overflow: hidden; }
  .sheet-foot { display: flex; flex: none; flex-wrap: wrap; align-items: center; justify-content: flex-end; gap: 8px; padding: 10px max(12px, env(safe-area-inset-right, 0px)) calc(10px + env(safe-area-inset-bottom, 0px)) max(12px, env(safe-area-inset-left, 0px)); border-top: 1px solid var(--color-border-subtle); }
  .sheet-body:has(+ .sheet-foot) { padding-bottom: 0; }
</style>
