<script lang="ts" module>
  export type ActionMenuEntry =
    | { kind: 'heading'; label: string }
    | { kind: 'separator' }
    | {
        kind: 'item';
        id: string;
        label: string;
        danger?: boolean;
        disabled?: boolean;
        /** Tooltip, e.g. why the item is disabled. */
        title?: string;
        /** Renders the item as a link (e.g. a download) instead of a button. */
        href?: string;
        download?: string;
        onselect?: () => void;
      };
</script>

<script lang="ts">
  import { popover, type PopoverAlign } from '$lib/actions/popover';

  interface Props {
    open: boolean;
    anchor: HTMLElement | null;
    items: ActionMenuEntry[];
    label: string;
    /** Which edge of the anchor the menu lines up with. */
    align?: PopoverAlign;
    onclose: () => void;
  }

  let { open, anchor, items, label, align = 'end', onclose }: Props = $props();

  let menuEl = $state<HTMLDivElement | null>(null);

  function menuItems(): HTMLElement[] {
    return Array.from(menuEl?.querySelectorAll<HTMLElement>('[role="menuitem"]:not([aria-disabled="true"])') ?? []);
  }

  function select(entry: Extract<ActionMenuEntry, { kind: 'item' }>): void {
    if (entry.disabled) return;
    onclose();
    entry.onselect?.();
  }

  // Escape is handled by the popover action, which also returns focus to the anchor.
  function onKeydown(event: KeyboardEvent): void {
    const list = menuItems();
    const index = list.indexOf(document.activeElement as HTMLElement);
    if (event.key === 'Tab') { onclose(); }
    else if (event.key === 'ArrowDown') { event.preventDefault(); list[(index + 1) % list.length]?.focus(); }
    else if (event.key === 'ArrowUp') { event.preventDefault(); list[(index - 1 + list.length) % list.length]?.focus(); }
    else if (event.key === 'Home') { event.preventDefault(); list[0]?.focus(); }
    else if (event.key === 'End') { event.preventDefault(); list[list.length - 1]?.focus(); }
  }

  $effect(() => {
    if (open && menuEl) menuItems()[0]?.focus();
  });
</script>

{#if open}
  <div
    use:popover={{ anchor, align, onclose }}
    bind:this={menuEl}
    class="action-menu ui-overlay"
    role="menu"
    tabindex="-1"
    aria-label={label}
    onkeydown={onKeydown}
  >
    {#each items as entry, index (index)}
      {#if entry.kind === 'heading'}
        <p class="action-menu-heading ui-area-label" role="presentation">{entry.label}</p>
      {:else if entry.kind === 'separator'}
        <div class="action-menu-separator" role="separator"></div>
      {:else if entry.href}
        <a
          role="menuitem"
          tabindex="-1"
          class="action-menu-item"
          href={entry.href}
          download={entry.download}
          data-action={entry.id}
          onclick={() => select(entry)}
        >{entry.label}</a>
      {:else}
        <button
          type="button"
          role="menuitem"
          tabindex="-1"
          class="action-menu-item"
          class:danger={entry.danger}
          aria-disabled={entry.disabled ? 'true' : undefined}
          title={entry.title}
          data-action={entry.id}
          onclick={() => select(entry)}
        >{entry.label}</button>
      {/if}
    {/each}
  </div>
{/if}

<style>
  .action-menu { z-index: 120; min-width: 200px; max-height: min(60vh, 22rem); overflow-y: auto; padding: 4px; }
  .action-menu-heading { padding: 6px 8px 3px; }
  .action-menu-separator { height: 1px; margin: 4px 0; background: var(--color-border-subtle); }
  .action-menu-item { display: flex; width: 100%; align-items: center; gap: 8px; border-radius: var(--radius-sm); padding: 6px 8px; text-align: left; font-size: var(--text-ui); color: var(--color-text-secondary); }
  .action-menu-item:hover, .action-menu-item:focus { background: var(--color-bg-surface-hover); color: var(--color-text-primary); outline: none; }
  .action-menu-item:focus-visible { box-shadow: inset 0 0 0 2px var(--color-primary-main); }
  .action-menu-item.danger { color: var(--color-error); }
  .action-menu-item.danger:hover, .action-menu-item.danger:focus { background: var(--color-error-surface); color: var(--color-error); }
  .action-menu-item[aria-disabled='true'] { opacity: 0.5; cursor: not-allowed; }
</style>
