<script lang="ts" module>
  import type { IconName } from '../atoms/Icon.svelte';

  export interface ActionMenuItem {
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
    /** Makes the item a radio: it shows a check while true and keeps the label's place while false. */
    checked?: boolean;
    /** Muted text after the label. */
    hint?: string;
    /** An icon before the label. */
    icon?: IconName;
    onselect?: () => void;
  }

  export type ActionMenuEntry =
    | { kind: 'heading'; label: string }
    /** A sentence under the title, such as why something failed; not an action. */
    | { kind: 'note'; text: string; tone?: 'error' }
    | { kind: 'separator' }
    | ActionMenuItem
    /** A row that shows or hides its items in place; the menu stays open. The state resets each time the menu opens. */
    | { kind: 'group'; id: string; label: string; items: ActionMenuItem[]; defaultOpen?: boolean };
</script>

<script lang="ts">
  import { popover, type PopoverAlign } from '$lib/actions/popover';
  import Icon from '../atoms/Icon.svelte';
  import { layout } from '$lib/state/layoutMode.svelte';
  import Sheet from './Sheet.svelte';

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
  // Groups the person toggled since the menu opened; a group not listed here follows its `defaultOpen`.
  let toggled = $state<Record<string, boolean>>({});

  // Every opening starts from the groups' defaults.
  $effect(() => {
    void open;
    toggled = {};
  });

  function isExpanded(group: Extract<ActionMenuEntry, { kind: 'group' }>): boolean {
    return toggled[group.id] ?? group.defaultOpen ?? false;
  }

  function menuItems(): HTMLElement[] {
    return Array.from(
      menuEl?.querySelectorAll<HTMLElement>('[role="menuitem"]:not([aria-disabled="true"]), [role="menuitemradio"]:not([aria-disabled="true"])') ?? [],
    );
  }

  function select(entry: ActionMenuItem): void {
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

  // Keys start on the first item; a finger starts on the menu itself, or the first row would look chosen.
  $effect(() => {
    if (!open || !menuEl) return;
    if (layout.compact) menuEl.focus();
    else menuItems()[0]?.focus();
  });

</script>

{#snippet row(entry: ActionMenuItem)}
  {#snippet content()}
    {#if entry.checked !== undefined}
      <span class="action-menu-check" aria-hidden="true">{#if entry.checked}<Icon name="check" size={14} />{/if}</span>
    {/if}
    {#if entry.icon}<Icon name={entry.icon} size={14} />{/if}
    <span class="action-menu-label">{entry.label}</span>
    {#if entry.hint}<span class="action-menu-hint">{entry.hint}</span>{/if}
  {/snippet}
  {#if entry.href}
    <a
      role="menuitem"
      tabindex="-1"
      class="action-menu-item"
      href={entry.href}
      download={entry.download}
      data-action={entry.id}
      onclick={() => select(entry)}
    >{@render content()}</a>
  {:else}
    <button
      type="button"
      role={entry.checked === undefined ? 'menuitem' : 'menuitemradio'}
      aria-checked={entry.checked === undefined ? undefined : entry.checked}
      tabindex="-1"
      class="action-menu-item"
      class:danger={entry.danger}
      aria-disabled={entry.disabled ? 'true' : undefined}
      title={entry.title}
      data-action={entry.id}
      onclick={() => select(entry)}
    >{@render content()}</button>
  {/if}
{/snippet}

{#snippet entries()}
    {#each items as entry, index (index)}
      {#if entry.kind === 'heading'}
        <p class="action-menu-heading ui-area-label" role="presentation">{entry.label}</p>
      {:else if entry.kind === 'note'}
        <p class="action-menu-note" class:error={entry.tone === 'error'} role="presentation">{entry.text}</p>
      {:else if entry.kind === 'separator'}
        <div class="action-menu-separator" role="separator"></div>
      {:else if entry.kind === 'group'}
        <button
          type="button"
          role="menuitem"
          tabindex="-1"
          class="action-menu-item action-menu-group"
          aria-expanded={isExpanded(entry)}
          data-action={entry.id}
          onclick={() => { toggled = { ...toggled, [entry.id]: !isExpanded(entry) }; }}
        >
          <span class="action-menu-label">{entry.label}</span>
          <Icon name={isExpanded(entry) ? 'chevdown' : 'chevright'} size={12} />
        </button>
        {#if isExpanded(entry)}
          <div class="action-menu-group-items" role="group" aria-label={entry.label}>
            {#each entry.items as child (child.id)}
              {@render row(child)}
            {/each}
          </div>
        {/if}
      {:else}
        {@render row(entry)}
      {/if}
    {/each}
{/snippet}

{#if layout.compact}
  <!-- On a phone a menu is a sheet from the bottom edge with finger-sized rows; the same items, the same keys. -->
  <Sheet {open} title={label} role="menu" layer="menu" bind:panel={menuEl} onkeydown={onKeydown} onclose={() => { if (open) onclose(); }}>
    <div class="action-sheet-body" data-testid="action-sheet">
      {@render entries()}
    </div>
  </Sheet>
{:else if open}
  <div
    use:popover={{ anchor, align, onclose }}
    bind:this={menuEl}
    class="action-menu ui-overlay"
    role="menu"
    tabindex="-1"
    aria-label={label}
    onkeydown={onKeydown}
  >
    {@render entries()}
  </div>
{/if}

<style>
  .action-sheet-body { padding: 4px max(12px, env(safe-area-inset-right, 0px)) 12px max(12px, env(safe-area-inset-left, 0px)); }
  .action-sheet-body .action-menu-item { min-height: 48px; padding: 0 12px; font-size: 15px; }
  .action-sheet-body .action-menu-heading { padding: 6px 12px 8px; }
  .action-menu { z-index: var(--z-menu); min-width: 200px; max-height: min(60vh, 22rem); overflow-y: auto; padding: 4px; }
  .action-menu-heading { padding: 6px 8px 3px; }
  .action-menu-note { max-width: 15rem; padding: 6px 8px 4px; font-size: var(--text-meta); line-height: 1.4; color: var(--color-text-muted); }
  .action-menu-note.error { color: var(--color-error); }
  .action-menu-separator { height: 1px; margin: 4px 0; background: var(--color-border-subtle); }
  .action-menu-item { display: flex; width: 100%; align-items: center; gap: 8px; border-radius: var(--radius-sm); padding: 6px 8px; text-align: left; font-size: var(--text-ui); color: var(--color-text-secondary); }
  .action-menu-item:focus { background: var(--color-bg-surface-hover); color: var(--color-text-primary); outline: none; }
  @media (hover: hover) { .action-menu-item:hover { background: var(--color-bg-surface-hover); color: var(--color-text-primary); outline: none; } }
  .action-menu-item:focus-visible { box-shadow: inset 0 0 0 2px var(--color-primary-main); }
  .action-menu-label { flex: 1; min-width: 0; }
  .action-menu-hint { flex: none; max-width: 45%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: var(--text-meta); color: var(--color-text-muted); }
  .action-menu-check { display: inline-grid; flex: none; place-items: center; width: 14px; height: 14px; color: var(--color-primary-main); }
  .action-menu-group { color: var(--color-text-muted); }
  .action-menu-group-items { padding-left: 12px; }
  .action-menu-item.danger { color: var(--color-error); }
  .action-menu-item.danger:focus { background: var(--color-error-surface); color: var(--color-error); }
  @media (hover: hover) { .action-menu-item.danger:hover { background: var(--color-error-surface); color: var(--color-error); } }
  .action-menu-item[aria-disabled='true'] { opacity: 0.5; cursor: not-allowed; }
</style>
