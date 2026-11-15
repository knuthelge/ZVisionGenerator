<script lang="ts">
  import { Icon } from '$lib/components/atoms';
  import type { GalleryAsset } from '$lib/types';
  import type { MenuContext } from './ActionMenu.svelte';
  import type { AssetPanel, PanelItem, PanelSection } from './assetMenu';
  import PanelControls from './PanelControls.svelte';
  import PanelFile from './PanelFile.svelte';

  interface Props {
    panel: AssetPanel;
    asset: GalleryAsset;
    /** The kind and size under the name in the head. */
    meta: string;
    /** How the menu picks items and where the panel is drawn. */
    ctx: MenuContext;
    /** Draw the head: a popover has it in the panel; a sheet has it in its own head row, except held sideways. */
    head: boolean;
  }

  let { panel, asset, meta, ctx, head }: Props = $props();

  // Held sideways the sheet spans the screen: one 8-column grid puts Make beside the head, and the sections side by side.
  const wide = $derived(ctx.wide);

  interface Placed {
    item: PanelItem;
    /** The row of cells the item sits in, which Left and Right stay inside. */
    row: string;
    /** Its place in a pair: 0 or 1, or null for a choice that spans the row. */
    slot: 0 | 1 | null;
  }

  /** Put a section's items in rows of two; a choice that spans its row stands alone. */
  function placeIn(id: string, items: PanelItem[]): Placed[] {
    const placed: Placed[] = [];
    let pair = 0;
    let slot: 0 | 1 = 0;
    for (const item of items) {
      if (item.span) {
        placed.push({ item, row: `${id}-${pair}`, slot: null });
        pair += 1;
      } else {
        placed.push({ item, row: `${id}-${pair}`, slot });
        if (slot === 1) { pair += 1; slot = 0; } else slot = 1;
      }
    }
    return placed;
  }

  const placedSections = $derived(panel.sections.map((section, index) => ({ section, index, placed: placeIn(section.id, section.items) })));
  // An image makes three things (Reuse settings, Edit, Adjust): Reuse takes its own row and the other two pair up under it. A video or a take
  // has Reuse beside its own action.
  const placedMake = $derived(placeIn('make', panel.make.length > 2 ? panel.make.map((item, index) => (index === 0 ? { ...item, span: true } : item)) : panel.make));

  /** The grid-column of an item in the 8-column sideways grid: its half, then its place in its pair. */
  function sideways(half: number, slot: 0 | 1 | null): string | undefined {
    if (!wide) return undefined;
    return slot === null ? `grid-column: ${half * 4 + 1} / span 4` : `grid-column: ${half * 4 + slot * 2 + 1} / span 2`;
  }
</script>

{#snippet choice(placed: Placed, half: number)}
  {@const item = placed.item}
  <button
    type="button"
    role="menuitem"
    tabindex="-1"
    class="ui-choice"
    class:ui-choice-primary={item.primary}
    class:fp-span={item.span}
    data-action={item.id}
    data-row={placed.row}
    data-initial={placed.row.startsWith('make') ? '' : undefined}
    aria-disabled={item.disabled ? 'true' : undefined}
    aria-busy={item.busy ? 'true' : undefined}
    aria-label={item.name}
    title={item.title}
    style={sideways(half, placed.slot)}
    onclick={(event) => ctx.select(item, event)}
  >
    {#if item.icon}<Icon name={item.icon} size={18} />{/if}
    <span class="ui-choice-text">
      <span class="ui-choice-name">{item.label}{#if item.warn}{' '}<em class="ui-choice-warn">· {item.warn}</em>{/if}</span>
      <!-- A greyed choice gives its reason; an enabled one says what it makes. -->
      {#if item.hint ?? item.description}<span class="ui-choice-hint">{item.hint ?? item.description}</span>{/if}
    </span>
  </button>
{/snippet}

{#snippet tool(item: PanelItem)}
  {#if item.href}
    <a
      role="menuitem"
      tabindex="-1"
      class="ui-tool"
      href={item.href}
      download={item.download}
      data-action={item.id}
      data-row="file"
      onclick={(event) => ctx.select(item, event)}
    ><Icon name={item.icon ?? 'download'} size={16} />{item.label}</a>
  {:else}
    <button
      type="button"
      role="menuitem"
      tabindex="-1"
      class="ui-tool"
      class:ui-tool-danger={item.danger}
      data-action={item.id}
      data-row="file"
      aria-disabled={item.disabled ? 'true' : undefined}
      aria-busy={item.busy ? 'true' : undefined}
      title={item.title}
      onclick={(event) => ctx.select(item, event)}
    >{#if item.icon}<Icon name={item.icon} size={16} />{/if}{item.label}</button>
  {/if}
{/snippet}

{#snippet headRow()}
  <PanelFile {asset} {meta} />
  <span class="fp-controls"><PanelControls view={panel.head.view} star={panel.head.star} select={ctx.select} onclose={wide ? ctx.close : undefined} /></span>
{/snippet}

<div class="file-panel" class:in-sheet={ctx.sheet} class:sideways={wide} data-testid="file-panel">
  {#if head && !wide}
    <div class="fp-head">{@render headRow()}</div>
  {/if}
  <div class="fp-body">
    <div class="fp-grid" class:sideways={wide}>
      {#each placedMake as placed (placed.item.id)}
        {@render choice(placed, 0)}
      {/each}
      {#if head && wide}
        <div class="fp-head fp-head-cell">{@render headRow()}</div>
      {/if}
      {#each placedSections as { section, index, placed } (section.id)}
        {@const half = index % 2}
        <div role="group" aria-label={section.label} class="fp-section">
          <span class="fp-label ui-area-label" aria-hidden="true" style={wide ? `grid-column: ${half * 4 + 1} / span 4` : undefined}>{section.label}</span>
          {#each placed as entry (entry.item.id)}
            {@render choice(entry, half)}
          {/each}
        </div>
      {/each}
    </div>
  </div>
  {#if panel.file.length > 0}
    <div class="fp-foot" role="group" aria-label="File">
      {#each panel.file as item (item.id)}
        {@render tool(item)}
      {/each}
    </div>
  {/if}
</div>

<style>
  .file-panel { display: flex; min-height: 0; flex: 1 1 auto; flex-direction: column; }
  /* A sheet: the grid's edge for its sides, and 16px under the file row before the safe area. */
  .file-panel.in-sheet { padding: var(--sp-8) var(--sp-12) var(--sp-16); }
  .file-panel.sideways { padding-inline: 0; }
  .fp-head { display: flex; min-height: 40px; align-items: center; gap: var(--sp-8); }
  .file-panel > .fp-head { margin-bottom: var(--sp-8); }
  .fp-controls { display: flex; flex: none; align-items: center; gap: var(--sp-4); }
  .fp-head-cell { grid-column: 5 / span 4; align-self: center; }
  /* The body scrolls when the room is short; the file row below it stays in view. */
  .fp-body { min-height: 0; flex: 0 1 auto; overflow-y: auto; overscroll-behavior: contain; }
  .fp-grid { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr) minmax(0, 1.2fr) minmax(0, 0.8fr); gap: var(--sp-6); }
  .fp-grid.sideways { grid-template-columns: repeat(8, minmax(0, 1fr)); grid-auto-flow: row dense; }
  .fp-section { display: contents; }
  .fp-grid :global(.ui-choice) { grid-column: span 2; }
  .fp-grid :global(.ui-choice.fp-span) { grid-column: 1 / -1; }
  .fp-label { display: flex; grid-column: 1 / -1; align-items: center; gap: var(--sp-8); margin-top: var(--sp-2); padding-left: var(--sp-8); }
  .fp-label::after { content: ''; height: 1px; flex: 1; background: var(--color-border-subtle); }
  /* The file row: the same four cells everywhere, pinned under the body, after a rule. Delete sits in the last, narrow column,
     with its own rule centred in the gap before it. */
  .fp-foot { display: grid; flex: none; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr) minmax(0, 1.2fr) minmax(0, 0.8fr); gap: var(--sp-6); margin-top: var(--sp-6); padding-top: var(--sp-6); border-top: 1px solid var(--color-border-subtle); }
  .fp-foot :global(.ui-tool-danger) { grid-column: 4; }
  .fp-foot :global(.ui-tool-danger)::before { content: ''; position: absolute; top: 8px; bottom: 8px; left: -3.5px; width: 1px; background: var(--color-border-strong); }
</style>
