<script lang="ts">
  import { Icon } from '$lib/components/atoms';
  import type { PanelItem } from './assetMenu';

  interface Props {
    /** View, where a click on the picture doesn't open the viewer. */
    view: PanelItem | null;
    star: PanelItem | null;
    /** Pick an item as the menu does: View closes the panel first, then runs. The star is a toggle and stays open, so it runs directly. */
    select: (item: PanelItem, event: MouseEvent) => void;
    /** Close, drawn when the panel is a sheet that has no head row of its own. */
    onclose?: () => void;
  }

  let { view, star, select, onclose }: Props = $props();
</script>

<!-- The head's controls as fixed icon buttons: only the star's glyph and colour change when it is toggled, never its box. -->
{#if view}
  <button type="button" role="menuitem" tabindex="-1" class="ui-btn ui-btn-icon panel-control" data-action="view" data-row="head" title="View" aria-label="View" onclick={(event) => select(view, event)}><Icon name="expand" size={16} /></button>
{/if}
{#if star}
  <button
    type="button"
    role="menuitemcheckbox"
    tabindex="-1"
    class="ui-btn ui-btn-icon panel-control panel-star"
    data-action="star"
    data-row="head"
    title={star.label}
    aria-label="Star"
    aria-checked={star.checked === true}
    onclick={() => star.onselect?.()}
  ><Icon name="star" size={16} solid={star.checked === true} /></button>
{/if}
{#if onclose}
  <button type="button" class="ui-btn ui-btn-icon" aria-label="Close" onclick={onclose}><Icon name="close" size={16} /></button>
{/if}

<style>
  .panel-control { background: transparent; color: var(--color-text-secondary); }
  .panel-star[aria-checked='true'] { color: var(--color-star); }
  /* An unlayered rule beats the button's layered ones, so the hover and pressed fills are restated here. */
  @media (hover: hover) {
    .panel-control:hover { background: var(--color-bg-control-hover); color: var(--color-text-primary); }
    .panel-star[aria-checked='true']:hover { color: var(--color-star); }
  }
  @media (hover: none) { .panel-control:active { background: var(--color-bg-control-hover); } }
</style>
