<script lang="ts">
  import type { Snippet } from 'svelte';

  interface Props {
    /** The group's accessible name. */
    label: string;
    /** The selected tool's key; the row scrolls it into view whenever it changes and when the row first shows. */
    selected: string;
    children: Snippet;
  }

  let { label, selected, children }: Props = $props();

  let row = $state<HTMLElement>();

  // The row scrolls sideways, so the selected dial is brought into view when the tab opens and whenever the tool changes.
  $effect(() => {
    void selected;
    row?.querySelector('[aria-pressed="true"]')?.scrollIntoView?.({ inline: 'nearest', block: 'nearest' });
  });
</script>

<div class="adj-tools" role="group" bind:this={row} aria-label={label}>
  {@render children()}
</div>

<style>
  /* The dials keep their size and the row scrolls sideways; the last dial peeks out at rest to show that it does. A parent sets --row-bleed
     to its own side padding, so the row runs to the screen's edge and a dial peeks out there. */
  .adj-tools { display: flex; gap: 2px; padding: 3px var(--row-bleed, 3px); margin: -3px calc(-1 * var(--row-bleed, 3px)); overflow-x: auto; overscroll-behavior-x: contain; scroll-snap-type: x proximity; scroll-padding-inline: 24px; }
  .adj-tools > :global(.adj-tool) { flex: 0 0 var(--tool-w, 64px); scroll-snap-align: start; }
  @media (pointer: coarse) { .adj-tools { scrollbar-width: none; } }
</style>
