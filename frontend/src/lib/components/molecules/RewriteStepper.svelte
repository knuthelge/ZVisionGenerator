<script lang="ts">
  import { Icon } from '$lib/components/atoms';

  interface Props {
    /** The shown version, 1-based, or null for "– of N" (nothing shown: Scratch emptied the Enhanced text); and how many versions there are. */
    index: number | null;
    count: number;
    /** With no version shown: the 1-based version that ‹ brings back. */
    back?: number;
    /** Show another version (1-based). */
    onstep: (index: number) => void;
    /** What the group is called to assistive technology: "Enhanced versions" or "Lyrics versions". */
    label?: string;
  }

  let { index, count, back = count, onstep, label = 'Enhanced versions' }: Props = $props();
</script>

<div class="rewrite-stepper" role="group" aria-label={label}>
  <button
    type="button"
    class="ui-btn ui-btn-sm ui-btn-icon ui-btn-quiet"
    aria-label={index === null ? `Bring back version ${back}` : 'Previous version'}
    disabled={index !== null && index <= 1}
    onclick={() => onstep(index === null ? back : index - 1)}
  ><Icon name="chevleft" size={14} /></button>
  <span class="rewrite-count" class:none={index === null} aria-live="polite">{index ?? '–'} of {count}</span>
  <button type="button" class="ui-btn ui-btn-sm ui-btn-icon ui-btn-quiet" aria-label="Next version" disabled={index === null || index >= count} onclick={() => onstep((index ?? 0) + 1)}><Icon name="chevright" size={14} /></button>
</div>

<style>
  /* The buttons and the counter share a centre line: the counter's line box is trimmed to the digit height, so no font's metrics shift it. */
  .rewrite-stepper { display: flex; flex: none; align-self: center; align-items: center; gap: 2px; height: var(--spacing-control-sm); }
  .rewrite-stepper :global(svg) { display: block; }
  /* Counts stop at 5 and are set in tabular figures, so the arrows hold still without a fixed width. */
  .rewrite-count { display: block; min-width: 3.25em; font-family: var(--font-mono); font-size: var(--text-meta); line-height: 1; text-align: center; color: var(--color-text-secondary); font-variant-numeric: tabular-nums; white-space: nowrap; text-box: trim-both cap alphabetic; }
  .rewrite-count.none { color: var(--color-text-muted); }
</style>
