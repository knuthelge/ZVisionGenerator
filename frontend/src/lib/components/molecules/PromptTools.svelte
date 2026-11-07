<script lang="ts">
  import type { Snippet } from 'svelte';
  import { Icon } from '$lib/components/atoms';

  interface Props {
    /** The left group: the tools that change the text (Enhance, JSON, Insert, Script helper, Improve lyrics). */
    children?: Snippet;
    /** The right group: the tools that bring text in or keep it (Triggers, Recent prompts, Save). It wraps as a whole when the row is too narrow, and stays right-aligned. */
    end?: Snippet;
    /** Clear the box. Given only while the box holds something and no helper is running in it; the button is first in the right group. */
    onclear?: () => void;
    /** What Clear empties, in the words of its name: "Clear the style". */
    clearWhat?: string;
  }

  let { children, end, onclear, clearWhat = 'text' }: Props = $props();
</script>

<div class="ui-prompt-tools">
  {#if children}<div class="group">{@render children()}</div>{/if}
  {#if end || onclear}
    <div class="group end">
      {#if onclear}
        <button type="button" class="ui-btn ui-btn-sm ui-btn-icon clear-button" aria-label="Clear the {clearWhat}" title="Clear" data-testid="clear-box" onclick={onclear}><Icon name="trash" size={14} /></button>
      {/if}
      {@render end?.()}
    </div>
  {/if}
</div>

<style>
  .group { display: flex; align-items: center; gap: 6px; }
  .group.end { margin-left: auto; }
  .clear-button { color: var(--color-text-muted); }
</style>
