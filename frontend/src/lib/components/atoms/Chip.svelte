<script lang="ts">
  import type { Snippet } from 'svelte';
  import Icon from './Icon.svelte';

  interface Props {
    /** On/off chip (a toggle button); leave undefined for a static chip. */
    pressed?: boolean;
    disabled?: boolean;
    title?: string;
    /** Adds a remove button; `removeLabel` names it for screen readers. */
    onremove?: () => void;
    removeLabel?: string;
    onclick?: (event: MouseEvent) => void;
    class?: string;
    children?: Snippet;
  }

  let { pressed, disabled = false, title, onremove, removeLabel = 'Remove', onclick, class: extraClass = '', children }: Props = $props();
</script>

{#if pressed !== undefined || onclick}
  <button type="button" class="ui-chip {extraClass}" aria-pressed={pressed} {disabled} {title} {onclick}>
    {@render children?.()}
  </button>
{:else}
  <span class="ui-chip {extraClass}" {title}>
    {@render children?.()}
    {#if onremove}
      <button type="button" class="chip-remove" aria-label={removeLabel} {disabled} onclick={onremove}>
        <Icon name="close" size={11} />
      </button>
    {/if}
  </span>
{/if}

<style>
  .chip-remove { display: inline-grid; place-items: center; width: 16px; height: 16px; margin-right: -3px; border-radius: var(--radius-xs); color: var(--color-text-muted); }
  .chip-remove:hover:not(:disabled) { background: var(--color-error-surface); color: var(--color-error); }
</style>
