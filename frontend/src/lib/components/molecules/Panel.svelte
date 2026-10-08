<script lang="ts">
  import type { Snippet } from 'svelte';
  import { Icon } from '$lib/components/atoms';
  import type { IconName } from '$lib/components/atoms/Icon.svelte';

  interface Props {
    /** Area name, shown in the one small-caps style. */
    title?: string;
    icon?: IconName;
    /** Controls or a count at the right end of the header. */
    actions?: Snippet;
    /** Drop the body padding, e.g. for a full-width table or rows. */
    flush?: boolean;
    /** Element for the panel, e.g. `form`. */
    as?: 'section' | 'form' | 'div';
    onsubmit?: (event: SubmitEvent) => void;
    class?: string;
    children?: Snippet;
  }

  let { title, icon, actions, flush = false, as = 'section', onsubmit, class: extraClass = '', children }: Props = $props();
</script>

<svelte:element this={as} class="ui-panel {extraClass}" {onsubmit}>
  {#if title || actions}
    <div class="ui-panel-head">
      {#if icon}<Icon name={icon} size={16} />{/if}
      {#if title}<h2 class="ui-area-label min-w-0 truncate">{title}</h2>{/if}
      {#if actions}<div class="ml-auto flex shrink-0 items-center gap-2">{@render actions()}</div>{/if}
    </div>
  {/if}
  <div class={flush ? '' : 'ui-panel-body'}>
    {@render children?.()}
  </div>
</svelte:element>
