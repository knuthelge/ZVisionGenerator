<script lang="ts">
  import type { Snippet } from 'svelte';

  interface Props {
    title: string;
    /** Buttons under the text. */
    actions?: Snippet;
    class?: string;
    children?: Snippet;
  }

  let { title, actions, class: extraClass = '', children }: Props = $props();
</script>

<div class="ui-empty {extraClass}">
  <p class="ui-empty-title">{title}</p>
  {#if children}<div class="max-w-md">{@render children()}</div>{/if}
  {#if actions}<div class="mt-2 flex flex-wrap justify-center gap-2">{@render actions()}</div>{/if}
</div>
