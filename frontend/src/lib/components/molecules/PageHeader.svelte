<script lang="ts">
  import type { Snippet } from 'svelte';

  interface Props {
    title: string;
    description?: string;
    /** Page-level controls at the right end of the bar. */
    actions?: Snippet;
    class?: string;
  }

  let { title, description, actions, class: extraClass = '' }: Props = $props();
</script>

<header class="ui-page-bar {extraClass}">
  <div class="min-w-0">
    <h1 class="ui-page-title">{title}</h1>
    {#if description}<p class="ui-page-sub">{description}</p>{/if}
  </div>
  {#if actions}<div class="ml-auto flex flex-wrap items-center gap-2">{@render actions()}</div>{/if}
</header>
