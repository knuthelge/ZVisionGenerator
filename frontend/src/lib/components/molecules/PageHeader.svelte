<script lang="ts">
  import type { Snippet } from 'svelte';
  import { layout } from '$lib/state/layoutMode.svelte';

  interface Props {
    title: string;
    description?: string;
    /** Page-level controls at the right end of the bar. */
    actions?: Snippet;
    /** Centre the title and actions at the reference-page width, lined up with a centred content block. */
    contained?: boolean;
    class?: string;
  }

  let { title, description, actions, contained = false, class: extraClass = '' }: Props = $props();
</script>

<header class="ui-page-bar {extraClass}">
  <div class="flex min-w-0 flex-1 flex-wrap items-center gap-x-3 gap-y-1.5 {contained ? 'mx-auto max-w-page' : ''}">
    <div class="min-w-0">
      <!-- On a phone the top strip shows the title; here it stays for assistive tech only. -->
      <h1 class="ui-page-title" class:sr-only={layout.compact}>{title}</h1>
      {#if description}<p class="ui-page-sub">{description}</p>{/if}
    </div>
    {#if actions}<div class="ml-auto flex flex-wrap items-center gap-2" class:w-full={layout.compact}>{@render actions()}</div>{/if}
  </div>
</header>
