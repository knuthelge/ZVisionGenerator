<script lang="ts">
  import type { Snippet } from 'svelte';

  interface Props {
    for?: string;
    required?: boolean;
    class?: string;
    children?: Snippet;
  }

  let {
    for: htmlFor,
    required = false,
    class: extraClass = '',
    children
  }: Props = $props();
</script>

<label
  for={htmlFor}
  class="block text-xs text-zinc-400 font-medium uppercase tracking-wider {extraClass}"
>
  {@render children?.()}
  {#if required}
    <span class="text-red-400 ml-0.5" aria-hidden="true">*</span>
  {/if}
</label>
