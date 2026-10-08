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

<label for={htmlFor} class="ui-label {extraClass}">
  {@render children?.()}
  {#if required}
    <span class="ml-0.5 text-error" aria-hidden="true">*</span>
  {/if}
</label>
