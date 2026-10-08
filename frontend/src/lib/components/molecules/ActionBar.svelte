<script lang="ts">
  import type { Snippet } from 'svelte';

  interface Props {
    class?: string;
    children?: Snippet;
  }

  let { class: extraClass = '', children }: Props = $props();
</script>

<!-- Every button in the bar is 36px; the one marked `main` takes the remaining width. -->
<div class="ui-action-bar {extraClass}">
  {@render children?.()}
</div>
