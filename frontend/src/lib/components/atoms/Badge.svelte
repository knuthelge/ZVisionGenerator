<script lang="ts">
  import type { Snippet } from 'svelte';

  interface Props {
    variant?: 'neutral' | 'success' | 'error' | 'warning' | 'info';
    class?: string;
    children?: Snippet;
  }

  let {
    variant = 'neutral',
    class: extraClass = '',
    children
  }: Props = $props();
</script>

<span class="ui-badge {variant === 'neutral' ? '' : `ui-badge-${variant}`} {extraClass}">
  {@render children?.()}
</span>
