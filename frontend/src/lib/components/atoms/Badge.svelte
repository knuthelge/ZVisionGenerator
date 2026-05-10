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

  const variants: Record<string, string> = {
    neutral: 'bg-zinc-800 text-zinc-400 border-zinc-700',
    success: 'bg-emerald-900/30 text-emerald-400 border-emerald-800/50',
    error: 'bg-red-900/30 text-red-400 border-red-800/50',
    warning: 'bg-amber-900/30 text-amber-400 border-amber-800/50',
    info: 'bg-teal-900/30 text-teal-400 border-teal-800/50'
  };

  const cls = $derived(`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium border ${variants[variant]} ${extraClass}`);
</script>

<span class={cls}>
  {@render children?.()}
</span>
