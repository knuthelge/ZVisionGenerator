<script lang="ts">
  import type { Snippet } from 'svelte';

  interface Props {
    variant?: 'primary' | 'secondary' | 'ghost' | 'danger';
    size?: 'sm' | 'md' | 'lg';
    disabled?: boolean;
    loading?: boolean;
    type?: 'button' | 'submit' | 'reset';
    class?: string;
    onclick?: (event: MouseEvent) => void;
    children?: Snippet;
  }

  let {
    variant = 'secondary',
    size = 'md',
    disabled = false,
    loading = false,
    type = 'button',
    class: extraClass = '',
    onclick,
    children
  }: Props = $props();

  const base = 'inline-flex items-center justify-center font-medium rounded-md transition-colors focus-visible:focus-ring cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed';

  const variants: Record<string, string> = {
    primary: 'bg-teal-600 hover:bg-teal-500 active:bg-teal-700 text-white',
    secondary: 'bg-zinc-800 hover:bg-zinc-700 active:bg-zinc-900 text-zinc-100',
    ghost: 'bg-transparent hover:bg-zinc-800 active:bg-zinc-700 text-zinc-400 hover:text-zinc-100',
    danger: 'bg-red-900/40 hover:bg-red-900/60 active:bg-red-900/80 text-red-400 border border-red-800'
  };

  const sizes: Record<string, string> = {
    sm: 'px-2.5 py-1.5 text-xs gap-1.5',
    md: 'px-4 py-2 text-sm gap-2',
    lg: 'px-6 py-3 text-base gap-2.5'
  };

  const cls = $derived(`${base} ${variants[variant]} ${sizes[size]} ${extraClass}`);
</script>

<button
  {type}
  class={cls}
  disabled={disabled || loading}
  {onclick}
>
  {#if loading}
    <svg class="animate-spin h-3.5 w-3.5 shrink-0" fill="none" viewBox="0 0 24 24" aria-hidden="true">
      <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
      <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"></path>
    </svg>
  {/if}
  {@render children?.()}
</button>
