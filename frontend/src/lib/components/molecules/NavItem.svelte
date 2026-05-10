<script lang="ts">
  import type { Snippet } from 'svelte';
  import type { PageId } from '$lib/types';

  interface Props {
    page: PageId;
    label: string;
    active?: boolean;
    href?: string;
    onclick?: (page: PageId) => void;
    children?: Snippet;
  }

  let {
    page,
    label,
    active = false,
    href,
    onclick,
    children
  }: Props = $props();

  const baseCls = 'inline-flex items-center gap-1.5 px-3 py-1.5 text-sm font-medium rounded-md transition-colors focus-visible:focus-ring';
  const activeCls  = 'text-teal-400 bg-teal-500/10 border-b-2 border-teal-400';
  const inactiveCls = 'text-zinc-500 hover:text-zinc-300 hover:bg-zinc-900 border-b-2 border-transparent';

  const cls = $derived(`${baseCls} ${active ? activeCls : inactiveCls}`);

  function handleClick(e: MouseEvent): void {
    if (onclick) {
      e.preventDefault();
      onclick(page);
    }
  }
</script>

{#if href}
  <a {href} class={cls} aria-current={active ? 'page' : undefined} onclick={handleClick}>
    {#if children}
      {@render children()}
    {:else}
      {label}
    {/if}
  </a>
{:else}
  <button type="button" class={cls} aria-current={active ? 'page' : undefined} onclick={handleClick}>
    {#if children}
      {@render children()}
    {:else}
      {label}
    {/if}
  </button>
{/if}
