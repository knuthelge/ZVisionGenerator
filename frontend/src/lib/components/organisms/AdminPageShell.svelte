<script lang="ts">
  import type { Snippet } from 'svelte';

  interface Props {
    title: string;
    description?: string;
    loading?: boolean;
    error?: string | null;
    class?: string;
    children?: Snippet;
  }

  let {
    title,
    description,
    loading = false,
    error = null,
    class: extraClass = '',
    children
  }: Props = $props();
</script>

<main class="flex-1 bg-zinc-900 overflow-y-auto p-6 custom-scrollbar {extraClass}">
  <div class="max-w-6xl mx-auto space-y-8">
    <!-- Header -->
    <header class="mb-8">
      <h1 class="text-2xl font-bold text-white tracking-tight">{title}</h1>
      {#if description}
        <p class="text-zinc-400 mt-2 text-sm">{description}</p>
      {/if}
    </header>

    <!-- Loading state -->
    {#if loading}
      <div class="flex items-center justify-center py-24">
        <div class="animate-spin rounded-full h-8 w-8 border-t-2 border-teal-500"></div>
      </div>
    <!-- Error state -->
    {:else if error}
      <div class="rounded-lg border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-100">
        {error}
      </div>
    <!-- Content -->
    {:else}
      {@render children?.()}
    {/if}
  </div>
</main>
