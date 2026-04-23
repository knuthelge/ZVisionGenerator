<script lang="ts">
  import { router } from '$lib/state/router.svelte';
  import { draft } from '$lib/state/draft.svelte';
  import type { PageId, Workflow } from '$lib/types';

  interface Props {
    currentPage: PageId;
  }
  let { currentPage }: Props = $props();

  const navItems: { id: PageId; label: string }[] = [
    { id: 'workspace', label: 'Workspace' },
    { id: 'models', label: 'Models' },
    { id: 'gallery', label: 'Gallery' },
    { id: 'config', label: 'Config' }
  ];

  const workflowItems: { id: Workflow; label: string }[] = [
    { id: 'txt2img', label: 'Text to Image' },
    { id: 'img2img', label: 'Image to Image' },
    { id: 'img2vid', label: 'Image to Video' },
    { id: 'txt2vid', label: 'Text to Video' }
  ];
</script>

<header class="h-12 bg-zinc-950 border-b border-zinc-900 flex items-center justify-between px-4 shrink-0">
  <div class="flex items-center gap-6 min-w-0">
    <h1 class="text-base font-bold tracking-tight text-white flex items-center gap-2 shrink-0">
      <img src="/docs/assets/zvision-white.png" alt="ziv" class="w-4 h-4 object-contain">
      ziv
    </h1>

    {#if currentPage === 'workspace'}
      <nav class="flex items-center gap-1 text-sm font-medium overflow-x-auto custom-scrollbar" aria-label="Workflow">
        {#each workflowItems as item}
          <button
            type="button"
            class="{draft.state.workflow === item.id
              ? 'text-teal-400 bg-teal-500/10'
              : 'text-zinc-500 hover:text-zinc-300 hover:bg-zinc-900'}
              py-1.5 px-3 rounded transition whitespace-nowrap"
            onclick={() => draft.update('workflow', item.id)}
          >
            {item.label}
          </button>
        {/each}
      </nav>
    {/if}
  </div>

  <nav class="flex items-center gap-2 text-sm font-medium shrink-0" aria-label="Main navigation">
    {#each navItems as item}
      <button
        type="button"
        class="{currentPage === item.id
          ? 'text-teal-400 bg-teal-500/10'
          : 'text-zinc-500 hover:text-zinc-300 hover:bg-zinc-900'}
          py-1.5 px-3 rounded transition"
        onclick={() => router.navigate(item.id)}
      >
        {item.label}
      </button>
    {/each}
  </nav>
</header>
