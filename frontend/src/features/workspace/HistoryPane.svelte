<script lang="ts">
  import { draft } from '$lib/state/draft.svelte';
  import { router } from '$lib/state/router.svelte';
  import type { GalleryAsset } from '$lib/types';

  interface Props {
    assets: GalleryAsset[];
    loading?: boolean;
  }

  let { assets, loading = false }: Props = $props();

  const collapsed = $derived(draft.state.historyCollapsed);

  function toggle(): void {
    draft.update('historyCollapsed', !draft.state.historyCollapsed);
  }

  function openGallery(): void {
    router.navigate('gallery');
  }
</script>

<section
  id="ws-history-shell"
  class="panel-shell panel-shell-right relative z-30 flex shrink-0 overflow-hidden transition-[width] duration-200 ease-out"
  style="width: {collapsed ? '3rem' : '20rem'}"
>
  {#if !collapsed}
    <div class="min-w-0 flex-1 overflow-y-auto flex flex-col custom-scrollbar transition-opacity duration-150 ease-out">
      <div class="panel-header sticky top-0 z-30 flex items-center justify-between px-4 py-3 backdrop-blur">
        <div>
          <h2 class="field-label">Session History</h2>
        </div>
        <div class="flex items-center gap-2">
          <button
            type="button"
            class="surface-link-muted text-xs"
            onclick={openGallery}
          >
            Open Gallery
          </button>
        </div>
      </div>

      {#if loading}
        <div class="flex items-center justify-center py-8">
          <div class="animate-spin rounded-full h-5 w-5 border-t-2 border-teal-500"></div>
        </div>
      {:else if assets.length === 0}
        <div class="p-4 text-center text-zinc-600 text-sm">
          <p>No history yet.</p>
          <p class="mt-1 text-xs">Generated assets will appear here.</p>
        </div>
      {:else}
        <div class="divide-y divide-zinc-900">
          {#each assets as asset (asset.path)}
            <div class="cursor-pointer p-3 transition group hover:bg-zinc-900/40">
              <div class="flex gap-3">
                <div class="surface-card h-14 w-14 shrink-0 overflow-hidden">
                  {#if asset.media_type === 'video'}
                    <video
                      src={asset.url}
                      class="w-full h-full object-cover"
                      muted
                      preload="none"
                    ></video>
                  {:else}
                    <img
                      src={asset.thumbnail_url || asset.url}
                      alt={asset.prompt}
                      class="w-full h-full object-cover"
                      loading="lazy"
                    >
                  {/if}
                </div>
                <div class="min-w-0 flex-1">
                  <p class="text-xs text-zinc-300 line-clamp-2 leading-relaxed">{asset.prompt || asset.filename}</p>
                  <p class="mt-1 text-[10px] text-zinc-600 font-mono truncate">{asset.filename}</p>
                  <p class="mt-0.5 text-[10px] text-zinc-600">{asset.created_at}</p>
                </div>
              </div>
            </div>
          {/each}
        </div>
      {/if}
    </div>
  {/if}

  <!-- Toggle handle -->
  <div class="panel-handle w-12 shrink-0 backdrop-blur">
    <button
      type="button"
      id="ws-history-toggle"
      class="panel-handle-button flex h-full w-full flex-col items-center justify-start gap-4 px-2 py-4"
      aria-label="{collapsed ? 'Expand' : 'Collapse'} history pane"
      aria-expanded={!collapsed}
      onclick={toggle}
    >
      <svg
        class="w-4 h-4 shrink-0 transition-transform duration-200 ease-out {collapsed ? 'rotate-180' : ''}"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"></path>
      </svg>
      <span class="sr-only">Toggle session history</span>
      <span class="text-[10px] font-semibold uppercase tracking-[0.25em] text-current whitespace-nowrap -rotate-90 origin-center select-none">
        History
      </span>
    </button>
  </div>
</section>
