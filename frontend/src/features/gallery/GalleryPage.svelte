<script lang="ts">
  import { onMount } from 'svelte';
  import { router } from '$lib/state/router.svelte';
  import { addToast } from '$lib/state/toasts.svelte';
  import { getGallery, deleteAsset } from '$lib/api/gallery';
  import type { GalleryAsset } from '$lib/types';

  let assets = $state<GalleryAsset[]>([]);
  let page = $state(1);
  let totalPages = $state(1);
  let loading = $state(true);
  let loadingMore = $state(false);
  let error = $state<string | null>(null);

  let mediaFilter = $state<'all' | 'image' | 'video'>('all');
  let sortOrder = $state<'newest' | 'oldest'>('newest');
  let selected = $state<Set<string>>(new Set());

  let selectedAsset = $state<GalleryAsset | null>(null);
  let lightboxOpen = $state(false);

  // Filtered + sorted assets (client-side since API may not support it yet)
  const filteredAssets = $derived(
    assets
      .filter((a) => mediaFilter === 'all' || a.media_type === mediaFilter)
      .sort((a, b) =>
        sortOrder === 'newest'
          ? new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
          : new Date(a.created_at).getTime() - new Date(b.created_at).getTime()
      )
  );

  const selectedCount = $derived(selected.size);
  const hasMore = $derived(page < totalPages);

  // Sentinel element for infinite scroll
  let sentinelEl = $state<HTMLDivElement | undefined>(undefined);
  let observer: IntersectionObserver | null = null;

  onMount(() => {
    // Check URL for selected asset
    const params = router.params;
    if (params.selected) {
      // Will be populated once assets load
    }

    loadPage(1);

    return () => {
      observer?.disconnect();
    };
  });

  $effect(() => {
    if (sentinelEl) {
      observer?.disconnect();
      observer = new IntersectionObserver(
        (entries) => {
          if (entries[0]?.isIntersecting && hasMore && !loadingMore) {
            loadMorePages();
          }
        },
        { threshold: 0.2 }
      );
      observer.observe(sentinelEl);
    }
  });

  async function loadPage(p: number): Promise<void> {
    loading = true;
    error = null;
    try {
      const result = await getGallery(p);
      assets = result.assets;
      page = result.page;
      totalPages = result.total_pages;
    } catch (e) {
      error = e instanceof Error ? e.message : 'Failed to load gallery';
    } finally {
      loading = false;
    }
  }

  async function loadMorePages(): Promise<void> {
    if (loadingMore || !hasMore) return;
    loadingMore = true;
    try {
      const result = await getGallery(page + 1);
      assets = [...assets, ...result.assets];
      page = result.page;
      totalPages = result.total_pages;
    } catch {
      // ignore load-more errors silently
    } finally {
      loadingMore = false;
    }
  }

  function toggleSelect(asset: GalleryAsset, isSelected: boolean): void {
    const next = new Set(selected);
    if (isSelected) {
      next.add(asset.path);
    } else {
      next.delete(asset.path);
    }
    selected = next;
  }

  function selectAsset(asset: GalleryAsset): void {
    selectedAsset = asset;
    router.replace('gallery', { selected: asset.path });
  }

  async function deleteSelected(): Promise<void> {
    if (!confirm(`Delete ${selectedCount} selected asset${selectedCount !== 1 ? 's' : ''}?`)) return;
    const paths = Array.from(selected);
    await Promise.allSettled(paths.map((p) => deleteAsset(p)));
    assets = assets.filter((a) => !selected.has(a.path));
    selected = new Set();
    if (selectedAsset && paths.includes(selectedAsset.path)) {
      selectedAsset = null;
    }
  }

  async function deleteSingle(asset: GalleryAsset): Promise<void> {
    if (!confirm(`Delete "${asset.filename}"?`)) return;
    try {
      await deleteAsset(asset.path);
      assets = assets.filter((a) => a.path !== asset.path);
      if (selectedAsset?.path === asset.path) selectedAsset = null;
      addToast('Deleted', 'success');
    } catch {
      addToast('Delete failed', 'error');
    }
  }

  function openLightbox(): void {
    lightboxOpen = true;
  }

  function closeLightbox(): void {
    lightboxOpen = false;
  }

  function reuseInWorkspace(asset: GalleryAsset): void {
    // Navigate to workspace with prefill from reuse URL
    if (asset.reuse_workspace_url) {
      window.location.hash = asset.reuse_workspace_url.replace(/^#/, '');
    } else {
      router.navigate('workspace');
    }
  }
</script>

<div id="gallery-view" class="flex-1 flex overflow-hidden">

  <!-- Left: scrollable grid -->
  <section id="gallery-scroll-region" class="flex-1 bg-zinc-900 overflow-y-auto p-6 custom-scrollbar">
    <div class="max-w-7xl mx-auto">
      <!-- Header -->
      <div class="flex flex-wrap items-start justify-between mb-6 gap-4">
        <div>
          <h2 class="text-xl font-semibold text-zinc-100">Gallery History</h2>
          <p class="text-sm text-zinc-500">
            Browsing {filteredAssets.length} loaded asset{filteredAssets.length !== 1 ? 's' : ''}, {sortOrder === 'newest' ? 'newest' : 'oldest'} first.
          </p>
        </div>
        <div class="flex flex-wrap items-center justify-end gap-2">
          <!-- Filter -->
          <select
            class="bg-zinc-950 border border-zinc-800 rounded px-3 py-1.5 text-sm text-zinc-300 focus:outline-none focus:border-teal-500 focus:ring-1 focus:ring-teal-500 transition"
            aria-label="Filter gallery media"
            onchange={(e) => { mediaFilter = (e.currentTarget as HTMLSelectElement).value as typeof mediaFilter; }}
          >
            <option value="all" selected={mediaFilter === 'all'}>All Media</option>
            <option value="image" selected={mediaFilter === 'image'}>Images Only</option>
            <option value="video" selected={mediaFilter === 'video'}>Videos Only</option>
          </select>

          <!-- Sort -->
          <select
            class="bg-zinc-950 border border-zinc-800 rounded px-3 py-1.5 text-sm text-zinc-300 focus:outline-none focus:border-teal-500 focus:ring-1 focus:ring-teal-500 transition"
            aria-label="Sort gallery assets"
            onchange={(e) => { sortOrder = (e.currentTarget as HTMLSelectElement).value as typeof sortOrder; }}
          >
            <option value="newest" selected={sortOrder === 'newest'}>Newest First</option>
            <option value="oldest" selected={sortOrder === 'oldest'}>Oldest First</option>
          </select>

          <button
            type="button"
            class="rounded-md border border-zinc-800 bg-zinc-950 px-3 py-1.5 text-sm text-zinc-300 transition hover:border-red-500/40 hover:text-red-300 disabled:opacity-50"
            disabled={selectedCount === 0}
            onclick={deleteSelected}
          >Delete Selected</button>
          <span class="text-xs font-mono uppercase tracking-[0.18em] text-zinc-500">{selectedCount} selected</span>
        </div>
      </div>

      {#if loading}
        <div class="flex items-center justify-center py-24">
          <div class="animate-spin rounded-full h-8 w-8 border-t-2 border-teal-500"></div>
        </div>
      {:else if error}
        <div class="rounded-lg border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-100">
          {error}
        </div>
      {:else}
        <!-- Grid -->
        <div class="grid grid-cols-1 gap-4 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5">
          {#each filteredAssets as asset (asset.path)}
            {@const isSelected = selected.has(asset.path)}
            {@const isActive = selectedAsset?.path === asset.path}
            <article
              class="relative group overflow-hidden rounded-lg border transition-all cursor-pointer
                {isActive || isSelected
                  ? 'border-teal-500/60 ring-2 ring-teal-500/50 bg-teal-500/5'
                  : 'border-zinc-800 hover:border-zinc-700 bg-zinc-950'}"
              onclick={() => selectAsset(asset)}
              role="button"
              tabindex="0"
              onkeydown={(e) => e.key === 'Enter' && selectAsset(asset)}
              aria-label="Asset: {asset.filename}"
            >
              <!-- Checkbox -->
              <!-- svelte-ignore a11y_click_events_have_key_events -->
              <div
                class="absolute top-2 left-2 z-10"
                onclick={(e) => { e.stopPropagation(); toggleSelect(asset, !isSelected); }}
              >
                <input
                  type="checkbox"
                  checked={isSelected}
                  class="rounded border-zinc-700 bg-zinc-900 text-teal-500 focus:ring-teal-500"
                  aria-label="Select {asset.filename}"
                >
              </div>

              <!-- Thumbnail -->
              <div class="aspect-square bg-zinc-900 overflow-hidden">
                {#if asset.media_type === 'video'}
                  <video
                    src={asset.thumbnail_url || asset.url}
                    class="w-full h-full object-cover"
                    muted
                    preload="none"
                  ></video>
                {:else}
                  <img
                    src={asset.thumbnail_url || asset.url}
                    alt={asset.prompt || asset.filename}
                    class="w-full h-full object-cover"
                    loading="lazy"
                  >
                {/if}
              </div>

              <!-- Footer -->
              <div class="p-2 bg-zinc-850">
                <p class="text-xs font-medium text-zinc-200 truncate">{asset.filename}</p>
                <p class="text-[10px] text-zinc-500 mt-0.5">{asset.created_at}</p>
              </div>

              <!-- Hover actions -->
              <div class="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center gap-2 pt-8">
                <button
                  type="button"
                  class="rounded-md border border-zinc-600 bg-zinc-900/80 px-2 py-1 text-xs text-zinc-200 hover:text-white transition"
                  onclick={(e) => { e.stopPropagation(); reuseInWorkspace(asset); }}
                  aria-label="Reuse settings"
                >Reuse</button>
                <a
                  href={asset.url}
                  download={asset.filename}
                  class="rounded-md border border-zinc-600 bg-zinc-900/80 px-2 py-1 text-xs text-zinc-200 hover:text-white transition"
                  onclick={(e) => e.stopPropagation()}
                  aria-label="Download"
                >↓</a>
              </div>
            </article>
          {/each}
        </div>

        <!-- Infinite scroll sentinel + pagination -->
        <div id="gallery-pagination" class="py-12 flex justify-center items-center">
          {#if hasMore}
            <div bind:this={sentinelEl} class="flex items-center gap-3 rounded-lg border border-zinc-800 bg-zinc-950 px-4 py-3 text-sm text-zinc-400">
              {#if loadingMore}
                <svg class="h-4 w-4 animate-spin text-teal-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                  <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                <span>Loading more...</span>
              {:else}
                <span>Scroll for more</span>
              {/if}
            </div>
          {:else if filteredAssets.length > 0}
            <div class="flex items-center gap-2 text-zinc-500 text-sm">
              <span>All assets loaded</span>
            </div>
          {/if}
        </div>
      {/if}
    </div>
  </section>

  <!-- Right: detail panel -->
  <section id="gallery-details" class="w-80 bg-zinc-950 border-l border-zinc-900 flex flex-col relative h-full overflow-y-auto custom-scrollbar">
    <div class="p-4 border-b border-zinc-900 sticky top-0 bg-zinc-950/95 backdrop-blur z-10 flex items-center justify-between">
      <h3 class="text-sm font-semibold text-zinc-100">Asset Details</h3>
      {#if selectedAsset}
        <a
          href={selectedAsset.url}
          download={selectedAsset.filename}
          class="text-zinc-500 hover:text-zinc-300"
          aria-label="Download selected asset"
        >
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path>
          </svg>
        </a>
      {/if}
    </div>

    <div class="p-4 space-y-6">
      {#if selectedAsset}
        <!-- Preview -->
        <div class="aspect-square bg-zinc-900 rounded-md border border-zinc-800 flex items-center justify-center overflow-hidden">
          {#if selectedAsset.media_type === 'video'}
            <video
              src={selectedAsset.url}
              controls
              muted
              preload="metadata"
              class="w-full h-full object-contain"
            ></video>
          {:else}
            <img
              src={selectedAsset.url}
              alt={selectedAsset.filename}
              class="w-full h-full object-contain"
            >
          {/if}
        </div>

        <button
          type="button"
          class="w-full rounded-md border border-zinc-800 bg-zinc-900 px-3 py-2 text-sm text-zinc-200 transition hover:border-teal-500 hover:text-white"
          onclick={openLightbox}
        >Open Fullscreen Viewer</button>

        <!-- Prompt -->
        <div>
          <label class="block text-[11px] font-semibold text-zinc-500 uppercase tracking-wider mb-2">Prompt</label>
          <div class="bg-zinc-900 border border-zinc-800 rounded p-3 text-sm text-zinc-300 leading-relaxed font-mono">
            {selectedAsset.prompt || '—'}
          </div>
        </div>

        <!-- Info grid -->
        <div>
          <label class="block text-[11px] font-semibold text-zinc-500 uppercase tracking-wider mb-2">Generation Info</label>
          <div class="grid grid-cols-2 gap-3 text-sm">
            <div class="bg-zinc-900 border border-zinc-800 rounded px-3 py-2">
              <p class="text-[10px] text-zinc-500 uppercase font-semibold mb-1">Model</p>
              <p class="text-zinc-200 truncate">{selectedAsset.model || '—'}</p>
            </div>
            <div class="bg-zinc-900 border border-zinc-800 rounded px-3 py-2">
              <p class="text-[10px] text-zinc-500 uppercase font-semibold mb-1">Type</p>
              <p class="text-zinc-200 font-mono">{selectedAsset.media_type}</p>
            </div>
            {#if selectedAsset.width}
              <div class="bg-zinc-900 border border-zinc-800 rounded px-3 py-2">
                <p class="text-[10px] text-zinc-500 uppercase font-semibold mb-1">Dimensions</p>
                <p class="text-zinc-200 font-mono">{selectedAsset.width}×{selectedAsset.height}</p>
              </div>
            {/if}
          </div>
        </div>

        <!-- Actions -->
        <div class="pt-4 border-t border-zinc-900 space-y-2 pb-4">
          <button
            type="button"
            class="w-full bg-teal-600 hover:bg-teal-500 text-white font-medium py-2 rounded shadow-sm transition flex items-center justify-center gap-2"
            onclick={() => reuseInWorkspace(selectedAsset!)}
          >
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 7v8a2 2 0 002 2h6M8 7V5a2 2 0 012-2h4.586a1 1 0 01.707.293l4.414 4.414a1 1 0 01.293.707V15a2 2 0 01-2 2h-2M8 7H6a2 2 0 00-2 2v10a2 2 0 002 2h8a2 2 0 002-2v-2"></path>
            </svg>
            Reuse in Workspace
          </button>
          <a
            href={selectedAsset.url}
            download={selectedAsset.filename}
            class="w-full bg-zinc-900 hover:bg-zinc-800 border border-zinc-800 text-zinc-200 font-medium py-2 rounded shadow-sm transition flex items-center justify-center gap-2"
          >
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path>
            </svg>
            Download Full Asset
          </a>
          <button
            type="button"
            class="w-full bg-zinc-900 hover:bg-red-500/10 hover:text-red-400 hover:border-red-500/30 border border-zinc-800 text-zinc-400 font-medium py-2 rounded transition flex items-center justify-center gap-2"
            onclick={() => deleteSingle(selectedAsset!)}
          >
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"></path>
            </svg>
            Delete
          </button>
        </div>
      {:else}
        <div class="aspect-square bg-zinc-900 rounded-md border border-dashed border-zinc-800 flex items-center justify-center text-zinc-700 text-sm">
          No asset selected
        </div>
      {/if}
    </div>
  </section>
</div>

<!-- Lightbox -->
{#if lightboxOpen && selectedAsset}
  <!-- svelte-ignore a11y_click_events_have_key_events a11y_no_static_element_interactions -->
  <div
    class="fixed inset-0 z-50 bg-black/90 p-6"
    onclick={(e) => { if (e.target === e.currentTarget) closeLightbox(); }}
  >
    <button
      type="button"
      class="absolute right-6 top-6 rounded-md border border-zinc-700 bg-zinc-950/80 px-3 py-2 text-sm text-zinc-200 hover:bg-zinc-800 transition"
      onclick={closeLightbox}
    >Close</button>
    <div class="flex h-full w-full items-center justify-center">
      {#if selectedAsset.media_type === 'video'}
        <video
          src={selectedAsset.url}
          controls
          class="max-h-full max-w-full object-contain"
        ></video>
      {:else}
        <img
          src={selectedAsset.url}
          alt={selectedAsset.filename}
          class="max-h-full max-w-full object-contain"
        >
      {/if}
    </div>
  </div>
{/if}
