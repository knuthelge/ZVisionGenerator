<script lang="ts">
  import { onMount } from 'svelte';
  import { router } from '$lib/state/router.svelte';
  import { addToast } from '$lib/state/toasts.svelte';
  import { getGallery, deleteAsset } from '$lib/api/gallery';
  import type { GalleryAsset } from '$lib/types';
  import ImageCard from '$lib/components/molecules/ImageCard.svelte';

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

  const selectedCount = $derived(selected.size);
  const hasMore = $derived(page < totalPages);

  // Pending URL-based selection to restore after the first page load.
  let _pendingSelected: string | null = null;

  // Sentinel element for infinite scroll
  let sentinelEl = $state<HTMLDivElement | undefined>(undefined);

  onMount(() => {
    // Record any URL-based selected asset so loadPage can restore it.
    const params = router.params;
    if (params.selected) {
      _pendingSelected = params.selected;
    }

    loadPage(1, mediaFilter, sortOrder);
  });

  $effect(() => {
    if (!lightboxOpen) return;
    function handleEsc(e: KeyboardEvent): void {
      if (e.key === 'Escape') { e.preventDefault(); closeLightbox(); }
    }
    document.addEventListener('keydown', handleEsc);
    return () => document.removeEventListener('keydown', handleEsc);
  });

  $effect(() => {
    if (!sentinelEl) return;
    const io = new IntersectionObserver(
      (entries) => {
        if (entries[0]?.isIntersecting && hasMore && !loadingMore) {
          loadMorePages();
        }
      },
      { threshold: 0.2 }
    );
    io.observe(sentinelEl);
    return () => io.disconnect();
  });

  async function loadPage(p: number, filter: string = mediaFilter, sort: string = sortOrder): Promise<void> {
    loading = true;
    error = null;
    try {
      const result = await getGallery(p, filter, sort);
      assets = result.assets;
      page = result.page;
      totalPages = result.total_pages;
      // Restore selection from a URL ?selected= param after the initial page load.
      if (_pendingSelected && p === 1) {
        const found = assets.find((a) => a.path === _pendingSelected) ?? null;
        if (found) {
          selectedAsset = found;
        }
        _pendingSelected = null;
      }
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
      const result = await getGallery(page + 1, mediaFilter, sortOrder);
      assets = [...assets, ...result.assets];
      page = result.page;
      totalPages = result.total_pages;
    } catch {
      // ignore load-more errors silently
    } finally {
      loadingMore = false;
    }
  }

  function onFilterChange(value: 'all' | 'image' | 'video'): void {
    mediaFilter = value;
    selected = new Set();
    selectedAsset = null;
    _pendingSelected = null;
    loadPage(1, value, sortOrder);
  }

  function onSortChange(value: 'newest' | 'oldest'): void {
    sortOrder = value;
    selected = new Set();
    loadPage(1, mediaFilter, value);
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
    // Extract query params from the reuse URL and navigate via the router so
    // the WorkspacePage receives them on mount. Setting window.location.hash
    // directly bypasses the router and drops the params when the workspace tab
    // is subsequently activated.
    if (asset.reuse_workspace_url) {
      const queryStr = asset.reuse_workspace_url.replace(/^#\/[^?]*\??/, '');
      const params: Record<string, string> = {};
      if (queryStr) {
        new URLSearchParams(queryStr).forEach((v, k) => { params[k] = v; });
      }
      router.navigate('workspace', params);
    } else {
      router.navigate('workspace');
    }
  }
</script>

<div id="gallery-view" class="flex-1 flex overflow-hidden">

  <!-- Left: scrollable grid -->
  <section id="gallery-scroll-region" class="panel-scroll-surface custom-scrollbar flex-1 overflow-y-auto p-6">
    <div class="max-w-7xl mx-auto">
      <!-- Header -->
      <div class="flex flex-wrap items-start justify-between mb-6 gap-4">
        <div>
          <h2 class="text-xl font-semibold text-zinc-100">Gallery History</h2>
          <p class="text-sm text-zinc-500">
            Browsing {assets.length} loaded asset{assets.length !== 1 ? 's' : ''}, {sortOrder === 'newest' ? 'newest' : 'oldest'} first.
          </p>
        </div>
        <div class="flex flex-wrap items-center justify-end gap-2">
          <!-- Filter -->
          <select
            class="surface-select"
            aria-label="Filter gallery media"
            bind:value={mediaFilter}
            onchange={() => onFilterChange(mediaFilter)}
          >
            <option value="all">All Media</option>
            <option value="image">Images Only</option>
            <option value="video">Videos Only</option>
          </select>

          <!-- Sort -->
          <select
            class="surface-select"
            aria-label="Sort gallery assets"
            bind:value={sortOrder}
            onchange={() => onSortChange(sortOrder)}
          >
            <option value="newest">Newest First</option>
            <option value="oldest">Oldest First</option>
          </select>

          <button
            type="button"
            class="surface-button-danger rounded-md px-3 py-1.5 text-sm disabled:opacity-50"
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
          {#each assets as asset (asset.path)}
            {@const isSelected = selected.has(asset.path)}
            {@const isActive = selectedAsset?.path === asset.path}
            <ImageCard
              {asset}
              selected={isSelected}
              active={isActive}
              onselect={toggleSelect}
              onview={selectAsset}
              onreuse={reuseInWorkspace}
              ondelete={deleteSingle}
            />
          {/each}
        </div>

        <!-- Infinite scroll sentinel + pagination -->
        <div id="gallery-pagination" class="py-12 flex justify-center items-center">
          {#if hasMore}
            <div bind:this={sentinelEl} class="surface-card-muted flex items-center gap-3 px-4 py-3 text-sm text-zinc-400">
              {#if loadingMore}
                <svg class="text-primary-main h-4 w-4 animate-spin" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                  <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                <span>Loading more...</span>
              {:else}
                <span>Scroll for more</span>
              {/if}
            </div>
          {:else if assets.length > 0}
            <div class="flex items-center gap-2 text-zinc-500 text-sm">
              <span>All assets loaded</span>
            </div>
          {/if}
        </div>
      {/if}
    </div>
  </section>

  <!-- Right: detail panel -->
  <section id="gallery-details" class="panel-shell panel-shell-right custom-scrollbar relative flex h-full w-80 flex-col overflow-y-auto">
    <div class="panel-header sticky top-0 z-10 flex items-center justify-between p-4 backdrop-blur">
      <h3 class="text-sm font-semibold text-zinc-100">Asset Details</h3>
      {#if selectedAsset}
        <a
          href={selectedAsset.url}
          download={selectedAsset.filename}
          class="surface-link-muted"
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
        <div class="surface-card aspect-square flex items-center justify-center overflow-hidden">
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
          class="surface-button-secondary w-full rounded-md px-3 py-2 text-sm"
          onclick={openLightbox}
        >Open Fullscreen Viewer</button>

        <!-- Prompt -->
        <div>
          <h4 class="block text-[11px] font-semibold text-zinc-500 uppercase tracking-wider mb-2">Prompt</h4>
          <div class="surface-card p-3 text-sm text-zinc-300 leading-relaxed font-mono">
            {selectedAsset.prompt || '—'}
          </div>
        </div>

        <!-- Info grid -->
        <div>
          <h4 class="block text-[11px] font-semibold text-zinc-500 uppercase tracking-wider mb-2">Generation Info</h4>
          <div class="grid grid-cols-2 gap-3 text-sm">
            <div class="surface-card px-3 py-2">
              <p class="text-[10px] text-zinc-500 uppercase font-semibold mb-1">Model</p>
              <p class="text-zinc-200 truncate">{selectedAsset.model || '—'}</p>
            </div>
            <div class="surface-card px-3 py-2">
              <p class="text-[10px] text-zinc-500 uppercase font-semibold mb-1">Type</p>
              <p class="text-zinc-200 font-mono">{selectedAsset.media_type}</p>
            </div>
            {#if selectedAsset.width}
              <div class="surface-card px-3 py-2">
                <p class="text-[10px] text-zinc-500 uppercase font-semibold mb-1">Dimensions</p>
                <p class="text-zinc-200 font-mono">{selectedAsset.width}×{selectedAsset.height}</p>
              </div>
            {/if}
          </div>
        </div>

        <!-- Actions -->
        <div class="space-y-2 border-t border-border-subtle pb-4 pt-4">
          <!-- Reuse fallback warning: shown when the backend indicates a model or
               workflow had to be substituted during history/gallery reuse. -->
          {#if selectedAsset.reuse_state?.fallback_reasons && selectedAsset.reuse_state.fallback_reasons.length > 0}
            <div
              class="surface-warning rounded-md border px-3 py-2 text-sm mb-3"
              role="alert"
            >
              <p class="font-semibold text-zinc-100 flex items-center gap-2 mb-1">
                <svg class="surface-warning-icon w-4 h-4 shrink-0" fill="currentColor" viewBox="0 0 20 20">
                  <path fill-rule="evenodd" d="M8.485 2.495c.673-1.167 2.357-1.167 3.03 0l6.28 10.875c.673 1.167-.17 2.625-1.516 2.625H3.72c-1.347 0-2.189-1.458-1.515-2.625L8.485 2.495zM10 5a.75.75 0 01.75.75v3.5a.75.75 0 01-1.5 0v-3.5A.75.75 0 0110 5zm0 9a1 1 0 100-2 1 1 0 000 2z" clip-rule="evenodd"></path>
                </svg>
                Reuse notice
              </p>
              <ul class="text-xs text-zinc-300 space-y-0.5 list-disc list-inside">
                {#each selectedAsset.reuse_state.fallback_reasons as reason}
                  <li>{reason}</li>
                {/each}
              </ul>
            </div>
          {/if}
          <button
            type="button"
            class="surface-button-primary w-full rounded-md py-2 font-medium shadow-sm transition flex items-center justify-center gap-2"
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
            class="surface-button-secondary flex w-full items-center justify-center gap-2 rounded-md py-2 font-medium shadow-sm"
          >
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path>
            </svg>
            Download Full Asset
          </a>
          <button
            type="button"
            class="surface-button-danger flex w-full items-center justify-center gap-2 rounded-md py-2 font-medium"
            onclick={() => deleteSingle(selectedAsset!)}
          >
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"></path>
            </svg>
            Delete
          </button>
        </div>
      {:else}
        <div class="surface-empty-state aspect-square flex items-center justify-center text-sm">
          No asset selected
        </div>
      {/if}
    </div>
  </section>
</div>

<!-- Lightbox -->
{#if lightboxOpen && selectedAsset}
  <div class="fixed inset-0 z-50">
    <!-- Backdrop: native button so no a11y suppression needed -->
    <button
      type="button"
      class="absolute inset-0 bg-black/90"
      onclick={closeLightbox}
      aria-label="Close fullscreen viewer"
      tabindex="-1"
    ></button>
    <!-- Dialog container: pointer-events-none so backdrop button receives clicks on empty areas -->
    <div
      role="dialog"
      aria-modal="true"
      aria-label="Fullscreen viewer"
      class="relative flex h-full w-full items-center justify-center p-6 pointer-events-none"
    >
      <button
        type="button"
        class="surface-overlay-action pointer-events-auto absolute right-6 top-6 z-10 rounded-md px-3 py-2 text-sm"
        onclick={closeLightbox}
      >Close</button>
      {#if selectedAsset.media_type === 'video'}
        <video
          src={selectedAsset.url}
          controls
          class="pointer-events-auto relative z-10 max-h-full max-w-full object-contain"
        >
          <track kind="captions" src="/app-static/empty.vtt" default srclang="en" label="No captions available">
        </video>
      {:else}
        <img
          src={selectedAsset.url}
          alt={selectedAsset.filename}
          class="pointer-events-auto relative z-10 max-h-full max-w-full object-contain"
        >
      {/if}
    </div>
  </div>
{/if}
