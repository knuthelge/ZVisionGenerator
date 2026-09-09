<script lang="ts">
  import { onMount } from 'svelte';
  import { router } from '$lib/state/router.svelte';
  import { addToast } from '$lib/state/toasts.svelte';
  import { getGallery, deleteAsset } from '$lib/api/gallery';
  import type { GalleryAsset } from '$lib/types';
  import ImageCard from '$lib/components/molecules/ImageCard.svelte';
  import Lightbox from '$lib/components/molecules/Lightbox.svelte';

  let assets = $state<GalleryAsset[]>([]);
  let page = $state(1);
  let totalPages = $state(1);
  let totalCount = $state(0);
  let loading = $state(true);
  let loadingMore = $state(false);
  let pageOnePending = $state(false);
  let error = $state<string | null>(null);

  let mediaFilter = $state<'all' | 'image' | 'video'>('all');
  let sortOrder = $state<'newest' | 'oldest'>('newest');
  let selected = $state<Set<string>>(new Set());
  let deletingIds = $state<Set<string>>(new Set());
  let bulkDeleteRuns = $state(0);

  let selectedAsset = $state<GalleryAsset | null>(null);
  let lightboxOpen = $state(false);

  const selectedCount = $derived(selected.size);
  const deletableSelectedCount = $derived(
    Array.from(selected).filter((id) => !deletingIds.has(id)).length
  );
  const hasMore = $derived(page < totalPages);
  const emptyState = $derived(assets.length === 0 && !loading && !error);
  const filteredEmptyState = $derived(emptyState && mediaFilter !== 'all');

  // Pending URL-based selection to restore after the first page load.
  let _pendingSelected: string | null = null;

  // View changes and mutations are independent commit authorities. Every request
  // captures both, as well as its exact query, before it may update the UI.
  let _viewGeneration = 0;
  let _mutationRevision = 0;
  let _pageOneRequestRevision = 0;
  const _successfullyDeletedIds = new Set<string>();

  // Sentinel element for infinite scroll
  let sentinelEl = $state<HTMLDivElement | undefined>(undefined);

  onMount(() => {
    // Record any URL-based selected asset so loadPage can restore it.
    const params = router.params;
    if (params.selected) {
      _pendingSelected = params.selected;
    }

    _viewGeneration += 1;
    void loadPageOne(mediaFilter, sortOrder, _viewGeneration, _mutationRevision, true);

    return () => {
      // Invalidate every request still awaiting a response after unmount.
      _viewGeneration += 1;
      _mutationRevision += 1;
    };
  });

  const viewerIndex = $derived(
    selectedAsset ? assets.findIndex((a) => a.id === selectedAsset!.id) : 0
  );

  $effect(() => {
    if (!sentinelEl || pageOnePending) return;
    const io = new IntersectionObserver(
      (entries) => {
        if (entries[0]?.isIntersecting && hasMore && !loadingMore && !pageOnePending) {
          loadMorePages();
        }
      },
      { threshold: 0.2 }
    );
    io.observe(sentinelEl);
    return () => io.disconnect();
  });

  function requestIsCurrent(
    viewGeneration: number,
    mutationRevision: number,
    filter: string,
    sort: string
  ): boolean {
    return viewGeneration === _viewGeneration
      && mutationRevision === _mutationRevision
      && filter === mediaFilter
      && sort === sortOrder;
  }

  function clearActiveAsset(removeRoute = true): void {
    selectedAsset = null;
    lightboxOpen = false;
    _pendingSelected = null;
    if (removeRoute) router.replace('gallery', {});
  }

  async function loadPageOne(
    filter: string,
    sort: string,
    viewGeneration: number,
    mutationRevision: number,
    restorePendingSelection = false,
    preserveLocalOnError = false
  ): Promise<void> {
    const pageOneRequestRevision = ++_pageOneRequestRevision;
    pageOnePending = true;
    if (!preserveLocalOnError) loading = true;
    loadingMore = false;
    // Preserve local assets during a mutation refill, but never let an error
    // from an invalidated replacement request keep hiding those assets.
    error = null;
    try {
      const result = await getGallery(1, filter, sort);
      if (!requestIsCurrent(viewGeneration, mutationRevision, filter, sort)) return;
      const returnedDeletedAssets = result.assets.filter((asset) => _successfullyDeletedIds.has(asset.id));
      assets = result.assets.filter((asset) => !_successfullyDeletedIds.has(asset.id));
      page = result.page;
      totalPages = result.total_pages;
      const staleDeletedCount = returnedDeletedAssets.filter(
        (asset) => filter === 'all' || asset.media_type === filter
      ).length;
      totalCount = Math.max(0, result.total_count - staleDeletedCount);
      const visibleIds = new Set(assets.map((asset) => asset.id));
      selected = new Set(Array.from(selected).filter((id) => visibleIds.has(id)));

      if (restorePendingSelection && _pendingSelected) {
        const found = assets.find((a) => a.id === _pendingSelected) ?? null;
        if (found) {
          selectedAsset = found;
        } else {
          clearActiveAsset();
        }
        _pendingSelected = null;
      } else if (selectedAsset) {
        const refreshedActive = assets.find((asset) => asset.id === selectedAsset!.id) ?? null;
        if (refreshedActive) {
          selectedAsset = refreshedActive;
        } else {
          clearActiveAsset();
        }
      }
    } catch (e) {
      if (!requestIsCurrent(viewGeneration, mutationRevision, filter, sort)) return;
      if (preserveLocalOnError) {
        addToast('Gallery refresh failed; deleted assets remain removed. Change the view to retry.', 'warning');
      } else {
        error = e instanceof Error ? e.message : 'Failed to load gallery';
      }
    } finally {
      if (pageOneRequestRevision === _pageOneRequestRevision) {
        pageOnePending = false;
      }
      if (requestIsCurrent(viewGeneration, mutationRevision, filter, sort)) {
        loading = false;
      }
    }
  }

  async function loadMorePages(): Promise<void> {
    if (loading || loadingMore || pageOnePending || !hasMore) return;
    const viewGeneration = _viewGeneration;
    const mutationRevision = _mutationRevision;
    const requestedPage = page + 1;
    const requestedFilter = mediaFilter;
    const requestedSort = sortOrder;
    loadingMore = true;
    try {
      const result = await getGallery(requestedPage, requestedFilter, requestedSort);
      if (!requestIsCurrent(viewGeneration, mutationRevision, requestedFilter, requestedSort)) return;
      // Page one can reset pagination depth without changing the query. Do not
      // append a response that would leave a gap in the current page sequence.
      if (requestedPage !== page + 1) return;
      const returnedDeletedAssets = result.assets.filter((asset) => _successfullyDeletedIds.has(asset.id));
      assets = [
        ...assets,
        ...result.assets.filter((asset) => !_successfullyDeletedIds.has(asset.id))
      ];
      page = result.page;
      totalPages = result.total_pages;
      const staleDeletedCount = returnedDeletedAssets.filter(
        (asset) => requestedFilter === 'all' || asset.media_type === requestedFilter
      ).length;
      totalCount = Math.max(0, result.total_count - staleDeletedCount);
    } catch {
      // ignore load-more errors silently
    } finally {
      if (requestIsCurrent(viewGeneration, mutationRevision, requestedFilter, requestedSort)) {
        loadingMore = false;
      }
    }
  }

  function beginReplacementView(
    filter: 'all' | 'image' | 'video',
    sort: 'newest' | 'oldest'
  ): void {
    mediaFilter = filter;
    sortOrder = sort;
    selected = new Set();
    clearActiveAsset();
    _viewGeneration += 1;
    void loadPageOne(filter, sort, _viewGeneration, _mutationRevision);
  }

  function onFilterChange(value: 'all' | 'image' | 'video'): void {
    beginReplacementView(value, sortOrder);
  }

  function onSortChange(value: 'newest' | 'oldest'): void {
    beginReplacementView(mediaFilter, value);
  }

  function clearMediaFilter(): void {
    beginReplacementView('all', sortOrder);
  }

  function openWorkspace(): void {
    router.navigate('workspace');
  }

  function toggleSelect(asset: GalleryAsset, isSelected: boolean): void {
    const next = new Set(selected);
    if (isSelected) {
      next.add(asset.id);
    } else {
      next.delete(asset.id);
    }
    selected = next;
  }

  async function deleteSelected(): Promise<void> {
    const targets = assets.filter(
      (asset) => selected.has(asset.id)
        && !deletingIds.has(asset.id)
        && !_successfullyDeletedIds.has(asset.id)
    );
    if (targets.length === 0) return;
    if (!confirm(`Delete ${targets.length} selected asset${targets.length !== 1 ? 's' : ''}?`)) return;

    markDeleting(targets.map((asset) => asset.id), true);
    bulkDeleteRuns += 1;
    try {
      const results = await Promise.allSettled(
        targets.map(async (asset) => deleteAsset(asset.id))
      );
      const deletedTargets = targets.filter((_, index) => results[index].status === 'fulfilled');
      const failedTargets = targets.filter((_, index) => results[index].status === 'rejected');
      reconcileSuccessfulDeletes(deletedTargets);

      if (failedTargets.length === 0) {
        addToast(
          `Deleted ${deletedTargets.length} selected asset${deletedTargets.length !== 1 ? 's' : ''}.`,
          'success'
        );
      } else if (deletedTargets.length > 0) {
        addToast(
          `Deleted ${deletedTargets.length}; ${failedTargets.length} failed and remain selected for retry.`,
          'warning'
        );
      } else {
        addToast(
          `Delete failed for ${failedTargets.length} selected asset${failedTargets.length !== 1 ? 's' : ''}; they remain selected for retry.`,
          'error'
        );
      }
    } finally {
      markDeleting(targets.map((asset) => asset.id), false);
      bulkDeleteRuns = Math.max(0, bulkDeleteRuns - 1);
    }
  }

  async function deleteSingle(asset: GalleryAsset): Promise<void> {
    if (deletingIds.has(asset.id) || _successfullyDeletedIds.has(asset.id)) return;
    if (!confirm(`Delete "${asset.filename}"?`)) return;
    markDeleting([asset.id], true);
    try {
      await deleteAsset(asset.id);
      reconcileSuccessfulDeletes([asset]);
      addToast('Deleted', 'success');
    } catch {
      addToast('Delete failed', 'error');
    } finally {
      markDeleting([asset.id], false);
    }
  }

  function markDeleting(ids: string[], pending: boolean): void {
    const next = new Set(deletingIds);
    for (const id of ids) {
      if (pending) next.add(id);
      else next.delete(id);
    }
    deletingIds = next;
  }

  function reconcileSuccessfulDeletes(targets: GalleryAsset[]): void {
    const newlyDeleted = targets.filter((asset) => !_successfullyDeletedIds.has(asset.id));
    if (newlyDeleted.length === 0) return;
    for (const asset of newlyDeleted) _successfullyDeletedIds.add(asset.id);

    const deletedIds = new Set(newlyDeleted.map((asset) => asset.id));
    assets = assets.filter((asset) => !deletedIds.has(asset.id));
    const visibleIds = new Set(assets.map((asset) => asset.id));
    selected = new Set(
      Array.from(selected).filter((id) => !deletedIds.has(id) && visibleIds.has(id))
    );

    const matchingCount = newlyDeleted.filter(
      (asset) => mediaFilter === 'all' || asset.media_type === mediaFilter
    ).length;
    totalCount = Math.max(0, totalCount - matchingCount);

    if (selectedAsset && deletedIds.has(selectedAsset.id)) {
      clearActiveAsset();
    }

    // A mutation invalidates every earlier replacement/pagination/refill. The
    // new refill deliberately captures the latest query, not the query at click time.
    _mutationRevision += 1;
    _viewGeneration += 1;
    const filter = mediaFilter;
    const sort = sortOrder;
    void loadPageOne(filter, sort, _viewGeneration, _mutationRevision, false, true);
  }

  function selectAsset(asset: GalleryAsset): void {
    selectedAsset = asset;
    router.replace('gallery', { selected: asset.id });
  }

  function openLightbox(): void {
    lightboxOpen = true;
  }

  function closeLightbox(): void {
    lightboxOpen = false;
  }

  function handleOpenLightbox(asset: GalleryAsset): void {
    selectAsset(asset);
    openLightbox();
  }

  function handleLightboxNavigate(index: number): void {
    const target = assets[index];
    if (target) selectAsset(target);
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
  <section id="gallery-scroll-region" class="panel-scroll-surface custom-scrollbar min-w-0 flex-1 overflow-y-auto p-4">
    <div class="max-w-7xl mx-auto">
      <!-- Header -->
      <div class="flex flex-wrap items-center justify-between mb-4 gap-3 border-b border-border-subtle pb-3">
        <div>
          <h2 class="text-base font-semibold text-zinc-100">Gallery History</h2>
          <p class="text-xs text-zinc-500 mt-1">
            Browsing {assets.length} loaded asset{assets.length !== 1 ? 's' : ''} of {totalCount}, {sortOrder === 'newest' ? 'newest' : 'oldest'} first.
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
            disabled={selectedCount === 0 || deletableSelectedCount === 0}
            onclick={deleteSelected}
          >{bulkDeleteRuns > 0 ? 'Deleting…' : 'Delete Selected'}</button>
          <span class="text-xs text-zinc-500">{selectedCount} selected</span>
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
      {:else if emptyState}
        <div class="surface-empty-state flex min-h-96 flex-col items-center justify-center gap-4 rounded-md border border-border-subtle px-6 py-12 text-center">
          <div>
            <h3 class="text-base font-semibold text-zinc-100">
              {filteredEmptyState ? 'No matching assets' : 'No generated assets yet'}
            </h3>
            <p class="mt-2 max-w-md text-sm text-zinc-400">
              {filteredEmptyState ? `No ${mediaFilter} assets match the current gallery filter.` : 'Generated outputs appear here after an image or video job completes.'}
            </p>
          </div>
          {#if filteredEmptyState}
            <button
              type="button"
              class="surface-button-secondary rounded-md px-3 py-2 text-sm"
              onclick={clearMediaFilter}
            >Show All Media</button>
          {:else}
            <button
              type="button"
              class="surface-button-primary rounded-md px-3 py-2 text-sm"
              onclick={openWorkspace}
            >Open Workspace</button>
          {/if}
        </div>
      {:else}
        <!-- Grid -->
        <div class="grid grid-cols-1 gap-4 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5">
          {#each assets as asset (asset.id)}
            {@const isSelected = selected.has(asset.id)}
            {@const isActive = selectedAsset?.id === asset.id}
            <ImageCard
              {asset}
              selected={isSelected}
              active={isActive}
              onselect={toggleSelect}
              onactivate={selectAsset}
              onopenlightbox={handleOpenLightbox}
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
  <section id="gallery-details" class="panel-shell panel-shell-right custom-scrollbar relative hidden h-full w-80 flex-col overflow-y-auto sm:flex">
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
          <!-- Reuse notice: shown when backend metadata reports reuse availability reasons. -->
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
          {#if selectedAsset.has_reusable_config === true}
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
          {:else}
            <button
              type="button"
              class="surface-button-secondary w-full rounded-md py-2 font-medium opacity-70"
              disabled
            >Reusable settings unavailable</button>
          {/if}
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
            disabled={deletingIds.has(selectedAsset.id)}
            onclick={() => deleteSingle(selectedAsset!)}
          >
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"></path>
            </svg>
            {deletingIds.has(selectedAsset.id) ? 'Deleting…' : 'Delete'}
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
<Lightbox
  {assets}
  currentIndex={viewerIndex}
  open={lightboxOpen}
  onclose={closeLightbox}
  onnavigate={handleLightboxNavigate}
/>
