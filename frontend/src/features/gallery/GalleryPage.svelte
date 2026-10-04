<script lang="ts">
  import { onMount } from 'svelte';
  import { router } from '$lib/state/router.svelte';
  import { addToast } from '$lib/state/toasts.svelte';
  import { getGallery, deleteAsset } from '$lib/api/gallery';
  import { historyStore } from '$lib/state/history.svelte';
  import { jobStore } from '$lib/state/job.svelte';
  import { draft } from '$lib/state/draft.svelte';
  import { referenceParams, reuseParams, type DeleteOptions, type ReferenceTarget } from '$lib/state/assetActions';
  import type { GalleryAsset } from '$lib/types';
  import { AssetTile, AssetViewer, requestConfirm } from '$lib/components/molecules';
  import { confirmDeleteAsset } from '$lib/state/assetActions';
  import { hasOpenModal, isCommandKey, isPlainKey, isTyping } from '$lib/keyboard';
  import { gridColumns, moveInGrid } from './gridNav';

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
  let viewerTrigger: HTMLElement | null = null;

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
  let _refillAfterViewer = false;
  // Deletes since the last page load shift later server pages forward; the next load re-reads
  // the last loaded page so the assets that moved onto it are not skipped.
  let _pagesShifted = false;

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
    document.addEventListener('keydown', handleGridKeydown);

    return () => {
      document.removeEventListener('keydown', handleGridKeydown);
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
    _refillAfterViewer = false;
    _pagesShifted = false;
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
          lightboxOpen = true;
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
    const rereadLastPage = _pagesShifted;
    const expectedPage = page + 1;
    const requestedPage = rereadLastPage ? page : expectedPage;
    const requestedFilter = mediaFilter;
    const requestedSort = sortOrder;
    loadingMore = true;
    try {
      const result = await getGallery(requestedPage, requestedFilter, requestedSort);
      if (!requestIsCurrent(viewGeneration, mutationRevision, requestedFilter, requestedSort)) return;
      // Page one can reset pagination depth without changing the query. Do not
      // append a response that would leave a gap in the current page sequence.
      if (expectedPage !== page + 1) return;
      const returnedDeletedAssets = result.assets.filter((asset) => _successfullyDeletedIds.has(asset.id));
      const loadedIds = new Set(assets.map((asset) => asset.id));
      assets = [
        ...assets,
        ...result.assets.filter((asset) => !_successfullyDeletedIds.has(asset.id) && !loadedIds.has(asset.id))
      ];
      if (rereadLastPage) _pagesShifted = false;
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

  function deletableSelection(): GalleryAsset[] {
    return assets.filter(
      (asset) => selected.has(asset.id)
        && !deletingIds.has(asset.id)
        && !_successfullyDeletedIds.has(asset.id)
    );
  }

  async function deleteSelected(): Promise<void> {
    const count = deletableSelection().length;
    if (count === 0) return;
    const approved = await requestConfirm({
      question: `Delete ${count} selected asset${count !== 1 ? 's' : ''}?`,
      info: 'This cannot be undone.',
      confirmLabel: 'Delete',
    });
    if (!approved) return;
    // The selection may have changed while the dialog was open.
    const targets = deletableSelection();
    if (targets.length === 0) return;

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

  async function deleteSingle(asset: GalleryAsset, options: DeleteOptions = {}): Promise<void> {
    if (deletingIds.has(asset.id) || _successfullyDeletedIds.has(asset.id)) return;
    if (options.confirm !== false && !(await confirmDeleteAsset(asset))) return;
    if (deletingIds.has(asset.id) || _successfullyDeletedIds.has(asset.id)) return;
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
    const deletedIndex = selectedAsset ? Math.max(0, assets.findIndex((asset) => asset.id === selectedAsset!.id)) : 0;
    for (const asset of newlyDeleted) _successfullyDeletedIds.add(asset.id);

    const deletedIds = new Set(newlyDeleted.map((asset) => asset.id));
    historyStore.removeAssets(deletedIds);
    jobStore.removeOutputs(deletedIds);
    // A workspace reference pointing at a deleted file would fail the next run.
    if (newlyDeleted.some((asset) => asset.file_path && asset.file_path === draft.state.referenceImagePath)) {
      draft.update('referenceImagePath', null);
    }
    assets = assets.filter((asset) => !deletedIds.has(asset.id));
    const visibleIds = new Set(assets.map((asset) => asset.id));
    selected = new Set(
      Array.from(selected).filter((id) => !deletedIds.has(id) && visibleIds.has(id))
    );

    const matchingCount = newlyDeleted.filter(
      (asset) => mediaFilter === 'all' || asset.media_type === mediaFilter
    ).length;
    totalCount = Math.max(0, totalCount - matchingCount);

    // The viewer moves on to the neighbouring asset; with nothing left it closes.
    if (selectedAsset && deletedIds.has(selectedAsset.id)) {
      const next = assets[Math.min(deletedIndex, assets.length - 1)] ?? null;
      if (lightboxOpen && next) selectAsset(next);
      else clearActiveAsset();
    }

    // A mutation invalidates every earlier replacement/pagination/refill. The
    // new refill deliberately captures the latest query, not the query at click time.
    _mutationRevision += 1;
    _viewGeneration += 1;
    // Refilling replaces the list with page one, which would pull the viewer back off later pages.
    // While the viewer is open the local removal stands, and the refill runs once it closes.
    if (lightboxOpen) {
      _refillAfterViewer = true;
      _pagesShifted = true;
      return;
    }
    refillAfterMutation();
  }

  function refillAfterMutation(): void {
    _refillAfterViewer = false;
    void loadPageOne(mediaFilter, sortOrder, _viewGeneration, _mutationRevision, false, true);
  }

  function selectAsset(asset: GalleryAsset): void {
    selectedAsset = asset;
    router.replace('gallery', { selected: asset.id });
  }

  function openViewer(asset: GalleryAsset, trigger: HTMLElement): void {
    viewerTrigger = trigger;
    selectAsset(asset);
    lightboxOpen = true;
  }

  function closeViewer(): void {
    const trigger = viewerTrigger;
    viewerTrigger = null;
    clearActiveAsset();
    if (_refillAfterViewer) {
      _viewGeneration += 1;
      refillAfterMutation();
    }
    queueMicrotask(() => { if (trigger?.isConnected) trigger.focus(); });
  }

  function handleViewerNavigate(index: number): void {
    const target = assets[index];
    if (target) selectAsset(target);
  }

  // Navigate through the router so the WorkspacePage receives the params on mount.
  // Setting window.location.hash directly bypasses the router and drops the params.
  function reuseInWorkspace(asset: GalleryAsset): void {
    router.navigate('workspace', reuseParams(asset));
  }

  function useAsReference(asset: GalleryAsset, target: ReferenceTarget): void {
    router.navigate('workspace', referenceParams(asset, target));
  }

  // --- Keyboard -------------------------------------------------------------
  let gridEl = $state<HTMLDivElement | null>(null);

  function gridTiles(): HTMLElement[] {
    return Array.from(gridEl?.querySelectorAll<HTMLElement>('.asset-tile') ?? []);
  }

  function focusTile(tile: HTMLElement | undefined): void {
    tile?.querySelector<HTMLElement>('.asset-tile-media')?.focus();
  }

  function handleGridKeydown(e: KeyboardEvent): void {
    if (e.defaultPrevented || lightboxOpen || isTyping(e.target) || hasOpenModal()) return;
    if (isCommandKey(e) && !e.altKey && !e.shiftKey && e.key.toLowerCase() === 'a') {
      if (assets.length === 0) return;
      e.preventDefault();
      selected = new Set(assets.map((asset) => asset.id));
      return;
    }
    if (!isPlainKey(e)) return;

    const tiles = gridTiles();
    const index = tiles.findIndex((tile) => tile.contains(document.activeElement));
    const focused = index >= 0 ? assets[index] : undefined;
    if (e.key === 'Escape') {
      if (selected.size === 0) return;
      e.preventDefault();
      selected = new Set();
    } else if (e.key === 'Delete' || e.key === 'Backspace') {
      // A held key must not open a second confirmation.
      if (e.repeat) {
        e.preventDefault();
        return;
      }
      if (selected.size > 0) void deleteSelected();
      else if (focused) void deleteSingle(focused);
      else return;
      e.preventDefault();
    } else if (!focused) {
      return;
    } else if (e.key === ' ' || e.key.toLowerCase() === 'x') {
      e.preventDefault();
      toggleSelect(focused, !selected.has(focused.id));
    } else {
      const next = moveInGrid(index, e.key, gridColumns(tiles.map((tile) => tile.offsetTop)), tiles.length);
      if (next === null) return;
      e.preventDefault();
      focusTile(tiles[next]);
    }
  }
</script>

<div id="gallery-view" class="flex-1 flex overflow-hidden">

  <!-- Left: scrollable grid -->
  <section id="gallery-scroll-region" class="panel-scroll-surface custom-scrollbar min-w-0 flex-1 overflow-y-auto p-4">
    <div class="mx-auto max-w-[120rem]">
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
        <div bind:this={gridEl} class="grid grid-cols-1 gap-4 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 2xl:grid-cols-6">
          {#each assets as asset (asset.id)}
            <AssetTile
              {asset}
              density="card"
              selected={selected.has(asset.id)}
              deleting={deletingIds.has(asset.id)}
              onselect={toggleSelect}
              onpreview={openViewer}
              onreuse={reuseInWorkspace}
              onreference={useAsReference}
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

</div>

<AssetViewer
  {assets}
  currentIndex={viewerIndex}
  open={lightboxOpen && selectedAsset !== null}
  setLabel="Gallery"
  {deletingIds}
  onclose={closeViewer}
  onnavigate={handleViewerNavigate}
  onnearend={() => { if (hasMore) void loadMorePages(); }}
  onreuse={reuseInWorkspace}
  onreference={useAsReference}
  ondelete={deleteSingle}
/>
