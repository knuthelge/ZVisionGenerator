<script lang="ts">
  import { onDestroy, onMount, tick, untrack } from 'svelte';
  import { errorText } from '$lib/api/client';
  import { keepLoadedTail } from '$lib/state/pageMerge';
  import { galleryMemory, workspaceMemory } from '$lib/state/pageMemory';
  import { emptySlotsHolding } from '$lib/state/assetRemoval';
  import { projectPick } from '$lib/state/projectPick.svelte';
  import { moveAndReconcile } from '$lib/state/moveAndReconcile';
  import { starAndReconcile } from '$lib/state/starAndReconcile';
  import { router } from '$lib/state/router.svelte';
  import { addToast, dismissToast } from '$lib/state/toasts.svelte';
  import { getGallery, getGalleryIds, deleteAssets, downloadAssets } from '$lib/api/gallery';
  import { getWorkspaceCoreContext } from '$lib/api/workspace';
  import { galleryStore, type GalleryQuery, type SortOrder } from '$lib/state/gallery.svelte';
  import { viewAdmits, type MediaFilter } from '$lib/state/galleryFilter';
  import { belongsToView } from '$lib/state/viewMoves';
  import { jobStore } from '$lib/state/job.svelte';
  import { draft } from '$lib/state/draft.svelte';
  import { audioTrackParams, editParams, extendParams, frameParams, referenceParams, reuseParams, frameSetNotice, useAsState, voiceParams, type AudioUseTarget, type DeleteOptions, type FrameTarget, type ReferenceTarget } from '$lib/state/assetActions';
  import type { FramePoint } from '$lib/state/framePoint';
  import { runFrameGrab } from '$lib/state/frameGrab.svelte';
  import { heldUploads } from '$lib/state/heldUploads.svelte';
  import { requestReference } from '$lib/state/pendingReference.svelte';
  import { startUpscale } from '$lib/state/upscale';
  import type { GalleryAsset, MoveResult, RemovedAsset, UpscaleFactor, UpscaleMethod, WorkspaceContext } from '$lib/types';
  import { Button, Icon, Spinner } from '$lib/components/atoms';
  import { Alert, AssetTile, AssetViewer, EmptyState, MoveMenu, PageHeader, SaveVoiceDialog, Select, requestConfirm } from '$lib/components/molecules';
  import { confirmDeleteAsset } from '$lib/state/assetActions';
  import { hasOpenModal, isCommandKey, isPlainKey, isTyping } from '$lib/keyboard';
  import { gridColumns, moveInGrid } from './gridNav';
  import { modeLine, rangeIds, selectionProject, tileClickIntent, withAdded, withToggled } from './selection';
  import { layout } from '$lib/state/layoutMode.svelte';

  // One Select mode on every layout: in it a click or tap toggles a tile, and the header and the phone bar carry the bulk actions.
  let selectMode = $state(false);
  // The last tile toggled; Shift-click selects the run from it.
  let anchorId = $state<string | null>(null);
  // Select all is fetching the matching IDs.
  let selectingAll = $state(false);
  const compact = $derived(layout.compact);

  let loading = $state(true);
  let loadingMore = $state(false);
  let pageOnePending = $state(false);
  let error = $state<string | null>(null);

  let mediaFilter = $state<MediaFilter>('all');
  let sortOrder = $state<SortOrder>('newest');
  let starredOnly = $state(false);
  // The list is the gallery store's view of the last query that loaded: shared with the Workspace strip, and kept
  // across visits. A view whose replacement failed keeps showing the previous one.
  let shownQuery = $state<GalleryQuery>({ filter: 'all', sort: 'newest', starred: false });
  const view = $derived(galleryStore.view(shownQuery));
  const assets = $derived(view.assets);
  const page = $derived(view.page);
  const totalPages = $derived(view.totalPages);
  const totalCount = $derived(view.totalCount ?? 0);
  let selected = $state<Set<string>>(new Set());
  let deletingIds = $state<Set<string>>(new Set());
  let bulkDeleteRuns = $state(0);
  // The moves started and not yet answered (they run in turn); the bulk Move button's list is anchored at the button in the header or the select bar.
  let movesPending = $state(0);
  const moving = $derived(movesPending > 0);
  let bulkMoveOpen = $state(false);
  let bulkMoveAnchor = $state<HTMLElement | null>(null);

  let selectedAsset = $state<GalleryAsset | null>(null);
  let lightboxOpen = $state(false);
  let viewerTrigger = $state<HTMLElement | null>(null);

  const selectedCount = $derived(selected.size);
  // The selection's files that no delete holds: every one, loaded or not.
  const actionableIds = $derived(Array.from(selected).filter((id) => !deletingIds.has(id)));
  // Where the selected files sit when they share it; undefined for a mix, which checks no row.
  const moveCurrent = $derived(selectionProject(selected, assets, projectPick.current, mediaFilter));
  const hasMore = $derived(page < totalPages);
  const emptyState = $derived(assets.length === 0 && !loading && !error);
  const starredEmptyState = $derived(emptyState && starredOnly);
  const filteredEmptyState = $derived(emptyState && !starredOnly && mediaFilter !== 'all');
  // Not in a project lists the files no project holds; it is offered with All files only, since a project's files are all in it.
  const mediaOptions = $derived([
    { value: 'all', label: 'All media' },
    { value: 'image', label: 'Images only' },
    { value: 'video', label: 'Videos only' },
    { value: 'audio', label: 'Audio only' },
    ...(projectPick.current === null ? [{ value: 'no_project', label: 'Not in a project' }] : []),
  ]);
  /** Whether the view shown keeps a moved file. */
  function inView(asset: GalleryAsset): boolean {
    return belongsToView(asset, mediaFilter, projectPick.current, starredOnly);
  }
  const emptyTitle = $derived(
    starredEmptyState ? 'No starred files'
      : filteredEmptyState ? 'No matching files'
      : projectPick.current !== null ? `Nothing in ${projectPick.current} yet`
      : 'No generated files yet'
  );
  const emptyBody = $derived(
    starredEmptyState ? 'Star a file from its ⋯ menu or the viewer.'
      : filteredEmptyState ? (mediaFilter === 'no_project' ? 'Every file is in a project.' : `No ${mediaFilter} files match the current gallery filter.`)
      : projectPick.current !== null ? 'New work saves here. To add existing files, switch to All files and use Move to project.'
      : 'Generated outputs appear here after a job completes.'
  );


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
    // Pages remember: the view a previous visit left is drawn at once, scrolled where it was, and page one refreshes
    // behind it; the first visit loads into placeholder tiles.
    const visit = galleryMemory.get();
    if (visit) {
      // A project picked elsewhere since the visit has no files outside a project to show.
      mediaFilter = visit.filter === 'no_project' && projectPick.current !== null ? 'all' : visit.filter;
      sortOrder = visit.sort;
      starredOnly = visit.starred;
      shownQuery = { filter: mediaFilter, sort: visit.sort, starred: starredOnly };
    }
    // The store may already hold this view: from the last visit, or the strip's list on the first one.
    const known = galleryStore.view({ filter: mediaFilter, sort: sortOrder, starred: starredOnly }).assets.length > 0;
    if (known) {
      loading = false;
      void tick().then(() => { if (scrollRegion && visit) scrollRegion.scrollTop = visit.scrollTop; });
    }

    _viewGeneration += 1;
    void loadPageOne(mediaFilter, sortOrder, starredOnly, _viewGeneration, _mutationRevision, false, known);
    document.addEventListener('keydown', handleGridKeydown);

    return () => {
      document.removeEventListener('keydown', handleGridKeydown);
      if (error || loading) galleryMemory.forget();
      else galleryMemory.set({ filter: mediaFilter, sort: sortOrder, starred: starredOnly, scrollTop });
      // Invalidate every request still awaiting a response after unmount.
      _viewGeneration += 1;
      _mutationRevision += 1;
    };
  });

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
    sort: string,
    starred: boolean
  ): boolean {
    return viewGeneration === _viewGeneration
      && mutationRevision === _mutationRevision
      && filter === mediaFilter
      && sort === sortOrder
      && starred === starredOnly;
  }

  function clearActiveAsset(): void {
    selectedAsset = null;
    lightboxOpen = false;
  }

  async function loadPageOne(
    filter: MediaFilter,
    sort: SortOrder,
    starred: boolean,
    viewGeneration: number,
    mutationRevision: number,
    preserveLocalOnError = false,
    keepLoaded = false
  ): Promise<void> {
    const pageOneRequestRevision = ++_pageOneRequestRevision;
    _refillAfterViewer = false;
    _pagesShifted = false;
    pageOnePending = true;
    if (!preserveLocalOnError && !keepLoaded) loading = true;
    loadingMore = false;
    // Preserve local assets during a mutation refill, but never let an error
    // from an invalidated replacement request keep hiding those assets.
    error = null;
    try {
      const result = await getGallery(1, filter, sort, projectPick.current, starred);
      if (!requestIsCurrent(viewGeneration, mutationRevision, filter, sort, starred)) return;
      const returnedDeletedAssets = result.assets.filter((asset) => _successfullyDeletedIds.has(asset.id));
      const pageOne = result.assets.filter((asset) => !_successfullyDeletedIds.has(asset.id));
      // A remembered view keeps the pages loaded after page one, so the grid stays where it was scrolled.
      const merged = keepLoaded ? keepLoadedTail(pageOne, assets) : pageOne;
      const staleDeletedCount = returnedDeletedAssets.filter((asset) => viewAdmits(filter, starred, asset)).length;
      galleryStore.setView({ filter, sort, starred }, {
        assets: merged,
        page: merged.length > pageOne.length ? page : result.page,
        totalPages: result.total_pages,
        totalCount: Math.max(0, result.total_count - staleDeletedCount),
      });
      shownQuery = { filter, sort, starred };

      if (selectedAsset) {
        const refreshedActive = assets.find((asset) => asset.id === selectedAsset!.id) ?? null;
        if (refreshedActive) {
          selectedAsset = refreshedActive;
        } else {
          clearActiveAsset();
        }
      }
    } catch (e) {
      if (!requestIsCurrent(viewGeneration, mutationRevision, filter, sort, starred)) return;
      if (preserveLocalOnError) {
        addToast('Gallery refresh failed; deleted files remain removed. Change the view to retry.', 'warning');
      } else {
        error = errorText(e, 'Failed to load gallery');
      }
    } finally {
      if (pageOneRequestRevision === _pageOneRequestRevision) {
        pageOnePending = false;
      }
      if (requestIsCurrent(viewGeneration, mutationRevision, filter, sort, starred)) {
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
    const requestedStarred = starredOnly;
    loadingMore = true;
    try {
      const result = await getGallery(requestedPage, requestedFilter, requestedSort, projectPick.current, requestedStarred);
      if (!requestIsCurrent(viewGeneration, mutationRevision, requestedFilter, requestedSort, requestedStarred)) return;
      // Page one can reset pagination depth without changing the query. Do not
      // append a response that would leave a gap in the current page sequence.
      if (expectedPage !== page + 1) return;
      const returnedDeletedAssets = result.assets.filter((asset) => _successfullyDeletedIds.has(asset.id));
      const loadedIds = new Set(assets.map((asset) => asset.id));
      if (rereadLastPage) _pagesShifted = false;
      const staleDeletedCount = returnedDeletedAssets.filter((asset) => viewAdmits(requestedFilter, requestedStarred, asset)).length;
      galleryStore.setView({ filter: requestedFilter, sort: requestedSort, starred: requestedStarred }, {
        assets: [...assets, ...result.assets.filter((asset) => !_successfullyDeletedIds.has(asset.id) && !loadedIds.has(asset.id))],
        page: result.page,
        totalPages: result.total_pages,
        totalCount: Math.max(0, result.total_count - staleDeletedCount),
      });
    } catch {
      // ignore load-more errors silently
    } finally {
      if (requestIsCurrent(viewGeneration, mutationRevision, requestedFilter, requestedSort, requestedStarred)) {
        loadingMore = false;
      }
    }
  }

  function beginReplacementView(
    filter: MediaFilter,
    sort: 'newest' | 'oldest',
    starred: boolean
  ): void {
    mediaFilter = filter;
    sortOrder = sort;
    starredOnly = starred;
    // A view change leaves Select mode: the selection belongs to the view it was made in.
    leaveSelectMode();
    clearActiveAsset();
    _viewGeneration += 1;
    void loadPageOne(filter, sort, starred, _viewGeneration, _mutationRevision);
  }

  // A new pick replaces the view like a filter change does (selection cleared, viewer closed, responses from the old
  // project dropped) and starts at the top; filter and sort stay. The first run only records the pick.
  let _shownPick = projectPick.current;
  $effect(() => {
    const pick = projectPick.current;
    if (pick === _shownPick) return;
    _shownPick = pick;
    untrack(() => {
      beginReplacementView(pick !== null && mediaFilter === 'no_project' ? 'all' : mediaFilter, sortOrder, starredOnly);
      if (scrollRegion) scrollRegion.scrollTop = 0;
    });
  });

  function onFilterChange(value: MediaFilter): void {
    beginReplacementView(value, sortOrder, starredOnly);
  }

  function onSortChange(value: 'newest' | 'oldest'): void {
    beginReplacementView(mediaFilter, value, starredOnly);
  }

  function toggleStarredOnly(): void {
    beginReplacementView(mediaFilter, sortOrder, !starredOnly);
  }

  function clearMediaFilter(): void {
    beginReplacementView('all', sortOrder, starredOnly);
  }

  /** The starred empty state's action: turn ★ off and keep the media filter. */
  function showAllStarred(): void {
    beginReplacementView(mediaFilter, sortOrder, false);
  }

  function openWorkspace(): void {
    router.navigate('workspace');
  }

  // --- Select mode -----------------------------------------------------------
  /** Move focus to one of the bulk controls once the DOM has caught up with the state change that swaps them. */
  function focusBulk(name: 'select' | 'done' | 'move' | 'delete'): void {
    void tick().then(() => document.querySelector<HTMLElement>(`#gallery-view [data-bulk="${name}"]`)?.focus());
  }

  function enterSelectMode(): void {
    selectMode = true;
  }

  /** Leave Select mode: the selection and the anchor go with it. */
  function leaveSelectMode(): void {
    selectMode = false;
    selected = new Set();
    anchorId = null;
  }

  function startSelecting(): void {
    enterSelectMode();
    focusBulk('done');
  }

  function doneSelecting(): void {
    leaveSelectMode();
    focusBulk('select');
  }

  /** Enter the mode with this tile selected: its checkbox ticked outside the mode, and the menu's Select. */
  function startSelectWith(asset: GalleryAsset): void {
    enterSelectMode();
    selected = withAdded(selected, [asset.id]);
    anchorId = asset.id;
  }

  /** Set one tile's selection, from its checkbox in the mode. */
  function setSelected(asset: GalleryAsset, isSelected: boolean): void {
    const next = new Set(selected);
    if (isSelected) next.add(asset.id);
    else next.delete(asset.id);
    selected = next;
    anchorId = asset.id;
  }

  /** A click on a tile's media in the mode: toggle it, or with Shift add the run from the anchor. */
  function onMediaClick(asset: GalleryAsset, event: MouseEvent): void {
    if (tileClickIntent(event.shiftKey) === 'range') {
      selected = withAdded(selected, rangeIds(assets.map((item) => item.id), anchorId, asset.id));
      anchorId = asset.id;
    } else {
      selected = withToggled(selected, asset.id);
      anchorId = asset.id;
    }
  }

  /** Space and X on a focused tile: enter the mode if need be, and toggle the tile. */
  function toggleTile(asset: GalleryAsset): void {
    enterSelectMode();
    selected = withToggled(selected, asset.id);
    anchorId = asset.id;
  }

  /** Select every file the view matches, loaded or not: the IDs the server lists now, as a snapshot. */
  async function selectAllMatching(): Promise<void> {
    if (selectingAll) return;
    enterSelectMode();
    selectingAll = true;
    const viewGeneration = _viewGeneration;
    try {
      const ids = await getGalleryIds(mediaFilter, projectPick.current, starredOnly);
      if (viewGeneration !== _viewGeneration || !selectMode) return;
      selected = new Set(ids.filter((id) => !_successfullyDeletedIds.has(id)));
      anchorId = null;
    } catch (e) {
      addToast(`Could not select every file: ${errorText(e, 'the request failed')}`, 'error');
    } finally {
      selectingAll = false;
    }
  }

  function downloadSelected(): void {
    if (selected.size === 0) return;
    // Synchronous, straight from the click: Safari drops a download that starts after an await.
    downloadAssets(Array.from(selected));
    addToast(`Downloading ${selected.size.toLocaleString()} file${selected.size !== 1 ? 's' : ''}.`, 'success');
  }

  /** The selected files that no delete holds and that are not already gone, loaded or not. */
  function deletableSelection(): string[] {
    return Array.from(selected).filter((id) => !deletingIds.has(id) && !_successfullyDeletedIds.has(id));
  }

  // Retry acts on this page's selection, so its toast must not outlive the page.
  let retryToastId: string | null = null;
  onDestroy(() => { if (retryToastId) dismissToast(retryToastId); });

  /** Delete the whole selection in one request. A clean run ends Select mode; failures stay selected for Retry. */
  async function deleteSelected(): Promise<void> {
    const count = deletableSelection().length;
    if (count === 0) return;
    const approved = await requestConfirm({
      question: `Delete ${count.toLocaleString()} file${count !== 1 ? 's' : ''}?`,
      info: 'This cannot be undone.',
      confirmLabel: 'Delete',
    });
    if (!approved) return;
    // The selection may have changed while the dialog was open.
    const targets = deletableSelection();
    if (targets.length === 0) return;

    markDeleting(targets, true);
    bulkDeleteRuns += 1;
    let leave = false;
    try {
      const result = await deleteAssets(targets);
      reconcileSuccessfulDeletes(result.deleted);
      const deleted = result.deleted.length;
      const failed = result.failed.length;
      if (failed === 0) {
        addToast(`Deleted ${deleted.toLocaleString()} file${deleted !== 1 ? 's' : ''}.`, 'success');
        leave = true;
      } else {
        retryToastId = addToast(
          `Deleted ${deleted.toLocaleString()}; ${failed.toLocaleString()} failed and remain selected for retry.`,
          deleted > 0 ? 'warning' : 'error',
          { action: { label: 'Retry', run: () => void deleteSelected() } }
        );
      }
    } catch (e) {
      retryToastId = addToast(errorText(e, 'Delete failed'), 'error', { action: { label: 'Retry', run: () => void deleteSelected() } });
    } finally {
      markDeleting(targets, false);
      bulkDeleteRuns = Math.max(0, bulkDeleteRuns - 1);
    }
    if (leave) leaveSelectMode();
    focusBulk(leave ? 'select' : 'delete');
  }

  async function deleteSingle(asset: GalleryAsset, options: DeleteOptions = {}): Promise<void> {
    if (deletingIds.has(asset.id) || _successfullyDeletedIds.has(asset.id)) return;
    if (options.confirm !== false && !(await confirmDeleteAsset(asset))) return;
    if (deletingIds.has(asset.id) || _successfullyDeletedIds.has(asset.id)) return;
    markDeleting([asset.id], true);
    try {
      const { failed } = await deleteAssets([asset.id]);
      if (failed.length > 0) {
        addToast(failed[0].detail || 'Delete failed', 'error');
        return;
      }
      // The client's own asset, not the server's answer, so the lists drop the file by the form of path they hold.
      reconcileSuccessfulDeletes([asset]);
      addToast('Deleted', 'success');
    } catch (e) {
      addToast(errorText(e, 'Delete failed'), 'error');
    } finally {
      markDeleting([asset.id], false);
    }
  }

  /**
   * Star or unstar a file from a tile, its menu or the viewer. In the starred view an unstarred file leaves the list at
   * once, and the viewer, when it showed that file, steps to its neighbour.
   */
  async function starAsset(asset: GalleryAsset, starred: boolean): Promise<void> {
    const activeIndex = selectedAsset ? Math.max(0, assets.findIndex((item) => item.id === selectedAsset!.id)) : 0;
    const viewerShowing = selectedAsset?.id === asset.id;
    if (!(await starAndReconcile(asset, starred))) return;
    if (starredOnly && !starred) {
      // The file left this view, so it leaves the selection too: a bulk action must not act on what the view no longer shows.
      if (selected.has(asset.id)) selected = withToggled(selected, asset.id);
      afterListShrank(activeIndex, viewerShowing, true);
    }
  }

  async function moveSingle(asset: GalleryAsset, project: string | null): Promise<void> {
    await moveFiles([asset.id], project);
  }

  /** Move the whole selection, loaded or not. A clean run ends Select mode; files that failed stay selected. */
  async function moveSelection(project: string | null): Promise<void> {
    const result = await moveFiles(actionableIds, project);
    if (result && result.failed.length === 0) {
      leaveSelectMode();
      focusBulk('select');
    } else {
      focusBulk('move');
    }
  }

  /** Move files by ID, loaded or not; resolves to the server's answer, or null when nothing was asked or the request failed. */
  async function moveFiles(ids: readonly string[], project: string | null): Promise<MoveResult | null> {
    const movable = ids.filter((id) => !deletingIds.has(id) && !_successfullyDeletedIds.has(id));
    if (movable.length === 0) return null;
    movesPending += 1;
    try {
      // The position the viewer steps from if its file leaves: read before the move changes the list.
      const activeIndex = selectedAsset ? Math.max(0, assets.findIndex((asset) => asset.id === selectedAsset!.id)) : 0;
      // A move started while another runs waits its turn. A file that left the list or was deleted meanwhile is the
      // server's to answer: it reports that file as failed and moves the rest.
      const result = await moveAndReconcile(movable, project, workspaceMemory.get());
      if (result) reconcileMoves(movable, result, activeIndex);
      return result;
    } finally {
      movesPending -= 1;
    }
  }

  /**
   * Bring the page in line with a move the shared state already took in. A moved file keeps its place under its new ID where
   * it still belongs to the view and leaves where it does not. Moved IDs are never recorded as deleted: a file moved back
   * regains its old ID and must show. Files already in the target are done with and leave the selection; failed ones stay.
   */
  function reconcileMoves(targetIds: readonly string[], result: MoveResult, activeIndex: number): void {
    const moves = result.moved;
    const failedIds = new Set(result.failed.map((failure) => failure.id));
    const settledIds = new Set(targetIds.filter((id) => !failedIds.has(id)));
    selected = new Set(Array.from(selected).filter((id) => !settledIds.has(id)));
    if (moves.length === 0) return;
    const movedFrom = new Set(moves.map((move) => move.from));

    // The viewer follows a file that stays under its new ID.
    let viewerLeft = false;
    if (selectedAsset && movedFrom.has(selectedAsset.id)) {
      const current = moves.find((move) => move.from === selectedAsset!.id)!.asset;
      const stays = assets.find((asset) => asset.id === current.id);
      if (stays) selectAsset(stays);
      else viewerLeft = true;
    }

    // Only a project view loses files to a move; there the list refills, as after a delete.
    afterListShrank(activeIndex, viewerLeft, moves.some((move) => !inView(move.asset)));
  }

  /**
   * Settle the page after files left the list. The viewer, when its file left, moves to the neighbour at its old position
   * and closes with none. With *refill*, a mutation invalidates every earlier replacement, pagination and refill, and page
   * one loads again at once, or once the viewer closes: refilling replaces the list, which would pull the viewer back off
   * later pages.
   */
  function afterListShrank(activeIndex: number, viewerLeft: boolean, refill: boolean): void {
    if (viewerLeft) {
      const next = assets[Math.min(activeIndex, assets.length - 1)] ?? null;
      if (lightboxOpen && next) selectAsset(next);
      else clearActiveAsset();
    }
    if (!refill) return;
    // The refill deliberately captures the latest query, not the query at click time.
    _mutationRevision += 1;
    _viewGeneration += 1;
    if (lightboxOpen) {
      _refillAfterViewer = true;
      _pagesShifted = true;
      return;
    }
    refillAfterMutation();
  }

  function markDeleting(ids: string[], pending: boolean): void {
    const next = new Set(deletingIds);
    for (const id of ids) {
      if (pending) next.add(id);
      else next.delete(id);
    }
    deletingIds = next;
  }

  function reconcileSuccessfulDeletes(targets: readonly RemovedAsset[]): void {
    const newlyDeleted = targets.filter((asset) => !_successfullyDeletedIds.has(asset.id));
    if (newlyDeleted.length === 0) return;
    const deletedIndex = selectedAsset ? Math.max(0, assets.findIndex((asset) => asset.id === selectedAsset!.id)) : 0;
    for (const asset of newlyDeleted) _successfullyDeletedIds.add(asset.id);

    const deletedIds = new Set(newlyDeleted.map((asset) => asset.id));
    // The store takes them out of every list and count: this view, the strip, and any other query loaded.
    galleryStore.removeAssets(newlyDeleted);
    jobStore.removeOutputs(deletedIds);
    // A workspace slot pointing at a deleted file would fail the next run. Without a context the model's capabilities
    // are unknown, so the start image is assumed usable. A reference row stays, as in the Workspace; the Workspace
    // flags it missing when it next checks its images.
    emptySlotsHolding(newlyDeleted.map((asset) => asset.file_path), null);
    selected = new Set(Array.from(selected).filter((id) => !deletedIds.has(id)));

    afterListShrank(deletedIndex, selectedAsset !== null && deletedIds.has(selectedAsset.id), true);
  }

  function refillAfterMutation(): void {
    _refillAfterViewer = false;
    void loadPageOne(mediaFilter, sortOrder, starredOnly, _viewGeneration, _mutationRevision, true);
  }

  function selectAsset(asset: GalleryAsset): void {
    selectedAsset = asset;
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

  // What the Use as items and the Save as voice dialog show comes from the Workspace context: the one the Workspace last
  // loaded, or, on a cold load of this page, one read here as soon as an audio take is listed. Until it is known the voice
  // items are unavailable, and stay so if the read fails.
  let context = $state<WorkspaceContext | null>(workspaceMemory.get());
  let contextRequested = false;
  $effect(() => {
    if (context !== null || contextRequested || !assets.some((asset) => asset.media_type === 'audio')) return;
    contextRequested = true;
    getWorkspaceCoreContext()
      .then((loaded) => {
        workspaceMemory.set(loaded);
        context = loaded;
      })
      .catch(() => undefined);
  });
  const useAs = $derived(useAsState(context, draft.state));

  function useAsReference(asset: GalleryAsset, target: ReferenceTarget): void {
    if (!asset.file_path) return;
    // A route's params replace the reference list, so a reference waits for the Workspace and joins it there.
    if (target === 'reference') {
      requestReference(asset.file_path);
      router.navigate('workspace');
    } else router.navigate('workspace', referenceParams(asset, target));
  }

  // The speech take being saved as a voice: the dialog stays on this page, over the viewer when it is open.
  let voiceTake = $state<GalleryAsset | null>(null);

  /** An audio take becomes the Video tab's audio track or the Speech tab's voice (the Workspace reads the params on mount), or opens the dialog that saves it as a voice. */
  function useAudio(asset: GalleryAsset, target: AudioUseTarget): void {
    if (!asset.file_path) return;
    if (target === 'save_voice') {
      voiceTake = asset;
      return;
    }
    router.navigate('workspace', target === 'audio_track' ? audioTrackParams(asset) : voiceParams(asset));
  }

  /** Extend a video in the Workspace, from the paused frame when the viewer has one; a frame too early is refused. */
  function extendAsset(asset: GalleryAsset, point?: FramePoint): void {
    if (point?.tooShort) return;
    router.navigate('workspace', extendParams(asset, point));
    addToast(`${asset.filename} is now the source video`, 'success');
  }

  /** The paused frame of a video becomes the Video tab's first or last frame: it is extracted, then the Workspace opens with it in the slot. */
  function useFrame(asset: GalleryAsset, point: FramePoint, target: FrameTarget): void {
    if (!asset.file_path || useAs[target]?.disabled) return;
    void runFrameGrab(asset, point, target, (saved) => {
      const notice = frameSetNotice(target, point, draft.state, heldUploads);
      router.navigate('workspace', frameParams(asset, saved.path, target));
      if (notice) addToast(notice.message, notice.type);
    });
  }

  /** Make an image the image to edit: the Workspace opens the Edit kind with it and no paint. */
  function editAsset(asset: GalleryAsset): void {
    router.navigate('workspace', editParams(asset));
  }

  async function upscaleAsset(asset: GalleryAsset, factor: UpscaleFactor, method: UpscaleMethod): Promise<void> {
    // The job panel and queue live in the Workspace; follow the upscale there.
    if (await startUpscale(asset, factor, method)) router.navigate('workspace');
  }

  // --- Keyboard -------------------------------------------------------------
  let gridEl = $state<HTMLDivElement | null>(null);
  let scrollRegion = $state<HTMLElement | null>(null);
  // Where the grid is scrolled, kept as it scrolls: the element is gone by the time the page is torn down.
  let scrollTop = 0;

  function gridTiles(): HTMLElement[] {
    return Array.from(gridEl?.querySelectorAll<HTMLElement>('.asset-tile') ?? []);
  }

  function focusTile(tile: HTMLElement | undefined): void {
    tile?.querySelector<HTMLElement>('.asset-tile-media')?.focus();
  }

  function handleGridKeydown(e: KeyboardEvent): void {
    if (e.defaultPrevented || isTyping(e.target) || hasOpenModal()) return;
    // Alt+3 (Option+3) focuses the first tile, as it focuses the newest strip tile in the Workspace. `code`, not `key`:
    // Alt+3 types "£" on a Mac keyboard.
    if (e.altKey && !isCommandKey(e) && !e.shiftKey && e.code === 'Digit3') {
      const first = gridTiles()[0];
      if (!first) return;
      e.preventDefault();
      focusTile(first);
      return;
    }
    if (isCommandKey(e) && !e.altKey && !e.shiftKey && e.key.toLowerCase() === 'a') {
      if (assets.length === 0) return;
      e.preventDefault();
      void selectAllMatching();
      return;
    }
    if (!isPlainKey(e)) return;

    const tiles = gridTiles();
    const index = tiles.findIndex((tile) => tile.contains(document.activeElement));
    const focused = index >= 0 ? assets[index] : undefined;
    if (e.key === 'Escape') {
      if (!selectMode) return;
      e.preventDefault();
      leaveSelectMode();
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
      toggleTile(focused);
    } else if (e.key.toLowerCase() === 's') {
      // S stars the focused file only, never the selection. A held key must not toggle it again and again.
      e.preventDefault();
      if (!e.repeat) void starAsset(focused, focused.starred !== true);
    } else {
      const next = moveInGrid(index, e.key, gridColumns(tiles.map((tile) => tile.offsetTop)), tiles.length);
      if (next === null) return;
      e.preventDefault();
      focusTile(tiles[next]);
    }
  }
</script>

{#snippet selectAllButton()}
  <!-- Busy while the matching IDs load: the label hides under the spinner so the button keeps its width, and aria-label keeps its name. It is not disabled, which would drop focus. -->
  <button type="button" class="ui-btn sel-all" aria-busy={selectingAll} aria-label={selectingAll ? 'Select all' : undefined} onclick={() => void selectAllMatching()}>
    <span class="sel-all-label">Select all</span>
    {#if selectingAll}<span class="sel-all-spin"><Spinner size="sm" /></span>{/if}
  </button>
{/snippet}

<div id="gallery-view" class="flex min-h-0 flex-1 flex-col bg-bg-base">
  <PageHeader
    title="Gallery"
    description={selectMode
      ? (compact ? undefined : modeLine({ project: projectPick.current, starred: starredOnly, filter: mediaFilter, coarse: layout.coarse }))
      : `${projectPick.current === null ? '' : `${projectPick.current} · `}Browsing ${assets.length} loaded file${assets.length !== 1 ? 's' : ''} of ${totalCount}, ${sortOrder === 'newest' ? 'newest' : 'oldest'} first.`}
  >
    {#snippet actions()}
      {#if !selectMode}
        <Select
          ariaLabel="Filter gallery media"
          class={compact ? 'min-w-0 flex-1' : 'w-36'}
          value={mediaFilter}
          options={mediaOptions}
          onchange={(next) => onFilterChange(next as MediaFilter)}
        />
        <Button
          icon
          class="star-toggle"
          aria-label="Starred"
          aria-pressed={starredOnly}
          title={starredOnly ? 'Showing starred only' : 'Show starred only'}
          onclick={toggleStarredOnly}
        ><Icon name="star" size={16} solid={starredOnly} /></Button>
        <Select
          ariaLabel="Sort gallery files"
          class={compact ? 'min-w-0 flex-1' : 'w-36'}
          value={sortOrder}
          options={[
            { value: 'newest', label: 'Newest first' },
            { value: 'oldest', label: 'Oldest first' },
          ]}
          onchange={(next) => onSortChange(next as 'newest' | 'oldest')}
        />
        <Button data-bulk="select" onclick={startSelecting}>Select</Button>
      {:else if compact}
        <!-- Phone: the mode line and Done share one row; the actions are in the bar at the foot. -->
        <p class="ui-page-sub min-w-0 flex-1" data-testid="mode-line">{modeLine({ project: projectPick.current, starred: starredOnly, filter: mediaFilter, coarse: layout.coarse })}</p>
        <Button data-bulk="done" aria-pressed="true" onclick={doneSelecting}>Done</Button>
      {:else}
        <Button data-bulk="done" aria-pressed="true" onclick={doneSelecting}>Done</Button>
        <span class="mode-count" role="status">{selectedCount.toLocaleString()} selected</span>
        {@render selectAllButton()}
        <span class="inline-flex" bind:this={bulkMoveAnchor}>
          <Button
            data-bulk="move"
            aria-haspopup="menu"
            aria-expanded={bulkMoveOpen}
            aria-label="Move selected"
            disabled={actionableIds.length === 0 || moving}
            onclick={() => { bulkMoveOpen = !bulkMoveOpen; }}
          ><Icon name="folder" size={14} />{moving ? 'Moving…' : 'Move'}<Icon name="chevdown" size={12} class="opacity-70" /></Button>
        </span>
        <Button aria-label="Download selected" disabled={selectedCount === 0} onclick={downloadSelected}><Icon name="download" size={14} />Download</Button>
        <Button data-bulk="delete" variant="danger" aria-label="Delete selected" disabled={actionableIds.length === 0} onclick={deleteSelected}>
          <Icon name="trash" size={14} />{bulkDeleteRuns > 0 ? 'Deleting…' : 'Delete'}
        </Button>
      {/if}
    {/snippet}
  </PageHeader>

  <section bind:this={scrollRegion} id="gallery-scroll-region" class="min-h-0 flex-1 overflow-y-auto p-4" onscroll={(event) => { scrollTop = event.currentTarget.scrollTop; }}>
    {#if loading}
      <!-- The frame first: the grid's shape is drawn before the assets arrive, so nothing moves when they do. -->
      <div class="grid grid-cols-2 gap-2 sm:gap-3 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 2xl:grid-cols-6" role="status" aria-label="Loading" data-testid="gallery-placeholders">
        {#each { length: 12 } as _, i (i)}
          <div class="tile-placeholder" aria-hidden="true"></div>
        {/each}
      </div>
    {:else if error}
      <Alert tone="error" live>{error}</Alert>
    {:else if emptyState}
      <EmptyState title={emptyTitle} class="min-h-96 justify-center">
        <p>{emptyBody}</p>
        {#snippet actions()}
          {#if starredEmptyState}
            <Button onclick={showAllStarred}>Show all</Button>
          {:else if filteredEmptyState}
            <Button onclick={clearMediaFilter}>Show all media</Button>
          {:else}
            <Button variant="primary" onclick={openWorkspace}>Open Workspace</Button>
          {/if}
        {/snippet}
      </EmptyState>
    {:else}
      <!-- Grid -->
      <div bind:this={gridEl} class="grid grid-cols-2 gap-2 sm:gap-3 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 2xl:grid-cols-6">
        {#each assets as asset (asset.id)}
          <AssetTile
            {asset}
            density="card"
            showProject={projectPick.current === null}
            selected={selected.has(asset.id)}
            {selectMode}
            deleting={deletingIds.has(asset.id)}
            onselect={setSelected}
            onmediaclick={onMediaClick}
            onstartselect={startSelectWith}
            onpreview={openViewer}
            onstar={starAsset}
            onreuse={reuseInWorkspace}
            onreference={useAsReference}
            onuseaudio={useAudio}
            {useAs}
            onextend={extendAsset}
            onedit={editAsset}
            onupscale={upscaleAsset}
            ondelete={deleteSingle}
            onmove={moveSingle}
          />
        {/each}
      </div>

      <!-- Infinite scroll sentinel + pagination -->
      <div id="gallery-pagination" class="flex items-center justify-center py-10">
        {#if hasMore}
          <div bind:this={sentinelEl} class="flex items-center gap-2 text-ui text-text-muted">
            {#if loadingMore}
              <Spinner size="sm" class="text-primary-main" />
              <span>Loading more…</span>
            {:else}
              <span>Scroll for more</span>
            {/if}
          </div>
        {:else if assets.length > 0}
          <p class="text-ui text-text-muted">All assets loaded</p>
        {/if}
      </div>
    {/if}
  </section>
  {#if selectMode && compact}
    <!-- Phone: the selection's actions, docked at the foot. -->
    <div class="ui-action-bar select-bar" data-testid="gallery-select-bar">
      <span class="sel-count" role="status">{selectedCount.toLocaleString()} selected</span>
      {@render selectAllButton()}
      <span class="inline-flex" bind:this={bulkMoveAnchor}>
        <Button
          icon
          data-bulk="move"
          aria-haspopup="menu"
          aria-expanded={bulkMoveOpen}
          aria-label="Move selected"
          title={moving ? 'Moving…' : 'Move selected'}
          disabled={actionableIds.length === 0 || moving}
          onclick={() => { bulkMoveOpen = !bulkMoveOpen; }}
        ><Icon name="folder" size={18} /></Button>
      </span>
      <Button icon aria-label="Download selected" title="Download selected" disabled={selectedCount === 0} onclick={downloadSelected}><Icon name="download" size={18} /></Button>
      <Button icon data-bulk="delete" variant="danger" aria-label="Delete selected" title={bulkDeleteRuns > 0 ? 'Deleting…' : 'Delete selected'} disabled={actionableIds.length === 0} onclick={deleteSelected}><Icon name="trash" size={18} /></Button>
    </div>
  {/if}
</div>

<MoveMenu
  open={bulkMoveOpen}
  anchor={bulkMoveAnchor}
  align="end"
  count={actionableIds.length}
  current={moveCurrent}
  onmove={(project) => void moveSelection(project)}
  onclose={() => { bulkMoveOpen = false; }}
/>

<AssetViewer
  {assets}
  origin={viewerTrigger}
  currentId={selectedAsset?.id ?? null}
  open={lightboxOpen && selectedAsset !== null}
  setLabel="Gallery"
  {deletingIds}
  onclose={closeViewer}
  onnavigate={handleViewerNavigate}
  onnearend={() => { if (hasMore) void loadMorePages(); }}
  onreuse={reuseInWorkspace}
  onreference={useAsReference}
  onuseaudio={useAudio}
  {useAs}
  audioModels={context?.audio_models}
  audioVocabulary={context?.audio_settings}
  onextend={extendAsset}
  onuseframe={useFrame}
  onedit={editAsset}
  onupscale={upscaleAsset}
  ondelete={deleteSingle}
  onmove={moveSingle}
  onstar={starAsset}
/>

<SaveVoiceDialog asset={voiceTake} model={useAs.voiceModel ?? null} vocabulary={context?.audio_settings} onclose={() => { voiceTake = null; }} />

<style>
  /* The ★ toggle beside the media filter: an outline star, muted; filled amber on an amber tint while on. */
  :global(.ui-btn.star-toggle) { color: var(--color-text-secondary); }
  :global(.ui-btn.star-toggle[aria-pressed='true']) { background: color-mix(in srgb, var(--color-star) 14%, transparent); color: var(--color-star); }
  @media (hover: hover) { :global(.ui-btn.star-toggle[aria-pressed='true']:hover:not(:disabled)) { background: color-mix(in srgb, var(--color-star) 22%, transparent); color: var(--color-star); } }
  @media (hover: none) { :global(.ui-btn.star-toggle[aria-pressed='true']:active:not(:disabled)) { background: color-mix(in srgb, var(--color-star) 22%, transparent); color: var(--color-star); } }
  /* The count in Select mode: status text between Done and Select all. */
  .mode-count { flex-shrink: 0; padding: 0 4px; font-size: var(--text-ui); font-weight: 600; font-variant-numeric: tabular-nums; white-space: nowrap; color: var(--color-text-primary); }
  /* Select all while the IDs load: the same width, the label hidden under a centred spinner. */
  .sel-all { position: relative; }
  .sel-all[aria-busy='true'] > .sel-all-label { visibility: hidden; }
  .sel-all-spin { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center; }
  /* The phone's bar: the selection's actions at the foot. */
  .select-bar { flex-shrink: 0; gap: 6px; padding: 8px 12px calc(8px + env(safe-area-inset-bottom, 0px)); border-top: 1px solid var(--color-border-subtle); background: var(--color-bg-overlay); }
  .sel-count { min-width: 0; flex: 1; font-size: var(--text-content); font-weight: 600; font-variant-numeric: tabular-nums; white-space: nowrap; color: var(--color-text-primary); }
  .select-bar :global(.ui-btn) { padding: 0 12px; }
  .select-bar :global(.ui-btn-icon) { padding: 0; }
  /* A tile's shape while the assets load: the image square over a caption, in the tile's surface. */
  .tile-placeholder { aspect-ratio: 1 / 1.32; border-radius: var(--radius-md); background: var(--color-bg-surface); }
</style>
