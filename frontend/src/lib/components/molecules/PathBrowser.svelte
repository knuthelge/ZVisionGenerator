<script lang="ts">
  import { tick, untrack } from 'svelte';
  import { browsePlaces, createFolder, errorText, listDirectory } from '$lib/api';
  import { Button, Icon, Toggle } from '$lib/components/atoms';
  import type { IconName } from '$lib/components/atoms/Icon.svelte';
  import { layout } from '$lib/state/layoutMode.svelte';
  import {
    breadcrumbs,
    DEFAULT_BROWSE_VIEW,
    formatBytes,
    formatModified,
    formatModifiedShort,
    gridMove,
    itemCountLabel,
    jumpToLetter,
    middleTruncate,
    orderAfterClick,
    readBrowseView,
    readRecentPaths,
    readShowHidden,
    rememberRecentPath,
    SORT_CHOICES,
    sortChoiceLabel,
    sortEntries,
    writeBrowseView,
    writeShowHidden,
    type BrowseOrder,
    type BrowseSortKey,
  } from '$lib/state/pathBrowser';
  import type { BrowseEntry, BrowseListing, BrowsePlace, BrowsePurpose, BrowsePurposeName } from '$lib/types';
  import ActionMenu, { type ActionMenuEntry } from './ActionMenu.svelte';
  import { fileName } from './assetDetails';
  import FilePreview from './FilePreview.svelte';
  import FileThumb from './FileThumb.svelte';
  import Modal from './Modal.svelte';
  import Segmented from './Segmented.svelte';
  import Sheet from './Sheet.svelte';

  interface Props {
    open?: boolean;
    purpose: BrowsePurposeName;
    /** The field's current value; the browser opens at it (or its folder). */
    initialPath?: string | null;
    onchoose: (path: string) => void;
    onclose?: () => void;
  }

  let { open = $bindable(false), purpose, initialPath = null, onchoose, onclose }: Props = $props();

  const PLACE_ICONS: Record<string, IconName> = { home: 'home', output: 'reference', models: 'cube', loras: 'cube' };
  // A skeleton only after this long, so quick folders don't flash.
  const SKELETON_DELAY_MS = 150;
  // The phone's Choose button cuts the file name in the middle to this many characters; the full name is its accessible name.
  const CHOOSE_NAME_MAX = 26;
  const PHONE_GRID_COLUMNS = 3;
  const LANDSCAPE_GRID_COLUMNS = 4;

  let info = $state<BrowsePurpose | null>(null);
  let places = $state<BrowsePlace[]>([]);
  let recent = $state<string[]>([]);
  let listing = $state<BrowseListing | null>(null);
  let history = $state<string[]>([]);
  // The selection is the path, not a position, so a re-sort keeps the same file selected.
  let selectedPath = $state<string | null>(null);
  let order = $state<BrowseOrder>({ sort: DEFAULT_BROWSE_VIEW.sort, dir: DEFAULT_BROWSE_VIEW.dir });
  let view = $state<'list' | 'grid'>(DEFAULT_BROWSE_VIEW.view);
  let previewOn = $state(DEFAULT_BROWSE_VIEW.preview);
  let sortMenuOpen = $state(false);
  let placesOpen = $state(false);
  let previewOpen = $state(false);
  let sortButton = $state<HTMLButtonElement | null>(null);
  let filter = $state('');
  let showHidden = $state(readShowHidden());
  let showAll = $state(false);
  let loading = $state(false);
  let showSkeleton = $state(false);
  let error = $state<string | null>(null);
  let editingPath = $state(false);
  let pathDraft = $state('');
  let creatingFolder = $state(false);
  let folderName = $state('');
  let listEl = $state<HTMLDivElement | null>(null);
  let crumbsEl = $state<HTMLDivElement | null>(null);
  let pathInputEl = $state<HTMLInputElement | null>(null);
  let folderInputEl = $state<HTMLInputElement | null>(null);
  let listPaneEl = $state<HTMLDivElement | null>(null);
  let requestVersion = 0;
  // The row clicked last in this listing; the second click of a double-click lands on the next listing's row and must not act on it.
  let lastClicked: string | null = null;

  const folderMode = $derived(info?.kind === 'directory');
  // The phone (compact layout, upright or sideways) has its own arrangement: a Places sheet, a sort sheet and a preview sheet instead of the pane.
  const compact = $derived(layout.compact);
  const landscape = $derived(compact && layout.short);
  // The pickers whose files are pictures, sound or clips get thumbnails, a grid and a preview; the rest keep icons and request nothing.
  const previewKind = $derived(info?.preview ?? null);
  const gridView = $derived(previewKind !== null && view === 'grid');
  const showPane = $derived(previewKind !== null && previewOn && !compact);
  const stripPane = $derived(showPane && layout.mode === 'tall');
  const sidePane = $derived(showPane && !stripPane);
  const sortItems = $derived<ActionMenuEntry[]>([
    ...SORT_CHOICES.map((choice) => ({
      kind: 'item' as const,
      id: choice.id,
      label: choice.label,
      checked: choice.order.sort === order.sort && choice.order.dir === order.dir,
      onselect: () => void applyOrder(choice.order),
    })),
    { kind: 'separator' },
    { kind: 'note', text: 'Folders stay on top' },
  ]);
  // "Output folder" reads as "Choose output folder"; a label starting with an acronym ("LoRA file") keeps its case.
  const title = $derived(info ? `Choose ${/^[A-Z][a-z]+(\s|$)/.test(info.label) ? info.label.charAt(0).toLowerCase() + info.label.slice(1) : info.label}` : 'Choose a path');
  const sorted = $derived(listing ? sortEntries(listing.entries, order) : []);
  const visible = $derived(sorted.filter((entry) => !filter || entry.name.toLowerCase().includes(filter.toLowerCase())));
  const active = $derived(selectedPath === null ? -1 : visible.findIndex((entry) => entry.path === selectedPath));
  const crumbs = $derived(listing ? breadcrumbs(listing.path, listing.separator) : []);
  const activeEntry = $derived(active >= 0 ? visible[active] ?? null : null);
  const chosenFile = $derived(activeEntry && activeEntry.kind === 'file' && activeEntry.selectable ? activeEntry : null);
  const mainLabel = $derived(folderMode ? `Use “${listing ? fileName(listing.path) : ''}”` : 'Choose file');
  // The phone's button names the file, cut in the middle; the full name is what a screen reader hears.
  const phoneChooseName = $derived(chosenFile ? middleTruncate(chosenFile.name, CHOOSE_NAME_MAX) : null);
  const mainText = $derived(!compact ? mainLabel : landscape ? (folderMode ? 'Use' : 'Choose') : phoneChooseName ? `Choose ${phoneChooseName}` : mainLabel);
  const mainAria = $derived(!compact ? undefined : chosenFile && !folderMode ? `Choose ${chosenFile.name}` : mainLabel);
  const countLabel = $derived(
    listing ? itemCountLabel(visible.filter((entry) => entry.kind === (folderMode ? 'directory' : 'file')).length, folderMode ? 'folder' : previewKind) : '',
  );
  const mainDisabled = $derived(loading || !listing || (!folderMode && !chosenFile));
  const footerName = $derived(folderMode ? listing?.path ?? '' : chosenFile?.name ?? '');

  $effect(() => {
    if (!open) return;
    const name = purpose;
    const start = initialPath ?? '';
    untrack(() => {
      history = [];
      filter = '';
      showAll = false;
      editingPath = false;
      creatingFolder = false;
      listing = null;
      info = null;
      error = null;
      restoreView();
      sortMenuOpen = false;
      placesOpen = false;
      previewOpen = false;
      recent = readRecentPaths(name);
      // The purpose says folder or file mode, which picks the first highlighted row, so it loads first.
      void loadPlaces(name).then(() => openAt(start)).then(() => listEl?.focus({ preventScroll: true }));
    });
  });

  /** Take the order, list or grid and preview switch this browser remembered, shared by every picker. */
  function restoreView(): void {
    const remembered = readBrowseView();
    order = { sort: remembered.sort, dir: remembered.dir };
    view = remembered.view;
    previewOn = remembered.preview;
  }

  function saveView(): void {
    writeBrowseView({ ...order, view, preview: previewOn });
  }

  async function loadPlaces(name: BrowsePurposeName): Promise<void> {
    try {
      const result = await browsePlaces(name);
      info = result.purpose;
      places = result.places;
    } catch (err) {
      error = errorText(err, 'Failed to load places.');
    }
  }

  /** Open the field's path; one that is gone (a moved file, an output folder not made yet) opens the purpose's place, saying why. */
  async function openAt(start: string): Promise<void> {
    if (await navigate(start, { push: false }) || !start) return;
    const reason = error;
    if (await navigate('', { push: false })) error = reason;
  }

  /** List *path*; returns whether it opened (on failure the error shows and the last folder stays). */
  async function navigate(path: string, options: { push?: boolean; focusName?: string } = {}): Promise<boolean> {
    const version = ++requestVersion;
    loading = true;
    const timer = setTimeout(() => { if (version === requestVersion) showSkeleton = true; }, SKELETON_DELAY_MS);
    try {
      const result = await listDirectory(purpose, path, { hidden: showHidden, all: showAll });
      if (version !== requestVersion) return false;
      if ((options.push ?? true) && listing && listing.path !== result.path) history = [...history, listing.path];
      const changedFolder = listing?.path !== result.path;
      listing = result;
      error = null;
      filter = '';
      lastClicked = null;
      selectedPath = firstSelection(sorted, options.focusName ?? result.focus);
      if (changedFolder) {
        await tick();
        if (listPaneEl) listPaneEl.scrollTop = 0;
        scrollToActive();
      }
      return true;
    } catch (err) {
      if (version === requestVersion) error = errorText(err, 'Failed to open the folder.');
      return false;
    } finally {
      clearTimeout(timer);
      if (version === requestVersion) {
        loading = false;
        showSkeleton = false;
      }
    }
  }

  /** Return the path to highlight on opening a folder: the named entry, else (in file mode) the first file the purpose takes, else the first row. */
  function firstSelection(entries: BrowseEntry[], focus: string | null): string | null {
    const named = focus ? entries.find((entry) => entry.name === focus) : undefined;
    if (named) return named.path;
    if (folderMode) return null;
    return ((focus ? undefined : entries.find((entry) => entry.selectable)) ?? entries[0])?.path ?? null;
  }

  function refresh(): void {
    if (listing) void navigate(listing.path, { push: false, focusName: activeEntry?.name });
  }

  function goBack(): void {
    const previous = history[history.length - 1];
    if (previous === undefined) return;
    history = history.slice(0, -1);
    void navigate(previous, { push: false });
  }

  function goUp(): void {
    if (listing?.parent) void navigate(listing.parent, { focusName: fileName(listing.path) });
  }

  function activate(entry: BrowseEntry): void {
    if (entry.kind === 'directory') void navigate(entry.path);
    else if (entry.selectable) choose(entry.path);
  }

  function choose(path: string): void {
    rememberRecentPath(purpose, path);
    placesOpen = false;
    previewOpen = false;
    open = false;
    onchoose(path);
  }

  function confirm(): void {
    if (folderMode && listing) choose(listing.path);
    else if (chosenFile) choose(chosenFile.path);
  }

  function close(): void {
    placesOpen = false;
    previewOpen = false;
    open = false;
    onclose?.();
  }

  function startEditingPath(): void {
    pathDraft = listing?.path ?? '';
    editingPath = true;
    queueMicrotask(() => pathInputEl?.select());
  }

  async function submitPath(): Promise<void> {
    const target = pathDraft.trim();
    if (!target) return;
    editingPath = false;
    if (!(await navigate(target))) return;
    // A typed file path chooses that file straight away when the purpose accepts it.
    if (!folderMode && listing?.focus && chosenFile?.name === listing.focus) choose(chosenFile.path);
  }

  function onPathKeydown(event: KeyboardEvent): void {
    if (event.key === 'Enter') { event.preventDefault(); void submitPath(); }
    else if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); editingPath = false; listEl?.focus({ preventScroll: true }); }
  }

  function startNewFolder(): void {
    folderName = '';
    creatingFolder = true;
    queueMicrotask(() => folderInputEl?.focus());
  }

  async function submitNewFolder(): Promise<void> {
    if (!listing || !folderName.trim()) return;
    try {
      const created = await createFolder(purpose, listing.path, folderName);
      creatingFolder = false;
      await navigate(created.path);
    } catch (err) {
      error = errorText(err, 'Failed to create the folder.');
    }
  }

  function onFolderKeydown(event: KeyboardEvent): void {
    if (event.key === 'Enter') { event.preventDefault(); void submitNewFolder(); }
    else if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); creatingFolder = false; }
  }

  // Enter in the filter must not submit a form the dialog sits in (Config, Models). A computer takes the first match; a phone only closes the keyboard.
  function onFilterKeydown(event: KeyboardEvent): void {
    if (event.key === 'Enter') {
      event.preventDefault();
      if (compact) (event.currentTarget as HTMLInputElement).blur();
      else if (activeEntry) activate(activeEntry);
    } else if (event.key === 'ArrowDown') {
      event.preventDefault();
      listEl?.focus({ preventScroll: true });
      move(Math.max(active, 0));
    }
  }

  function move(index: number): void {
    if (visible.length === 0) return;
    const target = Math.max(0, Math.min(visible.length - 1, index));
    selectedPath = visible[target].path;
    scrollToActive();
  }

  function scrollToActive(): void {
    listEl?.querySelector(`[data-index="${active}"]`)?.scrollIntoView?.({ block: 'nearest' });
  }

  /** Sort by *key* (a second click on the same heading reverses it) and remember the order for every picker. */
  async function sortBy(key: BrowseSortKey): Promise<void> {
    await applyOrder(orderAfterClick(order, key));
  }

  /** Take the order chosen from the sort menu and remember it for every picker. */
  async function applyOrder(next: BrowseOrder): Promise<void> {
    order = next;
    saveView();
    await tick();
    scrollToActive();
  }

  async function setView(next: 'list' | 'grid'): Promise<void> {
    view = next;
    saveView();
    await tick();
    scrollToActive();
    listEl?.focus({ preventScroll: true });
  }

  function togglePreview(): void {
    previewOn = !previewOn;
    saveView();
  }

  // Alt+P is matched by `code`: on macOS Option+P types a character, so `key` is not "p".
  function onBrowserKeydown(event: KeyboardEvent): void {
    if (previewKind === null || compact || event.code !== 'KeyP' || !event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return;
    event.preventDefault();
    togglePreview();
  }

  /** Return how many tiles the grid holds per row: fixed on the phone, else as the browser laid them out. */
  function gridColumns(): number {
    if (compact) return landscape ? LANDSCAPE_GRID_COLUMNS : PHONE_GRID_COLUMNS;
    if (!listEl) return 1;
    const tracks = getComputedStyle(listEl).gridTemplateColumns.split(/\s+/).filter((track) => /^[\d.]+px$/.test(track));
    return Math.max(1, tracks.length);
  }

  /** Whether *entry* shows a thumbnail: a file of the kind the picker takes. */
  function hasThumb(entry: BrowseEntry): boolean {
    return previewKind !== null && entry.kind === 'file' && entry.selectable;
  }

  function headingLabel(key: BrowseSortKey, label: string): string | undefined {
    if (order.sort !== key) return undefined;
    return `${label}, sorted ${order.dir === 'asc' ? 'ascending' : 'descending'}`;
  }

  function onListKeydown(event: KeyboardEvent): void {
    if (gridView && event.key.startsWith('Arrow') && !event.altKey && !event.ctrlKey && !event.metaKey) {
      event.preventDefault();
      move(gridMove(Math.max(active, 0), event.key, gridColumns(), visible.length));
    }
    else if (event.key === 'ArrowDown') { event.preventDefault(); move(active + 1); }
    else if (event.key === 'ArrowUp') { event.preventDefault(); move(active - 1); }
    else if (event.key === 'Home') { event.preventDefault(); move(0); }
    else if (event.key === 'End') { event.preventDefault(); move(visible.length - 1); }
    else if (event.key === 'Enter') {
      event.preventDefault();
      if (activeEntry) activate(activeEntry);
      else confirm();
    } else if (event.key === ' ' && compact && previewKind !== null) { event.preventDefault(); openPreviewSheet(); }
    else if (event.key === 'Backspace') { event.preventDefault(); goUp(); }
    else if (event.key === '/' || (event.key.toLowerCase() === 'l' && (event.ctrlKey || event.metaKey))) { event.preventDefault(); startEditingPath(); }
    else if (event.key.length === 1 && !event.ctrlKey && !event.metaKey && !event.altKey) {
      const index = jumpToLetter(visible, event.key, active);
      if (index >= 0) move(index);
    }
  }

  function onRowClick(entry: BrowseEntry, event: MouseEvent): void {
    // One click opens a folder. The later clicks of a double-click land on the new listing's rows, so a repeat click only counts on the row clicked before.
    const again = lastClicked === entry.path;
    if (event.detail > 1 && !again) return;
    lastClicked = entry.path;
    selectedPath = entry.path;
    if (entry.kind === 'directory') void navigate(entry.path);
    // A second tap on a row (or tile) tapped before opens the preview.
    else if (compact && again) openPreviewSheet();
  }

  /** A double-click on a file chooses it on a computer; a phone chooses with its Choose button. */
  function onRowDblClick(entry: BrowseEntry): void {
    if (!compact && entry.kind === 'file' && lastClicked === entry.path) activate(entry);
  }

  /** Open the preview sheet for the selected file; nothing for a folder or a file the picker does not take. */
  function openPreviewSheet(): void {
    if (activeEntry && hasThumb(activeEntry)) previewOpen = true;
  }

  /** The thumbnail button of a phone row: select the row and open its preview at once (a pointer shortcut; keys use Space). */
  function onThumbClick(entry: BrowseEntry, event: MouseEvent): void {
    event.stopPropagation();
    selectedPath = entry.path;
    openPreviewSheet();
  }

  function openPlaceFromSheet(path: string): void {
    placesOpen = false;
    void navigate(path);
  }

  /** The second line of a phone row: size and date. */
  function entryMeta(entry: BrowseEntry): string {
    return [entry.size === null ? '' : formatBytes(entry.size), entry.modified === null ? '' : formatModified(entry.modified)].filter(Boolean).join(' · ');
  }

  // The path shows its end: the folder you are in matters more than where the path starts.
  $effect(() => {
    if (!crumbsEl) return;
    void crumbs;
    const el = crumbsEl;
    void tick().then(() => { el.scrollLeft = el.scrollWidth; });
  });

  function setShowHidden(value: boolean): void {
    showHidden = value;
    writeShowHidden(value);
    refresh();
  }

  function showAllFiles(): void {
    showAll = true;
    refresh();
  }
</script>

{#snippet placeList(pick: (path: string) => unknown, iconSize: number)}
  {#each places as place (place.id)}
    <button type="button" class="place" class:place-on={listing?.path === place.path} title={place.path} aria-current={listing?.path === place.path ? 'location' : undefined} onclick={() => void pick(place.path)}>
      <Icon name={place.id.startsWith('drive:') ? 'drive' : PLACE_ICONS[place.id] ?? 'folder'} size={iconSize} /><span class="truncate">{place.label}</span>
    </button>
  {/each}
  {#if recent.length > 0}
    <p class="ui-area-label place-heading">Recent</p>
    {#each recent as path (path)}
      <button type="button" class="place" title={path} onclick={() => void pick(path)}>
        <Icon name="clock" size={iconSize} /><span class="truncate">{fileName(path)}</span>
      </button>
    {/each}
  {/if}
{/snippet}

{#snippet mainButton()}
  <Button variant="primary" disabled={mainDisabled} class="main" aria-label={mainAria} onclick={confirm}><span class="main-text">{mainText}</span></Button>
{/snippet}

<!-- The dialog opens once the purpose is known (or failed to load), so it opens at its final size. -->
<Modal open={open && (info !== null || error !== null || listing !== null)} {title} size={sidePane ? 'xl' : 'lg'} fill onclose={close}>
  <!-- Alt+P reaches the browser from any control in it; the list handles the rest of the keys itself. -->
  <!-- svelte-ignore a11y_no_static_element_interactions -->
  <div class="browser" class:compact class:landscape class:pane-side={sidePane} class:pane-strip={stripPane} data-testid="path-browser" data-layout={compact ? (landscape ? 'landscape' : 'portrait') : 'wide'} onkeydown={onBrowserKeydown}>
    <div class="toolbar" data-testid="browser-toolbar">
      {#if compact}
        <Button icon aria-label="Places" title="Places" aria-haspopup="dialog" onclick={() => { placesOpen = true; }}><Icon name="folder" size={14} /></Button>
      {/if}
      <Button icon aria-label="Back" title="Back" disabled={history.length === 0} onclick={goBack}><Icon name="chevleft" size={14} /></Button>
      <Button icon aria-label="Up one folder" title="Up one folder" disabled={!listing?.parent} onclick={goUp}><Icon name="up" size={14} /></Button>
      {#if editingPath}
        <input
          bind:this={pathInputEl}
          bind:value={pathDraft}
          class="ui-field path-edit"
          aria-label="Path"
          spellcheck="false"
          autocomplete="off"
          onkeydown={onPathKeydown}
          onblur={() => { editingPath = false; }}
        >
      {:else}
        <div bind:this={crumbsEl} class="crumbs" role="navigation" aria-label="Folder path">
          {#each crumbs as crumb, index (crumb.path)}
            {#if index > 0 && crumbs[index - 1].label !== '/'}<span class="crumb-sep" aria-hidden="true">{listing?.separator}</span>{/if}
            <button
              type="button"
              class="crumb"
              class:crumb-last={index === crumbs.length - 1}
              aria-current={index === crumbs.length - 1 ? 'location' : undefined}
              onclick={() => void navigate(crumb.path)}
            >{crumb.label}</button>
          {/each}
          <span class="flex-1"></span>
          <!-- A phone held sideways has no room to type a path; it is typed upright. -->
          {#if !landscape}
            <Button class="crumb-edit" variant="quiet" size="sm" icon aria-label="Type a path" title="Type a path (/)" onclick={startEditingPath}><Icon name="edit" size={12} /></Button>
          {/if}
        </div>
      {/if}
      {#if landscape}
        <Button icon aria-label="Sort: {sortChoiceLabel(order)}" title="Sort: {sortChoiceLabel(order)}" aria-haspopup="menu" aria-expanded={sortMenuOpen} onclick={() => { sortMenuOpen = !sortMenuOpen; }}><Icon name="sliders" size={14} /></Button>
      {/if}
      {#if previewKind !== null}
        <Segmented
          label="View"
          value={view}
          class={compact ? 'seg-icons' : ''}
          options={[{ value: 'list', label: 'List', icon: 'list' }, { value: 'grid', label: 'Grid', icon: 'grid' }]}
          onchange={(next) => void setView(next)}
        />
        {#if !compact}
          <Button aria-pressed={previewOn} title="Preview (Alt+P)" aria-keyshortcuts="Alt+P" onclick={togglePreview}><Icon name="reference" size={12} />Preview</Button>
        {/if}
      {/if}
      <label class="filter" class:filter-set={filter !== ''}>
        <Icon name="search" size={12} />
        <input bind:value={filter} class="filter-input" placeholder="Filter" aria-label="Filter this folder" spellcheck="false" autocomplete="off" oninput={() => { selectedPath = visible[0]?.path ?? null; }} onkeydown={onFilterKeydown}>
      </label>
      {#if folderMode}
        <Button icon={compact} aria-label="New folder" title="New folder" class="new-folder" onclick={startNewFolder}><Icon name="plus" size={12} />{#if !compact}New folder{/if}</Button>
      {/if}
      {#if landscape}
        {@render mainButton()}
      {/if}
    </div>

    {#if error}
      <p class="ui-help ui-help-error message" role="alert">{error}</p>
    {/if}
    {#if creatingFolder}
      <div class="new-folder-row">
        <Icon name="folder" size={14} />
        <input bind:this={folderInputEl} bind:value={folderName} class="ui-field flex-1" aria-label="New folder name" placeholder="Folder name" spellcheck="false" autocomplete="off" onkeydown={onFolderKeydown}>
        <Button variant="primary" disabled={!folderName.trim()} onclick={() => void submitNewFolder()}>Create</Button>
        <Button variant="quiet" onclick={() => { creatingFolder = false; }}>Cancel</Button>
      </div>
    {/if}

    <div class="body">
      {#if !compact}
        <nav class="places" aria-label="Places">
          <p class="ui-area-label place-heading">Places</p>
          {@render placeList(navigate, 14)}
        </nav>
      {/if}

      <div class="main">
      <div bind:this={listPaneEl} class="list-pane">
        {#if compact}
          <!-- Upright: how many items and in what order; it opens the sort sheet. Sideways the toolbar's sort button does. -->
          {#if !landscape}
            <div class="sort-bar">
              <button
                bind:this={sortButton}
                type="button"
                class="sort-row"
                data-testid="sort-row"
                aria-haspopup="menu"
                aria-expanded={sortMenuOpen}
                aria-label="Sort: {sortChoiceLabel(order)}"
                onclick={() => { sortMenuOpen = !sortMenuOpen; }}
              ><span class="sort-count" data-testid="sort-count">{countLabel}</span><span aria-hidden="true">·</span><span class="sort-order">{sortChoiceLabel(order)}</span><Icon name="chevdown" size={12} /></button>
            </div>
          {/if}
        {:else if gridView}
          <div class="sort-bar">
            <button
              bind:this={sortButton}
              type="button"
              class="ui-btn sort-btn"
              aria-haspopup="menu"
              aria-expanded={sortMenuOpen}
              aria-label="Sort: {sortChoiceLabel(order)}"
              onclick={() => { sortMenuOpen = !sortMenuOpen; }}
            >{sortChoiceLabel(order)}<Icon name="chevdown" size={12} /></button>
          </div>
        {:else}
          <div class="row head" role="toolbar" aria-label="Sort by">
            {#each [['name', 'Name'], ['size', 'Size'], ['modified', 'Modified']] as const as [key, label] (key)}
              <button
                type="button"
                class="ui-area-label head-btn"
                class:head-right={key !== 'name'}
                class:head-on={order.sort === key}
                aria-pressed={order.sort === key}
                aria-label={headingLabel(key, label)}
                onclick={() => void sortBy(key)}
              >{label}{#if order.sort === key}<span class="head-arrow" class:head-arrow-desc={order.dir === 'desc'} aria-hidden="true"><Icon name="up" size={10} /></span>{/if}</button>
            {/each}
          </div>
        {/if}
        {#if gridView || compact}
          <ActionMenu open={sortMenuOpen} anchor={sortButton} items={sortItems} label="Sort by" align="start" onclose={() => { sortMenuOpen = false; }} />
        {/if}
      <div
        bind:this={listEl}
        class="list"
        class:grid={gridView}
        data-view={gridView ? 'grid' : 'list'}
        role="listbox"
        tabindex="0"
        aria-label={listing ? `Contents of ${listing.path}` : 'Folder contents'}
        aria-activedescendant={active >= 0 ? `path-browser-row-${active}` : undefined}
        aria-busy={loading ? 'true' : undefined}
        onkeydown={onListKeydown}
      >
        {#if showSkeleton}
          {#if gridView}
            {#each [0, 1, 2, 3, 4, 5, 6, 7] as slot (slot)}
              <div class="tile skeleton-tile" data-testid="skeleton" aria-hidden="true">
                <span class="tile-thumb"></span>
                <span class="skeleton" style="width: {slot % 2 === 0 ? 80 : 55}%"></span>
              </div>
            {/each}
          {:else}
            {#each [40, 62, 30, 52] as width (width)}
              <div class="row skeleton-row" data-testid="skeleton" aria-hidden="true"><span class="skeleton" style="width: {width}%"></span></div>
            {/each}
          {/if}
        {:else if listing}
          {#each visible as entry, index (entry.path)}
            <!-- The listbox handles the keyboard for its options (aria-activedescendant). -->
            <!-- svelte-ignore a11y_click_events_have_key_events -->
            <div
              id="path-browser-row-{index}"
              data-index={index}
              data-name={entry.name}
              class={gridView ? 'tile' : 'row entry'}
              class:entry-dim={entry.kind === 'file' && !entry.selectable}
              class:entry-active={index === active}
              role="option"
              tabindex="-1"
              aria-selected={index === active}
              onclick={(event) => onRowClick(entry, event)}
              ondblclick={() => onRowDblClick(entry)}
            >
              {#if gridView}
                <span class="tile-thumb">
                  {#if previewKind && hasThumb(entry)}
                    <FileThumb {purpose} path={entry.path} kind={previewKind} iconSize={32} />
                  {:else}
                    <span class="tile-icon" class:entry-icon-pick={entry.kind === 'file' && entry.selectable}><Icon name={entry.kind === 'directory' ? 'folder' : 'file'} size={32} /></span>
                  {/if}
                </span>
                <span class="tile-name" title={entry.name} data-testid="entry-name">{entry.name}</span>
                <span class="tile-meta" data-testid="entry-meta">{entry.modified === null ? '' : formatModifiedShort(entry.modified)}</span>
              {:else}
                <span class="name">
                  {#if previewKind && hasThumb(entry) && compact}
                    <!-- A pointer shortcut to the preview; keys and screen readers use a second activation or Space, as a button inside an option is out of their reach. -->
                    <button type="button" class="row-thumb-btn" tabindex="-1" aria-hidden="true" data-testid="row-preview" onclick={(event) => onThumbClick(entry, event)} ondblclick={(event) => event.stopPropagation()}>
                      <FileThumb {purpose} path={entry.path} kind={previewKind} />
                    </button>
                  {:else if previewKind && hasThumb(entry)}
                    <span class="row-thumb"><FileThumb {purpose} path={entry.path} kind={previewKind} /></span>
                  {:else}
                    <span class="entry-icon" class:entry-icon-wide={previewKind !== null || compact} class:entry-icon-pick={entry.kind === 'file' && entry.selectable}><Icon name={entry.kind === 'directory' ? 'folder' : 'file'} size={compact ? 20 : 14} /></span>
                  {/if}
                  <span class="name-text">
                    <span class="truncate" title={entry.name} data-testid="entry-name">{entry.name}</span>
                    {#if compact}<span class="sub" data-testid="entry-meta">{entryMeta(entry)}</span>{/if}
                  </span>
                </span>
                {#if !compact}
                  <span class="num" data-testid="entry-size">{entry.size === null ? '—' : formatBytes(entry.size)}</span>
                  <span class="num" data-testid="entry-modified">{entry.modified === null ? '' : formatModified(entry.modified)}</span>
                {/if}
              {/if}
            </div>
          {/each}
          {#if visible.length === 0}
            <div class="empty" data-testid="empty-note">
              {#if filter}
                No names here contain “{filter}”.
              {:else if listing.hidden_count > 0}
                {folderMode ? 'No folders here.' : `No ${info?.accepted_extensions.join(' or ') ?? 'matching'} files here.`}
              {:else}
                This folder is empty.
              {/if}
            </div>
          {/if}
          {#if listing.hidden_count > 0 && !filter}
            <p class="hidden-note" data-testid="hidden-note">
              {listing.hidden_count} other {listing.hidden_count === 1 ? 'file' : 'files'} hidden ·
              <button type="button" class="ui-link" onclick={showAllFiles}>Show all</button>
            </p>
          {/if}
        {/if}
      </div>
      </div>
      {#if stripPane && previewKind}
        <aside class="preview-strip" aria-label="Preview"><FilePreview {purpose} kind={previewKind} entry={activeEntry} arrangement="strip" /></aside>
      {/if}
      </div>
      {#if sidePane && previewKind}
        <aside class="preview-pane" aria-label="Preview"><FilePreview {purpose} kind={previewKind} entry={activeEntry} /></aside>
      {/if}
    </div>

    {#if !landscape}
      <div class="footer" data-testid="footer">
        {#if !compact}
          <Toggle checked={showHidden} label="Hidden files" onchange={(event) => setShowHidden((event.currentTarget as HTMLInputElement).checked)} />
          {#if info && info.accepted_extensions.length > 0}
            <span class="ui-chip ext" data-testid="extensions">{info.accepted_extensions.join(' ')}</span>
          {/if}
          <span class="chosen" title={footerName} data-testid="chosen">{footerName}</span>
          <Button onclick={close}>Cancel</Button>
        {/if}
        {@render mainButton()}
      </div>
    {/if}
  </div>
</Modal>

{#if compact}
  <Sheet bind:open={placesOpen} title="Places" layer="modal-top">
    <div class="sheet-places">
      {@render placeList(openPlaceFromSheet, 16)}
      <div class="sheet-hidden">
        <Toggle checked={showHidden} label="Hidden files" onchange={(event) => setShowHidden((event.currentTarget as HTMLInputElement).checked)} />
      </div>
    </div>
  </Sheet>
  {#if previewKind}
    <Sheet bind:open={previewOpen} title="Preview" layer="modal-top">
      <FilePreview {purpose} kind={previewKind} entry={activeEntry} arrangement="sheet" />
      {#snippet footer()}
        <Button variant="primary" class="sheet-choose" disabled={!chosenFile} onclick={confirm}>Choose this file</Button>
      {/snippet}
    </Sheet>
  {/if}
{/if}

<style>
  .browser { --thumb: 24px; --tile: 132px; display: flex; flex-direction: column; height: min(560px, calc(100dvh - 6rem)); min-height: 0; }
  .browser.pane-side { height: min(600px, calc(100dvh - 6rem)); }
  .browser.pane-strip { height: min(780px, calc(100dvh - 6rem)); }
  @media (pointer: coarse) { .browser { --thumb: 32px; --tile: 150px; } }
  .toolbar { display: flex; align-items: center; gap: 6px; padding: 10px 16px; border-bottom: 1px solid var(--color-border-subtle); }
  .crumbs { display: flex; flex: 1; min-width: 0; align-items: center; gap: 2px; height: var(--spacing-control); padding: 0 4px; overflow: hidden; border-radius: var(--radius-sm); background: var(--color-bg-base); }
  .crumb { flex: none; height: 22px; padding: 0 6px; border-radius: var(--radius-xs); font-family: var(--font-mono); font-size: var(--text-ui); color: var(--color-text-secondary); white-space: nowrap; }
  @media (hover: hover) { .crumb:hover { background: var(--color-bg-surface-hover); color: var(--color-text-primary); } }
  .crumb-last { color: var(--color-text-primary); }
  .crumb-sep { color: var(--color-text-muted); font-size: var(--text-meta); }
  .path-edit { flex: 1; min-width: 0; font-family: var(--font-mono); }
  .filter { display: flex; align-items: center; gap: 6px; width: 160px; height: var(--spacing-control); padding: 0 8px; border-radius: var(--radius-sm); background: var(--color-bg-base); color: var(--color-text-muted); }
  .filter:focus-within { outline: 2px solid var(--color-primary-main); outline-offset: 2px; }
  .filter-input { flex: 1; min-width: 0; background: transparent; color: var(--color-text-primary); font-size: var(--text-ui); outline: none; }
  .message { margin: 8px 16px 0; }
  .new-folder-row { display: flex; align-items: center; gap: 6px; padding: 8px 16px; border-bottom: 1px solid var(--color-border-subtle); color: var(--color-text-secondary); }
  .body { display: flex; flex: 1; min-height: 0; }
  .places { display: flex; flex-direction: column; gap: 2px; width: 188px; flex: none; padding: 6px 8px; overflow-y: auto; border-right: 1px solid var(--color-border-subtle); }
  .place-heading { padding: 8px 8px 4px; }
  .place { display: flex; align-items: center; gap: 8px; height: var(--spacing-row); padding: 0 8px; border-radius: var(--radius-sm); color: var(--color-text-secondary); font-size: var(--text-ui); text-align: left; }
  @media (hover: hover) { .place:hover { background: var(--color-bg-surface-hover); color: var(--color-text-primary); } }
  .place-on { background: var(--color-primary-surface); color: var(--color-text-primary); }
  .place-on :global(svg) { color: var(--color-primary-main); }
  .main { display: flex; flex: 1; flex-direction: column; min-width: 0; min-height: 0; }
  .list-pane { flex: 1; min-width: 0; min-height: 0; overflow-y: auto; scroll-padding-top: var(--spacing-row); }
  .preview-pane { width: 300px; flex: none; min-height: 0; border-left: 1px solid var(--color-border-subtle); }
  .preview-strip { height: 236px; flex: none; min-height: 0; border-top: 1px solid var(--color-border-subtle); }
  .list { padding-bottom: 4px; }
  .list:focus-visible { outline: 2px solid var(--color-primary-main); outline-offset: -2px; }
  .row { display: grid; grid-template-columns: minmax(0, 1fr) 84px 124px; align-items: center; gap: 12px; padding: 0 12px; }
  .head { position: sticky; top: 0; z-index: 1; height: 28px; padding: 0 16px; background: var(--color-bg-overlay); border-bottom: 1px solid var(--color-border-subtle); }
  @media (pointer: coarse) { .head { height: var(--spacing-row); } }
  .head-btn { display: flex; align-items: center; gap: 4px; min-width: 0; height: 100%; }
  .head-right { justify-content: flex-end; }
  .head-on { color: var(--color-text-primary); }
  @media (hover: hover) { .head-btn:hover { color: var(--color-text-primary); } }
  .head-btn:focus-visible { outline: 2px solid var(--color-primary-main); outline-offset: -2px; }
  .head-arrow { display: flex; flex: none; color: var(--color-primary-main); }
  .head-arrow-desc { transform: rotate(180deg); }
  .entry { height: var(--spacing-row); margin: 1px 4px; border-radius: var(--radius-sm); color: var(--color-text-primary); font-size: var(--text-ui); cursor: default; }
  @media (hover: hover) { .entry:hover { background: var(--color-bg-surface-hover); } }
  .entry-active { background: var(--color-primary-surface); box-shadow: inset 0 0 0 1px var(--color-primary-border); }
  .list:focus-visible .entry-active { box-shadow: inset 0 0 0 2px var(--color-primary-main); }
  .entry-dim { color: var(--color-text-muted); }
  .entry-icon { display: flex; flex: none; color: var(--color-text-secondary); }
  .entry-icon-pick { color: var(--color-primary-main); }
  .sort-bar { position: sticky; top: 0; z-index: 1; display: flex; align-items: center; padding: 0 12px; height: var(--spacing-row); background: var(--color-bg-overlay); border-bottom: 1px solid var(--color-border-subtle); }
  .sort-btn { height: 22px; padding: 0 8px; gap: 4px; font-size: var(--text-ui); }
  .list.grid { display: grid; grid-template-columns: repeat(auto-fill, var(--tile)); justify-content: start; gap: 6px; padding: 8px 12px 8px; }
  .tile { display: flex; flex-direction: column; gap: 4px; width: var(--tile); padding: 6px; border-radius: var(--radius-sm); color: var(--color-text-primary); font-size: var(--text-ui); cursor: default; }
  @media (hover: hover) { .tile:hover { background: var(--color-bg-surface-hover); } }
  .tile-thumb { display: flex; align-items: center; justify-content: center; width: 100%; aspect-ratio: 1; border-radius: var(--radius-xs); background: var(--color-bg-fill); overflow: hidden; }
  .tile-icon { display: flex; color: var(--color-text-secondary); }
  .tile-name { display: -webkit-box; -webkit-box-orient: vertical; -webkit-line-clamp: 2; line-clamp: 2; overflow: hidden; overflow-wrap: anywhere; line-height: 1.3; min-height: 2.6em; }
  .tile-meta { font-family: var(--font-mono); font-size: var(--text-meta); color: var(--color-text-muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .row-thumb { display: flex; flex: none; width: var(--thumb); height: var(--thumb); }
  .entry-icon-wide { justify-content: center; width: var(--thumb); }
  .name { display: flex; align-items: center; gap: 8px; min-width: 0; }
  .num { text-align: right; font-family: var(--font-mono); font-size: var(--text-meta); color: var(--color-text-muted); }
  .skeleton-row { height: var(--spacing-row); }
  .skeleton { display: block; height: 10px; border-radius: var(--radius-xs); background: var(--color-bg-fill); }
  .skeleton-tile { pointer-events: none; }
  .empty { grid-column: 1 / -1; padding: 32px 16px; text-align: center; color: var(--color-text-muted); }
  .hidden-note { grid-column: 1 / -1; padding: 8px 16px; color: var(--color-text-muted); font-size: var(--text-ui); }
  .footer { display: flex; align-items: center; gap: 8px; padding: 10px 16px; border-top: 1px solid var(--color-border-subtle); }
  .footer :global(.ui-btn) { height: var(--spacing-bar); }
  .chosen { flex: 1; min-width: 0; text-align: right; overflow-wrap: anywhere; font-family: var(--font-mono); font-size: var(--text-ui); color: var(--color-text-muted); }

  .main-text { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .name-text { display: flex; flex-direction: column; gap: 2px; min-width: 0; }
  .sub, .sort-row { font-size: var(--text-ui); }
  .sub { font-family: var(--font-mono); font-size: var(--text-meta); color: var(--color-text-muted); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .row-thumb-btn { display: flex; flex: none; width: var(--thumb); height: var(--thumb); padding: 0; border-radius: var(--radius-xs); }
  .sort-row { display: flex; align-items: center; gap: 6px; width: 100%; height: 100%; color: var(--color-text-secondary); text-align: left; }
  .sort-order { color: var(--color-text-primary); }
  .sheet-places { display: flex; flex-direction: column; gap: 2px; padding: 4px max(12px, env(safe-area-inset-right, 0px)) 12px max(12px, env(safe-area-inset-left, 0px)); }
  .sheet-places .place { height: 48px; font-size: 15px; }
  .sheet-places .place-heading { padding: 12px 8px 4px; }
  .sheet-hidden { padding: 12px 8px 4px; }
  .sheet-hidden :global(.toggle) { min-height: 40px; }
  :global(.sheet-choose) { flex: 1; height: 44px; justify-content: center; }

  /* The phone: a full-height sheet, keyed on the layout rather than the width, so a phone held sideways gets it too. */
  .browser.compact { --thumb: 40px; height: auto; flex: 1; }
  .compact .toolbar { padding: 8px 12px; gap: 4px; }
  .compact .crumbs { overflow-x: auto; scrollbar-width: none; }
  .compact .crumbs::-webkit-scrollbar { display: none; }
  .compact :global(.crumb-edit) { position: sticky; right: 0; flex: none; background: var(--color-bg-base); }
  .compact :global(.seg-icons > button) { gap: 0; padding: 0 9px; font-size: 0; }
  .compact .filter { width: 36px; padding: 0 11px; }
  .compact .filter:focus-within, .compact .filter-set { width: 140px; }
  .compact .toolbar :global(.main) { flex: none; height: var(--spacing-control); padding: 0 14px; }
  .compact .sort-bar { padding: 0 16px; }
  .compact .row { grid-template-columns: minmax(0, 1fr); }
  .compact .entry { height: 60px; margin: 0; border-radius: 0; border-bottom: 1px solid var(--color-border-subtle); padding: 0 16px; font-size: var(--text-content); }
  .compact .skeleton-row { height: 60px; }
  .compact .name { gap: 12px; }
  .compact .list.grid { grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 6px; padding: 8px; }
  .landscape .list.grid { grid-template-columns: repeat(4, minmax(0, 1fr)); }
  .compact .tile { width: auto; }
  /* Three or four tiles across leave no room for "1.9 MB · Today 23:03" on one line, so it wraps. */
  .compact .tile-meta { white-space: normal; overflow-wrap: anywhere; }
  .compact .footer { padding: 10px 12px 18px; }
  .compact .footer :global(.main) { flex: 1; min-width: 0; height: 44px; justify-content: center; }
</style>
