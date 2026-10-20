<script lang="ts">
  import { untrack } from 'svelte';
  import { browsePlaces, createFolder, errorText, listDirectory } from '$lib/api';
  import { Button, Icon, Toggle } from '$lib/components/atoms';
  import type { IconName } from '$lib/components/atoms/Icon.svelte';
  import {
    
    breadcrumbs,
    formatBytes,
    formatModified,
    jumpToLetter,
    readRecentPaths,
    readShowHidden,
    rememberRecentPath,
    writeShowHidden,
  } from '$lib/state/pathBrowser';
  import type { BrowseEntry, BrowseListing, BrowsePlace, BrowsePurpose, BrowsePurposeName } from '$lib/types';
  import { fileName } from './assetDetails';
  import Modal from './Modal.svelte';

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

  let info = $state<BrowsePurpose | null>(null);
  let places = $state<BrowsePlace[]>([]);
  let recent = $state<string[]>([]);
  let listing = $state<BrowseListing | null>(null);
  let history = $state<string[]>([]);
  let active = $state(-1);
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
  let pathInputEl = $state<HTMLInputElement | null>(null);
  let folderInputEl = $state<HTMLInputElement | null>(null);
  let requestVersion = 0;

  const folderMode = $derived(info?.kind === 'directory');
  // "Output folder" reads as "Choose output folder"; a label starting with an acronym ("LoRA file") keeps its case.
  const title = $derived(info ? `Choose ${/^[A-Z][a-z]+(\s|$)/.test(info.label) ? info.label.charAt(0).toLowerCase() + info.label.slice(1) : info.label}` : 'Choose a path');
  const visible = $derived(
    listing ? listing.entries.filter((entry) => !filter || entry.name.toLowerCase().includes(filter.toLowerCase())) : []
  );
  const crumbs = $derived(listing ? breadcrumbs(listing.path, listing.separator) : []);
  const activeEntry = $derived(active >= 0 ? visible[active] ?? null : null);
  const chosenFile = $derived(activeEntry && activeEntry.kind === 'file' && activeEntry.selectable ? activeEntry : null);
  const mainLabel = $derived(folderMode ? `Use “${listing ? fileName(listing.path) : ''}”` : 'Choose file');
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
      error = null;
      recent = readRecentPaths(name);
      // The purpose says folder or file mode, which picks the first highlighted row, so it loads first.
      void loadPlaces(name).then(() => openAt(start)).then(() => listEl?.focus());
    });
  });

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
      listing = result;
      error = null;
      filter = '';
      const focus = options.focusName ?? result.focus;
      active = focus ? result.entries.findIndex((entry) => entry.name === focus) : folderMode ? -1 : result.entries.findIndex((entry) => entry.selectable);
      if (active < 0 && !folderMode && result.entries.length > 0) active = 0;
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
    open = false;
    onchoose(path);
  }

  function confirm(): void {
    if (folderMode && listing) choose(listing.path);
    else if (chosenFile) choose(chosenFile.path);
  }

  function close(): void {
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
    else if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); editingPath = false; listEl?.focus(); }
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

  // Enter in the filter takes the first match; it must not submit a form the dialog sits in (Config, Models).
  function onFilterKeydown(event: KeyboardEvent): void {
    if (event.key === 'Enter') {
      event.preventDefault();
      if (activeEntry) activate(activeEntry);
    } else if (event.key === 'ArrowDown') {
      event.preventDefault();
      listEl?.focus();
      move(Math.max(active, 0));
    }
  }

  function move(index: number): void {
    if (visible.length === 0) return;
    active = Math.max(0, Math.min(visible.length - 1, index));
    listEl?.querySelector(`[data-index="${active}"]`)?.scrollIntoView?.({ block: 'nearest' });
  }

  function onListKeydown(event: KeyboardEvent): void {
    if (event.key === 'ArrowDown') { event.preventDefault(); move(active + 1); }
    else if (event.key === 'ArrowUp') { event.preventDefault(); move(active - 1); }
    else if (event.key === 'Home') { event.preventDefault(); move(0); }
    else if (event.key === 'End') { event.preventDefault(); move(visible.length - 1); }
    else if (event.key === 'Enter') {
      event.preventDefault();
      if (activeEntry) activate(activeEntry);
      else confirm();
    } else if (event.key === 'Backspace') { event.preventDefault(); goUp(); }
    else if (event.key === '/' || (event.key.toLowerCase() === 'l' && (event.ctrlKey || event.metaKey))) { event.preventDefault(); startEditingPath(); }
    else if (event.key.length === 1 && !event.ctrlKey && !event.metaKey && !event.altKey) {
      const index = jumpToLetter(visible, event.key, active);
      if (index >= 0) move(index);
    }
  }

  function onRowClick(entry: BrowseEntry, index: number, event: MouseEvent): void {
    // One click opens a folder; the second click of a double-click lands on the new listing and is ignored.
    if (event.detail > 1) return;
    active = index;
    if (entry.kind === 'directory') void navigate(entry.path);
  }

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

<Modal bind:open {title} size="lg" fill onclose={close}>
  <div class="browser" data-testid="path-browser">
    <div class="toolbar">
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
        <div class="crumbs" role="navigation" aria-label="Folder path">
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
          <Button variant="quiet" size="sm" icon aria-label="Type a path" title="Type a path (/)" onclick={startEditingPath}><Icon name="edit" size={12} /></Button>
        </div>
      {/if}
      <label class="filter">
        <Icon name="search" size={12} />
        <input bind:value={filter} class="filter-input" placeholder="Filter" aria-label="Filter this folder" spellcheck="false" autocomplete="off" oninput={() => { active = visible.length > 0 ? 0 : -1; }} onkeydown={onFilterKeydown}>
      </label>
      {#if folderMode}
        <Button aria-label="New folder" title="New folder" class="new-folder" onclick={startNewFolder}><Icon name="plus" size={12} /><span class="max-sm:hidden">New folder</span></Button>
      {/if}
    </div>

    <div class="place-chips" aria-label="Places">
      {#each places as place (place.id)}
        <button type="button" class="ui-chip" aria-pressed={listing?.path === place.path} onclick={() => void navigate(place.path)}>
          <Icon name={place.id.startsWith('drive:') ? 'drive' : PLACE_ICONS[place.id] ?? 'folder'} size={12} />{place.label}
        </button>
      {/each}
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
      <nav class="places" aria-label="Places">
        <p class="ui-area-label place-heading">Places</p>
        {#each places as place (place.id)}
          <button type="button" class="place" class:place-on={listing?.path === place.path} title={place.path} onclick={() => void navigate(place.path)}>
            <Icon name={place.id.startsWith('drive:') ? 'drive' : PLACE_ICONS[place.id] ?? 'folder'} size={14} /><span class="truncate">{place.label}</span>
          </button>
        {/each}
        {#if recent.length > 0}
          <p class="ui-area-label place-heading">Recent</p>
          {#each recent as path (path)}
            <button type="button" class="place" title={path} onclick={() => void navigate(path)}>
              <Icon name="clock" size={14} /><span class="truncate">{fileName(path)}</span>
            </button>
          {/each}
        {/if}
      </nav>

      <div
        bind:this={listEl}
        class="list"
        role="listbox"
        tabindex="0"
        aria-label={listing ? `Contents of ${listing.path}` : 'Folder contents'}
        aria-activedescendant={active >= 0 ? `path-browser-row-${active}` : undefined}
        aria-busy={loading ? 'true' : undefined}
        onkeydown={onListKeydown}
      >
        <div class="row head" aria-hidden="true">
          <span class="ui-area-label">Name</span><span class="ui-area-label num">Size</span><span class="ui-area-label num">Modified</span>
        </div>
        {#if showSkeleton}
          {#each [40, 62, 30, 52] as width (width)}
            <div class="row skeleton-row" aria-hidden="true"><span class="skeleton" style="width: {width}%"></span></div>
          {/each}
        {:else if listing}
          {#each visible as entry, index (entry.path)}
            <!-- The listbox handles the keyboard for its options (aria-activedescendant). -->
            <!-- svelte-ignore a11y_click_events_have_key_events -->
            <div
              id="path-browser-row-{index}"
              data-index={index}
              class="row entry"
              class:entry-dim={entry.kind === 'file' && !entry.selectable}
              class:entry-active={index === active}
              role="option"
              tabindex="-1"
              aria-selected={index === active}
              onclick={(event) => onRowClick(entry, index, event)}
              ondblclick={() => { if (entry.kind === 'file') activate(entry); }}
            >
              <span class="name">
                <span class="entry-icon" class:entry-icon-pick={entry.kind === 'file' && entry.selectable}><Icon name={entry.kind === 'directory' ? 'folder' : 'file'} size={14} /></span>
                <span class="truncate">{entry.name}</span>
              </span>
              <span class="num">{entry.size === null ? '—' : formatBytes(entry.size)}</span>
              <span class="num">{entry.modified === null ? '' : formatModified(entry.modified)}</span>
            </div>
          {/each}
          {#if visible.length === 0}
            <div class="empty">
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
            <p class="hidden-note">
              {listing.hidden_count} other {listing.hidden_count === 1 ? 'file' : 'files'} hidden ·
              <button type="button" class="ui-link" onclick={showAllFiles}>Show all</button>
            </p>
          {/if}
        {/if}
      </div>
    </div>

    <div class="footer">
      <Toggle checked={showHidden} label="Hidden files" onchange={(event) => setShowHidden((event.currentTarget as HTMLInputElement).checked)} class="max-sm:hidden" />
      {#if info && info.accepted_extensions.length > 0}
        <span class="ui-chip ext max-sm:hidden">{info.accepted_extensions.join(' ')}</span>
      {/if}
      <span class="chosen truncate" title={footerName}>{footerName}</span>
      <Button class="max-sm:hidden" onclick={close}>Cancel</Button>
      <Button variant="primary" disabled={mainDisabled} class="main" onclick={confirm}>{mainLabel}</Button>
    </div>
  </div>
</Modal>

<style>
  .browser { display: flex; flex-direction: column; height: min(560px, calc(100dvh - 6rem)); min-height: 0; }
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
  .place-chips { display: none; }
  .message { margin: 8px 16px 0; }
  .new-folder-row { display: flex; align-items: center; gap: 6px; padding: 8px 16px; border-bottom: 1px solid var(--color-border-subtle); color: var(--color-text-secondary); }
  .body { display: flex; flex: 1; min-height: 0; }
  .places { display: flex; flex-direction: column; gap: 2px; width: 188px; flex: none; padding: 6px 8px; overflow-y: auto; border-right: 1px solid var(--color-border-subtle); }
  .place-heading { padding: 8px 8px 4px; }
  .place { display: flex; align-items: center; gap: 8px; height: var(--spacing-row); padding: 0 8px; border-radius: var(--radius-sm); color: var(--color-text-secondary); font-size: var(--text-ui); text-align: left; }
  @media (hover: hover) { .place:hover { background: var(--color-bg-surface-hover); color: var(--color-text-primary); } }
  .place-on { background: var(--color-primary-surface); color: var(--color-text-primary); }
  .place-on :global(svg) { color: var(--color-primary-main); }
  .list { flex: 1; min-width: 0; overflow-y: auto; padding-bottom: 4px; }
  .list:focus-visible { outline: 2px solid var(--color-primary-main); outline-offset: -2px; }
  .row { display: grid; grid-template-columns: minmax(0, 1fr) 84px 96px; align-items: center; gap: 12px; padding: 0 12px; }
  .head { position: sticky; top: 0; z-index: 1; height: 28px; padding: 0 16px; background: var(--color-bg-overlay); border-bottom: 1px solid var(--color-border-subtle); }
  .entry { height: var(--spacing-row); margin: 1px 4px; border-radius: var(--radius-sm); color: var(--color-text-primary); font-size: var(--text-ui); cursor: default; }
  @media (hover: hover) { .entry:hover { background: var(--color-bg-surface-hover); } }
  .entry-active { background: var(--color-primary-surface); box-shadow: inset 0 0 0 1px var(--color-primary-border); }
  .list:focus-visible .entry-active { box-shadow: inset 0 0 0 2px var(--color-primary-main); }
  .entry-dim { color: var(--color-text-muted); }
  .entry-icon { display: flex; flex: none; color: var(--color-text-secondary); }
  .entry-icon-pick { color: var(--color-primary-main); }
  .name { display: flex; align-items: center; gap: 8px; min-width: 0; }
  .num { text-align: right; font-family: var(--font-mono); font-size: var(--text-meta); color: var(--color-text-muted); }
  .skeleton-row { height: var(--spacing-row); }
  .skeleton { display: block; height: 10px; border-radius: var(--radius-xs); background: var(--color-bg-fill); }
  .empty { padding: 32px 16px; text-align: center; color: var(--color-text-muted); }
  .hidden-note { padding: 8px 16px; color: var(--color-text-muted); font-size: var(--text-ui); }
  .footer { display: flex; align-items: center; gap: 8px; padding: 10px 16px; border-top: 1px solid var(--color-border-subtle); }
  .footer :global(.ui-btn) { height: var(--spacing-bar); }
  .chosen { flex: 1; min-width: 0; text-align: right; font-family: var(--font-mono); font-size: var(--text-ui); color: var(--color-text-muted); }

  @media (max-width: 639px) {
    .browser { height: auto; flex: 1; }
    .toolbar { padding: 8px 12px; }
    .filter { width: 36px; padding: 0 11px; }
    .filter:focus-within { width: 140px; }
    .place-chips { display: flex; gap: 6px; padding: 8px 12px; overflow-x: auto; border-bottom: 1px solid var(--color-border-subtle); }
    .place-chips .ui-chip { flex: none; height: 30px; gap: 6px; }
    .places, .head { display: none; }
    .row { grid-template-columns: minmax(0, 1fr) auto; }
    .row .num:last-child { display: none; }
    .entry { height: 48px; margin: 0; border-radius: 0; border-bottom: 1px solid var(--color-border-subtle); padding: 0 16px; }
    .footer { padding: 10px 12px 18px; }
    .chosen { display: none; }
    .footer :global(.main) { flex: 1; height: 44px; justify-content: center; }
  }
</style>
