<script lang="ts">
  import { onDestroy, onMount } from 'svelte';
  import { ApiError } from '$lib/api/client';
  import { loadPromptDocument, openPathPicker, previewPromptDocument, savePromptDocument } from '$lib/api/promptFiles';
  import { Icon, Spinner } from '$lib/components/atoms';
  import { ActionMenu, EmptyState, requestConfirm } from '$lib/components/molecules';
  import type { ActionMenuEntry } from '$lib/components/molecules/ActionMenu.svelte';
  import { hasOpenModal, isCommandKey } from '$lib/keyboard';
  import { draft } from '$lib/state/draft.svelte';
  import { requestPromptRun } from '$lib/state/pendingPromptRun';
  import { router } from '$lib/state/router.svelte';
  import { addToast } from '$lib/state/toasts.svelte';
  import type { PromptDocument, PromptDocumentLoad, PromptDocumentSaved } from '$lib/types';
  import { PromptBuilder } from './builder.svelte';
  import { optionId, remapOptionIds } from './document';
  import { DragDrop } from './dragDrop.svelte';
  import EntryPreview from './EntryPreview.svelte';
  import NewFileDialog from './NewFileDialog.svelte';
  import PromptSetSection from './PromptSetSection.svelte';
  import RepairView from './RepairView.svelte';
  import SnippetsPanel from './SnippetsPanel.svelte';
  import { clearUnsaved, forgetRecent, loadUnsaved, recentFiles, rememberRecent, saveUnsaved } from './storage';

  const UNSAVED_DELAY_MS = 400;

  let unsavedTimer: ReturnType<typeof setTimeout> | undefined;
  const builder = new PromptBuilder((document, roll) => previewPromptDocument(document, roll), keepUnsaved);
  const dnd = new DragDrop(builder);

  let phase = $state<'empty' | 'loading' | 'ready' | 'repair' | 'error'>('empty');
  let loadingPath: string | null = null;
  let lastLoad = $state<(PromptDocumentLoad & { document: PromptDocument }) | null>(null);
  let repair = $state<{ path: string; problem: string; rawText: string; revision: string } | null>(null);
  let loadError = $state<string | null>(null);
  let restoreOffer = $state<PromptDocument | null>(null);
  let saving = $state(false);
  let fileMenuOpen = $state(false);
  let fileMenuAnchor = $state<HTMLButtonElement | null>(null);
  let newFileOpen = $state(false);

  const fileName = $derived(builder.path ? (builder.path.split(/[\\/]/).pop() ?? builder.path) : '');
  const folder = $derived(builder.path ? builder.path.slice(0, builder.path.length - fileName.length) : '');
  const errorCount = $derived(builder.errors.length);
  const saveTitle = $derived(
    errorCount > 0 ? `Fix before saving: ${[...new Set(builder.errors.map((problem) => problem.message))].join(' ')}` : 'Save (⌘S)'
  );
  const fileMenuItems = $derived<ActionMenuEntry[]>([
    ...(() => {
      const recent = recentFiles().filter((path) => path !== builder.path);
      return recent.length > 0
        ? [
            { kind: 'heading', label: 'Recent' } as const,
            ...recent.map((path): ActionMenuEntry => ({ kind: 'item', id: `recent-${path}`, label: path, onselect: () => void openFile(path) })),
            { kind: 'separator' } as const,
          ]
        : [];
    })(),
    { kind: 'item', id: 'browse', label: 'Browse…', onselect: () => void browse() },
    { kind: 'item', id: 'new', label: 'New file…', onselect: () => { newFileOpen = true; } },
    { kind: 'separator' },
    { kind: 'item', id: 'reload', label: 'Reload from disk', disabled: !builder.path, onselect: () => void reload() },
  ]);

  /** Keep unsaved work in browser storage now (leaving the file or the page). */
  function flushUnsaved(): void {
    clearTimeout(unsavedTimer);
    if (builder.path && builder.dirty) saveUnsaved(builder.path, { revision: builder.revision, document: $state.snapshot(builder.doc) as PromptDocument });
  }

  function keepUnsaved(): void {
    clearTimeout(unsavedTimer);
    unsavedTimer = setTimeout(() => {
      if (builder.path && builder.dirty) saveUnsaved(builder.path, { revision: builder.revision, document: $state.snapshot(builder.doc) as PromptDocument });
    }, UNSAVED_DELAY_MS);
  }

  function detail(err: unknown, fallback: string): string {
    return err instanceof ApiError ? err.detail || fallback : err instanceof Error ? err.message : fallback;
  }

  async function openFile(path: string): Promise<void> {
    if (loadingPath === path) return;
    if (builder.dirty && builder.path && builder.path !== path) {
      flushUnsaved();
      addToast(`Unsaved changes to ${fileName} are kept; reopen it to restore them.`, 'info');
    }
    loadingPath = path;
    phase = 'loading';
    loadError = null;
    restoreOffer = null;
    try {
      const payload = await loadPromptDocument(path);
      rememberRecent(payload.path);
      if (router.params.path !== payload.path) router.replace('prompts', { path: payload.path });
      if (!payload.document) {
        // Nothing of the previous file stays loaded: it can't be saved or kept under this file's name.
        flushUnsaved();
        builder.close(payload.path);
        repair = { path: payload.path, problem: payload.problem ?? 'The file could not be read.', rawText: payload.raw_text, revision: payload.revision };
        phase = 'repair';
        return;
      }
      const loaded = { ...payload, document: payload.document };
      lastLoad = loaded;
      repair = null;
      builder.load(loaded);
      const unsaved = loadUnsaved(payload.path);
      if (unsaved && unsaved.revision === payload.revision && JSON.stringify(unsaved.document) !== JSON.stringify(payload.document)) {
        restoreOffer = unsaved.document;
      } else if (unsaved) {
        clearUnsaved(payload.path);
        if (unsaved.revision !== payload.revision) addToast('Unsaved changes from an earlier visit were dropped: the file has changed since.', 'info');
      }
      phase = 'ready';
    } catch (err) {
      loadError = detail(err, 'Could not open the prompt file.');
      if (err instanceof ApiError && err.status === 422) forgetRecent(path);
      phase = 'error';
    } finally {
      loadingPath = null;
    }
  }

  function restoreUnsaved(): void {
    if (!lastLoad || !restoreOffer) return;
    builder.load(lastLoad, restoreOffer);
    restoreOffer = null;
  }

  function discardUnsaved(): void {
    if (builder.path) clearUnsaved(builder.path);
    restoreOffer = null;
  }

  async function reload(): Promise<void> {
    if (!builder.path) return;
    if (builder.dirty) {
      const confirmed = await requestConfirm({ question: 'Reload the file from disk?', info: 'Your unsaved changes are dropped.', confirmLabel: 'Reload', danger: true });
      if (!confirmed) return;
      clearUnsaved(builder.path);
    }
    await openFile(builder.path);
  }

  async function browse(): Promise<void> {
    try {
      const result = await openPathPicker({ kind: 'existing_file', purpose: 'prompt_file', initial_path: builder.path });
      if (result.status === 'selected' && result.path) await openFile(result.path);
      else if (result.status === 'error' && result.message) addToast(result.message, 'error');
    } catch (err) {
      addToast(detail(err, 'Could not open the file picker.'), 'error');
    }
  }

  /** Save the document; returns the saved payload, or null when nothing was saved. */
  async function save(force = false): Promise<PromptDocumentSaved | null> {
    const path = builder.path;
    // Only a loaded document can be saved; the repair view saves its raw text itself.
    if (!path || saving || phase !== 'ready') return null;
    if (errorCount > 0) {
      addToast(saveTitle, 'error');
      return null;
    }
    saving = true;
    try {
      const sent = builder.snapshot();
      const saved = await savePromptDocument({
        path,
        revision: builder.revision,
        document: JSON.parse(sent) as PromptDocument,
        ...(force ? { force: true, base_text: builder.baseText } : {}),
      });
      builder.applySaved(saved, sent);
      clearTimeout(unsavedTimer);
      // Edits typed while saving stay unsaved and are kept for the next visit like any other.
      if (builder.dirty) keepUnsaved();
      else clearUnsaved(path);
      if (draft.state.promptFilePath === saved.path) {
        draft.patch({ promptFileOptionIds: remapOptionIds(draft.state.promptFileOptionIds, saved.option_id_map) });
      }
      for (const warning of saved.warnings) addToast(warning, 'warning');
      addToast(`Saved ${fileName}.`, 'success', 2500);
      return saved;
    } catch (err) {
      if (err instanceof ApiError && err.status === 409) {
        saving = false;
        const overwrite = await requestConfirm({
          question: 'The file changed on disk since you opened it.',
          info: 'Overwrite it with your version? To see the other version instead, choose Reload from disk in the file menu (your changes are dropped).',
          confirmLabel: 'Overwrite',
          danger: true,
        });
        return overwrite ? save(true) : null;
      }
      addToast(detail(err, 'Saving failed.'), 'error');
      return null;
    } finally {
      saving = false;
    }
  }

  /** Save if needed, then return the entry's current id (ids change on save). */
  async function saveFirst(entryId: string): Promise<string | null> {
    if (!builder.dirty) return entryId;
    const saved = await save();
    return saved ? (saved.ids[entryId] ?? null) : null;
  }

  async function generate(entryId: string): Promise<void> {
    const currentId = await saveFirst(entryId);
    const id = currentId ? optionId(builder.doc, currentId) : null;
    if (!builder.path || !id) return;
    requestPromptRun({ path: builder.path, optionId: id });
    router.navigate('workspace');
  }

  async function useInWorkspace(): Promise<void> {
    const path = builder.path;
    if (!path || (builder.dirty && !(await save()))) return;
    const sameFile = draft.state.promptFilePath === path;
    draft.patch({ promptSource: 'file', promptFilePath: path, promptFileOptionIds: sameFile ? draft.state.promptFileOptionIds : [] });
    router.navigate('workspace');
  }

  function onKeydown(event: KeyboardEvent): void {
    if (event.defaultPrevented || hasOpenModal()) return;
    if (isCommandKey(event) && !event.shiftKey && !event.altKey && event.key.toLowerCase() === 's') {
      event.preventDefault();
      void save();
      return;
    }
    const target = event.target as HTMLElement | null;
    const typing = target?.closest?.('input, textarea, select, [contenteditable="true"]');
    if (isCommandKey(event) && !event.shiftKey && !event.altKey && event.key.toLowerCase() === 'z' && !typing && phase === 'ready') {
      event.preventDefault();
      if (!builder.undo()) addToast('Nothing to undo.', 'info', 2000);
    }
  }

  // Follow the URL: `#/prompts?path=…` opens that file; without one, reopen the last file.
  $effect(() => {
    const wanted = router.params.path;
    if (wanted && wanted !== builder.path && wanted !== loadingPath) void openFile(wanted);
  });

  onMount(() => {
    if (!router.params.path) {
      const initial = recentFiles()[0] ?? draft.state.promptFilePath;
      if (initial) void openFile(initial);
    }
    document.addEventListener('keydown', onKeydown);
    return () => document.removeEventListener('keydown', onKeydown);
  });

  onDestroy(() => {
    builder.dispose();
    // Leaving the page keeps unsaved work for the next visit.
    flushUnsaved();
  });
</script>

<div class="prompts-page">
  <div class="ui-page-bar file-bar">
    <button bind:this={fileMenuAnchor} type="button" class="ui-btn file-button" class:file-dirty={builder.dirty} aria-haspopup="menu" aria-expanded={fileMenuOpen} aria-label="Prompt file menu" title={builder.path ?? 'Open a prompt file'} onclick={() => { fileMenuOpen = !fileMenuOpen; }}>
      <span class="dirty-dot" aria-hidden="true"></span>
      {#if builder.path}
        <span class="file-folder">{folder}</span><span class="file-name">{fileName}</span>
      {:else}
        <span class="file-name">Open a prompt file</span>
      {/if}
      <Icon name="chevdown" size={14} class="file-caret" />
    </button>
    <ActionMenu open={fileMenuOpen} anchor={fileMenuAnchor} items={fileMenuItems} label="Prompt file" onclose={() => { fileMenuOpen = false; }} />
    <span class="spacer"></span>
    {#if phase === 'ready'}
      <span class="save-status" role="status">
        {#if saving}Saving…{:else if builder.dirty}{errorCount > 0 ? `${errorCount} ${errorCount === 1 ? 'problem' : 'problems'} to fix` : 'Unsaved changes'}{:else}Saved{/if}
      </span>
      <button type="button" class="ui-btn" title="Save if needed, then select this file in the Workspace" onclick={() => void useInWorkspace()}>Use in Workspace</button>
      <button type="button" class="ui-btn ui-btn-primary" disabled={!builder.dirty || saving || errorCount > 0} title={saveTitle} onclick={() => void save()}>Save <kbd>⌘S</kbd></button>
    {/if}
  </div>

  {#if restoreOffer}
    <div class="restore-bar" role="status">
      <span>You have unsaved changes to this file from an earlier visit.</span>
      <button type="button" class="ui-btn ui-btn-sm ui-btn-primary" onclick={restoreUnsaved}>Restore</button>
      <button type="button" class="ui-btn ui-btn-sm" onclick={discardUnsaved}>Discard</button>
    </div>
  {/if}

  {#if phase === 'ready'}
    <div class="builder">
      <aside class="column column-snippets" aria-label="Snippets">
        <SnippetsPanel {builder} {dnd} />
      </aside>
      <main class="column column-sets" aria-label="Prompt sets">
        <div class="sets-head">
          <h2 class="ui-area-label">Sets</h2>
          <p class="sets-hint">Click a prompt to edit it. Type <b class="hint-snippet">$</b> for a snippet and <b class="hint-choice">{'{a|b}'}</b> for a random choice.</p>
        </div>
        <div class="sets">
          {#each builder.doc.sets as set, index (set.id)}
            <PromptSetSection {builder} {dnd} {set} {index} ongenerate={(entryId) => void generate(entryId)} />
          {:else}
            <p class="sets-empty">No sets yet. A set is a group of prompts; its name starts the output filenames.</p>
          {/each}
          <button type="button" class="ui-btn ui-btn-add new-set" onclick={() => builder.addSet()}>＋ New set</button>
        </div>
      </main>
      <aside class="column column-preview" aria-label="Preview">
        <EntryPreview {builder} />
      </aside>
    </div>
  {:else if phase === 'repair' && repair}
    {#key `${repair.path}:${repair.revision}`}
      <RepairView path={repair.path} problem={repair.problem} rawText={repair.rawText} onreload={() => repair && void openFile(repair.path)} />
    {/key}
  {:else if phase === 'loading'}
    <p class="page-loading" role="status"><Spinner size="sm" />Opening…</p>
  {:else if phase === 'error'}
    <EmptyState title="This file could not be opened">
      <p class="page-error" role="alert">{loadError}</p>
      <p>Choose another file from the file menu, or create a new one.</p>
    </EmptyState>
  {:else}
    <EmptyState title="No prompt file open">
      <p>Build prompt files from snippets and sets of prompts.</p>
      {#snippet actions()}
        <button type="button" class="ui-btn ui-btn-primary" onclick={() => void browse()}>Open a prompt file…</button>
        <button type="button" class="ui-btn" onclick={() => { newFileOpen = true; }}>New file…</button>
      {/snippet}
    </EmptyState>
  {/if}
</div>

<NewFileDialog bind:open={newFileOpen} initialDirectory={folder || null} oncreated={(path) => void openFile(path)} />

<style>
  .prompts-page { display: flex; flex: 1; flex-direction: column; min-height: 0; }
  .file-bar { flex-wrap: nowrap; gap: 6px; padding-inline: 12px; }
  .file-button { flex-shrink: 1; min-width: 0; gap: 6px; color: var(--color-text-primary); }
  .dirty-dot { display: none; width: 7px; height: 7px; flex-shrink: 0; border-radius: 50%; background: var(--color-warning); }
  .file-dirty .dirty-dot { display: inline-block; }
  .file-folder { overflow: hidden; text-overflow: ellipsis; font-family: var(--font-mono); font-size: var(--text-ui); font-weight: 400; color: var(--color-text-muted); direction: rtl; }
  .file-name { font-weight: 700; }
  .file-button :global(.file-caret) { color: var(--color-text-muted); }
  .spacer { flex: 1; }
  .save-status { margin-right: 2px; font-size: var(--text-ui); color: var(--color-text-muted); white-space: nowrap; }
  .restore-bar { display: flex; flex-shrink: 0; flex-wrap: wrap; align-items: center; gap: 6px; padding: 6px 12px; border-bottom: 1px solid var(--color-warning-border); background: var(--color-warning-surface); font-size: var(--text-ui); color: var(--color-warning); }
  .restore-bar span { margin-right: 4px; }
  .builder { display: grid; flex: 1; grid-template-columns: 250px minmax(0, 1fr) 310px; min-height: 0; }
  .column { min-height: 0; overflow-y: auto; padding: 10px 12px 24px; }
  .column-snippets, .column-preview { background: var(--color-bg-surface); }
  .column-snippets { border-right: 1px solid var(--color-border-strong); }
  .column-preview { border-left: 1px solid var(--color-border-strong); }
  .sets-head { display: flex; align-items: center; gap: 10px; min-height: 26px; margin-bottom: 8px; }
  .sets-hint { overflow: hidden; font-size: var(--text-meta); text-overflow: ellipsis; white-space: nowrap; color: var(--color-text-muted); }
  .hint-snippet { color: var(--color-accent-coral); font-family: var(--font-mono); font-weight: 500; }
  .hint-choice { color: var(--color-accent-amber); font-family: var(--font-mono); font-weight: 500; }
  .sets { display: flex; flex-direction: column; gap: 10px; }
  .sets-empty { font-size: var(--text-ui); color: var(--color-text-muted); }
  .new-set { align-self: flex-start; }
  .page-loading { display: flex; align-items: center; justify-content: center; gap: 8px; padding: 48px 16px; font-size: var(--text-ui); color: var(--color-text-muted); }
  .page-error { margin-bottom: 6px; color: var(--color-error); }
  @media (max-width: 1100px) {
    .builder { grid-template-columns: 230px minmax(0, 1fr); }
    .column-preview { display: none; }
  }
  @media (max-width: 720px) {
    .builder { display: block; overflow-y: auto; }
    .column { overflow: visible; padding-bottom: 16px; }
    .column-snippets { border-right: none; border-bottom: 1px solid var(--color-border-strong); }
    .file-folder { display: none; }
  }
</style>
