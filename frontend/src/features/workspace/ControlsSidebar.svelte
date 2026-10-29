<script lang="ts">
  import { Icon, Spinner } from '$lib/components/atoms';
  import { ActionBar } from '$lib/components/molecules';
  import { draft } from '$lib/state/draft.svelte';
  import { extendGuard, extendModelSupport } from '$lib/state/extend';
  import { extendInspection } from '$lib/state/extendInspection.svelte';
  import type { WorkspaceContext } from '$lib/types';
  import ComposePane from './ComposePane.svelte';
  import SettingsPane from './SettingsPane.svelte';

  interface Props {
    context: WorkspaceContext | null;
    /** A submit is in flight. */
    busy: boolean;
    /** A job is running or queued: Generate adds to the queue. */
    jobsActive?: boolean;
    /** Jobs waiting behind the active one, shown as a badge on the collapsed strip. */
    queuedCount?: number;
    /** Why the server refused the last Generate. */
    submitError?: string | null;
    imageFile: File | null;
    referencePreviewUrl?: string | null;
    /** The last frame: an uploaded image and the preview of a known path. */
    endImageFile?: File | null;
    endPreviewUrl?: string | null;
    lastSeed?: number | null;
    onImageFileChange: (file: File | null) => void;
    onEndImageFileChange?: (file: File | null) => void;
    /** The images uploaded as references, by row. */
    referenceFiles?: (File | null)[];
    /** Why a reference blocks Generate (a row whose image is gone), computed once by the Workspace; null when none does. */
    referenceBlock?: string | null;
    onReferenceFilesChange?: (files: (File | null)[]) => void;
    /** Copy the CLI command for the current settings. */
    oncopycli?: () => void;
    /** Portrait tablet: Compose and Settings sit side by side under the stage, and the sidebar never collapses. */
    tall?: boolean;
    /** `settings`: Settings alone (a phone's All sheet; the composer owns Compose there). */
    panes?: 'both' | 'settings';
    /** Off: the Settings footer closes with `onsettingsdone` instead of generating. */
    settingsSubmit?: boolean;
    onsettingsdone?: () => void;
  }

  let { context, busy, jobsActive = false, queuedCount = 0, submitError = null, imageFile, referencePreviewUrl = null, endImageFile = null, endPreviewUrl = null, lastSeed = null, onImageFileChange, onEndImageFileChange = () => {}, referenceFiles = [], referenceBlock = null, onReferenceFilesChange = () => {}, oncopycli, tall = false, panes = 'both', settingsSubmit = true, onsettingsdone }: Props = $props();
  const generateLabel = $derived(jobsActive ? 'Add to queue' : 'Generate');
  const authorityReady = $derived(context !== null && draft.authorityReady);
  const collapsed = $derived(draft.state.sidebarCollapsed && !tall);
  // An extension waits for a source it can read; the strip's Generate follows the same rule as the footer's.
  const sourceGuard = $derived(extendGuard(draft.state, draft.extendSource, extendInspection.statusFor(draft.state.extendSourcePath), extendModelSupport(context, draft.state.model)));
  const blockNote = $derived(sourceGuard.note ?? referenceBlock);
  const stripTitle = $derived(blockNote ? `${generateLabel}: ${blockNote.charAt(0).toLowerCase()}${blockNote.slice(1).replace(/\.$/, '')}` : `${generateLabel} (⌘↵)`);

  function setCollapsed(value: boolean): void {
    draft.update('sidebarCollapsed', value);
  }
</script>

<!-- Left column: Compose on top, Settings below with Generate pinned at its foot. When collapsed, the panes stay
     in the DOM (hidden) because their fields belong to the generate form; a narrow strip takes their place. -->
<aside id="ws-controls-sidebar" class="workspace-left ui-pane ui-pane-left" class:collapsed class:tall aria-label="Compose and settings">
  <div class="sidebar-panes">
    {#if authorityReady && context}
      {#if panes === 'both'}
        <ComposePane {context} {busy} {jobsActive} endImagePending={endImageFile !== null} oncollapse={tall ? undefined : () => setCollapsed(true)} />
      {/if}
      <SettingsPane {context} {busy} {jobsActive} {submitError} {imageFile} {referencePreviewUrl} {endImageFile} {endPreviewUrl} {lastSeed} {onImageFileChange} {onEndImageFileChange} {referenceFiles} {referenceBlock} {onReferenceFilesChange} {oncopycli} submit={settingsSubmit} ondone={onsettingsdone} />
    {:else}
      <div class="flex-1 overflow-y-auto p-3">
        <p class="flex items-center gap-2 text-ui text-text-muted" role="status"><Spinner size="sm" />Loading settings…</p>
      </div>
      <ActionBar class="ui-pane-footer z-10 w-full shrink-0 p-3">
        <button id="ws-submit" type="submit" disabled={true} class="ui-btn ui-btn-primary ui-btn-main">
          <Icon name="bolt" size={16} />
          <span>Generate</span>
          <kbd>⌘↵</kbd>
        </button>
      </ActionBar>
    {/if}
  </div>

  {#if collapsed}
    <div class="sidebar-strip" data-testid="sidebar-strip">
      <button
        type="button"
        class="strip-expand"
        aria-label="Expand sidebar"
        title="Expand sidebar"
        aria-controls="ws-controls-sidebar"
        aria-expanded="false"
        onclick={() => setCollapsed(false)}
      >
        <Icon name="uncollapse" size={16} />
      </button>
      <div class="strip-generate-wrap">
        <button
          type="submit"
          class="strip-generate"
          aria-label={generateLabel}
          title={stripTitle}
          disabled={busy || !authorityReady || sourceGuard.blocked || referenceBlock !== null}
        >
          <Icon name="bolt" size={18} />
        </button>
        {#if queuedCount > 0}
          <span class="strip-badge" data-testid="strip-queue-count" aria-label="{queuedCount} queued">{queuedCount}</span>
        {/if}
      </div>
    </div>
  {/if}
</aside>

<style>
  .workspace-left { display: flex; min-height: 0; flex-direction: column; }
  .sidebar-panes { display: flex; flex: 1; min-height: 0; flex-direction: column; }
  .sidebar-strip { display: none; }
  @media (min-width: 640px) {
    .workspace-left.collapsed .sidebar-panes { display: none; }
    .workspace-left.collapsed .sidebar-strip { display: flex; flex: 1; flex-direction: column; align-items: center; justify-content: space-between; padding: 8px 0 12px; }
  }
  .strip-expand { display: grid; place-items: center; width: 36px; height: 36px; border-radius: var(--radius-sm); color: var(--color-text-secondary); }
  @media (hover: hover) { .strip-expand:hover { background: var(--color-bg-surface-hover); color: var(--color-text-primary); } }
  .strip-generate { display: grid; place-items: center; width: 36px; height: 36px; border-radius: var(--radius-sm); background: var(--color-primary-main); color: var(--color-primary-ink); }
  @media (hover: hover) { .strip-generate:hover { background: var(--color-primary-hover); } }
  .strip-generate:disabled { background: var(--color-bg-surface-hover); color: var(--color-text-muted); cursor: not-allowed; }
  .strip-generate-wrap { position: relative; }
  .strip-badge { position: absolute; top: -6px; right: -6px; display: grid; place-items: center; min-width: 18px; height: 18px; padding: 0 5px; border: 2px solid var(--color-bg-base); border-radius: var(--radius-xs); background: var(--color-primary-main); color: var(--color-primary-ink); font-size: var(--text-meta); font-weight: 700; }
  @media (max-width: 639px) {
    .workspace-left { flex: none; }
  }
  /* Portrait tablet: the two panes share the width under the stage, each scrolling on its own. */
  .workspace-left.tall .sidebar-panes { flex-direction: row; }
  .workspace-left.tall .sidebar-panes > :global(.compose-pane) { flex: 1 1 50%; max-height: none; border-bottom: 0; border-right: 1px solid var(--color-border-subtle); }
  .workspace-left.tall .sidebar-panes > :global(.settings-pane) { flex: 1 1 50%; min-width: 0; }
</style>
