<script lang="ts">
  import { Icon, Spinner } from '$lib/components/atoms';
  import { ActionBar } from '$lib/components/molecules';
  import { draft } from '$lib/state/draft.svelte';
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
    imageFile: File | null;
    referencePreviewUrl?: string | null;
    lastSeed?: number | null;
    onImageFileChange: (file: File | null) => void;
  }

  let { context, busy, jobsActive = false, queuedCount = 0, imageFile, referencePreviewUrl = null, lastSeed = null, onImageFileChange }: Props = $props();
  const generateLabel = $derived(jobsActive ? 'Add to queue' : 'Generate');
  const authorityReady = $derived(context !== null && draft.authorityReady);
  const collapsed = $derived(draft.state.sidebarCollapsed);

  function setCollapsed(value: boolean): void {
    draft.update('sidebarCollapsed', value);
  }
</script>

<!-- Left column: Compose on top, Settings below with Generate pinned at its foot. When collapsed, the panes stay
     in the DOM (hidden) because their fields belong to the generate form; a narrow strip takes their place. -->
<aside id="ws-controls-sidebar" class="workspace-left panel-shell panel-shell-left" class:collapsed aria-label="Compose and settings">
  <div class="sidebar-panes">
    {#if authorityReady && context}
      <ComposePane {context} {busy} {jobsActive} oncollapse={() => setCollapsed(true)} />
      <SettingsPane {context} {busy} {jobsActive} {imageFile} {referencePreviewUrl} {lastSeed} {onImageFileChange} />
    {:else}
      <div class="flex-1 overflow-y-auto p-3 custom-scrollbar">
        <p class="flex items-center gap-2 text-ui text-text-muted" role="status"><Spinner size="sm" />Loading settings…</p>
      </div>
      <ActionBar class="panel-footer z-10 w-full shrink-0 p-3">
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
          title="{generateLabel} (⌘↵)"
          disabled={busy || !authorityReady}
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
  .strip-expand:hover { background: var(--color-bg-surface-hover); color: var(--color-text-primary); }
  .strip-generate { display: grid; place-items: center; width: 36px; height: 36px; border-radius: var(--radius-sm); background: var(--color-primary-main); color: var(--color-primary-ink); }
  .strip-generate:hover { background: var(--color-primary-hover); }
  .strip-generate:disabled { border: 1px solid var(--color-border-strong); background: var(--color-bg-base); color: var(--color-text-muted); cursor: not-allowed; }
  .strip-generate-wrap { position: relative; }
  .strip-badge { position: absolute; top: -6px; right: -6px; display: grid; place-items: center; min-width: 18px; height: 18px; padding: 0 5px; border: 2px solid var(--color-bg-base); border-radius: var(--radius-xs); background: var(--color-primary-main); color: var(--color-primary-ink); font-size: var(--text-meta); font-weight: 700; }
  @media (max-width: 639px) {
    .workspace-left { flex: none; }
  }
</style>
