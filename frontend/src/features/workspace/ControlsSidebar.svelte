<script lang="ts">
  import { Icon } from '$lib/components/atoms';
  import { draft } from '$lib/state/draft.svelte';
  import type { WorkspaceContext } from '$lib/types';
  import ComposePane from './ComposePane.svelte';
  import SettingsPane from './SettingsPane.svelte';

  interface Props {
    context: WorkspaceContext | null;
    busy: boolean;
    imageFile: File | null;
    referencePreviewUrl?: string | null;
    lastSeed?: number | null;
    onImageFileChange: (file: File | null) => void;
  }

  let { context, busy, imageFile, referencePreviewUrl = null, lastSeed = null, onImageFileChange }: Props = $props();
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
      <ComposePane {context} {busy} oncollapse={() => setCollapsed(true)} />
      <SettingsPane {context} {busy} {imageFile} {referencePreviewUrl} {lastSeed} {onImageFileChange} />
    {:else}
      <div class="flex-1 overflow-y-auto p-3 custom-scrollbar">
        <div class="surface-card-muted space-y-3 rounded-md p-4">
          <p class="field-label">Loading Workspace Controls</p>
          <p class="text-sm text-zinc-400">Loading editable defaults and controls.</p>
        </div>
      </div>
      <div class="panel-footer z-10 w-full shrink-0 p-3">
        <button
          id="ws-submit"
          type="submit"
          disabled={true}
          class="surface-button surface-button-primary flex w-full items-center justify-center gap-2 rounded-full py-2.5"
        >
          <span>Generate</span>
          <span class="surface-shortcut ml-2 px-1.5 py-0.5 text-xs font-mono opacity-80">⌘↵</span>
        </button>
      </div>
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
      <button
        type="submit"
        class="strip-generate"
        aria-label="Generate"
        title="Generate (⌘↵)"
        disabled={busy || !authorityReady}
      >
        <Icon name="bolt" size={18} />
      </button>
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
  .strip-expand { display: grid; place-items: center; width: 32px; height: 32px; border-radius: 6px; color: var(--color-text-secondary); }
  .strip-expand:hover { background: var(--color-bg-surface-hover); color: var(--color-text-primary); }
  .strip-generate { display: grid; place-items: center; width: 44px; height: 44px; border-radius: 999px; background: var(--color-primary-main); color: var(--color-primary-ink); }
  .strip-generate:hover { background: var(--color-primary-hover); }
  .strip-generate:disabled { opacity: 0.5; cursor: not-allowed; }
  @media (max-width: 639px) {
    .workspace-left { flex: none; }
  }
</style>
