<script lang="ts">
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
</script>

<!-- Left column: Compose on top, Settings below with Generate pinned at its foot. -->
<aside id="ws-controls-sidebar" class="workspace-left panel-shell panel-shell-left" aria-label="Compose and settings">
  {#if authorityReady && context}
    <ComposePane {context} {busy} />
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
</aside>

<style>
  .workspace-left { display: flex; min-height: 0; flex-direction: column; }
  @media (max-width: 639px) {
    .workspace-left { flex: none; }
  }
</style>
