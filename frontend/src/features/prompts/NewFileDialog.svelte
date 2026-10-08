<script lang="ts">
  import { createPromptFile } from '$lib/api/promptFiles';
  import { ApiError } from '$lib/api/client';
  import { Modal, PathField } from '$lib/components/molecules';

  interface Props {
    open?: boolean;
    /** Folder to suggest, e.g. the current file's. */
    initialDirectory: string | null;
    oncreated: (path: string) => void;
  }

  let { open = $bindable(false), initialDirectory, oncreated }: Props = $props();

  let directory = $state<string | null>(null);
  let name = $state('prompts');
  let creating = $state(false);
  let error = $state<string | null>(null);

  const fileName = $derived(/\.ya?ml$/i.test(name.trim()) ? name.trim() : `${name.trim() || 'name'}.yaml`);

  $effect(() => {
    if (open) {
      directory = initialDirectory;
      error = null;
    }
  });

  async function create(): Promise<void> {
    if (!directory || !name.trim()) return;
    creating = true;
    error = null;
    try {
      const { path } = await createPromptFile(directory, name.trim());
      open = false;
      oncreated(path);
    } catch (err) {
      error = err instanceof ApiError ? err.detail : err instanceof Error ? err.message : 'Could not create the file.';
    } finally {
      creating = false;
    }
  }
</script>

<Modal bind:open title="New prompt file">
  {#snippet children()}
    <form class="new-file" onsubmit={(event) => { event.preventDefault(); void create(); }}>
      <PathField
        id="prompts-new-folder"
        label="Folder"
        bind:value={directory}
        placeholder="/absolute/path/to/folder"
        pickerKind="directory"
        pickerPurpose="prompt_file_folder"
        onresolve={async (candidate) => candidate}
        onvaluechange={(value) => { directory = value.trim() || null; }}
      />
      <label class="new-file-label" for="prompts-new-name">Name</label>
      <input id="prompts-new-name" class="surface-input new-file-name" bind:value={name} spellcheck="false" autocomplete="off">
      <p class="new-file-hint">Creates an empty <code>{fileName}</code>.</p>
      {#if error}
        <p class="new-file-error" role="alert">{error}</p>
      {/if}
    </form>
  {/snippet}
  {#snippet footer()}
    <button type="button" class="surface-button-secondary new-file-button" onclick={() => { open = false; }}>Cancel</button>
    <button type="button" class="surface-button-primary new-file-button" disabled={creating || !directory || !name.trim()} onclick={() => void create()}>Create</button>
  {/snippet}
</Modal>

<style>
  .new-file { display: flex; flex-direction: column; gap: 8px; }
  .new-file-label { margin-top: 4px; font-size: 12px; font-weight: 600; color: var(--color-text-secondary); }
  .new-file-name { padding: 6px 8px; font-size: 13px; }
  .new-file-hint { font-size: 12px; color: var(--color-text-muted); }
  .new-file-error { font-size: 12.5px; color: var(--color-error); }
  .new-file-button { padding: 6px 14px; border-radius: var(--radius-md); font-size: 13px; }
</style>
