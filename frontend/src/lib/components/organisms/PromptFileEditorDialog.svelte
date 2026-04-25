<script lang="ts">
  import { readPromptFile, writePromptFile } from '$lib/api/promptFiles';
  import { Button, Textarea } from '$lib/components/atoms';
  import FormField from '../molecules/FormField.svelte';
  import Modal from '../molecules/Modal.svelte';
  import type { PromptFileInspection } from '$lib/types';

  interface Props {
    open?: boolean;
    path: string | null;
    acceptedExtensions: string[];
    onsaved?: (inspection: PromptFileInspection, rawText: string) => void;
    onclose?: () => void;
  }

  let {
    open = $bindable(false),
    path,
    acceptedExtensions,
    onsaved,
    onclose,
  }: Props = $props();

  let rawText = $state('');
  let reading = $state(false);
  let saving = $state(false);
  let error = $state<string | null>(null);
  let lastLoadedPath = $state<string | null>(null);

  $effect(() => {
    if (open && path && path !== lastLoadedPath) {
      void loadDocument(path);
    }
    if (!open) {
      error = null;
    }
  });

  async function loadDocument(nextPath: string): Promise<void> {
    reading = true;
    saving = false;
    error = null;
    try {
      const document = await readPromptFile(nextPath);
      rawText = document.raw_text;
      lastLoadedPath = document.path;
    } catch (err) {
      error = err instanceof Error ? err.message : 'Failed to read prompt file.';
    } finally {
      reading = false;
    }
  }

  async function save(): Promise<void> {
    if (!path) return;
    saving = true;
    error = null;
    try {
      const inspection = await writePromptFile(path, rawText);
      onsaved?.(inspection, rawText);
      open = false;
      onclose?.();
    } catch (err) {
      error = err instanceof Error ? err.message : 'Failed to save prompt file.';
    } finally {
      saving = false;
    }
  }

  function close(): void {
    open = false;
    onclose?.();
  }
</script>

<Modal bind:open title="Edit Prompt File" onclose={close}>
  {#snippet children()}
    <div class="space-y-4">
      <div class="surface-card-muted rounded-md px-3 py-2 text-xs text-zinc-400">
        <p class="font-medium text-zinc-200">{path ?? 'No prompt file selected'}</p>
        <p class="mt-1">Accepted extensions: {acceptedExtensions.join(', ')}</p>
      </div>

      <FormField
        label="Prompt YAML"
        for="ws-prompt-file-editor"
        helper="Edits are written atomically on the host machine. Invalid YAML stays in the dialog and does not touch the file on disk."
        error={error}
      >
        <Textarea
          id="ws-prompt-file-editor"
          value={rawText}
          rows={18}
          mono={true}
          disabled={reading || saving || !path}
          class="rounded-md focus:border-primary-main focus:ring-4 focus:ring-primary-main"
          oninput={(event) => {
            rawText = (event.currentTarget as HTMLTextAreaElement).value;
          }}
        />
      </FormField>
    </div>
  {/snippet}

  {#snippet footer()}
    <Button type="button" variant="ghost" onclick={close}>Cancel</Button>
    <Button type="button" variant="primary" disabled={reading || !path} loading={saving} onclick={() => void save()}>
      Save File
    </Button>
  {/snippet}
</Modal>