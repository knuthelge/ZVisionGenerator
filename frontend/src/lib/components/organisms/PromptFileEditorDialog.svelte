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
    helperText: string;
    revision?: number;
    onsaved?: (inspection: PromptFileInspection, rawText: string) => void;
    onclose?: () => void;
  }

  let {
    open = $bindable(false),
    path,
    acceptedExtensions,
    helperText,
    revision = 0,
    onsaved,
    onclose,
  }: Props = $props();

  let rawText = $state('');
  let reading = $state(false);
  let saving = $state(false);
  let error = $state<string | null>(null);
  let lastLoadedKey = $state<string | null>(null);

  $effect(() => {
    const nextLoadKey = open && path ? `${path}:${revision}` : null;
    if (nextLoadKey && path && nextLoadKey !== lastLoadedKey) {
      void loadDocument(path, nextLoadKey);
    }
    if (!open) {
      error = null;
      rawText = '';
      lastLoadedKey = null;
    }
  });

  async function loadDocument(nextPath: string, loadKey: string): Promise<void> {
    lastLoadedKey = loadKey;
    reading = true;
    saving = false;
    error = null;
    rawText = '';
    try {
      const document = await readPromptFile(nextPath);
      rawText = document.raw_text;
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
      open = false;
      onsaved?.(inspection, rawText);
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
        helper={helperText}
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