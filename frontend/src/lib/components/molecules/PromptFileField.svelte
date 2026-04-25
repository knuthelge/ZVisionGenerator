<script lang="ts">
  import { inspectPromptFile } from '$lib/api/promptFiles';
  import { Button, Select } from '$lib/components/atoms';
  import PromptFileEditorDialog from '$lib/components/organisms/PromptFileEditorDialog.svelte';
  import type { PromptFileContract, PromptFileInspection, PromptFileOption, PromptSource, WorkflowMode } from '$lib/types';
  import FormField from './FormField.svelte';
  import PathField from './PathField.svelte';

  interface Props {
    contract: PromptFileContract;
    promptSource: PromptSource;
    path: string | null;
    selectedOptionId: string | null;
    workflowMode: WorkflowMode;
    negativePromptSupported: boolean;
    disabled?: boolean;
    onPathChange: (path: string | null) => void;
    onOptionChange: (optionId: string | null) => void;
  }

  let {
    contract,
    promptSource,
    path,
    selectedOptionId,
    workflowMode,
    negativePromptSupported,
    disabled = false,
    onPathChange,
    onOptionChange,
  }: Props = $props();

  let options = $state<PromptFileOption[]>([]);
  let loadingOptions = $state(false);
  let optionsError = $state<string | null>(null);
  let optionsStatus = $state<string | null>(null);
  let optionsStatusTone = $state<'muted' | 'success' | 'warning' | 'error'>('muted');
  let loadedPath = $state<string | null>(null);
  let editorOpen = $state(false);

  const selectOptions = $derived(
    options.map((option) => ({ value: option.id, label: option.label }))
  );
  const selectedOption = $derived(
    options.find((option) => option.id === selectedOptionId) ?? null
  );

  $effect(() => {
    if (!path) {
      options = [];
      loadedPath = null;
      optionsError = null;
      return;
    }

    if (promptSource === 'file' && path !== loadedPath) {
      void refreshPath(path, null).catch(() => undefined);
    }
  });

  async function applyInspection(inspection: PromptFileInspection, successMessage: string | null): Promise<string> {
    const previousSelection = selectedOptionId;
    const selectionStillActive = previousSelection !== null && inspection.options.some((option) => option.id === previousSelection);

    options = inspection.options;
    loadedPath = inspection.path;
    optionsError = null;
    optionsStatus = successMessage;
    optionsStatusTone = successMessage ? 'success' : 'muted';
    onPathChange(inspection.path);

    if (!selectionStillActive) {
      if (previousSelection !== null) {
        optionsStatus = 'The previously selected prompt option is no longer active. Select a new one before generating.';
        optionsStatusTone = 'warning';
      }
      onOptionChange(null);
    }

    if (inspection.options.length === 0) {
      optionsStatus = 'This prompt file has no active prompt options.';
      optionsStatusTone = 'warning';
      onOptionChange(null);
    }

    return inspection.path;
  }

  async function refreshPath(candidate: string, successMessage: string | null): Promise<string> {
    loadingOptions = true;
    optionsError = null;
    try {
      const inspection = await inspectPromptFile(candidate);
      return await applyInspection(inspection, successMessage);
    } catch (err) {
      optionsError = err instanceof Error ? err.message : 'Failed to inspect prompt file.';
      throw err instanceof Error ? err : new Error(optionsError);
    } finally {
      loadingOptions = false;
    }
  }

  function clear(): void {
    options = [];
    loadedPath = null;
    optionsError = null;
    optionsStatus = null;
    onPathChange(null);
    onOptionChange(null);
  }

  function handleSaved(inspection: PromptFileInspection): void {
    void applyInspection(inspection, 'Prompt file saved on the host machine.');
  }
</script>

<div class="space-y-4 border-t border-border-subtle pt-4">
  <div class="flex items-center justify-between gap-3">
    <span class="field-label block">Prompt File</span>
    <Button type="button" size="sm" variant="ghost" disabled={disabled || !path} onclick={() => (editorOpen = true)}>
      Edit YAML
    </Button>
  </div>

  <PathField
    id="ws-prompts-file"
    name="prompts_file"
    label="Prompt File Path"
    value={path}
    placeholder="/absolute/path/to/prompts.yaml"
    helper="Browse and edits operate on files on the machine running the Z Vision Generator server. The workspace stores the normalized path returned by the backend."
    pickerKind={contract.browse_kind}
    pickerPurpose="prompt_file"
    {disabled}
    onresolve={(candidate) => refreshPath(candidate, 'Loaded prompt file from the host machine.')}
    onclear={clear}
  />

  <FormField
    label="Prompt Option"
    for="ws-prompt-option"
    helper={contract.selection_required ? 'Choose one active prompt option from the file before generating.' : 'Choose an active prompt option from the file.'}
    error={optionsError}
    status={optionsStatus}
    statusTone={optionsStatusTone}
  >
    <Select
      id="ws-prompt-option"
      name="prompt_option_id"
      value={selectedOptionId ?? ''}
      options={selectOptions}
      placeholder={loadingOptions ? 'Loading prompt options…' : 'Select a prompt option'}
      disabled={disabled || loadingOptions || options.length === 0}
      class="rounded-md focus:border-primary-main focus:ring-4 focus:ring-primary-main"
      onchange={(event) => {
        const nextValue = (event.currentTarget as HTMLSelectElement).value;
        onOptionChange(nextValue || null);
      }}
    />
  </FormField>

  {#if selectedOption}
    <div class="surface-card-muted space-y-3 rounded-md p-3">
      <div>
        <p class="field-hint-label mb-1 block">Prompt Preview</p>
        <p class="text-sm text-zinc-200">{selectedOption.prompt_preview}</p>
      </div>

      {#if selectedOption.negative_preview}
        <div>
          <p class="field-hint-label mb-1 block">Negative Preview</p>
          <p class="text-sm text-zinc-400">{selectedOption.negative_preview}</p>
        </div>

        {#if workflowMode === 'video' || !negativePromptSupported}
          <p class="text-xs text-amber-400">
            {workflowMode === 'video'
              ? 'Negative prompt entries in the file are ignored for video workflows.'
              : 'The current image model does not support negative prompts, so the file negative entry will be ignored.'}
          </p>
        {/if}
      {/if}
    </div>
  {/if}

  <PromptFileEditorDialog
    bind:open={editorOpen}
    path={path}
    acceptedExtensions={contract.accepted_extensions}
    onsaved={handleSaved}
  />
</div>