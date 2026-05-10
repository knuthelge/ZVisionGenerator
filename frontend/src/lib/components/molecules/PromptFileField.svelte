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
  let manualPath = $state<string | null>(null);
  let editorOpen = $state(false);
  let editorRevision = $state(0);

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

    if (promptSource === 'file' && path !== loadedPath && path !== manualPath) {
      void refreshPath(path, null).catch(() => undefined);
    }
  });

  async function applyInspection(inspection: PromptFileInspection, successMessage: string | null): Promise<string> {
    const previousSelection = selectedOptionId;
    const selectionStillActive = previousSelection !== null && inspection.options.some((option) => option.id === previousSelection);

    options = inspection.options;
    loadedPath = inspection.path;
    manualPath = null;
    optionsError = null;
    optionsStatus = successMessage;
    optionsStatusTone = successMessage ? 'success' : 'muted';
    onPathChange(inspection.path);
    editorRevision += 1;

    if (!selectionStillActive) {
      if (previousSelection !== null) {
        optionsStatus = contract.help.stale_selection;
        optionsStatusTone = 'warning';
      }
      onOptionChange(null);
    }

    if (inspection.options.length === 0) {
      optionsStatus = contract.help.empty_options;
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
    manualPath = null;
    optionsError = null;
    optionsStatus = null;
    optionsStatusTone = 'muted';
    onPathChange(null);
    onOptionChange(null);
  }

  function handleManualPathChange(value: string): void {
    const nextPath = value.trim() ? value : null;
    if (nextPath === path) return;
    manualPath = nextPath;
    if (nextPath !== loadedPath) {
      options = [];
      optionsError = null;
      optionsStatus = null;
      optionsStatusTone = 'muted';
      loadedPath = null;
      onOptionChange(null);
    }
    onPathChange(nextPath);
  }

  function handleSaved(inspection: PromptFileInspection): void {
    void applyInspection(inspection, contract.help.saved);
  }

  function handleOptionChange(optionId: string | null): void {
    onOptionChange(optionId);
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
    helper={contract.help.path}
    pickerKind={contract.browse_kind}
    pickerPurpose="prompt_file"
    {disabled}
    onresolve={(candidate) => refreshPath(candidate, contract.help.loaded)}
    onvaluechange={handleManualPathChange}
    onclear={clear}
  />

  <FormField
    label="Prompt Option"
    for="ws-prompt-option"
    helper={contract.selection_required ? contract.help.option_required : contract.help.option_optional}
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
        handleOptionChange(nextValue || null);
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
              ? contract.help.ignored_negative_video
              : contract.help.ignored_negative_unsupported}
          </p>
        {/if}
      {/if}
    </div>
  {/if}

  <PromptFileEditorDialog
    bind:open={editorOpen}
    path={path}
    acceptedExtensions={contract.accepted_extensions}
    helperText={contract.help.editor}
    revision={editorRevision}
    onsaved={handleSaved}
  />
</div>