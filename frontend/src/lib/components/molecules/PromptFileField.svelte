<script lang="ts">
  import { inspectPromptFile } from '$lib/api/promptFiles';
  import { Button } from '$lib/components/atoms';
  import PromptFileEditorDialog from '$lib/components/organisms/PromptFileEditorDialog.svelte';
  import type { PromptFileContract, PromptFileInspection, PromptFileOption, PromptSource, WorkflowMode } from '$lib/types';
  import FormField from './FormField.svelte';
  import PathField from './PathField.svelte';

  interface Props {
    contract: PromptFileContract;
    promptSource: PromptSource;
    path: string | null;
    selectedOptionIds: string[];
    workflowMode: WorkflowMode;
    negativePromptSupported: boolean;
    disabled?: boolean;
    onPathChange: (path: string | null) => void;
    onOptionChange: (optionIds: string[]) => void;
  }

  let {
    contract,
    promptSource,
    path,
    selectedOptionIds,
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
  let expandedOptionIds = $state<string[]>([]);

  const selectedOptions = $derived(
    options.filter((option) => selectedOptionIds.includes(option.id))
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
    const previousSelection = selectedOptionIds;
    const activeSelection = previousSelection.filter((id) => inspection.options.some((option) => option.id === id));

    options = inspection.options;
    expandedOptionIds = [];
    loadedPath = inspection.path;
    manualPath = null;
    optionsError = null;
    optionsStatus = successMessage;
    optionsStatusTone = successMessage ? 'success' : 'muted';
    onPathChange(inspection.path);
    editorRevision += 1;

    if (activeSelection.length !== previousSelection.length) {
      if (previousSelection.length > 0) {
        optionsStatus = contract.help.stale_selection;
        optionsStatusTone = 'warning';
      }
      onOptionChange(activeSelection);
    }

    if (inspection.options.length === 0) {
      optionsStatus = contract.help.empty_options;
      optionsStatusTone = 'warning';
      onOptionChange([]);
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
    onOptionChange([]);
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
      onOptionChange([]);
    }
    onPathChange(nextPath);
  }

  function handleSaved(inspection: PromptFileInspection): void {
    void applyInspection(inspection, contract.help.saved);
  }

  function handleOptionChange(optionId: string, checked: boolean): void {
    const selected = new Set(selectedOptionIds);
    if (checked) selected.add(optionId);
    else selected.delete(optionId);
    onOptionChange(options.filter((option) => selected.has(option.id)).map((option) => option.id));
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
    helper={contract.selection_required ? contract.help.option_required : contract.help.option_optional}
    error={optionsError}
    status={optionsStatus}
    statusTone={optionsStatusTone}
  >
    <fieldset disabled={disabled || loadingOptions || options.length === 0} class="space-y-2">
      <legend class="field-label mb-2">Prompts to Run</legend>
      {#if loadingOptions}
        <p class="text-xs text-zinc-400">Loading prompts…</p>
      {:else if options.length > 0}
        <div class="flex items-center gap-3 text-xs">
          <button type="button" class="surface-link-muted" onclick={() => onOptionChange(options.map((option) => option.id))}>Select all</button>
          <button type="button" class="surface-link-muted" onclick={() => onOptionChange([])}>Clear selection</button>
          <span class="text-zinc-400">{selectedOptions.length} of {options.length} selected</span>
        </div>
        <div class="max-h-[60vh] space-y-2 overflow-y-auto" aria-label="Prompt choices">
          {#each options as option (option.id)}
            {@const expanded = expandedOptionIds.includes(option.id)}
            <div class="rounded-md border p-3 text-sm text-zinc-200 transition-colors {selectedOptionIds.includes(option.id) ? 'border-teal-500/30 bg-teal-500/5' : 'border-transparent hover:bg-zinc-800'}">
              <label class="flex cursor-pointer items-start gap-2">
                <input
                  type="checkbox"
                  name="prompt_option_id"
                  value={option.id}
                  checked={selectedOptionIds.includes(option.id)}
                  class="surface-checkbox mt-0.5 shrink-0"
                  onchange={(event) => handleOptionChange(option.id, event.currentTarget.checked)}
                >
                <span class="min-w-0 flex-1 space-y-1.5">
                  <span class="block text-xs font-medium text-zinc-400">{option.set_name} · #{option.source_index + 1}</span>
                  <span id={`prompt-detail-${option.id}`} class="whitespace-pre-wrap break-words" class:block={expanded} class:line-clamp-2={!expanded}>{option.prompt_preview}</span>
                  {#if expanded && option.negative_preview}
                    <span class="block border-t border-zinc-700/50 pt-2 text-xs text-zinc-400">
                      <span class="font-medium">Negative:</span> {option.negative_preview}
                    </span>
                  {/if}
                </span>
              </label>
              <button
                type="button"
                class="surface-link-muted ml-5 mt-2 text-xs"
                aria-expanded={expanded}
                aria-controls={`prompt-detail-${option.id}`}
                aria-label={`${expanded ? 'Show less' : 'Show more'} for ${option.set_name} #${option.source_index + 1}`}
                onclick={() => {
                  expandedOptionIds = expanded
                    ? expandedOptionIds.filter((id) => id !== option.id)
                    : [...expandedOptionIds, option.id];
                }}
              >{expanded ? 'Show less' : 'Show more'}</button>
            </div>
          {/each}
        </div>
      {/if}
    </fieldset>
  </FormField>

  {#if selectedOptions.some((option) => option.negative_preview) && (workflowMode === 'video' || !negativePromptSupported)}
    <p class="text-xs text-amber-400">
      {workflowMode === 'video'
        ? contract.help.ignored_negative_video
        : contract.help.ignored_negative_unsupported}
    </p>
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