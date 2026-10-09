<script lang="ts">
  import { inspectPromptFile } from '$lib/api/promptFiles';
  import type { Snippet } from 'svelte';
  import { Icon } from '$lib/components/atoms';
  import { router } from '$lib/state/router.svelte';
  import type { PromptFileContract, PromptFileInspection, PromptFileOption, PromptSource, WorkflowMode } from '$lib/types';
  import PathField from './PathField.svelte';
  import PromptChooserDialog from './PromptChooserDialog.svelte';

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
    /** Extra tools rendered first in the box's tool row (e.g. the Enhance popover trigger). */
    tools?: Snippet;
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
    tools,
  }: Props = $props();

  let options = $state<PromptFileOption[]>([]);
  let loadingOptions = $state(false);
  let optionsError = $state<string | null>(null);
  let optionsStatus = $state<string | null>(null);
  let optionsStatusTone = $state<'muted' | 'success' | 'warning' | 'error'>('muted');
  let loadedPath = $state<string | null>(null);
  let manualPath = $state<string | null>(null);
  let chooserOpen = $state(false);
  const SUMMARY_LIMIT = 3;

  const selectedOptions = $derived(
    options.filter((option) => selectedOptionIds.includes(option.id))
  );
  const selfEnhancingCount = $derived(selectedOptions.filter((option) => option.enhance).length);
  const summaryCount = $derived(
    `${selectedOptions.length} of ${options.length} prompts${selfEnhancingCount > 0 ? ` · ✨ ${selfEnhancingCount} enhance themselves` : ''}`
  );
  const fileLabel = $derived(path ? (path.split('/').pop() ?? path) : '');

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
    loadedPath = inspection.path;
    manualPath = null;
    optionsError = null;
    optionsStatus = successMessage;
    optionsStatusTone = successMessage ? 'success' : 'muted';
    onPathChange(inspection.path);

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

  function openBuilder(): void {
    if (path) router.navigate('prompts', { path });
  }

</script>

<div class="prompt-file">
  <div class="ui-prompt-box">
    <div class="prompt-file-path">
      <PathField
        id="ws-prompts-file"
        name="prompts_file"
        label="Prompt file"
        value={path}
        placeholder="/absolute/path/to/prompts.yaml"
        helper={path ? undefined : contract.help.path}
        pickerKind={contract.browse_kind}
        pickerPurpose="prompt_file"
        {disabled}
        onresolve={(candidate) => refreshPath(candidate, contract.help.loaded)}
        onvaluechange={handleManualPathChange}
        onclear={clear}
      />
    </div>

    {#if loadingOptions}
      <p class="prompt-file-note">Loading prompts…</p>
    {:else if options.length > 0}
      <button
        type="button"
        class="prompt-file-summary"
        aria-label="Choose prompts, {selectedOptions.length} of {options.length} selected"
        {disabled}
        onclick={() => { chooserOpen = true; }}
      >
        <span class="prompt-file-count">{summaryCount}</span>
        {#if selectedOptions.length === 0}
          <span class="prompt-file-line text-warning">No prompts selected. Choose the prompts to run.</span>
        {/if}
        {#each selectedOptions.slice(0, SUMMARY_LIMIT) as option (option.id)}
          <span class="prompt-file-line"><b>{option.set_name} #{option.source_index + 1}</b> {option.prompt_preview}</span>
        {/each}
        {#if selectedOptions.length > SUMMARY_LIMIT}
          <span class="prompt-file-more">+{selectedOptions.length - SUMMARY_LIMIT} more</span>
        {/if}
      </button>
    {/if}

    <div class="ui-prompt-tools">
      {@render tools?.()}
      <button
        type="button"
        class="ui-btn ui-btn-sm"
        data-action="choose-prompts"
        title="Choose prompts"
        aria-label="Choose prompts"
        disabled={disabled || options.length === 0}
        onclick={() => { chooserOpen = true; }}
      ><Icon name="list" size={13} />Prompts…</button>
      <button type="button" class="ui-btn ui-btn-sm" data-action="edit-prompts" title="Edit the file on the Prompts page" disabled={disabled || !path} onclick={openBuilder}>
        Edit
      </button>
    </div>
  </div>

  {#each selectedOptions as option (option.id)}
    <input type="hidden" name="prompt_option_id" value={option.id}>
  {/each}

  {#if optionsError}
    <p class="prompt-file-status text-error" role="alert">{optionsError}</p>
  {:else if optionsStatus}
    <p class="prompt-file-status" data-tone={optionsStatusTone} role="status">{optionsStatus}</p>
  {/if}
  {#if selectedOptions.some((option) => option.negative_preview) && (workflowMode === 'video' || !negativePromptSupported)}
    <p class="prompt-file-status text-warning">
      {workflowMode === 'video' ? contract.help.ignored_negative_video : contract.help.ignored_negative_unsupported}
    </p>
  {/if}

  <PromptChooserDialog
    bind:open={chooserOpen}
    {fileLabel}
    {options}
    {workflowMode}
    {selectedOptionIds}
    onconfirm={onOptionChange}
    onedit={path && !disabled ? openBuilder : undefined}
  />
</div>

<style>
  .prompt-file { display: flex; flex-direction: column; gap: 6px; }
  .prompt-file-path { padding: 8px 10px; }
  .prompt-file-note { padding: 4px 10px 8px; font-size: var(--text-ui); color: var(--color-text-muted); }
  .prompt-file-summary { display: flex; width: 100%; flex-direction: column; gap: 4px; padding: 8px 10px; border-top: 1px solid var(--color-border-subtle); text-align: left; }
  .prompt-file-summary:hover:not(:disabled) { background: color-mix(in srgb, var(--color-bg-surface) 60%, transparent); }
  .prompt-file-summary:focus-visible { outline: 2px solid var(--color-primary-main); outline-offset: -2px; }
  .prompt-file-count { font-size: var(--text-meta); font-weight: 600; color: var(--color-text-secondary); }
  .prompt-file-line { display: -webkit-box; overflow: hidden; -webkit-box-orient: vertical; -webkit-line-clamp: 2; line-clamp: 2; font-size: var(--text-ui); line-height: 1.45; color: var(--color-text-secondary); }
  .prompt-file-line b { margin-right: 4px; font-weight: 600; color: var(--color-text-muted); }
  .prompt-file-more { font-size: var(--text-meta); color: var(--color-text-muted); }
  .prompt-file-status { font-size: var(--text-ui); color: var(--color-text-muted); }
  .prompt-file-status[data-tone='success'] { color: var(--color-success); }
  .prompt-file-status[data-tone='warning'] { color: var(--color-warning); }
  .prompt-file-status[data-tone='error'] { color: var(--color-error); }
</style>
