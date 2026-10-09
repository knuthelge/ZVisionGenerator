<script lang="ts">
  import { Icon } from '$lib/components/atoms';
  import type { PromptFileOption, WorkflowMode } from '$lib/types';
  import Modal from './Modal.svelte';

  interface Props {
    open?: boolean;
    /** File name shown in the title. */
    fileLabel: string;
    options: PromptFileOption[];
    /** Image runs ignore an entry's `motion`, so it is left out of the tooltip. */
    workflowMode: WorkflowMode;
    selectedOptionIds: string[];
    /** Called with the confirmed selection, in file order. */
    onconfirm: (optionIds: string[]) => void;
    /** Opens the YAML editor; omitted when editing is unavailable. */
    onedit?: () => void;
  }

  let { open = $bindable(false), fileLabel, options, workflowMode, selectedOptionIds, onconfirm, onedit }: Props = $props();

  function enhanceSummary(spec: string): string {
    return workflowMode === 'image' ? spec.replace(/,?motion=[^,]*/, '') : spec;
  }

  // Choices are a working copy until confirmed, so Cancel leaves the run untouched.
  let draftIds = $state<Set<string>>(new Set());
  let filter = $state('');
  let collapsedSets = $state<Set<string>>(new Set());
  let expandedIds = $state<Set<string>>(new Set());
  let wasOpen = false;

  $effect(() => {
    if (open && !wasOpen) {
      draftIds = new Set(selectedOptionIds);
      filter = '';
      expandedIds = new Set();
    }
    wasOpen = open;
  });

  const query = $derived(filter.trim().toLowerCase());
  const visible = $derived(
    query
      ? options.filter((option) => `${option.set_name} ${option.prompt_preview} ${option.negative_preview ?? ''}`.toLowerCase().includes(query))
      : options
  );
  const groups = $derived.by(() => {
    const bySet = new Map<string, PromptFileOption[]>();
    for (const option of visible) bySet.set(option.set_name, [...(bySet.get(option.set_name) ?? []), option]);
    return Array.from(bySet, ([name, items]) => ({ name, items }));
  });
  const selectedCount = $derived(options.filter((option) => draftIds.has(option.id)).length);

  function setSelected(items: PromptFileOption[], selected: boolean): void {
    const next = new Set(draftIds);
    for (const item of items) {
      if (selected) next.add(item.id);
      else next.delete(item.id);
    }
    draftIds = next;
  }

  function toggleSet(name: string): void {
    const next = new Set(collapsedSets);
    if (next.has(name)) next.delete(name);
    else next.add(name);
    collapsedSets = next;
  }

  function toggleExpanded(id: string): void {
    const next = new Set(expandedIds);
    if (next.has(id)) next.delete(id);
    else next.add(id);
    expandedIds = next;
  }

  function confirm(): void {
    onconfirm(options.filter((option) => draftIds.has(option.id)).map((option) => option.id));
    open = false;
  }
</script>

<Modal bind:open title="Choose prompts · {fileLabel}" size="lg">
  <div class="chooser" data-testid="prompt-chooser">
    <div class="chooser-bar">
      <label class="chooser-filter">
        <span class="sr-only">Filter prompts</span>
        <input type="search" class="ui-field" placeholder="Filter prompts…" bind:value={filter} onkeydown={(event) => { if (event.key === 'Enter') event.preventDefault(); }}>
      </label>
      <span class="chooser-count">{selectedCount} of {options.length} selected</span>
      <button type="button" class="ui-btn ui-btn-sm ui-btn-quiet chooser-link" data-action="select-all" onclick={() => setSelected(visible, true)}>All</button>
      <button type="button" class="ui-btn ui-btn-sm ui-btn-quiet chooser-link" data-action="select-none" onclick={() => setSelected(visible, false)}>None</button>
    </div>

    {#if groups.length === 0}
      <p class="chooser-empty">No prompts match “{filter}”.</p>
    {/if}

    {#each groups as group (group.name)}
      {@const collapsed = collapsedSets.has(group.name)}
      {@const groupSelected = group.items.filter((item) => draftIds.has(item.id)).length}
      <section class="chooser-set" aria-label="Prompt set {group.name}">
        <div class="chooser-set-head">
          <button type="button" class="chooser-set-toggle" aria-expanded={!collapsed} onclick={() => toggleSet(group.name)}>
            <Icon name="chevright" size={12} class="chooser-chev" />
            <span class="font-semibold text-text-primary">{group.name}</span>
            <span class="text-text-muted">{groupSelected} of {group.items.length}</span>
          </button>
          <button type="button" class="ui-btn ui-btn-sm ui-btn-quiet chooser-link" aria-label="Select all in {group.name}" onclick={() => setSelected(group.items, true)}>All</button>
          <button type="button" class="ui-btn ui-btn-sm ui-btn-quiet chooser-link" aria-label="Select none in {group.name}" onclick={() => setSelected(group.items, false)}>None</button>
        </div>
        {#if !collapsed}
          {#each group.items as option (option.id)}
            {@const expanded = expandedIds.has(option.id)}
            {@const checked = draftIds.has(option.id)}
            <div class="chooser-option" data-selected={checked}>
              <label class="flex min-w-0 flex-1 cursor-pointer items-start gap-2">
                <input
                  type="checkbox"
                  class="accent-primary-main mt-0.5 shrink-0"
                  value={option.id}
                  {checked}
                  aria-label="Select {option.set_name} #{option.source_index + 1}"
                  onchange={(event) => setSelected([option], event.currentTarget.checked)}
                >
                <span class="min-w-0 flex-1">
                  <span class="mr-1.5 font-mono text-meta text-text-muted">#{option.source_index + 1}</span>
                  {#if option.enhance}<span class="mr-1.5 text-meta text-primary-main" title="Enhanced each image: {enhanceSummary(option.enhance)}">✨ enhanced</span>{/if}
                  <span id="prompt-detail-{option.id}" class="whitespace-pre-wrap break-words text-content text-text-primary" class:line-clamp-2={!expanded} class:block={expanded}>{option.prompt_preview}</span>
                  {#if expanded && option.negative_preview}
                    <span class="mt-2 block border-t border-border-subtle pt-2 text-ui text-text-secondary"><span class="font-medium">Negative:</span> {option.negative_preview}</span>
                  {/if}
                </span>
              </label>
              <button
                type="button"
                class="ui-btn ui-btn-sm ui-btn-quiet shrink-0"
                aria-expanded={expanded}
                aria-controls="prompt-detail-{option.id}"
                aria-label="{expanded ? 'Show less' : 'Show more'} for {option.set_name} #{option.source_index + 1}"
                onclick={() => toggleExpanded(option.id)}
              >{expanded ? 'Show less' : 'Show more'}</button>
            </div>
          {/each}
        {/if}
      </section>
    {/each}
  </div>

  {#snippet footer()}
    {#if onedit}
      <button type="button" class="ui-btn mr-auto" onclick={() => { open = false; onedit(); }}>Edit file</button>
    {/if}
    <button type="button" class="ui-btn" data-action="cancel-prompts" onclick={() => { open = false; }}>Cancel</button>
    <button type="button" class="ui-btn ui-btn-primary" data-action="confirm-prompts" data-count={selectedCount} onclick={confirm}>
      Use {selectedCount} prompt{selectedCount === 1 ? '' : 's'}
    </button>
  {/snippet}
</Modal>

<style>
  .chooser { display: flex; flex-direction: column; gap: 12px; }
  .chooser-bar { position: sticky; top: -12px; z-index: 1; display: flex; align-items: center; gap: 10px; padding: 4px 0 8px; background: var(--color-bg-overlay); }
  .chooser-filter { flex: 1; min-width: 0; }
  .chooser-count { flex: none; font-size: var(--text-ui); color: var(--color-text-muted); }
  .chooser-link { flex: none; }
  .chooser-empty { padding: 16px 0; text-align: center; font-size: var(--text-content); color: var(--color-text-muted); }
  .chooser-set { border-top: 1px solid var(--color-border-subtle); padding-top: 6px; }
  .chooser-set-head { display: flex; align-items: center; gap: 10px; padding: 4px 0; }
  .chooser-set-toggle { display: inline-flex; flex: 1; min-width: 0; align-items: center; gap: 6px; font-size: var(--text-content); text-align: left; }
  .chooser-set-toggle :global(.chooser-chev) { transition: transform 0.12s ease; color: var(--color-text-muted); }
  .chooser-set-toggle[aria-expanded='true'] :global(.chooser-chev) { transform: rotate(90deg); }
  .chooser-option { display: flex; align-items: flex-start; gap: 12px; margin: 4px 0; padding: 8px 10px; border: 1px solid transparent; border-radius: var(--radius-sm); }
  .chooser-option:hover { background: var(--color-bg-surface-hover); }
  .chooser-option[data-selected='true'] { background: color-mix(in srgb, var(--color-primary-main) 10%, transparent); }
  .chooser-set-toggle:focus-visible, .chooser-link:focus-visible { outline: 2px solid var(--color-primary-main); outline-offset: 2px; }
</style>
