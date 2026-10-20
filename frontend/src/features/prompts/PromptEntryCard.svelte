<script lang="ts">
  import { Toggle } from '$lib/components/atoms';
  import { ActionMenu } from '$lib/components/molecules';
  import type { ActionMenuEntry } from '$lib/components/molecules/ActionMenu.svelte';
  import type { DocumentEntry, DocumentSet } from '$lib/types';
  import type { PromptBuilder } from './builder.svelte';
  import { reorderDrag, type DragDrop } from './dragDrop.svelte';
  import EnhanceChip from './EnhanceChip.svelte';
  import PromptValueEditor from './PromptValueEditor.svelte';

  interface Props {
    builder: PromptBuilder;
    dnd: DragDrop;
    set: DocumentSet;
    entry: DocumentEntry;
    index: number;
    ongenerate: (entryId: string) => void;
  }

  let { builder, dnd, set, entry, index, ongenerate }: Props = $props();

  let rowEl = $state<HTMLElement | null>(null);
  let menuAnchor = $state<HTMLButtonElement | null>(null);
  let menuOpen = $state(false);

  const selected = $derived(builder.selectedId === entry.id);
  const problems = $derived(builder.problemsFor(entry.id));
  const mark = $derived(dnd.markFor('entry', entry.id));
  const mode = $derived(entry.prompt.kind);

  const menuItems = $derived<ActionMenuEntry[]>([
    { kind: 'item', id: 'scene', label: entry.scene ? 'Remove scene' : 'Add scene', onselect: () => builder.setScene(entry.id, !entry.scene) },
    { kind: 'item', id: 'negative', label: entry.negative ? 'Remove negative prompt' : 'Add negative prompt', onselect: () => builder.setNegative(entry.id, entry.negative === null) },
    // The Text and Fields switch, for a narrow column where it does not fit on the row.
    ...(mode === 'structured' ? [] : [{ kind: 'item', id: 'layout', label: mode === 'text' ? 'Write as fields' : 'Write as text', onselect: () => builder.setValueMode(entry.id, 'prompt', mode === 'text' ? 'fields' : 'text') } as const]),
    { kind: 'separator' },
    { kind: 'item', id: 'duplicate', label: 'Duplicate', onselect: () => builder.duplicateEntry(entry.id) },
    { kind: 'item', id: 'up', label: 'Move up', disabled: index === 0, onselect: () => builder.moveEntry(entry.id, set.id, set.entries[index - 1]?.id ?? null, 'before') },
    { kind: 'item', id: 'down', label: 'Move down', disabled: index === set.entries.length - 1, onselect: () => builder.moveEntry(entry.id, set.id, set.entries[index + 1]?.id ?? null, 'after') },
    ...(builder.doc.sets.length > 1
      ? [
          { kind: 'separator' } as const,
          { kind: 'heading', label: 'Move to set' } as const,
          ...builder.doc.sets
            .filter((other) => other.id !== set.id)
            .map((other): ActionMenuEntry => ({ kind: 'item', id: `move-${other.id}`, label: other.name || '(unnamed)', onselect: () => builder.moveEntry(entry.id, other.id, null, 'after') })),
        ]
      : []),
    { kind: 'separator' },
    { kind: 'item', id: 'delete', label: 'Delete prompt', danger: true, onselect: () => builder.deleteEntry(entry.id) },
  ]);

  function onRowClick(event: MouseEvent): void {
    if ((event.target as HTMLElement).closest('button, input, textarea, label, [role="textbox"]')) return;
    builder.select(entry.id);
  }
</script>

<!-- Clicking anywhere on the row selects it for the preview; keyboard users select it by focusing inside. -->
<!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_noninteractive_element_interactions -->
<div
  role="group"
  bind:this={rowEl}
  class="entry"
  class:entry-selected={selected}
  class:entry-inactive={!entry.active}
  class:entry-dragging={dnd.dragging?.id === entry.id}
  data-drop={mark}
  data-entry-id={entry.id}
  data-dnd-entry={entry.id}
  data-dnd-set={set.id}
  aria-label="{set.name} #{index + 1}"
  onclick={onRowClick}
  onfocusin={() => builder.select(entry.id)}
>
  <div class="gutter">
    <span
      class="grip"
      use:reorderDrag={{ dnd, kind: 'entry', id: entry.id }}
      role="button"
      tabindex="-1"
      aria-label="Drag to reorder or move to another set"
      title="Drag to reorder or move to another set"
    >⠿</span>
    <span class="ordinal">{index + 1}</span>
    <Toggle
      checked={entry.active}
      ariaLabel="Active"
      title={entry.active ? 'Active: runs with the file' : 'Inactive: skipped when the file runs'}
      onchange={() => builder.toggleActive(entry.id)}
    />
  </div>

  <div class="main">
    {#if entry.scene}
      <div class="labelled" data-testid="entry-scene">
        <span class="labelled-label">Scene</span>
        <div class="labelled-value"><PromptValueEditor {builder} {entry} valueKey="scene" label="Scene" placeholder="Who and where, look, style, camera." /></div>
      </div>
    {/if}
    <PromptValueEditor {builder} {entry} valueKey="prompt" label="Prompt" placeholder="Describe the image…" />
    {#if entry.negative}
      <div class="labelled">
        <span class="labelled-label">Negative</span>
        <div class="labelled-value"><PromptValueEditor {builder} {entry} valueKey="negative" label="Negative prompt" placeholder="What to avoid…" /></div>
      </div>
    {/if}
    {#each problems as problem, problemIndex (problemIndex)}
      <p class="problem" data-severity={problem.severity}>{problem.severity === 'error' ? '✕' : '⚠'} {problem.message}</p>
    {/each}
  </div>

  <div class="actions">
    {#if mode !== 'structured'}
      <div class="ui-segmented ui-segmented-sm on-hover" role="radiogroup" aria-label="Prompt layout">
        <button type="button" role="radio" aria-checked={mode === 'text'} onclick={() => builder.setValueMode(entry.id, 'prompt', 'text')}>Text</button>
        <button type="button" role="radio" aria-checked={mode === 'fields'} onclick={() => builder.setValueMode(entry.id, 'prompt', 'fields')}>Fields</button>
      </div>
    {/if}
    <EnhanceChip
      enhance={entry.enhance}
      axes={builder.axes}
      defaults={builder.defaults}
      idPrefix="enhance-{entry.id}"
      onchange={(enhance) => builder.setEnhance(entry.id, enhance)}
    />
    <button
      type="button"
      class="ui-btn ui-btn-sm ui-btn-icon generate"
      aria-label="Generate this one"
      title={entry.active ? 'Generate this one: save, then queue it with the Workspace settings' : 'Turn the prompt on to generate it'}
      disabled={!entry.active}
      onclick={() => ongenerate(entry.id)}
    >▶</button>
    <button bind:this={menuAnchor} type="button" class="ui-btn ui-btn-sm ui-btn-icon on-hover" aria-label="Prompt actions" aria-haspopup="menu" aria-expanded={menuOpen} onclick={() => { menuOpen = !menuOpen; }}>⋯</button>
    <ActionMenu open={menuOpen} anchor={menuAnchor} items={menuItems} label="Prompt actions" onclose={() => { menuOpen = false; }} />
  </div>
</div>

<style>
  .entry { position: relative; display: grid; grid-template-columns: auto minmax(0, 1fr) auto; align-items: start; gap: 8px 14px; padding: 8px 8px 8px 4px; border-bottom: 1px solid var(--color-border-subtle); }
  @media (hover: hover) { .entry:hover { background: color-mix(in srgb, var(--color-bg-surface-hover) 50%, transparent); } }
  .entry-selected { background: color-mix(in srgb, var(--color-primary-main) 7%, var(--color-bg-surface)); box-shadow: inset 2px 0 0 var(--color-primary-main); }
  @media (hover: hover) { .entry-selected:hover { background: color-mix(in srgb, var(--color-primary-main) 7%, var(--color-bg-surface)); box-shadow: inset 2px 0 0 var(--color-primary-main); } }
  .entry-inactive .main { color: var(--color-text-muted); }
  .entry-inactive .main :global(.prompt-text-display) { color: var(--color-text-muted); }
  .entry-dragging { opacity: 0.35; }
  .entry[data-drop='into'] { box-shadow: inset 0 0 0 1px var(--color-accent-coral); }
  .entry[data-drop='before']::before, .entry[data-drop='after']::after { content: ''; position: absolute; left: 4px; right: 4px; height: 2px; border-radius: 2px; background: var(--color-primary-main); pointer-events: none; z-index: 1; }
  .entry[data-drop='before']::before { top: -1px; }
  .entry[data-drop='after']::after { bottom: -1px; }

  .gutter { display: flex; align-items: center; gap: 6px; height: 26px; }
  .grip { width: 10px; cursor: grab; font-size: var(--text-ui); line-height: 1; color: var(--color-text-muted); opacity: 0; user-select: none; }
  @media (hover: hover) { .entry:hover .grip { opacity: 0.6; } }
  /* A finger drags by the grip, so it is always there on a touch screen, finger-sized, and never scrolls the page. */
  .grip { touch-action: none; }
  @media (hover: none) { .grip { opacity: 0.6; } }
  @media (pointer: coarse) { .grip { width: 24px; font-size: 16px; text-align: center; } }
  .ordinal { width: 14px; text-align: right; font-family: var(--font-heading); font-size: var(--text-meta); font-weight: 700; color: var(--color-text-muted); }

  .main { min-width: 0; padding-top: 1px; }
  .labelled { display: flex; align-items: baseline; gap: 10px; margin-top: 2px; }
  .labelled-label { flex-shrink: 0; font-family: var(--font-heading); font-size: var(--text-meta); font-weight: 800; letter-spacing: 0.06em; text-transform: uppercase; color: var(--color-text-muted); }
  .labelled-value { flex: 1; min-width: 0; font-size: var(--text-ui); }
  .problem { margin-top: 2px; font-size: var(--text-meta); }
  .problem[data-severity='error'] { color: var(--color-error); }
  .problem[data-severity='warning'] { color: var(--color-warning); }

  .actions { display: flex; align-items: center; gap: 6px; }
  .on-hover { opacity: 0; transition: opacity 120ms; }
  .entry-selected .on-hover, .entry:focus-within .on-hover, .on-hover[aria-expanded='true'] { opacity: 1; }
  @media (hover: hover) { .entry:hover .on-hover { opacity: 1; } }
  /* No hover on a touch screen: the row's actions are always there. */
  @media (hover: none) { .on-hover { opacity: 1; } }
  .generate:not(:disabled) { color: var(--color-primary-main); }
  /* A narrow column (a phone) has no room for the actions beside the text: they share the top row with the toggle,
     and the text takes the full width under them. */
  @container prompt-sets (max-width: 560px) {
    .entry { grid-template-columns: auto minmax(0, 1fr); grid-template-areas: 'gutter actions' 'main main'; gap: 6px 8px; }
    .gutter { grid-area: gutter; height: 32px; }
    .actions { grid-area: actions; flex-wrap: wrap; justify-content: flex-end; min-width: 0; }
    .actions > .ui-segmented { display: none; }
    .main { grid-area: main; padding-left: 6px; }
  }
</style>
