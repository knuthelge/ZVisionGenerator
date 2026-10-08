<script lang="ts">
  import { ActionMenu } from '$lib/components/molecules';
  import type { ActionMenuEntry } from '$lib/components/molecules/ActionMenu.svelte';
  import type { DocumentEntry, DocumentSet } from '$lib/types';
  import type { PromptBuilder } from './builder.svelte';
  import type { DragDrop } from './dragDrop.svelte';
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
    { kind: 'item', id: 'negative', label: entry.negative ? 'Remove negative prompt' : 'Add negative prompt', onselect: () => builder.setNegative(entry.id, entry.negative === null) },
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
    if ((event.target as HTMLElement).closest('button, input, textarea, [role="textbox"]')) return;
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
  aria-label="{set.name} #{index + 1}"
  onclick={onRowClick}
  onfocusin={() => builder.select(entry.id)}
  ondragover={(event) => rowEl && dnd.overEntry(event, set.id, entry.id, rowEl)}
  ondrop={(event) => dnd.drop(event)}
>
  <div class="gutter">
    <span
      class="grip"
      draggable="true"
      role="button"
      tabindex="-1"
      aria-label="Drag to reorder or move to another set"
      title="Drag to reorder or move to another set"
      ondragstart={(event) => dnd.start(event, 'entry', entry.id, rowEl)}
      ondragend={() => dnd.end()}
    >⠿</span>
    <span class="ordinal">{index + 1}</span>
    <button
      type="button"
      class="switch"
      class:switch-on={entry.active}
      role="switch"
      aria-checked={entry.active}
      aria-label="Active"
      title={entry.active ? 'Active: runs with the file' : 'Inactive: skipped when the file runs'}
      onclick={() => builder.toggleActive(entry.id)}
    ></button>
  </div>

  <div class="main">
    <PromptValueEditor {builder} {entry} valueKey="prompt" label="Prompt" placeholder="Describe the image…" />
    {#if entry.negative}
      <div class="negative">
        <span class="negative-label">Negative</span>
        <div class="negative-value"><PromptValueEditor {builder} {entry} valueKey="negative" label="Negative prompt" placeholder="What to avoid…" /></div>
      </div>
    {/if}
    {#each problems as problem, problemIndex (problemIndex)}
      <p class="problem" data-severity={problem.severity}>{problem.severity === 'error' ? '✕' : '⚠'} {problem.message}</p>
    {/each}
  </div>

  <div class="actions">
    {#if mode !== 'structured'}
      <div class="panel-segment on-hover" role="radiogroup" aria-label="Prompt layout">
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
      class="panel-button panel-button-icon generate"
      aria-label="Generate this one"
      title={entry.active ? 'Generate this one: save, then queue it with the Workspace settings' : 'Turn the prompt on to generate it'}
      disabled={!entry.active}
      onclick={() => ongenerate(entry.id)}
    >▶</button>
    <button bind:this={menuAnchor} type="button" class="panel-button panel-button-icon on-hover" aria-label="Prompt actions" aria-haspopup="menu" aria-expanded={menuOpen} onclick={() => { menuOpen = !menuOpen; }}>⋯</button>
    <ActionMenu open={menuOpen} anchor={menuAnchor} items={menuItems} label="Prompt actions" onclose={() => { menuOpen = false; }} />
  </div>
</div>

<style>
  .entry { position: relative; display: grid; grid-template-columns: auto minmax(0, 1fr) auto; align-items: start; gap: 8px 14px; padding: 8px 8px 8px 4px; border-bottom: 1px solid var(--color-border-subtle); }
  .entry:hover { background: color-mix(in srgb, var(--color-bg-surface-hover) 50%, transparent); }
  .entry-selected, .entry-selected:hover { background: color-mix(in srgb, var(--color-primary-main) 7%, var(--color-bg-surface)); box-shadow: inset 2px 0 0 var(--color-primary-main); }
  .entry-inactive .main { color: var(--color-text-muted); }
  .entry-inactive .main :global(.prompt-text-display) { color: var(--color-text-muted); }
  .entry-dragging { opacity: 0.35; }
  .entry[data-drop='into'] { box-shadow: inset 0 0 0 1px var(--color-accent-blush); }
  .entry[data-drop='before']::before, .entry[data-drop='after']::after { content: ''; position: absolute; left: 4px; right: 4px; height: 2px; border-radius: 2px; background: var(--color-primary-main); pointer-events: none; z-index: 1; }
  .entry[data-drop='before']::before { top: -1px; }
  .entry[data-drop='after']::after { bottom: -1px; }

  .gutter { display: flex; align-items: center; gap: 6px; height: 26px; }
  .grip { width: 10px; cursor: grab; font-size: 12px; line-height: 1; color: var(--color-text-muted); opacity: 0; user-select: none; }
  .entry:hover .grip { opacity: 0.6; }
  .ordinal { width: 14px; text-align: right; font-family: var(--font-heading); font-size: 11.5px; font-weight: 700; color: var(--color-text-muted); }
  .switch { position: relative; flex-shrink: 0; width: 26px; height: 15px; border-radius: 9999px; background: var(--color-zinc-700); transition: background-color 150ms; }
  .switch::after { content: ''; position: absolute; top: 2px; left: 2px; width: 11px; height: 11px; border-radius: 50%; background: var(--color-zinc-300); transition: transform 150ms; }
  .switch-on { background: var(--color-primary-main); }
  .switch-on::after { transform: translateX(11px); background: var(--color-primary-ink); }
  .switch:focus-visible { outline: 2px solid var(--color-primary-main); outline-offset: 2px; }

  .main { min-width: 0; padding-top: 1px; }
  .negative { display: flex; align-items: baseline; gap: 10px; margin-top: 2px; }
  .negative-label { flex-shrink: 0; font-family: var(--font-heading); font-size: 10px; font-weight: 800; letter-spacing: 0.06em; text-transform: uppercase; color: var(--color-text-muted); }
  .negative-value { flex: 1; min-width: 0; font-size: 12px; }
  .problem { margin-top: 2px; font-size: 11.5px; }
  .problem[data-severity='error'] { color: var(--color-error); }
  .problem[data-severity='warning'] { color: var(--color-warning); }

  .actions { display: flex; align-items: center; gap: 6px; }
  .on-hover { opacity: 0; transition: opacity 120ms; }
  .entry:hover .on-hover, .entry-selected .on-hover, .entry:focus-within .on-hover, .on-hover[aria-expanded='true'] { opacity: 1; }
  .generate:not(:disabled) { color: var(--color-primary-main); }
</style>
