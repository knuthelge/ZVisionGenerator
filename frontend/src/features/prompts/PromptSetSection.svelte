<script lang="ts">
  import { Icon } from '$lib/components/atoms';
  import { ActionMenu, requestConfirm } from '$lib/components/molecules';
  import type { ActionMenuEntry } from '$lib/components/molecules/ActionMenu.svelte';
  import { addToast } from '$lib/state/toasts.svelte';
  import type { DocumentSet } from '$lib/types';
  import type { PromptBuilder } from './builder.svelte';
  import type { DragDrop } from './dragDrop.svelte';
  import PromptEntryCard from './PromptEntryCard.svelte';

  interface Props {
    builder: PromptBuilder;
    dnd: DragDrop;
    set: DocumentSet;
    index: number;
    ongenerate: (entryId: string) => void;
  }

  let { builder, dnd, set, index, ongenerate }: Props = $props();

  let collapsed = $state(false);
  let sectionEl = $state<HTMLElement | null>(null);
  let menuAnchor = $state<HTMLButtonElement | null>(null);
  let menuOpen = $state(false);

  const activeCount = $derived(set.entries.filter((entry) => entry.active).length);
  const problems = $derived(builder.problemsFor(set.id));
  const mark = $derived(dnd.markFor('set', set.id));
  const lastIndex = $derived(builder.doc.sets.length - 1);

  const menuItems = $derived<ActionMenuEntry[]>([
    { kind: 'item', id: 'duplicate', label: 'Duplicate set', onselect: () => builder.duplicateSet(set.id) },
    { kind: 'item', id: 'up', label: 'Move up', disabled: index === 0, onselect: () => builder.moveSet(set.id, builder.doc.sets[index - 1].id, 'before') },
    { kind: 'item', id: 'down', label: 'Move down', disabled: index === lastIndex, onselect: () => builder.moveSet(set.id, builder.doc.sets[index + 1].id, 'after') },
    { kind: 'separator' },
    { kind: 'item', id: 'all-on', label: 'Turn all prompts on', disabled: activeCount === set.entries.length, onselect: () => builder.setAllActive(set.id, true) },
    { kind: 'item', id: 'all-off', label: 'Turn all prompts off', disabled: activeCount === 0, onselect: () => builder.setAllActive(set.id, false) },
    { kind: 'separator' },
    { kind: 'item', id: 'delete', label: 'Delete set', danger: true, onselect: () => void deleteSet() },
  ]);

  async function deleteSet(): Promise<void> {
    if (set.entries.length > 0) {
      const confirmed = await requestConfirm({
        question: `Delete set ${set.name} and its ${set.entries.length} ${set.entries.length === 1 ? 'entry' : 'entries'}?`,
        info: 'Undo (⌘Z) brings it back until you save.',
        confirmLabel: 'Delete set',
        danger: true,
      });
      if (!confirmed) return;
    }
    builder.deleteSet(set.id);
  }

  function rename(input: HTMLInputElement): void {
    const name = input.value.trim();
    if (name === set.name) return;
    const problem = builder.renameSet(set.id, name);
    if (problem) {
      input.value = set.name;
      addToast(problem, 'error');
    } else {
      addToast(`Renamed. New images from this set start with ${name}_`, 'info');
    }
  }

  function addEntry(): void {
    collapsed = false;
    builder.addEntry(set.id);
  }
</script>

<section
  bind:this={sectionEl}
  class="set"
  class:set-dragging={dnd.dragging?.id === set.id}
  data-drop={mark}
  aria-label="Set {set.name}"
  ondragover={(event) => sectionEl && dnd.overSet(event, set.id, sectionEl, (event.target as HTMLElement).closest('.set-head') !== null, set.entries.length === 0 || collapsed)}
  ondrop={(event) => dnd.drop(event)}
>
  <header class="set-head">
    <span
      class="grip"
      draggable="true"
      role="button"
      tabindex="-1"
      aria-label="Drag to reorder sets"
      title="Drag to reorder sets"
      ondragstart={(event) => dnd.start(event, 'set', set.id, sectionEl)}
      ondragend={() => dnd.end()}
    >⠿</span>
    <button type="button" class="caret" class:caret-closed={collapsed} aria-expanded={!collapsed} aria-label={collapsed ? 'Expand set' : 'Collapse set'} onclick={() => { collapsed = !collapsed; }}><Icon name="chevdown" size={14} /></button>
    <input
      class="set-name"
      aria-label="Set name"
      value={set.name}
      spellcheck="false"
      onchange={(event) => rename(event.currentTarget)}
      onkeydown={(event) => { if (event.key === 'Enter') event.currentTarget.blur(); }}
    >
    <span class="count">{activeCount} of {set.entries.length} active</span>
    <span class="spacer"></span>
    <button type="button" class="ui-btn ui-btn-sm ui-btn-icon" aria-label="Add prompt to {set.name}" title="Add prompt" onclick={addEntry}>＋</button>
    <button bind:this={menuAnchor} type="button" class="ui-btn ui-btn-sm ui-btn-icon" aria-label="Set actions" aria-haspopup="menu" aria-expanded={menuOpen} onclick={() => { menuOpen = !menuOpen; }}>⋯</button>
    <ActionMenu open={menuOpen} anchor={menuAnchor} items={menuItems} label="Set actions" onclose={() => { menuOpen = false; }} />
  </header>
  {#if problems.length > 0}
    <p class="set-problem" role="alert">{problems.map((problem) => problem.message).join(' ')}</p>
  {/if}

  {#if !collapsed}
    <div class="entries">
      {#each set.entries as entry, entryIndex (entry.id)}
        <PromptEntryCard {builder} {dnd} {set} {entry} index={entryIndex} {ongenerate} />
      {/each}
      <button type="button" class="ui-btn ui-btn-sm ui-btn-quiet add-entry" onclick={addEntry}>＋ Add prompt</button>
    </div>
  {/if}
</section>

<style>
  .set { position: relative; overflow: hidden; border: 1px solid var(--color-border-strong); border-radius: var(--radius-md); background: var(--color-bg-surface); }
  .set-dragging { opacity: 0.35; }
  .set[data-drop='before']::before, .set[data-drop='after']::after { content: ''; position: absolute; left: 0; right: 0; height: 2px; background: var(--color-primary-main); pointer-events: none; z-index: 2; }
  .set[data-drop='before']::before { top: 0; }
  .set[data-drop='after']::after { bottom: 0; }
  .set[data-drop='into'] > .set-head { background: var(--color-primary-subtle); box-shadow: inset 0 0 0 1px var(--color-primary-main); }
  .set-head { display: flex; align-items: center; gap: 6px; height: 36px; padding: 0 6px 0 4px; background: var(--color-bg-raised); border-bottom: 1px solid var(--color-border-strong); }
  .grip { width: 10px; cursor: grab; font-size: var(--text-ui); line-height: 1; color: var(--color-text-muted); opacity: 0; user-select: none; }
  .set-head:hover .grip { opacity: 0.6; }
  .caret { display: grid; place-items: center; width: 20px; height: 20px; border-radius: var(--radius-xs); color: var(--color-text-muted); transition: transform 120ms; }
  .caret:hover { color: var(--color-text-primary); }
  .caret-closed { transform: rotate(-90deg); }
  .set-name { min-width: 4ch; max-width: 100%; field-sizing: content; height: 26px; padding: 0 6px; border: 1px solid transparent; border-radius: var(--radius-sm); background: none; font-family: var(--font-heading); font-size: var(--text-content); font-weight: 800; color: var(--color-text-primary); }
  .set-name:hover { border-color: var(--color-border-strong); }
  .set-name:focus { outline: 2px solid var(--color-primary-main); outline-offset: 2px; background: var(--color-bg-base); }
  .count { font-size: var(--text-meta); color: var(--color-text-muted); white-space: nowrap; }
  .spacer { flex: 1; }
  .set-problem { padding: 4px 10px; border-bottom: 1px solid var(--color-border-subtle); font-size: var(--text-meta); color: var(--color-error); }
  .entries { display: flex; flex-direction: column; }
  .add-entry { align-self: flex-start; margin: 4px 6px 6px 22px; }
  .add-entry:hover { color: var(--color-primary-main); }
</style>
