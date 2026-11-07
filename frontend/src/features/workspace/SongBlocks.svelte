<script lang="ts">
  import { tick, type Snippet } from 'svelte';
  import { autogrow } from '$lib/actions/autogrow';
  import { Icon } from '$lib/components/atoms';
  import { ActionMenu, PromptTools, TextBoxHeader, type ActionMenuEntry, type ActionMenuItem } from '$lib/components/molecules';
  import type { StepperState } from '$lib/state/versions';
  import { layout } from '$lib/state/layoutMode.svelte';
  import { SECTION_NAMES, add, duplicate, hasWords, lineCount, moveDown, moveUp, parseSong, remove, serializeSong, setLead, setTag, setText, tagName, type Song } from '$lib/state/songBlocks';

  interface Props {
    /** The tagged lyrics text. The blocks are a view over it: every edit writes the whole text back through `oninput`. */
    value: string;
    /** What the lyrics box says when it holds nothing. */
    placeholder: string;
    /** The id of the element that says what the lyrics make (the status line). */
    describedby?: string;
    /** The tools at the right of the box's foot (Recent prompts). */
    tools?: Snippet;
    /** The tools at the left of the box's foot (the lyrics helper). */
    lead?: Snippet;
    /** The version shown was written by a helper: the heading carries a ✦. Typing in it makes it the user's, and the ✦ goes. */
    written?: boolean;
    /** The versions of the lyrics, stepped at the right end of the head, after Edit as text; null for none. */
    stepper?: StepperState | null;
    onstep?: (index: number) => void;
    /** The tools in each block's header, before its ⋯ menu; the block's position is given. */
    blockTools?: Snippet<[number]>;
    /**
     * Open the lyrics helper on block *index*: to write it when it has no words, else to improve it. *anchor* is the element it opens from.
     * Without it the ⋯ menu has no such item.
     */
    onhelp?: (index: number, write: boolean, anchor: HTMLElement) => void;
    /** Clear the song; given only while the box holds something and no helper is running in it. */
    onclear?: () => void;
    oninput: (text: string) => void;
  }

  let { value, placeholder, describedby, tools, lead, written = false, stepper = null, onstep, blockTools, onhelp, onclear, oninput }: Props = $props();

  /** The sections the Add row offers directly; the rest are in its More menu. */
  const QUICK_NAMES = ['Verse', 'Chorus', 'Bridge', 'Outro'] as const;
  const BLANK_SECTION = 'Section';

  type MenuKind = 'actions' | 'type' | 'add';

  let asText = $state(false);
  let menu = $state<{ kind: MenuKind; index: number; anchor: HTMLElement } | null>(null);
  let renaming = $state<number | null>(null);
  let renameCancelled = false;
  // A block that was removed with words in it: the text it left, and the text to put back.
  let removed = $state<{ name: string; before: string; after: string } | null>(null);

  const song = $derived(parseSong(value));
  /** The words before the first tag are an untitled block of their own; an empty song has only that block. */
  const showLead = $derived(song.blocks.length === 0 || song.lead.text !== '');
  const undoable = $derived(removed !== null && value === removed.after ? removed : null);

  const nameOf = (index: number): string => tagName(song.blocks[index].tag) || BLANK_SECTION;
  const meta = (index: number): string => {
    const lines = lineCount(song.blocks[index]);
    return lines === 0 ? 'no words' : `${lines} ${lines === 1 ? 'line' : 'lines'}`;
  };

  function commit(next: Song): void {
    const text = serializeSong(next);
    if (text !== value) oninput(text);
  }

  /** Put the caret in block *index*'s words, where a finger is not made to raise its keyboard. */
  async function focusWords(index: number): Promise<void> {
    await tick();
    if (!layout.coarse) document.getElementById(`ws-lyrics-block-${index}`)?.focus({ preventScroll: false });
  }

  /** Give the focus to the element *id* once the menu that held it has handed it back to its anchor. */
  async function focusSoon(id: string): Promise<void> {
    await tick();
    setTimeout(() => document.getElementById(id)?.focus(), 0);
  }

  /** Give the focus to block *index*'s ⋯ button, so the keys follow a block that moved. */
  function focusActions(index: number): Promise<void> {
    return focusSoon(`ws-lyrics-actions-${index}`);
  }

  /** Put the caret at the start of block *index*'s words, whatever the device: the user is typing there. */
  async function caretToBlock(index: number): Promise<void> {
    await tick();
    const words = document.getElementById(`ws-lyrics-block-${index}`) as HTMLTextAreaElement | null;
    words?.focus();
    words?.setSelectionRange(0, 0);
  }

  /**
   * Write an edit made by typing in the words of block *at* (-1 for the words before the first tag). A tag typed by hand cuts
   * the words into more blocks: the keys that follow it belong under it, so the caret goes to the new block. The words box
   * disappears when it is left empty, so the caret goes to the first block then.
   */
  function typed(at: number, edit: (song: Song) => Song): void {
    const next = edit(song);
    const text = serializeSong(next);
    const after = parseSong(text);
    const added = after.blocks.length - song.blocks.length;
    commit(next);
    if (added > 0) void caretToBlock(at + added);
    else if (at === -1 && after.lead.text === '' && after.blocks.length > 0) void caretToBlock(0);
  }

  function addBlock(after: number, name: string): void {
    removed = null;
    commit(add(song, after, name));
    void focusWords(after + 1);
  }

  function removeBlock(index: number): void {
    const name = nameOf(index);
    const before = value;
    const left = song.blocks.length - 1;
    const next = serializeSong(remove(song, index));
    removed = hasWords(song.blocks[index]) ? { name, before, after: next } : null;
    oninput(next);
    void focusSoon(left > 0 ? `ws-lyrics-actions-${Math.min(index, left - 1)}` : removed !== null ? 'ws-lyrics-undo' : 'ws-lyrics');
  }

  function undoRemove(): void {
    if (undoable) oninput(undoable.before);
    removed = null;
  }

  function openMenu(kind: MenuKind, index: number, event: MouseEvent): void {
    const anchor = event.currentTarget as HTMLElement;
    menu = menu !== null && menu.kind === kind && menu.index === index ? null : { kind, index, anchor };
  }

  const nameItems = (pick: (name: string) => void, checkedName?: string): ActionMenuItem[] =>
    SECTION_NAMES.map((name) => ({
      kind: 'item',
      id: `song-section-${name.toLowerCase()}`,
      label: name,
      ...(checkedName === undefined ? {} : { checked: name.toLowerCase() === checkedName.toLowerCase() }),
      onselect: () => pick(name),
    }));

  const menuItems = $derived.by((): ActionMenuEntry[] => {
    if (menu === null) return [];
    const { index, anchor } = menu;
    if (menu.kind === 'add') return nameItems((name) => addBlock(song.blocks.length - 1, name));
    if (menu.kind === 'type') {
      const current = nameOf(index);
      return [
        { kind: 'heading', label: 'Section' },
        ...nameItems((name) => commit(setTag(song, index, name)), current),
        { kind: 'separator' },
        {
          kind: 'item',
          id: 'song-section-custom',
          label: 'Other name…',
          checked: !SECTION_NAMES.some((name) => name.toLowerCase() === current.toLowerCase()),
          onselect: () => { renameCancelled = false; renaming = index; void tick().then(() => { const input = document.getElementById(`ws-lyrics-name-${index}`) as HTMLInputElement | null; input?.focus(); input?.select(); }); },
        },
      ];
    }
    const block = song.blocks[index];
    const writes = !hasWords(block);
    const help: ActionMenuEntry[] = onhelp
      ? [
          {
            kind: 'item',
            id: 'song-help',
            label: writes ? 'Write' : 'Improve this section…',
            icon: 'edit',
            onselect: () => onhelp(index, writes, anchor),
          },
          { kind: 'separator' },
        ]
      : [];
    return [
      ...help,
      { kind: 'item', id: 'song-move-up', label: 'Move up', icon: 'up', disabled: index === 0, onselect: () => { commit(moveUp(song, index)); void focusActions(index - 1); } },
      { kind: 'item', id: 'song-move-down', label: 'Move down', icon: 'down', disabled: index === song.blocks.length - 1, onselect: () => { commit(moveDown(song, index)); void focusActions(index + 1); } },
      { kind: 'item', id: 'song-duplicate', label: 'Duplicate', icon: 'copy', onselect: () => { removed = null; commit(duplicate(song, index)); void focusActions(index + 1); } },
      { kind: 'group', id: 'song-add-below', label: 'Add a section below', items: nameItems((name) => addBlock(index, name)) },
      { kind: 'separator' },
      { kind: 'item', id: 'song-remove', label: 'Remove', icon: 'trash', danger: true, onselect: () => removeBlock(index) },
    ];
  });

  const menuLabel = $derived(menu === null ? 'Section' : menu.kind === 'add' ? 'Add a section' : menu.kind === 'type' ? 'Section name' : 'Section actions');

  function closeMenu(): void {
    menu = null;
  }

  function commitName(index: number, name: string): void {
    renaming = null;
    if (!renameCancelled) commit(setTag(song, index, name));
    renameCancelled = false;
    void tick().then(() => document.getElementById(`ws-lyrics-type-${index}`)?.focus());
  }

  function onRenameKeydown(event: KeyboardEvent): void {
    if (event.key === 'Enter') { event.preventDefault(); (event.currentTarget as HTMLInputElement).blur(); }
    else if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); renameCancelled = true; (event.currentTarget as HTMLInputElement).blur(); }
  }
</script>

<div class="ui-prompt-box song" data-testid="song-blocks" data-view={asText ? 'text' : 'blocks'}>
  <TextBoxHeader heading="Lyrics" {stepper} stepperLabel="Lyrics versions" {onstep}>
    {#snippet headingMark()}
      {#if written}<span class="star" role="img" aria-label="written by a helper"><Icon name="sparkle" size={12} /></span>{/if}
    {/snippet}
    {#snippet right()}
      <button
        type="button"
        id="ws-lyrics-as-text"
        class="ui-btn ui-btn-sm view-switch"
        aria-pressed={asText}
        title="Show the lyrics as one text with its [section] tags, to edit them directly"
        onclick={() => { asText = !asText; menu = null; renaming = null; }}
      >Edit as text</button>
    {/snippet}
  </TextBoxHeader>

  {#if asText}
    <label for="ws-lyrics" class="sr-only">Lyrics</label>
    <textarea
      id="ws-lyrics"
      class="song-text"
      {placeholder}
      aria-describedby={describedby}
      {value}
      use:autogrow={value}
      oninput={(e) => oninput(e.currentTarget.value)}
    ></textarea>
  {:else}
    <div class="song-body" class:song-empty={song.blocks.length === 0}>
      {#if showLead}
        <div class="blk blk-lead" data-testid="song-lead">
          <textarea
            id="ws-lyrics"
            class="blk-text"
            aria-label={song.blocks.length === 0 ? 'Lyrics' : 'Lyrics before the first section'}
            aria-describedby={describedby}
            placeholder={song.blocks.length === 0 ? placeholder : ''}
            rows="1"
            value={song.lead.text}
            use:autogrow={song.lead.text}
            oninput={(e) => typed(-1, (current) => setLead(current, e.currentTarget.value))}
          ></textarea>
        </div>
      {/if}
      {#each song.blocks as block, index (index)}
        {@const name = nameOf(index)}
        <div class="blk" class:inst={!hasWords(block)} data-testid="song-block" data-section={name}>
          <div class="blk-head">
            {#if renaming === index}
              <input
                id={`ws-lyrics-name-${index}`}
                class="blk-rename"
                aria-label={`Name of section ${index + 1}`}
                maxlength="40"
                value={tagName(block.tag)}
                onkeydown={onRenameKeydown}
                onblur={(e) => commitName(index, e.currentTarget.value)}
              />
            {:else}
              <button
                type="button"
                id={`ws-lyrics-type-${index}`}
                class="blk-type"
                class:open={menu?.kind === 'type' && menu.index === index}
                aria-haspopup="menu"
                aria-expanded={menu?.kind === 'type' && menu.index === index}
                aria-label={`Section ${index + 1}: ${name}. Change it`}
                onclick={(e) => openMenu('type', index, e)}
              >{name}<Icon name="chevdown" size={11} /></button>
            {/if}
            <span class="blk-meta">{meta(index)}</span>
            <div class="blk-tools">
              {@render blockTools?.(index)}
              <button
                type="button"
                id={`ws-lyrics-actions-${index}`}
                class="ui-btn ui-btn-sm ui-btn-icon"
                aria-haspopup="menu"
                aria-expanded={menu?.kind === 'actions' && menu.index === index}
                aria-label={`Actions for section ${index + 1}: ${name}`}
                onclick={(e) => openMenu('actions', index, e)}
              ><Icon name="more" size={15} /></button>
            </div>
          </div>
          <textarea
            id={`ws-lyrics-block-${index}`}
            class="blk-text"
            class:blk-empty={!hasWords(block)}
            aria-label={`Words of section ${index + 1}: ${name}`}
            placeholder="Add words"
            rows="1"
            value={block.text}
            use:autogrow={block.text}
            oninput={(e) => typed(index, (current) => setText(current, index, e.currentTarget.value))}
          ></textarea>
        </div>
      {/each}
      <div class="blk-add" role="group" aria-label="Add a section">
        <span class="blk-add-label">Add</span>
        {#each QUICK_NAMES as name (name)}
          <button type="button" class="ui-btn ui-btn-sm" data-action={`song-add-${name.toLowerCase()}`} onclick={() => addBlock(song.blocks.length - 1, name)}><Icon name="plus" size={12} />{name}</button>
        {/each}
        <button
          type="button"
          id="ws-lyrics-add-more"
          class="ui-btn ui-btn-sm"
          aria-haspopup="menu"
          aria-expanded={menu?.kind === 'add'}
          onclick={(e) => openMenu('add', -1, e)}
        >More<Icon name="chevdown" size={12} /></button>
      </div>
      {#if undoable}
        <p class="blk-undo" role="status">Removed {undoable.name}. <button type="button" id="ws-lyrics-undo" class="blk-undo-link" onclick={undoRemove}>Undo</button></p>
      {/if}
    </div>
  {/if}

  <PromptTools end={tools} {onclear} clearWhat="lyrics">{@render lead?.()}</PromptTools>
</div>
<ActionMenu open={menu !== null} anchor={menu?.anchor ?? null} items={menuItems} label={menuLabel} align={menu?.kind === 'actions' ? 'end' : 'start'} onclose={closeMenu} />

<style>
  /* The focus ring belongs to the block or the control that has the focus, not to the whole box. */
  .song:focus-within { outline: none; }
  .star { display: inline-flex; align-items: center; margin-left: 5px; color: var(--color-primary-main); }
  /* Edit as text is quiet like the stepper's buttons, and teal while on (the pressed state of every toggle). */
  .view-switch:not([aria-pressed='true']) { background: transparent; color: var(--color-text-secondary); }
  .song-text { display: block; width: 100%; min-height: 120px; max-height: 320px; overflow-y: auto; resize: none; padding: 9px 10px; border: 0; background: transparent; font-size: var(--text-content); line-height: 1.5; color: var(--color-text-primary); }
  .song-text:focus, .song-text:focus-visible { outline: none; }
  .song-text::placeholder { color: var(--color-zinc-600); }

  .song-body { display: flex; flex-direction: column; gap: 6px; padding: 6px; }
  .blk { position: relative; border-radius: var(--radius-md); background: color-mix(in srgb, var(--color-bg-surface) 60%, var(--color-bg-fill)); }
  .blk:focus-within { outline: 2px solid var(--color-primary-main); outline-offset: 1px; }
  .blk.inst { background: transparent; box-shadow: inset 0 0 0 1px var(--color-border-strong); }
  .blk-lead { background: transparent; }
  .blk-head { display: flex; align-items: center; gap: 6px; min-height: 30px; padding: 0 4px 0 6px; }
  .blk-type { display: inline-flex; align-items: center; gap: 3px; max-width: 100%; height: 24px; padding: 0 6px; border-radius: var(--radius-xs); font-family: var(--font-heading); font-size: var(--text-meta); font-weight: 800; letter-spacing: 0.06em; text-transform: uppercase; color: var(--color-text-secondary); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .blk-type :global(svg) { flex: none; color: var(--color-text-muted); }
  @media (hover: hover) { .blk-type:hover { background: var(--color-bg-surface-hover); } }
  .blk-type.open { background: var(--color-bg-surface-hover); }
  .blk-type:focus-visible { outline: 2px solid var(--color-primary-main); outline-offset: -2px; }
  .blk-rename { height: 24px; width: 12em; max-width: 60%; padding: 0 6px; border: 1px solid var(--color-primary-main); border-radius: var(--radius-xs); background: var(--color-bg-base); font-size: var(--text-content); color: var(--color-text-primary); }
  .blk-rename:focus, .blk-rename:focus-visible { outline: none; }
  .blk-meta { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: var(--text-meta); color: var(--color-text-muted); }
  .blk-tools { display: flex; align-items: center; gap: 2px; margin-left: auto; }
  .blk-text { display: block; width: 100%; resize: none; overflow: hidden; padding: 2px 12px 9px 12px; border: 0; background: transparent; font-size: var(--text-content); line-height: 1.5; color: var(--color-text-primary); }
  .blk-lead .blk-text { padding-top: 6px; padding-bottom: 6px; }
  .song-body.song-empty .blk-lead .blk-text { min-height: 72px; }
  .blk-text.blk-empty { padding-top: 0; padding-bottom: 7px; }
  .blk-text:focus, .blk-text:focus-visible { outline: none; }
  .blk-text::placeholder { color: var(--color-zinc-600); }
  .blk-add { display: flex; flex-wrap: wrap; align-items: center; gap: 6px; padding: 2px 0 0 2px; }
  .blk-add-label { margin-right: 2px; font-size: var(--text-meta); font-weight: 600; color: var(--color-text-muted); }
  .blk-undo { margin: 0; padding: 0 2px; font-size: var(--text-meta); color: var(--color-text-secondary); }
  .blk-undo-link { font-weight: 600; text-decoration: underline; color: var(--color-text-primary); }

  /* Touch: 16px text, below which iOS zooms into a field as it focuses, and targets a finger can hit. */
  @media (pointer: coarse) {
    .song-text, .blk-text, .blk-rename { font-size: 16px; }
    .blk-head { min-height: 44px; }
    .blk-type { height: 36px; padding: 0 10px; font-size: 12px; }
    .blk-meta { font-size: 12px; }
    .blk-tools :global(.ui-btn-icon) { width: 40px; height: 40px; }
    .blk-add :global(.ui-btn) { height: 36px; padding: 0 12px; }
    .blk-rename { height: 36px; }
  }
</style>
