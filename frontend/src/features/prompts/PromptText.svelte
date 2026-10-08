<script lang="ts">
  import { tick } from 'svelte';
  import { popover } from '$lib/actions/popover';
  import { autogrow } from '$lib/actions/autogrow';
  import type { DocumentSnippet } from '$lib/types';
  import { canMakeChoice, choiceAt, choiceOptions, replaceChoice, snippetQueryAt, tokenizePrompt, valueAsText, type TextPart } from './document';

  interface Props {
    text: string;
    editing: boolean;
    label: string;
    placeholder?: string;
    snippets: DocumentSnippet[];
    onstartedit: () => void;
    onendedit: () => void;
    /** Each keystroke while editing. */
    ontype: (text: string) => void;
    /** A whole-value change made outside typing (a choice edited from its pill). */
    onreplace: (text: string) => void;
  }

  let { text, editing, label, placeholder = '', snippets, onstartedit, onendedit, ontype, onreplace }: Props = $props();

  let textareaEl = $state<HTMLTextAreaElement | null>(null);
  let suggestions = $state<DocumentSnippet[]>([]);
  let query = $state('');
  let highlighted = $state(0);
  /**
   * The open choice editor. `start..end` is an existing `{…}` group, or (with `wrap`) selected text, or the bare
   * caret (`start === end`), that becomes a choice on Apply. Opened while typing, it keeps the text box in edit mode.
   */
  let choice = $state<{ start: number; end: number; anchor: HTMLElement; wrap: boolean; whileEditing: boolean } | null>(null);
  /** The options being edited, one per line (kept apart from `choice` so the box never reads a closed choice). */
  let draft = $state('');
  /** While typing: the choice around the caret, or a selection that could become one. */
  let caretChoice = $state<{ start: number; end: number } | null>(null);
  let selection = $state<{ start: number; end: number } | null>(null);

  const tokens = $derived(tokenizePrompt(text));
  const known = $derived(new Set(snippets.map((snippet) => snippet.name)));

  $effect(() => {
    if (editing && textareaEl && document.activeElement !== textareaEl) {
      textareaEl.focus();
      textareaEl.setSelectionRange(textareaEl.value.length, textareaEl.value.length);
    }
  });

  function snippetTitle(name: string): string {
    const snippet = snippets.find((item) => item.name === name);
    return snippet ? valueAsText(snippet.value) || '(empty)' : 'Not defined';
  }

  function onDisplayClick(event: MouseEvent): void {
    if ((event.target as HTMLElement).closest('[data-choice]')) return;
    onstartedit();
  }

  function onDisplayKeydown(event: KeyboardEvent): void {
    if (event.target !== event.currentTarget) return;
    if (event.key === 'Enter' || event.key === 'F2') {
      event.preventDefault();
      onstartedit();
    }
  }

  function updateSuggestions(): void {
    if (!textareaEl) return;
    const partial = snippetQueryAt(textareaEl.value, textareaEl.selectionStart);
    query = partial ?? '';
    suggestions = partial === null ? [] : snippets.filter((snippet) => snippet.name.startsWith(partial));
    highlighted = 0;
  }

  function onInput(event: Event): void {
    ontype((event.currentTarget as HTMLTextAreaElement).value);
    updateSuggestions();
    updateCaret();
  }

  function updateCaret(): void {
    if (!textareaEl) return;
    const { value, selectionStart, selectionEnd } = textareaEl;
    caretChoice = selectionStart === selectionEnd ? choiceAt(value, selectionStart) : null;
    selection = selectionStart !== selectionEnd && canMakeChoice(value.slice(selectionStart, selectionEnd)) ? { start: selectionStart, end: selectionEnd } : null;
  }

  /** Open the choice editor from the text box: the choice around the caret, the selection as a new one, or a new one at the caret. */
  function editChoiceWhileTyping(): void {
    if (!textareaEl) return;
    if (caretChoice) {
      draft = choiceOptions(text, caretChoice.start, caretChoice.end).join('\n');
      choice = { ...caretChoice, anchor: textareaEl, wrap: false, whileEditing: true };
    } else if (selection) {
      draft = text.slice(selection.start, selection.end);
      choice = { ...selection, anchor: textareaEl, wrap: true, whileEditing: true };
    } else if (textareaEl.selectionStart === textareaEl.selectionEnd) {
      const caret = textareaEl.selectionStart;
      draft = '';
      choice = { start: caret, end: caret, anchor: textareaEl, wrap: true, whileEditing: true };
    }
  }

  function onTextareaBlur(): void {
    suggestions = [];
    // Opening the choice editor moves focus into it; the text box stays in edit mode.
    if (!choice) onendedit();
  }

  async function closeChoice(): Promise<void> {
    const wasEditing = choice?.whileEditing ?? false;
    choice = null;
    if (!wasEditing) return;
    await tick();
    // Escape returns focus to the text box; a click elsewhere leaves it, which ends editing.
    if (textareaEl && document.activeElement !== textareaEl) onendedit();
  }

  async function insertSnippet(name: string): Promise<void> {
    if (!textareaEl) return;
    const caret = textareaEl.selectionStart;
    const start = caret - query.length;
    const next = textareaEl.value.slice(0, start) + name + textareaEl.value.slice(caret);
    ontype(next);
    suggestions = [];
    await tick();
    textareaEl.value = next;
    textareaEl.setSelectionRange(start + name.length, start + name.length);
    textareaEl.focus();
  }

  function onTextareaKeydown(event: KeyboardEvent): void {
    if (suggestions.length > 0) {
      if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
        event.preventDefault();
        highlighted = (highlighted + (event.key === 'ArrowDown' ? 1 : -1) + suggestions.length) % suggestions.length;
        return;
      }
      if (event.key === 'Enter' || event.key === 'Tab') {
        event.preventDefault();
        void insertSnippet(suggestions[highlighted].name);
        return;
      }
      if (event.key === 'Escape') {
        event.preventDefault();
        event.stopPropagation();
        suggestions = [];
        return;
      }
    }
    if (event.key === 'Enter' && event.altKey) {
      event.preventDefault();
      editChoiceWhileTyping();
      return;
    }
    if (event.key === 'Escape' || (event.key === 'Enter' && (event.metaKey || event.ctrlKey))) {
      event.preventDefault();
      onendedit();
    }
  }

  function openChoice(start: number, end: number, anchor: HTMLElement): void {
    draft = choiceOptions(text, start, end).join('\n');
    choice = { start, end, anchor, wrap: false, whileEditing: false };
  }

  /** Focus the options box once the popover is in place, caret at the end, so the keyboard flow continues. */
  function focusAtEnd(node: HTMLTextAreaElement): void {
    queueMicrotask(() => {
      node.focus();
      node.setSelectionRange(node.value.length, node.value.length);
    });
  }

  /** ⌘↵ / Ctrl↵ (or ⌥↵ again) applies the options; Enter alone adds a line. */
  function onOptionsKeydown(event: KeyboardEvent): void {
    if (event.key === 'Enter' && (event.metaKey || event.ctrlKey || event.altKey) && choice) {
      event.preventDefault();
      void applyChoice(draft.split('\n'));
    }
  }

  async function applyChoice(options: string[]): Promise<void> {
    if (!choice) return;
    const { start, end, whileEditing } = choice;
    const before = text;
    const next = replaceChoice(before, start, end, options);
    // Back to typing, with the caret just after the choice. `text` follows the parent once it applies the change.
    const caret = next.length - (before.length - end);
    onreplace(next);
    choice = null;
    if (!whileEditing) return;
    await tick();
    if (!textareaEl) return;
    textareaEl.value = next;
    textareaEl.focus();
    textareaEl.setSelectionRange(caret, caret);
    updateCaret();
  }
</script>

{#snippet parts(items: TextPart[])}
  {#each items as part, index (index)}
    {#if part.kind === 'snippet'}
      <span class="tok-snippet" class:tok-undefined={!known.has(part.name)} title={snippetTitle(part.name)}>${part.name}</span>
    {:else}{part.text}{/if}
  {/each}
{/snippet}

{#if editing}
  <div class="prompt-text-edit">
    <textarea
      bind:this={textareaEl}
      use:autogrow={text}
      class="prompt-textarea"
      aria-label={label}
      {placeholder}
      value={text}
      rows="2"
      oninput={onInput}
      onkeydown={onTextareaKeydown}
      onkeyup={updateCaret}
      onselect={updateCaret}
      onfocus={updateCaret}
      onclick={() => { updateSuggestions(); updateCaret(); }}
      onblur={onTextareaBlur}
    ></textarea>
    <div class="choice-bar">
      {#if caretChoice}
        <span>In a random choice of {choiceOptions(text, caretChoice.start, caretChoice.end).length} options.</span>
        <button type="button" class="ui-btn ui-btn-sm ui-btn-quiet choice-bar-button" onmousedown={(event) => event.preventDefault()} onclick={editChoiceWhileTyping}>Edit choice <kbd>⌥↵</kbd></button>
      {:else if selection}
        <span>Selected text.</span>
        <button type="button" class="ui-btn ui-btn-sm ui-btn-quiet choice-bar-button" onmousedown={(event) => event.preventDefault()} onclick={editChoiceWhileTyping}>Make a random choice <kbd>⌥↵</kbd></button>
      {:else}
        <span>Type <b>$</b> for a snippet.</span>
        <button type="button" class="ui-btn ui-btn-sm ui-btn-quiet choice-bar-button" onmousedown={(event) => event.preventDefault()} onclick={editChoiceWhileTyping}>New random choice <kbd>⌥↵</kbd></button>
      {/if}
    </div>
    {#if suggestions.length > 0 && textareaEl}
      <div use:popover={{ anchor: textareaEl, onclose: () => { suggestions = []; } }} class="ui-overlay snippet-suggestions" role="listbox" aria-label="Snippets">
        {#each suggestions as snippet, index (snippet.id)}
          <button
            type="button"
            role="option"
            aria-selected={index === highlighted}
            class="suggestion"
            class:suggestion-on={index === highlighted}
            onmousedown={(event) => { event.preventDefault(); void insertSnippet(snippet.name); }}
          >
            <span class="suggestion-name">${snippet.name}</span>
            <span class="suggestion-value">{valueAsText(snippet.value)}</span>
          </button>
        {/each}
      </div>
    {/if}
  </div>
{:else}
  <div
    class="prompt-text-display"
    class:prompt-text-empty={!text}
    role="textbox"
    aria-readonly="false"
    aria-multiline="true"
    aria-label={label}
    tabindex="0"
    onclick={onDisplayClick}
    onkeydown={onDisplayKeydown}
  >
    {#if !text}
      {placeholder}
    {:else}
      {#each tokens as token, index (index)}
        {#if token.kind === 'choice'}
          <button
            type="button"
            class="tok-choice"
            data-choice
            title="Random choice: edit its options"
            onclick={(event) => openChoice(token.start, token.end, event.currentTarget)}
          >
            {#each token.options as option, optionIndex (optionIndex)}
              <span class="tok-option">{#if option.length === 0}∅{:else}{@render parts(option)}{/if}</span>
            {/each}
          </button>
        {:else}
          {@render parts([token])}
        {/if}
      {/each}
    {/if}
  </div>
{/if}

{#if choice}
  <div use:popover={{ anchor: choice.anchor, onclose: () => void closeChoice() }} class="ui-overlay choice-editor" role="dialog" aria-label="Random choice">
    <p class="ui-area-label">{choice.wrap ? 'New random choice' : 'Random choice'}</p>
    <p class="choice-hint">One option per line; one is picked for each image. An empty line is an empty option. <kbd>⌘↵</kbd> applies, <kbd>Esc</kbd> cancels.</p>
    <textarea
      use:focusAtEnd
      class="choice-textarea"
      aria-label="Options, one per line"
      rows={Math.max(3, draft.split('\n').length + 1)}
      bind:value={draft}
      onkeydown={onOptionsKeydown}
    ></textarea>
    <div class="choice-actions">
      {#if !choice.wrap}
        <button type="button" class="ui-btn" title="Replace the choice with its first option" onclick={() => void applyChoice([choiceOptions(text, choice!.start, choice!.end)[0]])}>Remove choice</button>
      {/if}
      <button type="button" class="ui-btn ui-btn-primary" onclick={() => void applyChoice(draft.split('\n'))}>Apply</button>
    </div>
  </div>
{/if}

<style>
  .prompt-text-display {
    min-height: 24px;
    margin: 0 -6px;
    padding: 2px 6px;
    border-radius: var(--radius-sm);
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    line-height: 1.6;
    font-size: var(--text-content);
    color: var(--color-text-primary);
    cursor: text;
  }
  .prompt-text-display:hover { background: var(--color-bg-raised); }
  .prompt-text-display:focus-visible { outline: 2px solid var(--color-primary-main); outline-offset: 2px; }
  .prompt-text-empty { color: var(--color-text-muted); font-style: italic; }
  .prompt-text-edit { position: relative; }
  .prompt-textarea {
    display: block;
    width: 100%;
    min-height: 48px;
    max-height: 420px;
    resize: vertical;
    padding: 2px 6px;
    margin: 0 -6px;
    width: calc(100% + 12px);
    border: 1px solid var(--color-border-strong);
    border-radius: var(--radius-sm);
    background: var(--color-bg-base);
    color: var(--color-text-primary);
    font-size: var(--text-content);
    line-height: 1.6;
  }
  .prompt-textarea:focus { outline: 2px solid var(--color-primary-main); outline-offset: 2px; }
  .tok-snippet {
    display: inline-flex;
    align-items: center;
    padding: 0 7px;
    border: 1px solid color-mix(in srgb, var(--color-accent-blush) 30%, transparent);
    border-radius: var(--radius-xs);
    background: color-mix(in srgb, var(--color-accent-blush) 12%, transparent);
    color: var(--color-accent-blush);
    font-family: var(--font-mono);
    font-size: var(--text-ui);
    line-height: 1.45;
  }
  .tok-undefined {
    border-color: color-mix(in srgb, var(--color-error) 45%, transparent);
    background: color-mix(in srgb, var(--color-error) 12%, transparent);
    color: var(--color-error);
    text-decoration: wavy underline;
  }
  .tok-choice {
    display: inline-flex;
    align-items: center;
    padding: 0 2px;
    border: 1px solid color-mix(in srgb, var(--color-accent-amber) 30%, transparent);
    border-radius: var(--radius-sm);
    background: color-mix(in srgb, var(--color-accent-amber) 10%, transparent);
    font: inherit;
    line-height: 1.45;
  }
  .tok-choice:hover { border-color: var(--color-accent-amber); }
  .tok-choice:focus-visible { outline: 2px solid var(--color-accent-amber); outline-offset: 2px; }
  .tok-option { padding: 0 4px; color: var(--color-accent-amber); }
  .tok-option + .tok-option { border-left: 1px solid color-mix(in srgb, var(--color-accent-amber) 30%, transparent); }
  .choice-bar { display: flex; align-items: center; gap: 6px; min-height: 24px; margin-top: 2px; font-size: var(--text-meta); color: var(--color-text-muted); }
  .choice-bar b { font-family: var(--font-mono); font-weight: 500; color: var(--color-text-secondary); }
  .choice-bar-button { height: 22px; padding: 0 6px; font-size: var(--text-meta); color: var(--color-accent-amber); }
  .choice-bar-button kbd { font-family: var(--font-mono); font-size: var(--text-meta); opacity: 0.7; }
  .snippet-suggestions { z-index: 60; min-width: 220px; max-width: 360px; padding: 4px; }
  .suggestion { display: flex; width: 100%; align-items: baseline; gap: 10px; padding: 5px 8px; border-radius: var(--radius-sm); text-align: left; }
  .suggestion-on, .suggestion:hover { background: var(--color-bg-surface-hover); }
  .suggestion-name { font-family: var(--font-mono); font-size: var(--text-ui); color: var(--color-accent-blush); }
  .suggestion-value { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: var(--text-meta); color: var(--color-text-muted); }
  .choice-editor { z-index: 60; width: 300px; padding: 10px; }
  .choice-hint { margin-top: 2px; font-size: var(--text-meta); color: var(--color-text-muted); }
  .choice-textarea { width: 100%; margin-top: 8px; padding: 6px 8px; border: 1px solid var(--color-border-strong); border-radius: var(--radius-sm); background: var(--color-bg-surface); font-size: var(--text-content); resize: vertical; }
  .choice-textarea:focus { outline: 2px solid var(--color-primary-main); outline-offset: 2px; }
  .choice-hint kbd { font-family: var(--font-mono); font-size: var(--text-meta); color: var(--color-text-secondary); }
  .choice-actions { display: flex; justify-content: flex-end; gap: 6px; margin-top: 8px; }
</style>
