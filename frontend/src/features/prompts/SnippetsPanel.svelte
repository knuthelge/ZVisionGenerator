<script lang="ts">
  import { requestConfirm } from '$lib/components/molecules';
  import { addToast } from '$lib/state/toasts.svelte';
  import type { DocumentSnippet } from '$lib/types';
  import type { PromptBuilder } from './builder.svelte';
  import { findSnippet, flattenValue, valueAsText, valueData } from './document';
  import type { DragDrop } from './dragDrop.svelte';
  import PromptText from './PromptText.svelte';

  interface Props {
    builder: PromptBuilder;
    dnd: DragDrop;
  }

  let { builder, dnd }: Props = $props();

  let openId = $state<string | null>(null);

  function uses(snippet: DocumentSnippet): number {
    return builder.preview?.snippet_uses[snippet.id] ?? 0;
  }

  function add(): void {
    openId = builder.addSnippet();
  }

  function rename(snippet: DocumentSnippet, input: HTMLInputElement): void {
    const name = input.value.trim();
    if (name === snippet.name) return;
    const from = snippet.name;
    const problem = builder.renameSnippet(snippet.id, name);
    if (problem) {
      input.value = snippet.name;
      addToast(problem, 'error');
    } else if (uses(snippet) > 0) {
      addToast(`Renamed $${from} to $${name} everywhere.`, 'info');
    }
  }

  async function remove(snippet: DocumentSnippet): Promise<void> {
    const count = uses(snippet);
    if (count > 0) {
      const confirmed = await requestConfirm({
        question: `Delete $${snippet.name}?`,
        info: `It is used ${count} ${count === 1 ? 'time' : 'times'}; those references will be undefined. Undo (⌘Z) brings it back until you save.`,
        confirmLabel: 'Delete snippet',
        danger: true,
      });
      if (!confirmed) return;
    }
    openId = null;
    builder.deleteSnippet(snippet.id);
  }

  function setText(snippetId: string, text: string, undoable: boolean): void {
    const change = () => {
      const live = findSnippet(builder.doc, snippetId);
      if (live?.value.kind === 'text') live.value.text = text;
    };
    if (undoable) builder.edit(change);
    else builder.type(change);
  }
</script>

<div class="snippets-head">
  <h2 class="ui-area-label">Snippets</h2>
  <button type="button" class="ui-btn ui-btn-sm ui-btn-icon" aria-label="New snippet" title="New snippet" onclick={add}>＋</button>
</div>
<p class="snippets-hint">Reusable prompt text. Use one in a prompt as <b>$name</b>, or drag it onto an entry.</p>

<ul class="snippets" aria-label="Snippets">
  {#each builder.doc.snippets as snippet (snippet.id)}
    {@const open = openId === snippet.id}
    {@const count = uses(snippet)}
    {@const problems = builder.problemsFor(snippet.id)}
    {@const mark = dnd.markFor('snippet', snippet.id)}
    <li
      class="snippet"
      class:snippet-open={open}
      class:snippet-unused={count === 0 && !open}
      class:snippet-dragging={dnd.dragging?.id === snippet.id}
      data-drop={mark}
      draggable={!open}
      ondragstart={(event) => dnd.start(event, 'snippet', snippet.id, event.currentTarget)}
      ondragend={() => dnd.end()}
      ondragover={(event) => dnd.overSnippet(event, snippet.id, event.currentTarget)}
      ondrop={(event) => dnd.drop(event)}
    >
      {#if open}
        <input
          class="snippet-name-input"
          aria-label="Snippet name"
          value={snippet.name}
          spellcheck="false"
          onchange={(event) => rename(snippet, event.currentTarget)}
          onkeydown={(event) => { if (event.key === 'Enter') event.currentTarget.blur(); }}
        >
        {#if snippet.value.kind === 'text'}
          <div class="snippet-value">
            <PromptText
              text={snippet.value.text}
              editing={builder.editingKey === `snippet:${snippet.id}`}
              label="Value of ${snippet.name}"
              placeholder="Snippet text…"
              snippets={builder.doc.snippets.filter((item) => item.id !== snippet.id)}
              onstartedit={() => builder.beginTyping(`snippet:${snippet.id}`)}
              onendedit={() => { if (builder.editingKey === `snippet:${snippet.id}`) builder.endTyping(); }}
              ontype={(text) => setText(snippet.id, text, false)}
              onreplace={(text) => setText(snippet.id, text, true)}
            />
          </div>
        {:else}
          <p class="snippet-structured">{flattenValue(valueData(snippet.value))}</p>
          <p class="snippet-note">Structured snippet: shown flattened, kept as written.</p>
        {/if}
        <p class="snippet-note">{count === 0 ? 'Not used yet.' : `Used ${count} ${count === 1 ? 'time' : 'times'}.`} Renaming updates every reference.</p>
        <div class="snippet-actions">
          <button type="button" class="ui-btn ui-btn-sm ui-btn-danger" onclick={() => void remove(snippet)}>Delete</button>
          <button type="button" class="ui-btn ui-btn-sm" onclick={() => { openId = null; }}>Done</button>
        </div>
      {:else}
        <button type="button" class="snippet-row" aria-label="Edit snippet ${snippet.name}" onclick={() => { openId = snippet.id; }}>
          <span class="grip" aria-hidden="true">⠿</span>
          <span class="snippet-name">${snippet.name}</span>
          <span class="snippet-uses">{count ? `${count}×` : 'unused'}</span>
        </button>
        <p class="snippet-preview">{valueAsText(snippet.value) || '(empty)'}</p>
      {/if}
      {#if problems.length > 0}
        <p class="snippet-problem" role="alert">{problems.map((problem) => problem.message).join(' ')}</p>
      {/if}
    </li>
  {:else}
    <li class="snippets-empty">No snippets yet.</li>
  {/each}
</ul>

<style>
  .snippets-head { display: flex; align-items: center; justify-content: space-between; min-height: 26px; margin-bottom: 2px; }
  .snippets-hint { margin-bottom: 8px; font-size: var(--text-meta); color: var(--color-text-muted); }
  .snippets-hint b { color: var(--color-text-secondary); font-family: var(--font-mono); font-weight: 500; }
  .snippets { overflow: hidden; border-radius: var(--radius-md); background: var(--color-bg-surface); }
  .snippet { position: relative; padding: 5px 8px 6px; cursor: grab; }
  .snippet + .snippet { border-top: 1px solid var(--color-border-subtle); }
  .snippet:hover { background: var(--color-bg-surface-hover); }
  .snippet-open, .snippet-open:hover { background: var(--color-bg-raised); cursor: default; }
  .snippet-unused { opacity: 0.55; }
  .snippet-dragging { opacity: 0.35; }
  .snippet[data-drop='before']::before, .snippet[data-drop='after']::after { content: ''; position: absolute; left: 0; right: 0; height: 2px; background: var(--color-primary-main); pointer-events: none; }
  .snippet[data-drop='before']::before { top: -1px; }
  .snippet[data-drop='after']::after { bottom: -1px; }
  .snippet-row { display: flex; width: 100%; align-items: center; gap: 4px; text-align: left; cursor: inherit; }
  .grip { width: 10px; font-size: var(--text-ui); color: var(--color-text-muted); opacity: 0; }
  .snippet:hover .grip { opacity: 0.6; }
  .snippet-name { font-family: var(--font-mono); font-size: var(--text-ui); font-weight: 500; color: var(--color-accent-coral); }
  .snippet-uses { margin-left: auto; font-size: var(--text-meta); color: var(--color-text-muted); }
  .snippet-preview { overflow: hidden; padding-left: 14px; font-size: var(--text-ui); line-height: 1.4; text-overflow: ellipsis; white-space: nowrap; color: var(--color-text-secondary); }
  .snippet-name-input { width: 100%; height: 26px; padding: 0 6px; border: 1px solid transparent; border-radius: var(--radius-sm); background: var(--color-bg-base); font-family: var(--font-mono); font-size: var(--text-ui); color: var(--color-accent-coral); }
  .snippet-name-input:focus { outline: 2px solid var(--color-primary-main); outline-offset: 2px; }
  .snippet-value { margin-top: 4px; padding: 0 6px; font-size: var(--text-ui); }
  .snippet-structured { margin-top: 4px; white-space: pre-wrap; font-size: var(--text-ui); color: var(--color-text-secondary); }
  .snippet-note { margin-top: 4px; font-size: var(--text-meta); color: var(--color-text-muted); }
  .snippet-problem { margin-top: 2px; font-size: var(--text-meta); color: var(--color-error); }
  .snippet-actions { display: flex; justify-content: flex-end; gap: 4px; margin-top: 6px; }
  .snippets-empty { padding: 8px; font-size: var(--text-ui); color: var(--color-text-muted); }
</style>
