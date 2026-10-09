<script lang="ts">
  import { requestConfirm } from '$lib/components/molecules';
  import type { DocumentEntry } from '$lib/types';
  import type { EntryValueKey, PromptBuilder } from './builder.svelte';
  import { findEntry, flattenValue } from './document';
  import PromptText from './PromptText.svelte';

  interface Props {
    builder: PromptBuilder;
    entry: DocumentEntry;
    valueKey: EntryValueKey;
    label: string;
    placeholder: string;
  }

  let { builder, entry, valueKey, label, placeholder }: Props = $props();

  const value = $derived(entry[valueKey]);
  const baseKey = $derived(`${entry.id}:${valueKey}`);

  /** The live value object, so typing writes into the document even after an undo replaced it. */
  function liveValue() {
    return findEntry(builder.doc, entry.id)?.entry[valueKey] ?? null;
  }

  function startEdit(key: string): void {
    builder.select(entry.id);
    builder.beginTyping(key);
  }

  function endEdit(key: string): void {
    if (builder.editingKey === key) builder.endTyping();
  }

  async function convertToText(): Promise<void> {
    const confirmed = await requestConfirm({
      question: 'Convert this value to text?',
      info: 'Its lists and nested fields become one line of text. The prompt the model receives stays the same.',
      confirmLabel: 'Convert',
    });
    if (confirmed) builder.setValueMode(entry.id, valueKey, 'text');
  }
</script>

{#if value?.kind === 'text'}
  <PromptText
    text={value.text}
    editing={builder.editingKey === baseKey}
    {label}
    {placeholder}
    snippets={builder.doc.snippets}
    onstartedit={() => startEdit(baseKey)}
    onendedit={() => endEdit(baseKey)}
    ontype={(text) => builder.type(() => { const live = liveValue(); if (live?.kind === 'text') live.text = text; })}
    onreplace={(text) => builder.edit(() => { const live = liveValue(); if (live?.kind === 'text') live.text = text; })}
  />
{:else if value?.kind === 'fields'}
  <div class="fields">
    {#each value.fields as field, index (index)}
      {@const fieldKey = `${baseKey}:${index}`}
      <input
        class="field-key"
        aria-label="{label} field name"
        placeholder="name"
        value={field.key}
        spellcheck="false"
        onchange={(event) => builder.renameField(entry.id, valueKey, index, (event.currentTarget as HTMLInputElement).value.trim())}
      >
      <PromptText
        text={field.value}
        editing={builder.editingKey === fieldKey}
        label="{label}: {field.key}"
        placeholder="value…"
        snippets={builder.doc.snippets}
        onstartedit={() => startEdit(fieldKey)}
        onendedit={() => endEdit(fieldKey)}
        ontype={(text) => builder.type(() => { const live = liveValue(); if (live?.kind === 'fields' && live.fields[index]) live.fields[index].value = text; })}
        onreplace={(text) => builder.edit(() => { const live = liveValue(); if (live?.kind === 'fields' && live.fields[index]) live.fields[index].value = text; })}
      />
      <button type="button" class="ui-btn ui-btn-row field-remove" aria-label="Remove field {field.key}" title="Remove field" onclick={() => builder.removeField(entry.id, valueKey, index)}>×</button>
    {/each}
    <button type="button" class="ui-btn ui-btn-sm ui-btn-quiet add-field" onclick={() => builder.addField(entry.id, valueKey)}>＋ Field</button>
  </div>
{:else if value?.kind === 'structured'}
  <div class="structured">
    <p class="structured-text">{flattenValue(value.data)}</p>
    <p class="structured-note">
      Lists or nested fields: shown flattened, kept as written.
      <button type="button" class="ui-link" onclick={() => void convertToText()}>Convert to text</button>
    </p>
  </div>
{/if}

<style>
  .fields { display: grid; grid-template-columns: max-content minmax(0, 1fr) auto; gap: 0 8px; align-items: start; }
  .field-key { min-width: 6ch; max-width: 18ch; field-sizing: content; height: 22px; margin-top: 2px; padding: 0 4px; border: 1px solid transparent; border-radius: var(--radius-sm); background: none; font-family: var(--font-heading); font-size: var(--text-meta); font-weight: 800; letter-spacing: 0.06em; text-transform: uppercase; color: var(--color-text-muted); }
  .field-key::placeholder { color: var(--color-error); opacity: 0.7; }
  .field-key:hover { background: var(--color-bg-surface-hover); }
  .field-key:focus { outline: 2px solid var(--color-primary-main); outline-offset: 2px; background: var(--color-bg-base); color: var(--color-text-primary); }
  .field-remove { opacity: 0; }
  .fields:hover .field-remove, .field-remove:focus-visible { opacity: 1; }
  .add-field { grid-column: 1 / -1; justify-self: start; height: 22px; padding: 0 4px; font-size: var(--text-meta); }
  .add-field:hover { color: var(--color-primary-main); }
  .structured { padding: 2px 0; }
  .structured-text { white-space: pre-wrap; font-size: var(--text-content); line-height: 1.6; color: var(--color-text-secondary); }
  .structured-note { margin-top: 2px; font-size: var(--text-meta); color: var(--color-text-muted); }
</style>
