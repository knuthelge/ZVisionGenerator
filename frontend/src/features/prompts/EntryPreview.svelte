<script lang="ts">
  import type { PromptBuilder } from './builder.svelte';
  import { selectedEntry } from './builder.svelte';
  import { enhanceSettingsFor } from './document';

  interface Props {
    builder: PromptBuilder;
  }

  let { builder }: Props = $props();

  const selected = $derived(selectedEntry(builder));
  const preview = $derived(selected ? builder.preview?.entries[selected.entry.id] ?? null : null);
  const problems = $derived(selected ? builder.problemsFor(selected.entry.id) : []);
  const rolled = $derived(
    selected && preview && builder.rolled?.entryId === selected.entry.id && builder.rolled.base === preview.prompt ? builder.rolled.prompt : null
  );
  const choiceParts = $derived(preview ? splitChoices(preview.prompt) : []);
  const enhanceRows = $derived.by(() => {
    const enhance = selected?.entry.enhance ?? null;
    if (enhance === null || enhance === true) return [];
    const settings = enhanceSettingsFor(enhance, builder.axes, builder.defaults);
    return builder.axes
      .map((axis) => {
        const value = settings[axis.key];
        const slugs = Array.isArray(value) ? value : [value];
        const labels = slugs.map((slug) => axis.options.find((option) => option.slug === slug)?.label ?? slug);
        return { label: axis.label, value: labels.join(', ') };
      })
      .filter((row) => row.value);
  });

  /** Split text into plain parts and `{a|b}` choices, for highlighting. */
  function splitChoices(text: string): { text: string; choice: boolean }[] {
    const parts: { text: string; choice: boolean }[] = [];
    let last = 0;
    for (const match of text.matchAll(/\{[^{}]+\}/g)) {
      const index = match.index ?? 0;
      if (index > last) parts.push({ text: text.slice(last, index), choice: false });
      parts.push({ text: match[0], choice: true });
      last = index + match[0].length;
    }
    if (last < text.length) parts.push({ text: text.slice(last), choice: false });
    return parts;
  }
</script>

<div class="preview-head"><h2 class="panel-label">Preview</h2></div>
{#if !selected}
  <p class="preview-empty">Select a prompt to see what the model receives.</p>
{:else}
  <div class="preview">
    <p class="preview-id">
      <span class="preview-name">{selected.setName || '(unnamed)'} #{selected.index + 1}</span>
      <span class="preview-code">{selected.entry.active ? `${selected.setName}:${selected.index}` : 'inactive, no prompt id'}</span>
    </p>

    <section class="box" aria-label="Prompt">
      <div class="box-head">
        <span class="panel-label" title="What the model receives: snippets resolved, fields flattened">Prompt</span>
        <span class="box-tools">
          {#if rolled}
            <button type="button" class="panel-button panel-button-quiet small" onclick={() => builder.clearRoll()}>Show choices</button>
          {/if}
          <button type="button" class="panel-button small" title="Pick one option of every choice, as a run would" disabled={!preview} onclick={() => selected && void builder.roll(selected.entry.id)}>🎲 Roll</button>
        </span>
      </div>
      <div class="box-body" aria-live="polite">
        {#if rolled}
          {rolled}
        {:else if preview}
          {#each choiceParts as part, index (index)}{#if part.choice}<mark>{part.text}</mark>{:else}{part.text}{/if}{/each}
        {:else}
          <span class="muted">No preview: fix the problems below.</span>
        {/if}
      </div>
    </section>

    {#if preview?.negative}
      <section class="box" aria-label="Negative prompt">
        <div class="box-head"><span class="panel-label">Negative</span></div>
        <div class="box-body">{preview.negative}</div>
      </section>
    {/if}

    <section class="box" aria-label="Settings">
      <div class="box-head"><span class="panel-label">Settings</span></div>
      <dl class="box-body settings">
        <dt>Runs</dt>
        <dd>{selected.entry.active ? 'Yes, active' : 'No, inactive'}</dd>
        <dt>Enhance</dt>
        <dd class:on={selected.entry.enhance !== null}>{selected.entry.enhance === null ? 'Off' : selected.entry.enhance === true ? 'Default options' : 'Custom'}</dd>
        {#each enhanceRows as row (row.label)}
          <dt>{row.label}</dt>
          <dd class="on">{row.value}</dd>
        {/each}
      </dl>
    </section>

    <section class="box" aria-label="Problems">
      <div class="box-head"><span class="panel-label">Problems</span></div>
      <div class="box-body">
        {#if problems.length === 0}
          <span class="muted">None.</span>
        {:else}
          {#each problems as problem, index (index)}
            <p class="problem" data-severity={problem.severity}>{problem.severity === 'error' ? '✕' : '⚠'} {problem.message}</p>
          {/each}
        {/if}
      </div>
    </section>
  </div>
{/if}

<style>
  .preview-head { display: flex; align-items: center; min-height: 26px; margin-bottom: 4px; }
  .preview { display: flex; flex-direction: column; gap: 8px; }
  .preview-empty { padding: 16px 0; font-size: 12px; color: var(--color-text-muted); }
  .preview-id { display: flex; align-items: baseline; gap: 6px; }
  .preview-name { font-family: var(--font-heading); font-size: 13.5px; font-weight: 800; }
  .preview-code { font-size: 11.5px; color: var(--color-text-muted); }
  .box { overflow: hidden; border: 1px solid var(--color-border-strong); border-radius: var(--radius-md); background: var(--color-bg-base); }
  .box-head { display: flex; align-items: center; justify-content: space-between; gap: 8px; height: 32px; padding: 0 5px 0 10px; border-bottom: 1px solid var(--color-border-subtle); }
  .box-head .panel-label { white-space: nowrap; }
  .box-tools { display: flex; gap: 4px; }
  .small { height: 22px; font-size: 11.5px; }
  .box-body { padding: 8px 10px; white-space: pre-wrap; overflow-wrap: anywhere; font-size: 12.5px; line-height: 1.6; }
  .box-body mark { padding: 0 2px; border-radius: 3px; background: color-mix(in srgb, var(--color-accent-amber) 18%, transparent); color: var(--color-accent-amber); }
  .settings { display: grid; grid-template-columns: max-content minmax(0, 1fr); gap: 4px 12px; white-space: normal; }
  .settings dt { color: var(--color-text-muted); }
  .settings dd { color: var(--color-text-secondary); }
  .settings dd.on { color: var(--color-primary-main); }
  .muted { color: var(--color-text-muted); }
  .problem { font-size: 12px; white-space: normal; }
  .problem + .problem { margin-top: 2px; }
  .problem[data-severity='error'] { color: var(--color-error); }
  .problem[data-severity='warning'] { color: var(--color-warning); }
</style>
