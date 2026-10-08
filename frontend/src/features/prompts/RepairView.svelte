<script lang="ts">
  import { untrack } from 'svelte';
  import { writePromptFile } from '$lib/api/promptFiles';
  import { ApiError } from '$lib/api/client';

  interface Props {
    path: string;
    problem: string;
    rawText: string;
    /** Reload the file after a successful save or on request. */
    onreload: () => void;
  }

  let { path, problem, rawText, onreload }: Props = $props();

  // The page re-creates this view for each load, so the box starts from the text it was given.
  let text = $state(untrack(() => rawText));
  let saving = $state(false);
  let error = $state<string | null>(null);

  async function save(): Promise<void> {
    saving = true;
    error = null;
    try {
      await writePromptFile(path, text);
      onreload();
    } catch (err) {
      error = err instanceof ApiError ? err.detail : err instanceof Error ? err.message : 'Saving failed.';
    } finally {
      saving = false;
    }
  }
</script>

<section class="repair" aria-label="Repair prompt file">
  <h2 class="repair-title">This file can't be opened in the builder</h2>
  <p class="repair-problem" role="alert">{problem}</p>
  <p class="repair-hint">Fix it below, then save. The builder opens it once it is valid YAML shaped like a prompt file: sets that are lists of entries, each with a <code>prompt</code>.</p>
  <textarea class="repair-text" aria-label="Prompt file text" spellcheck="false" bind:value={text}></textarea>
  {#if error}
    <p class="repair-error" role="alert">{error}</p>
  {/if}
  <div class="repair-actions">
    <button type="button" class="panel-button panel-button-bar" onclick={onreload}>Reload from disk</button>
    <button type="button" class="panel-button panel-button-bar panel-button-primary" disabled={saving} onclick={() => void save()}>Save and open</button>
  </div>
</section>

<style>
  .repair { display: flex; flex-direction: column; gap: 10px; max-width: 900px; margin: 0 auto; padding: 24px 16px; }
  .repair-title { font-size: 16px; font-weight: 800; }
  .repair-problem { padding: 8px 12px; border-radius: var(--radius-md); background: color-mix(in srgb, var(--color-error) 10%, transparent); color: var(--color-error); font-family: var(--font-mono); font-size: 12.5px; white-space: pre-wrap; }
  .repair-hint { font-size: 12.5px; color: var(--color-text-muted); }
  .repair-text { min-height: 50vh; padding: 10px 12px; border: 1px solid var(--color-zinc-700); border-radius: var(--radius-md); background: var(--color-zinc-900); font-family: var(--font-mono); font-size: 12.5px; line-height: 1.6; tab-size: 2; resize: vertical; }
  .repair-text:focus { outline: none; border-color: var(--color-primary-main); }
  .repair-error { font-size: 12.5px; color: var(--color-error); }
  .repair-actions { display: flex; justify-content: flex-end; gap: 8px; }
</style>
