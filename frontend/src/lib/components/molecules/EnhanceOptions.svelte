<script lang="ts">
  import type { Snippet } from 'svelte';
  import type { EnhanceAxis, EnhanceAxisKey, EnhanceSettings, WorkflowMode } from '$lib/types';
  import EnhancePicker from './EnhancePicker.svelte';
  import InspectorRow from './InspectorRow.svelte';

  interface Props {
    axes: EnhanceAxis[];
    settings: EnhanceSettings;
    mode: WorkflowMode;
    /** The video has an audio track; audio-only axes (Sound) are hidden without it. */
    audio?: boolean;
    disabled?: boolean;
    idPrefix?: string;
    /** Rows shown first in the Output section (Rewrite, Write as), above Length. */
    output?: Snippet;
    onchange: (settings: EnhanceSettings) => void;
  }

  let { axes, settings, mode, audio = false, disabled = false, idPrefix = 'enhance', output, onchange }: Props = $props();

  /** Matches the server's limit on a direction. */
  const NOTE_MAX_CHARS = 300;

  const video = $derived(mode === 'video');

  function axisFor(key: EnhanceAxisKey): EnhanceAxis | undefined {
    const axis = axes.find((candidate) => candidate.key === key);
    if (!axis || (axis.video_only && !video) || (axis.audio_only && !audio)) return undefined;
    return axis;
  }

  function selectedList(axis: EnhanceAxis): string[] {
    const value = settings[axis.key];
    return Array.isArray(value) ? value : [];
  }

  function allSelected(axis: EnhanceAxis): boolean {
    const value = selectedList(axis);
    return axis.options.every((option) => value.includes(option.slug));
  }

  // Pick-any axes switch every option on, or all off once they are all on.
  function toggleAll(axis: EnhanceAxis): void {
    onchange({ ...settings, [axis.key]: allSelected(axis) ? [] : axis.options.map((option) => option.slug) });
  }

  function toggle(axis: EnhanceAxis, slug: string): void {
    const current = selectedList(axis);
    const next = current.includes(slug) ? current.filter((item) => item !== slug) : [...current, slug];
    // Keep the matrix order so the spec and backend instructions stay stable.
    const order = axis.options.map((option) => option.slug);
    onchange({ ...settings, [axis.key]: next.sort((a, b) => order.indexOf(a) - order.indexOf(b)) });
  }
</script>

{#snippet picker(key: EnhanceAxisKey)}
  {@const axis = axisFor(key)}
  {#if axis}
    <InspectorRow label={axis.label} forId={`${idPrefix}-${key}`}>
      <EnhancePicker {axis} value={String(settings[key])} id={`${idPrefix}-${key}`} {disabled} onchange={(slug) => onchange({ ...settings, [key]: slug })} />
    </InspectorRow>
  {/if}
{/snippet}

{#snippet chips(key: EnhanceAxisKey, title: string)}
  {@const axis = axisFor(key)}
  {#if axis}
    <fieldset class="enhance-chips" {disabled}>
      <legend id={`${idPrefix}-${key}-label`} class="enhance-subtitle">
        <span class="ui-area-label">{title} · pick any</span>
        <button
          type="button"
          class="toggle-all ui-btn ui-btn-sm ui-btn-quiet"
          data-toggle-all={axis.key}
          data-all-selected={allSelected(axis)}
          aria-label="{allSelected(axis) ? 'Clear all' : 'Select all'} {axis.label.toLowerCase()}"
          {disabled}
          onclick={() => toggleAll(axis)}
        >{allSelected(axis) ? 'None' : 'All'}</button>
      </legend>
      <div class="chip-list" role="group" aria-labelledby={`${idPrefix}-${key}-label`}>
        {#each axis.options as option (option.slug)}
          <button type="button" class="ui-chip enhance-chip" aria-pressed={selectedList(axis).includes(option.slug)} title={option.description || undefined} {disabled} onclick={() => toggle(axis, option.slug)}>
            {option.label}
          </button>
        {/each}
      </div>
    </fieldset>
  {/if}
{/snippet}

<div class="enhance-options">
  <section class="enhance-section" aria-label="Look">
    <h3 class="enhance-title ui-area-label">Look</h3>
    {@render picker('style')}
    {@render picker('mood')}
    {@render picker('theme')}
  </section>

  {#if axisFor('details')}
    <section class="enhance-section">{@render chips('details', 'Focus on')}</section>
  {/if}

  {#if video && (axisFor('camera') || axisFor('pace') || axisFor('motion'))}
    <section class="enhance-section" aria-label="Motion">
      <h3 class="enhance-title ui-area-label">Motion</h3>
      {@render picker('camera')}
      {@render picker('pace')}
      {@render chips('motion', 'Describe')}
    </section>
  {/if}

  {#if axisFor('sound')}
    <section class="enhance-section">{@render chips('sound', 'Sound')}</section>
  {/if}

  <section class="enhance-section" aria-label="Output">
    <h3 class="enhance-title ui-area-label">Output</h3>
    {@render output?.()}
    {@render picker('length')}
  </section>

  <input
    id={`${idPrefix}-note`}
    type="text"
    class="ui-field enhance-note"
    maxlength={NOTE_MAX_CHARS}
    placeholder="Direction (optional): focus on the dog, make her look surprised"
    aria-label="Direction for the enhancer"
    value={settings.note}
    {disabled}
    oninput={(e) => onchange({ ...settings, note: (e.currentTarget as HTMLInputElement).value })}
  />
</div>

<style>
  .enhance-options { display: grid; gap: 0; min-width: 0; }
  .enhance-section { display: grid; gap: 2px; padding: 8px 0; }
  .enhance-section + .enhance-section { border-top: 1px solid var(--color-border-subtle); }
  .enhance-title { margin: 0 0 4px 12px; }
  .enhance-chips { display: grid; gap: 6px; min-width: 0; margin: 0; padding: 0 12px; border: 0; }
  .enhance-subtitle { display: flex; width: 100%; align-items: center; gap: 8px; padding: 0; }
  .toggle-all { margin-left: auto; }
  .chip-list { display: flex; flex-wrap: wrap; gap: 4px; }
  .enhance-chip { cursor: pointer; }
  .enhance-chip:disabled { cursor: not-allowed; opacity: 0.4; }
  .enhance-note { margin-top: 8px; }
</style>
