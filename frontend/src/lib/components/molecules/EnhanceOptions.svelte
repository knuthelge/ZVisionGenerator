<script lang="ts" module>
  /** A Sound's sections of options and the axes each lists, in order; an axis the matrix does not offer is skipped. */
  const SOUND_SECTIONS: { title: string; keys: string[] }[] = [
    { title: 'Place', keys: ['space', 'distance'] },
    { title: 'Motion', keys: ['over_time', 'movement'] },
  ];
  /** A Music style's pickers above and below its two instrument lists (Instruments, Without). */
  const MUSIC_SOUND: { title: string; keys: string[] } = { title: 'Sound', keys: ['genre', 'music_mood', 'production'] };
  const MUSIC_VOICE: { title: string; keys: string[] } = { title: 'Voice', keys: ['singer'] };
</script>

<script lang="ts" generics="S extends EnhanceSettings | AudioEnhanceSettings">
  import type { Snippet } from 'svelte';
  import { Icon } from '$lib/components/atoms';
  import { CUSTOM_PALETTE } from '$lib/state/promptEnhance';
  import type { AudioEnhanceAxis, AudioEnhanceSettings, EnhanceAxis, EnhanceSettings, WorkflowMode } from '$lib/types';
  import ColorSwatches from './ColorSwatches.svelte';
  import EnhancePicker from './EnhancePicker.svelte';
  import InspectorRow from './InspectorRow.svelte';

  interface Props {
    /** The axes to offer. Audio passes the axes of its task only. */
    axes: (EnhanceAxis | AudioEnhanceAxis)[];
    settings: S;
    /** `audio` lays out the audio task's sections (Sound's Place and Motion) in place of the image and video ones. */
    mode: WorkflowMode;
    /** The video has an audio track; audio-only axes (Sound) are hidden without it. */
    audio?: boolean;
    disabled?: boolean;
    idPrefix?: string;
    /** Rows shown first in the Output section (Rewrite, Write as), above Length. */
    output?: Snippet;
    onchange: (settings: S) => void;
  }

  let { axes, settings, mode, audio = false, disabled = false, idPrefix = 'enhance', output, onchange }: Props = $props();

  /** Matches the server's limit on a direction. */
  const NOTE_MAX_CHARS = 300;
  /** Matches the server's limit on Custom colours. */
  const COLORS_MAX = 5;

  const video = $derived(mode === 'video');
  const audioMode = $derived(mode === 'audio');
  const NOTE_PLACEHOLDER = 'Direction (optional): focus on the dog, make her look surprised';
  const NOTE_PLACEHOLDER_AUDIO_SOUND = 'Direction (optional): keep the thunder far off';
  const NOTE_PLACEHOLDER_AUDIO_MUSIC = 'Direction (optional): keep the guitar fingerpicked';
  /** The image and video options, for the rows that only they have (Palette). */
  const image = $derived(settings as EnhanceSettings);
  /** The settings as a lookup by axis key; image, video and audio settings are all keyed by their axes. */
  const record = $derived(settings as unknown as Record<string, unknown>);

  function axisFor(key: string): EnhanceAxis | undefined {
    const axis = axes.find((candidate) => candidate.key === key) as EnhanceAxis | undefined;
    if (!axis || (axis.video_only && !video) || (axis.audio_only && !audio)) return undefined;
    return axis;
  }

  /** Report *settings* with one axis changed. */
  function change(key: string, value: unknown): void {
    onchange({ ...settings, [key]: value });
  }

  function selectedList(axis: EnhanceAxis): string[] {
    const value = record[axis.key];
    return Array.isArray(value) ? value : [];
  }

  /** Music: the instruments left out, and the ones asked for, which Without cannot also list. */
  const leftOut = $derived(Array.isArray(record.without) ? (record.without as string[]) : []);
  const featured = $derived(Array.isArray(record.instruments) ? (record.instruments as string[]) : []);

  function allSelected(axis: EnhanceAxis): boolean {
    const value = selectedList(axis);
    return axis.options.every((option) => value.includes(option.slug));
  }

  // Pick-any axes switch every option on, or all off once they are all on.
  function toggleAll(axis: EnhanceAxis): void {
    changeList(axis.key, allSelected(axis) ? [] : axis.options.map((option) => option.slug));
  }

  /** Report *settings* with the picks of *key* changed; an instrument picked is no longer left out. */
  function changeList(key: string, next: string[]): void {
    const patch: Record<string, unknown> = { [key]: next };
    if (audioMode && key === 'instruments') patch.without = leftOut.filter((slug) => !next.includes(slug));
    onchange({ ...settings, ...patch });
  }

  function toggle(axis: EnhanceAxis, slug: string): void {
    const current = selectedList(axis);
    const next = current.includes(slug) ? current.filter((item) => item !== slug) : [...current, slug];
    // Keep the matrix order so the spec and backend instructions stay stable.
    const order = axis.options.map((option) => option.slug);
    changeList(axis.key, next.sort((a, b) => order.indexOf(a) - order.indexOf(b)));
  }

  // A palette other than Custom owns no colours, so leaving Custom (or clearing back to Keep) drops them.
  function choosePalette(slug: string): void {
    onchange({ ...settings, palette: slug, colors: slug === CUSTOM_PALETTE ? image.colors : [] });
  }
</script>

{#snippet picker(key: string, choose?: (slug: string) => void, swatches?: string[])}
  {@const axis = axisFor(key)}
  {#if axis}
    <InspectorRow label={axis.label} forId={`${idPrefix}-${key}`}>
      <EnhancePicker {axis} value={String(record[key])} id={`${idPrefix}-${key}`} {disabled} {swatches} onchange={choose ?? ((slug) => change(key, slug))} />
    </InspectorRow>
  {/if}
{/snippet}

{#snippet chips(key: string, title: string, minus = false)}
  {@const axis = axisFor(key)}
  {#if axis}
    <fieldset class="enhance-chips" {disabled}>
      <legend id={`${idPrefix}-${key}-label`} class="enhance-subtitle">
        <span class="ui-area-label">{title} · pick any</span>
        <!-- Leaving out everything is no choice worth a button. -->
        {#if !minus}
          <button
            type="button"
            class="toggle-all ui-btn ui-btn-sm ui-btn-quiet"
            data-toggle-all={axis.key}
            data-all-selected={allSelected(axis)}
            aria-label="{allSelected(axis) ? 'Clear all' : 'Select all'} {axis.label.toLowerCase()}"
            {disabled}
            onclick={() => toggleAll(axis)}
          >{allSelected(axis) ? 'None' : 'All'}</button>
        {/if}
      </legend>
      <div class="chip-list" role="group" aria-labelledby={`${idPrefix}-${key}-label`}>
        {#each axis.options as option (option.slug)}
          {@const on = selectedList(axis).includes(option.slug)}
          <!-- An instrument asked for cannot also be left out. -->
          {@const taken = minus && featured.includes(option.slug)}
          <button
            type="button"
            class="ui-chip enhance-chip"
            aria-pressed={on}
            aria-label={minus ? `Without ${option.label.toLowerCase()}` : undefined}
            title={taken ? `${option.label} is asked for above` : option.description || undefined}
            disabled={disabled || taken}
            onclick={() => toggle(axis, option.slug)}
          >
            {#if minus && on}<Icon name="minus" size={12} />{/if}{option.label}
          </button>
        {/each}
      </div>
    </fieldset>
  {/if}
{/snippet}

{#snippet audioSection(section: { title: string; keys: string[] })}
  {#if section.keys.some((key) => axisFor(key))}
    <section class="enhance-section" aria-label={section.title}>
      <h3 class="enhance-title ui-area-label">{section.title}</h3>
      {#each section.keys as key (key)}{@render picker(key)}{/each}
    </section>
  {/if}
{/snippet}

<div class="enhance-options">
  {#if audioMode}
    {#each SOUND_SECTIONS as section (section.title)}{@render audioSection(section)}{/each}
    {@render audioSection(MUSIC_SOUND)}
    {#if axisFor('instruments')}<section class="enhance-section">{@render chips('instruments', 'Instruments')}</section>{/if}
    {#if axisFor('without')}<section class="enhance-section">{@render chips('without', 'Without', true)}</section>{/if}
    {@render audioSection(MUSIC_VOICE)}
  {:else}
  <section class="enhance-section" aria-label="Look">
    <h3 class="enhance-title ui-area-label">Look</h3>
    {@render picker('style')}
    {@render picker('mood')}
    {@render picker('theme')}
  </section>

  {#if axisFor('shot') || axisFor('angle')}
    <section class="enhance-section" aria-label="Framing">
      <h3 class="enhance-title ui-area-label">Framing</h3>
      {@render picker('shot')}
      {@render picker('angle')}
    </section>
  {/if}

  {#if axisFor('palette')}
    <section class="enhance-section" aria-label="Color">
      <h3 class="enhance-title ui-area-label">Color</h3>
      {@render picker('palette', choosePalette, image.palette === CUSTOM_PALETTE ? image.colors : undefined)}
      {#if image.palette === CUSTOM_PALETTE}
        <div class="enhance-colors">
          <ColorSwatches colors={image.colors} max={COLORS_MAX} {disabled} onchange={(colors) => change('colors', colors)} />
        </div>
      {/if}
    </section>
  {/if}

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
  {/if}

  <input
    id={`${idPrefix}-note`}
    type="text"
    class="ui-field enhance-note"
    maxlength={NOTE_MAX_CHARS}
    placeholder={audioMode ? (axisFor('genre') ? NOTE_PLACEHOLDER_AUDIO_MUSIC : NOTE_PLACEHOLDER_AUDIO_SOUND) : NOTE_PLACEHOLDER}
    aria-label="Direction for the enhancer"
    value={settings.note}
    {disabled}
    oninput={(e) => change('note', (e.currentTarget as HTMLInputElement).value)}
  />
</div>

<style>
  .enhance-options { display: grid; gap: 0; min-width: 0; }
  .enhance-section { display: grid; gap: 2px; padding: 8px 0; }
  .enhance-section + .enhance-section { border-top: 1px solid var(--color-border-subtle); }
  .enhance-title { margin: 0 0 4px 12px; }
  .enhance-colors { padding: 2px 8px 4px 116px; }
  @media (pointer: coarse) { .enhance-colors { padding: 4px 12px; } }
  .enhance-chips { display: grid; gap: 6px; min-width: 0; margin: 0; padding: 0 12px; border: 0; }
  .enhance-subtitle { display: flex; width: 100%; align-items: center; gap: 8px; padding: 0; }
  .toggle-all { margin-left: auto; }
  .chip-list { display: flex; flex-wrap: wrap; gap: 4px; }
  .enhance-chip { cursor: pointer; }
  .enhance-chip:disabled { cursor: not-allowed; opacity: 0.4; }
  .enhance-note { margin-top: 8px; }
</style>
