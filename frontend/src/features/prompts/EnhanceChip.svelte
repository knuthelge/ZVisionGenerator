<script lang="ts">
  import { popover } from '$lib/actions/popover';
  import { EnhanceOptions } from '$lib/components/molecules';
  import type { EnhanceAxis, EnhanceSettings, EntryEnhance } from '$lib/types';
  import { enhanceSettingsFor, enhanceShortLabel, enhanceSummary, isNoOpEnhance } from './document';

  interface Props {
    enhance: EntryEnhance;
    axes: EnhanceAxis[];
    defaults: EnhanceSettings;
    idPrefix: string;
    onchange: (enhance: EntryEnhance) => void;
  }

  let { enhance, axes, defaults, idPrefix, onchange }: Props = $props();

  let open = $state(false);
  let anchor = $state<HTMLButtonElement | null>(null);
  let useDefaults = $state(false);
  let settings = $state<EnhanceSettings>({ style: 'keep', mood: 'keep', details: [], length: 'same', motion: [] });

  const noop = $derived(!useDefaults && isNoOpEnhance(settings));

  function toggle(): void {
    if (open) {
      open = false;
      return;
    }
    useDefaults = enhance === true;
    settings = enhanceSettingsFor(enhance, axes, defaults);
    open = true;
  }

  function apply(): void {
    open = false;
    onchange(useDefaults ? true : $state.snapshot(settings));
  }

  function turnOff(): void {
    open = false;
    onchange(null);
  }
</script>

<button
  bind:this={anchor}
  type="button"
  class="ui-btn ui-btn-sm enhance-button"
  class:ui-btn-on={enhance !== null}
  aria-haspopup="dialog"
  aria-expanded={open}
  title={enhance === null ? 'Enhance this prompt on every run' : `Enhanced: ${enhanceSummary(enhance, axes)}. Click to change.`}
  onclick={toggle}
><span aria-hidden="true">✨</span><span class="enhance-text">{enhance === null ? 'Enhance' : enhanceShortLabel(enhance, axes)}</span></button>

{#if open}
  <div use:popover={{ anchor, align: 'end', onclose: () => { open = false; } }} class="ui-overlay enhance-popover" role="dialog" aria-label="Enhance this prompt">
    <p class="ui-area-label enhance-title">Enhance this prompt</p>
    <div class="ui-segmented enhance-mode" role="radiogroup" aria-label="Enhance options">
      <button type="button" role="radio" aria-checked={useDefaults} onclick={() => { useDefaults = true; }}>Default options</button>
      <button type="button" role="radio" aria-checked={!useDefaults} onclick={() => { useDefaults = false; }}>Choose</button>
    </div>
    {#if useDefaults}
      <p class="enhance-hint">Uses the enhancer's default options (<code>enhance: true</code>).</p>
    {:else}
      <EnhanceOptions {axes} {settings} mode="video" {idPrefix} onchange={(next) => { settings = next; }} />
      <p class="enhance-hint">Motion only applies to video; image runs ignore it.</p>
    {/if}
    {#if noop}
      <p class="enhance-warning" role="status">Nothing to enhance: pick a style, a mood, a detail or a length.</p>
    {/if}
    <div class="enhance-actions">
      {#if enhance !== null}
        <button type="button" class="ui-btn" onclick={turnOff}>Turn off</button>
      {/if}
      <button type="button" class="ui-btn ui-btn-primary" disabled={noop} onclick={apply}>{enhance === null ? 'Enhance' : 'Apply'}</button>
    </div>
  </div>
{/if}

<style>
  .enhance-button { max-width: 210px; justify-content: flex-start; gap: 5px; padding: 0 9px; }
  .enhance-text { min-width: 0; overflow: hidden; text-overflow: ellipsis; }
  .enhance-popover { z-index: 60; width: 360px; max-height: 80vh; overflow-y: auto; padding: 10px 12px 12px; }
  .enhance-title { margin-bottom: 8px; }
  .enhance-mode { margin-bottom: 10px; }
  .enhance-hint { margin-top: 8px; font-size: var(--text-meta); color: var(--color-text-muted); }
  .enhance-warning { margin-top: 8px; font-size: var(--text-meta); color: var(--color-warning); }
  .enhance-actions { display: flex; justify-content: flex-end; gap: 6px; margin-top: 12px; }
</style>
