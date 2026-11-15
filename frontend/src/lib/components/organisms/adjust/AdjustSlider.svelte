<script lang="ts">
  import { Icon } from '$lib/components/atoms';
  import type { IconName } from '$lib/components/atoms/Icon.svelte';
  import { SLIDER_MAX, SLIDER_MIN, SLIDER_STEP } from '$lib/state/adjustMath';
  import { centredFill, signedLabel, signedValueText } from './adjustFormat';
  import BandSwatch from './BandSwatch.svelte';
  import { bandColour, type MixTool } from './adjustTools';

  interface Props {
    label: string;
    /** The row's icon. A band's row has a swatch instead. */
    icon?: IconName;
    /** A Colour mixer band: the row shows its swatch, and the track is the scale from grey to the band's colour (no centre fill). */
    band?: MixTool['band'];
    /** An integer from -100 to 100; 0 is neutral. */
    value: number;
    oninput: (value: number) => void;
    /** Called for a double click or a double tap: the caller sets the slider back to 0. */
    onreset: () => void;
    /** The one large slider of a phone: no name or value row (the caller shows them), a taller track and a bigger thumb. */
    large?: boolean;
  }

  let { label, icon, band, value, oninput, onreset, large = false }: Props = $props();

  const uid = $props.id();
  const fill = $derived(centredFill(value));

  /** A second pointer-up this close in time and place is a double tap (a touch screen sends no dblclick for a range). */
  const DOUBLE_MS = 300;
  const DOUBLE_PX = 10;
  let lastUp: { at: number; x: number; y: number } | null = null;

  function onpointerup(event: PointerEvent): void {
    const now = Date.now();
    if (lastUp && now - lastUp.at <= DOUBLE_MS && Math.hypot(event.clientX - lastUp.x, event.clientY - lastUp.y) <= DOUBLE_PX) {
      lastUp = null;
      onreset();
      return;
    }
    lastUp = { at: now, x: event.clientX, y: event.clientY };
  }
</script>

{#snippet range()}
  <input
    id="{uid}-range"
    class="ui-range"
    class:ui-range-centred={!band}
    class:ui-range-hue={band}
    class:adj-big={large}
    type="range"
    aria-label={large ? label : undefined}
    min={SLIDER_MIN}
    max={SLIDER_MAX}
    step={SLIDER_STEP}
    {value}
    aria-valuetext={signedValueText(value)}
    style={band ? `--band: ${bandColour(band)}` : `--lo: ${fill.lo}%; --hi: ${fill.hi}%`}
    oninput={(event) => oninput(event.currentTarget.valueAsNumber)}
    ondblclick={() => { lastUp = null; onreset(); }}
    {onpointerup}
  />
{/snippet}

{#if large}
  {@render range()}
{:else}
  <div class="adj-row" data-set={value !== 0 ? '' : undefined}>
    <div class="adj-row-head">
      {#if band}
        <span class="adj-swatch-slot"><BandSwatch {band} /></span>
      {:else if icon}
        <Icon name={icon} size={14} />
      {/if}
      <label for="{uid}-range">{label}</label>
      <span class="adj-value" aria-hidden="true">{signedLabel(value)}</span>
    </div>
    {@render range()}
  </div>
{/if}

<style>
  .adj-row { display: flex; flex-direction: column; gap: 4px; }
  .adj-row-head { display: flex; align-items: center; gap: 6px; min-height: 18px; }
  /* The swatch takes the icon's 14px slot. */
  .adj-swatch-slot { display: grid; flex: none; width: 14px; place-items: center; }
  .adj-row-head :global(svg) { color: var(--color-text-muted); }
  .adj-row-head label { flex: 1; font-size: var(--text-ui); font-weight: 500; color: var(--color-text-secondary); }
  .adj-value { min-width: 4ch; font-family: var(--font-mono); font-size: var(--text-ui); text-align: right; color: var(--color-text-muted); }
  .adj-row[data-set] .adj-value { color: var(--color-text-primary); }
  .adj-row[data-set] .adj-row-head :global(svg) { color: var(--color-primary-main); }
  .adj-row input, .adj-big { width: 100%; }
  .adj-big { height: 44px; }
  .adj-big::-webkit-slider-runnable-track { height: 6px; }
  .adj-big::-webkit-slider-thumb { width: 28px; height: 28px; margin-top: -11px; box-shadow: 0 0 0 3px var(--color-bg-surface); }
  .adj-big::-moz-range-track { height: 6px; }
  .adj-big::-moz-range-thumb { width: 28px; height: 28px; box-shadow: 0 0 0 3px var(--color-bg-surface); }
  .adj-big.ui-range-centred, .adj-big.ui-range-hue { background-size: 2px 16px; }
</style>
