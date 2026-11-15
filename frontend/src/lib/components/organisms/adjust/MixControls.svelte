<script lang="ts">
  import type { LightSettings } from '$lib/state/adjustMath';
  import AdjustSlider from './AdjustSlider.svelte';
  import { MIX_TOOLS } from './adjustTools';

  interface Props {
    light: LightSettings;
    /** Whether the pointer is a finger: the help line says tap instead of click. */
    coarse?: boolean;
    /** The sliders in two columns of four with no help line, for the docked panel of a portrait tablet. */
    dock?: boolean;
    onchange: (key: keyof LightSettings, value: number) => void;
  }

  let { light, coarse = false, dock = false, onchange }: Props = $props();
</script>

{#snippet rows()}
  {#each MIX_TOOLS as row (row.key)}
    <AdjustSlider label={row.label} band={row.band} value={light[row.key]} oninput={(value) => onchange(row.key, value)} onreset={() => onchange(row.key, 0)} />
  {/each}
{/snippet}

{#if dock}
  <div class="dock-sliders">{@render rows()}</div>
{:else}
  {@render rows()}
  <p class="adjust-help">Saturation per colour. {coarse ? 'Double-tap' : 'Double-click'} a slider to set it back to 0.</p>
{/if}

<style>
  .adjust-help { font-size: var(--text-meta); line-height: 1.45; color: var(--color-text-muted); }
  /* Two columns that fill down: Red to Green, then Aqua to Magenta. */
  .dock-sliders { display: grid; grid-auto-flow: column; grid-template-rows: repeat(4, auto); grid-template-columns: 1fr 1fr; gap: 12px 28px; }
</style>
