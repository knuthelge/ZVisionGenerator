<script lang="ts">
  import type { LightSettings } from '$lib/state/adjustMath';
  import AdjustSlider from './AdjustSlider.svelte';
  import { LIGHT_TOOLS, TONE_COUNT } from './adjustTools';

  interface Props {
    light: LightSettings;
    /** Whether the pointer is a finger: the help line says tap instead of click. */
    coarse?: boolean;
    /** The sliders in two columns with no divider or help line, for the docked panel of a portrait tablet. */
    dock?: boolean;
    onchange: (key: keyof LightSettings, value: number) => void;
  }

  let { light, coarse = false, dock = false, onchange }: Props = $props();

  const TONE = LIGHT_TOOLS.slice(0, TONE_COUNT);
  const COLOUR = LIGHT_TOOLS.slice(TONE_COUNT);
</script>

{#snippet slider(row: (typeof LIGHT_TOOLS)[number])}
  <AdjustSlider label={row.label} icon={row.icon} value={light[row.key]} oninput={(value) => onchange(row.key, value)} onreset={() => onchange(row.key, 0)} />
{/snippet}

{#if dock}
  <div class="dock-sliders">
    {#each LIGHT_TOOLS as row (row.key)}
      {@render slider(row)}
    {/each}
  </div>
{:else}
  {#each TONE as row (row.key)}
    {@render slider(row)}
  {/each}
  <div class="adjust-sep"></div>
  {#each COLOUR as row (row.key)}
    {@render slider(row)}
  {/each}
  <p class="adjust-help">{coarse ? 'Double-tap' : 'Double-click'} a slider to set it back to 0.</p>
{/if}

<style>
  .adjust-sep { height: 1px; background: var(--color-border-subtle); }
  .adjust-help { font-size: var(--text-meta); line-height: 1.45; color: var(--color-text-muted); }
  /* Two columns that fill down: the four tone sliders, then the three colour sliders. */
  .dock-sliders { display: grid; grid-auto-flow: column; grid-template-rows: repeat(4, auto); grid-template-columns: 1fr 1fr; gap: 12px 28px; }
</style>
