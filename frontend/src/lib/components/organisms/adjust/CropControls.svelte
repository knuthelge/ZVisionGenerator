<script lang="ts">
  import { Icon } from '$lib/components/atoms';
  import AdjustDial from './AdjustDial.svelte';
  import { aspectLabel, canSwapRatio, type Aspect, type AspectPreset, type Size } from '$lib/state/adjustGeometry';

  interface Props {
    aspect: Aspect;
    /** The size of the picture the crop gives, in pixels. */
    output: Size | null;
    /** The id of the help line, which the crop frame names as its description. */
    helpId: string;
    /**
     * `panel` is the side panel (headed sections, a size line and the help). `dock` is the tablet's docked panel: the shapes in one
     * row, then the buttons with the size at their right. `phone` is the phone's dock: the shapes in a grid and one row of three
     * buttons. `column` is the column of a phone held sideways: the shapes in a grid, then three round buttons.
     */
    variant?: 'panel' | 'dock' | 'phone' | 'column';
    onpreset: (preset: AspectPreset) => void;
    onswap: () => void;
    onrotate: (direction: 'left' | 'right') => void;
    onflip: () => void;
  }

  let { aspect, output, helpId, variant = 'panel', onpreset, onswap, onrotate, onflip }: Props = $props();

  const panel = $derived(variant === 'panel');
  const helpText = $derived(`Drag the frame${aspect.preset === 'free' ? ', its corners or its edges' : ' or its corners'}. With the frame focused, the arrow keys move it (Shift for 10×).`);

  const PRESETS: { preset: AspectPreset; label: string }[] = [
    { preset: 'free', label: 'Free' },
    { preset: 'original', label: 'Original' },
    { preset: '1:1', label: '1:1' },
    { preset: '4:5', label: '4:5' },
    { preset: '3:2', label: '3:2' },
    { preset: '16:9', label: '16:9' },
    { preset: '9:16', label: '9:16' },
  ];

  /** A swapped shape shows its own turned label ("5:4", "2:3") on its chip. */
  const labelOf = (preset: AspectPreset, label: string): string => (preset === aspect.preset ? aspectLabel(aspect) : label);
</script>

<div class="adj-crop" data-variant={variant}>
  <section>
    {#if panel}<h4 class="ui-area-label adjust-area">Shape</h4>{/if}
    <div class="adj-shapes" role="group" aria-label="Shape">
      {#each PRESETS as item (item.preset)}
        <button type="button" class="ui-chip" aria-pressed={item.preset === aspect.preset} onclick={() => onpreset(item.preset)}>{labelOf(item.preset, item.label)}</button>
      {/each}
      <button type="button" class="ui-chip swap" aria-label="Swap orientation" disabled={!canSwapRatio(aspect)} onclick={onswap}>
        <Icon name="swap-orientation" size={14} />Swap
      </button>
    </div>
  </section>
  {#if variant === 'column'}
    <div class="adj-tools">
      <AdjustDial icon="rotate-left" label="Rotate left" onclick={() => onrotate('left')} />
      <AdjustDial icon="rotate-right" label="Rotate right" onclick={() => onrotate('right')} />
      <AdjustDial icon="flip-horizontal" label="Flip" onclick={onflip} />
    </div>
  {:else}
    <section class="adj-geo-row">
      {#if panel}<h4 class="ui-area-label adjust-area">Rotate and flip</h4>{/if}
      <div class="adj-geo">
        <button type="button" class="ui-btn" onclick={() => onrotate('left')}><Icon name="rotate-left" size={variant === 'phone' ? 16 : 14} />Rotate left</button>
        <button type="button" class="ui-btn" onclick={() => onrotate('right')}><Icon name="rotate-right" size={variant === 'phone' ? 16 : 14} />Rotate right</button>
        <button type="button" class="ui-btn" onclick={onflip}><Icon name="flip-horizontal" size={variant === 'phone' ? 16 : 14} />Flip</button>
      </div>
      {#if variant === 'dock' && output}
        <span class="adjust-size" data-testid="adjust-crop-size">Crop {output.width} × {output.height} px</span>
      {/if}
    </section>
  {/if}
  {#if panel && output}
    <p class="adjust-size" data-testid="adjust-crop-size">Crop {output.width} × {output.height} px</p>
  {/if}
  <!-- The crop frame names this line as its description; the phone and the tablet dock have no room for it on screen. -->
  <p id={helpId} class={panel ? 'adjust-help' : 'sr-only'}>{helpText}</p>
</div>

<style>
  .adjust-area { margin: 0 0 8px; }
  .adj-shapes { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 6px; }
  .adj-shapes .ui-chip { justify-content: center; height: var(--spacing-control); padding: 0 6px; border-radius: var(--radius-sm); font-family: var(--font-mono); font-weight: 500; }
  .adj-shapes .ui-chip.swap { gap: 4px; font-family: var(--font-sans); font-weight: 600; }
  .adj-shapes .ui-chip:disabled { opacity: 0.4; cursor: not-allowed; }
  .adj-geo { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }
  .adj-geo .ui-btn { width: 100%; }
  .adj-crop { display: contents; }
  .adj-tools { display: flex; justify-content: space-between; gap: 2px; }

  /* The tablet's dock: the shapes in one row; the buttons in three columns with the size at their right. */
  [data-variant='dock'] .adj-shapes { grid-template-columns: repeat(8, minmax(0, 1fr)); }
  [data-variant='dock'] .adj-geo-row { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
  [data-variant='dock'] .adj-geo { flex: 1; grid-template-columns: repeat(3, minmax(0, 1fr)); max-width: 520px; }
  [data-variant='dock'] .adj-geo .ui-btn { height: var(--spacing-control); }
  /* The phone: chips and buttons at the content size. */
  [data-variant='phone'] .adj-shapes .ui-chip, [data-variant='column'] .adj-shapes .ui-chip { font-size: var(--text-content); }
  [data-variant='column'] .adj-shapes .ui-chip.swap { padding: 0 4px; }
  [data-variant='phone'] .adj-geo { grid-template-columns: repeat(3, minmax(0, 1fr)); }
  [data-variant='phone'] .adj-geo .ui-btn { height: var(--spacing-control); padding: 0 8px; font-size: var(--text-content); }
  .adjust-size { font-family: var(--font-mono); font-size: var(--text-ui); color: var(--color-text-secondary); }
  .adjust-help { font-size: var(--text-meta); line-height: 1.45; color: var(--color-text-muted); }
</style>
