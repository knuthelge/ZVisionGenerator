<script lang="ts">
  import { Icon } from '$lib/components/atoms';
  import { COLOR_ADD_RESET } from '$lib/state/promptEnhance';

  interface Props {
    colors: string[];
    /** Most colours the row holds; the add control is hidden at this count. */
    max: number;
    disabled?: boolean;
    onchange: (colors: string[]) => void;
  }

  let { colors, max, disabled = false, onchange }: Props = $props();

  const atMax = $derived(colors.length >= max);

  function addColor(event: Event & { currentTarget: HTMLInputElement }): void {
    const input = event.currentTarget;
    // The server drops a repeated colour, so the row never shows one either.
    const picked = input.value.toLowerCase();
    if (!colors.some((color) => color.toLowerCase() === picked)) onchange([...colors, input.value]);
    // A colour input fires no change when it is set to the value it already holds, so park it on a value nobody picks.
    input.value = COLOR_ADD_RESET;
  }

  function replaceAt(index: number, value: string): void {
    onchange(colors.map((color, i) => (i === index ? value : color)));
  }

  function removeAt(index: number): void {
    onchange(colors.filter((_, i) => i !== index));
  }
</script>

<div class="swatches" class:is-disabled={disabled} role="group" aria-label="Custom colors">
  <!-- Keyed by index: the <input> survives live edits, so the open system picker stays linked to it. -->
  {#each colors as color, index (index)}
    <span class="color">
      <label class="dot" data-color-dot={index} style:background={color}>
        <input
          type="color"
          aria-label="Color {index + 1}"
          value={color}
          {disabled}
          oninput={(event) => replaceAt(index, event.currentTarget.value)}
        />
      </label>
      <button
        type="button"
        class="remove"
        data-color-remove={index}
        aria-label="Remove color {index + 1}"
        {disabled}
        onclick={() => removeAt(index)}
      >
        <Icon name="close" size={12} />
      </button>
    </span>
  {/each}
  <!-- Hidden at the limit, so a full row never wraps a lone greyed-out + onto its own line. -->
  {#if !atMax}
    <label class="dot-add" data-color-add>
      <Icon name="plus" size={12} />
      <input type="color" aria-label="Add a color" value={COLOR_ADD_RESET} {disabled} onchange={addColor} />
    </label>
  {/if}
  {#if colors.length === 0}<span class="hint" data-color-hint>Add up to {max} colors</span>{/if}
</div>

<style>
  .swatches {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 4px;
  }
  .swatches.is-disabled {
    opacity: 0.4;
  }

  .color {
    display: inline-flex;
    align-items: center;
    height: var(--spacing-control-sm);
    padding: 0 2px;
    border-radius: 9999px;
    background: var(--color-bg-surface-hover);
  }

  /* The colour circle. The native input covers the label, so a click opens the system picker. */
  .dot {
    position: relative;
    flex: none;
    width: 18px;
    height: 18px;
    border-radius: 9999px;
    box-shadow: inset 0 0 0 1px color-mix(in srgb, var(--color-text-primary) 24%, transparent);
    cursor: pointer;
  }
  .dot input,
  .dot-add input {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    margin: 0;
    padding: 0;
    border: 0;
    opacity: 0;
    cursor: pointer;
  }
  .dot input:disabled,
  .dot-add input:disabled {
    cursor: default;
  }
  .dot:has(input:focus-visible),
  .dot-add:has(input:focus-visible) {
    outline: 2px solid var(--color-primary-main);
    outline-offset: 2px;
  }

  .hint {
    font-size: var(--text-meta);
    color: var(--color-text-muted);
  }

  .remove {
    display: inline-flex;
    flex: none;
    align-items: center;
    justify-content: center;
    width: 18px;
    height: 18px;
    padding: 0;
    border: 0;
    border-radius: 9999px;
    background: transparent;
    color: var(--color-text-muted);
    cursor: pointer;
  }
  .remove:disabled {
    cursor: default;
  }

  .dot-add {
    position: relative;
    display: inline-flex;
    flex: none;
    align-items: center;
    justify-content: center;
    width: var(--spacing-control-sm);
    height: var(--spacing-control-sm);
    border: 1px dashed var(--color-border-strong);
    border-radius: 9999px;
    color: var(--color-text-secondary);
    cursor: pointer;
  }
  .dot-add:has(input:disabled) {
    opacity: 0.4;
    cursor: default;
  }

  @media (hover: hover) {
    .color:has(.dot:hover) {
      background: var(--color-bg-control-hover);
    }
    .remove:hover:not(:disabled) {
      background: var(--color-error-surface);
      color: var(--color-error);
    }
    .dot-add:hover:not(:has(input:disabled)) {
      border-color: var(--color-primary-main);
      color: var(--color-primary-main);
    }
  }

  @media (hover: none) {
    .color:has(.dot:active) {
      background: var(--color-bg-control-hover);
    }
    .remove:active:not(:disabled) {
      color: var(--color-error);
    }
    .dot-add:active:not(:has(input:disabled)) {
      border-color: var(--color-primary-main);
    }
  }

  /* Touch: every dot, remove and add control gets a 44 x 44 hit area; the circle stays 28 px. */
  @media (pointer: coarse) {
    .swatches {
      gap: 6px;
    }
    .color {
      height: 44px;
      padding: 0;
    }
    .dot {
      width: 28px;
      height: 28px;
      margin: 8px;
    }
    .dot input {
      inset: -8px;
      width: 44px;
      height: 44px;
    }
    .remove {
      width: 44px;
      height: 44px;
    }
    .dot-add {
      width: 44px;
      height: 44px;
    }
  }
</style>
