<script lang="ts" module>
  export interface EnhanceChoiceOption<V extends string> {
    value: V;
    label: string;
    /** What the option does; shown as the chip's tooltip. */
    help: string;
    /** Why the option cannot be chosen here; the chip is disabled and the reason becomes its tooltip. */
    unavailable?: string | null;
  }
</script>

<script lang="ts" generics="T extends string">
  interface Props {
    /** Short inline label, e.g. "Enhancer". */
    label: string;
    options: EnhanceChoiceOption<T>[];
    /** The choice the run will use. */
    value: T;
    disabled?: boolean;
    /** Prefix of the element ids, as in `EnhanceOptions`. */
    idPrefix: string;
    onchange: (value: T) => void;
  }

  let { label, options, value, disabled = false, idPrefix, onchange }: Props = $props();
</script>

<div class="choice" role="radiogroup" aria-labelledby={`${idPrefix}-label`}>
  <span id={`${idPrefix}-label`} class="ui-label choice-label">{label}</span>
  {#each options as option (option.value)}
    <!-- The tooltip sits on a wrapper: browsers do not show one on a disabled button. -->
    <span class="chip-wrap" title={option.unavailable ?? option.help}>
      <button
        type="button"
        class="ui-chip enhance-chip"
        class:enhance-chip-on={value === option.value}
        role="radio"
        aria-checked={value === option.value}
        aria-describedby={option.unavailable ? `${idPrefix}-${option.value}-reason` : undefined}
        data-choice={option.value}
        disabled={disabled || Boolean(option.unavailable)}
        onclick={() => onchange(option.value)}
      >
        {option.label}
      </button>
      {#if option.unavailable}<span id={`${idPrefix}-${option.value}-reason`} class="sr-only">{option.unavailable}</span>{/if}
    </span>
  {/each}
</div>

<style>
  .choice { display: inline-flex; align-items: center; gap: 0.375rem; }
  .choice-label { margin: 0; }
  .chip-wrap { display: inline-flex; }
  .enhance-chip { cursor: pointer; transition: background-color 0.12s ease, color 0.12s ease; }
  .enhance-chip-on { background: color-mix(in srgb, var(--color-primary-main) 16%, transparent); color: var(--color-primary-main); }
  .enhance-chip:disabled { cursor: not-allowed; opacity: 0.4; }
</style>
