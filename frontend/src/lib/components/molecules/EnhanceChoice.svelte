<script lang="ts" module>
  export interface EnhanceChoiceOption<V extends string> {
    value: V;
    label: string;
    /** What the option does; shown as the segment's tooltip. */
    help: string;
    /** Why the option cannot be chosen here; the segment is unavailable and the reason becomes its tooltip. */
    unavailable?: string | null;
  }
</script>

<script lang="ts" generics="T extends string">
  interface Props {
    /** Short label, e.g. "Enhancer"; the group's accessible name. */
    label: string;
    options: EnhanceChoiceOption<T>[];
    /** The choice the run will use. */
    value: T;
    disabled?: boolean;
    /** Prefix of the element ids, as in `EnhanceOptions`. */
    idPrefix: string;
    /** Show the label beside the control; off where a row label already names it. */
    showLabel?: boolean;
    /** Stretch the segments across the available width. */
    full?: boolean;
    onchange: (value: T) => void;
  }

  let { label, options, value, disabled = false, idPrefix, showLabel = true, full = false, onchange }: Props = $props();
</script>

<div class="choice" class:full>
  <span id={`${idPrefix}-label`} class="ui-label choice-label" class:sr-only={!showLabel}>{label}</span>
  <div class="ui-segmented choice-segments" role="radiogroup" aria-labelledby={`${idPrefix}-label`}>
    {#each options as option (option.value)}
      <!-- An unavailable segment stays focusable (aria-disabled) so its tooltip can say why: browsers show none on disabled buttons. -->
      <button
        type="button"
        role="radio"
        aria-checked={value === option.value}
        aria-disabled={option.unavailable ? 'true' : undefined}
        aria-describedby={option.unavailable ? `${idPrefix}-${option.value}-reason` : undefined}
        title={option.unavailable ?? option.help}
        data-choice={option.value}
        {disabled}
        onclick={() => { if (!option.unavailable && option.value !== value) onchange(option.value); }}
      >{option.label}</button>
    {/each}
  </div>
  {#each options as option (option.value)}
    {#if option.unavailable}<span id={`${idPrefix}-${option.value}-reason`} class="sr-only">{option.unavailable}</span>{/if}
  {/each}
</div>

<style>
  .choice { display: inline-flex; align-items: center; gap: 8px; min-width: 0; }
  .choice.full { display: flex; flex: 1; }
  .choice-label { margin: 0; white-space: nowrap; }
  .choice.full .choice-segments { display: flex; flex: 1; }
  .choice.full .choice-segments > button { flex: 1; }
  .choice-segments > button[aria-disabled='true'] { opacity: 0.4; cursor: not-allowed; }
</style>
