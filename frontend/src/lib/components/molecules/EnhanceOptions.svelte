<script lang="ts">
  import type { EnhanceAxis, EnhanceSettings, WorkflowMode } from '$lib/types';

  interface Props {
    axes: EnhanceAxis[];
    settings: EnhanceSettings;
    mode: WorkflowMode;
    disabled?: boolean;
    idPrefix?: string;
    onchange: (settings: EnhanceSettings) => void;
  }

  let { axes, settings, mode, disabled = false, idPrefix = 'enhance', onchange }: Props = $props();

  const visibleAxes = $derived(axes.filter((axis) => !axis.video_only || mode === 'video'));

  function selected(axis: EnhanceAxis, slug: string): boolean {
    const value = settings[axis.key];
    return Array.isArray(value) ? value.includes(slug) : value === slug;
  }

  function allSelected(axis: EnhanceAxis): boolean {
    const value = settings[axis.key];
    return Array.isArray(value) && axis.options.every((option) => value.includes(option.slug));
  }

  // Multi-pick axes switch every option on, or all off once they are all on.
  function toggleAll(axis: EnhanceAxis): void {
    onchange({ ...settings, [axis.key]: allSelected(axis) ? [] : axis.options.map((option) => option.slug) });
  }

  function choose(axis: EnhanceAxis, slug: string): void {
    const current = settings[axis.key];
    if (Array.isArray(current)) {
      const next = current.includes(slug) ? current.filter((item) => item !== slug) : [...current, slug];
      // Keep the matrix order so the spec and backend instructions stay stable.
      const order = axis.options.map((option) => option.slug);
      onchange({ ...settings, [axis.key]: next.sort((a, b) => order.indexOf(a) - order.indexOf(b)) });
    } else {
      onchange({ ...settings, [axis.key]: slug });
    }
  }
</script>

<div class="space-y-3">
  {#each visibleAxes as axis (axis.key)}
    <fieldset class="min-w-0" {disabled}>
      <legend id={`${idPrefix}-${axis.key}-label`} class="ui-label mb-1.5 flex w-full items-center gap-2">
        <span>{axis.label}{axis.multi ? ' · pick any' : ''}</span>
        {#if axis.multi}
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
      <div class="flex flex-wrap gap-1.5" role={axis.multi ? 'group' : 'radiogroup'} aria-labelledby={`${idPrefix}-${axis.key}-label`}>
        {#each axis.options as option (option.slug)}
          {@const isOn = selected(axis, option.slug)}
          <button
            type="button"
            class="ui-chip enhance-chip"
            class:enhance-chip-on={isOn}
            role={axis.multi ? undefined : 'radio'}
            aria-checked={axis.multi ? undefined : isOn}
            aria-pressed={axis.multi ? isOn : undefined}
            {disabled}
            onclick={() => choose(axis, option.slug)}
          >
            {option.label}
          </button>
        {/each}
      </div>
    </fieldset>
  {/each}
</div>

<style>
  .toggle-all { margin-left: auto; }
  /* Single-choice chips mark their choice with aria-checked, multi-choice ones with aria-pressed. */
  .enhance-chip { cursor: pointer; transition: border-color 0.12s ease, color 0.12s ease; }
  .enhance-chip-on { border-color: var(--color-primary-border); color: var(--color-primary-main); }
  .enhance-chip:disabled { cursor: not-allowed; opacity: 0.4; }
</style>
