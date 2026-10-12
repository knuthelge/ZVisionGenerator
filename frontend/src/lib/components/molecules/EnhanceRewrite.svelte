<script lang="ts">
  import type { EnhanceParts } from '$lib/types';

  interface Props {
    /** The choice the run will use. */
    value: EnhanceParts;
    /** "Action only" waits for a scene: with none, Enhance splits the text into both parts. */
    actionOnlyBlocked?: boolean;
    disabled?: boolean;
    /** Prefix of the element ids, as in `EnhanceOptions`. */
    idPrefix?: string;
    onchange: (parts: EnhanceParts) => void;
  }

  let { value, actionOnlyBlocked = false, disabled = false, idPrefix = 'ws-enhance', onchange }: Props = $props();

  const OPTIONS: { value: EnhanceParts; label: string; help: string }[] = [
    { value: 'both', label: 'Scene and action', help: 'Enhance rewrites both from your text.' },
    { value: 'action', label: 'Action only', help: 'Your scene is kept as written.' },
  ];
  const BLOCKED_REASON = 'Write a scene first. With no scene, Enhance splits your text into both.';

  const help = $derived(actionOnlyBlocked ? BLOCKED_REASON : (OPTIONS.find((option) => option.value === value)?.help ?? ''));
</script>

<fieldset class="min-w-0" {disabled}>
  <legend id={`${idPrefix}-rewrite-label`} class="ui-label mb-1.5">Rewrite</legend>
  <div class="flex flex-wrap gap-1.5" role="radiogroup" aria-labelledby={`${idPrefix}-rewrite-label`}>
    {#each OPTIONS as option (option.value)}
      {@const blocked = actionOnlyBlocked && option.value === 'action'}
      <button
        type="button"
        class="ui-chip enhance-chip"
        class:enhance-chip-on={value === option.value}
        role="radio"
        aria-checked={value === option.value}
        aria-describedby={blocked ? `${idPrefix}-rewrite-help` : undefined}
        disabled={blocked}
        onclick={() => onchange(option.value)}
      >
        {option.label}
      </button>
    {/each}
  </div>
  <p id={`${idPrefix}-rewrite-help`} class="ui-help mt-1.5">{help}</p>
</fieldset>

<style>
  .enhance-chip { cursor: pointer; transition: background-color 0.12s ease, color 0.12s ease; }
  .enhance-chip-on { background: color-mix(in srgb, var(--color-primary-main) 16%, transparent); color: var(--color-primary-main); }
  .enhance-chip:disabled { cursor: not-allowed; opacity: 0.4; }
</style>
