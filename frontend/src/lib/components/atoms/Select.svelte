<script lang="ts">
  import Icon from './Icon.svelte';

  interface SelectOption {
    value: string;
    label: string;
    disabled?: boolean;
  }

  interface Props {
    id?: string;
    name?: string;
    value?: string;
    options: SelectOption[];
    placeholder?: string;
    disabled?: boolean;
    required?: boolean;
    error?: string | null;
    /** Accessible name when there is no visible label. */
    ariaLabel?: string;
    class?: string;
    onchange?: (event: Event) => void;
  }

  let {
    id,
    name,
    value = $bindable(''),
    options,
    placeholder,
    disabled = false,
    required = false,
    error = null,
    ariaLabel,
    class: extraClass = '',
    onchange
  }: Props = $props();
</script>

<div class="relative min-w-0">
  <select
    {id}
    {name}
    bind:value
    {disabled}
    {required}
    class="ui-field {extraClass}"
    aria-label={ariaLabel}
    aria-invalid={error ? 'true' : undefined}
    aria-describedby={error ? `${id}-error` : undefined}
    {onchange}
  >
    {#if placeholder}
      <option value="" disabled selected={!value}>{placeholder}</option>
    {/if}
    {#each options as opt (opt.value)}
      <option value={opt.value} disabled={opt.disabled}>{opt.label}</option>
    {/each}
  </select>
  <Icon name="chevdown" size={14} class="pointer-events-none absolute right-2 top-1/2 -translate-y-1/2 text-text-muted" />
</div>
{#if error}
  <p id="{id}-error" class="ui-help ui-help-error mt-1">{error}</p>
{/if}
