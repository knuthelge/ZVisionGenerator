<script lang="ts">
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
    error?: string | null;
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
    error = null,
    class: extraClass = '',
    onchange
  }: Props = $props();

  const baseCls = 'surface-select w-full appearance-none pr-8 transition-colors bg-no-repeat';
  const errorCls = 'border-red-500';
  const cls = $derived(`${baseCls} ${error ? errorCls : ''} ${extraClass}`);
</script>

<div class="relative">
  <select
    {id}
    {name}
    bind:value
    {disabled}
    class={cls}
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
  <!-- Chevron icon -->
  <div class="pointer-events-none absolute inset-y-0 right-0 flex items-center pr-2.5 text-zinc-500">
    <svg class="h-4 w-4" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
      <path stroke-linecap="round" stroke-linejoin="round" d="M4 6l4 4 4-4" />
    </svg>
  </div>
</div>
{#if error}
  <p id="{id}-error" class="mt-1 text-xs text-red-400">{error}</p>
{/if}
