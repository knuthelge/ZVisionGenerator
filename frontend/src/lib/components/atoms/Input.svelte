<script lang="ts">
  interface Props {
    id?: string;
    name?: string;
    type?: 'text' | 'number' | 'email' | 'password' | 'search' | 'url';
    value?: string | number;
    placeholder?: string;
    disabled?: boolean;
    readonly?: boolean;
    error?: string | null;
    class?: string;
    min?: number | string;
    max?: number | string;
    step?: number | string;
    autocomplete?: HTMLInputElement['autocomplete'];
    oninput?: (event: Event) => void;
    onchange?: (event: Event) => void;
    onblur?: (event: FocusEvent) => void;
  }

  let {
    id,
    name,
    type = 'text',
    value = $bindable(''),
    placeholder,
    disabled = false,
    readonly = false,
    error = null,
    class: extraClass = '',
    min,
    max,
    step,
    autocomplete,
    oninput,
    onchange,
    onblur
  }: Props = $props();

  const baseCls = 'surface-input w-full transition-colors';
  const errorCls = 'border-red-500 focus:border-red-500';
  const cls = $derived(`${baseCls} ${error ? errorCls : ''} ${extraClass}`);
</script>

<input
  {id}
  {name}
  {type}
  bind:value
  {placeholder}
  {disabled}
  {readonly}
  {min}
  {max}
  {step}
  {autocomplete}
  class={cls}
  aria-invalid={error ? 'true' : undefined}
  aria-describedby={error ? `${id}-error` : undefined}
  {oninput}
  {onchange}
  {onblur}
/>
{#if error}
  <p id="{id}-error" class="mt-1 text-xs text-red-400">{error}</p>
{/if}
