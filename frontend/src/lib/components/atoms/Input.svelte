<script lang="ts">
  interface Props {
    id?: string;
    name?: string;
    type?: 'text' | 'number' | 'email' | 'password' | 'search' | 'url';
    value?: string | number;
    placeholder?: string;
    disabled?: boolean;
    readonly?: boolean;
    required?: boolean;
    error?: string | null;
    class?: string;
    min?: number | string;
    max?: number | string;
    step?: number | string;
    autocomplete?: HTMLInputElement['autocomplete'];
    ariaDescribedby?: string;
    ariaInvalid?: boolean;
    ariaBusy?: boolean;
    oninput?: (event: Event) => void;
    onchange?: (event: Event) => void;
    onblur?: (event: FocusEvent) => void;
    onkeydown?: (event: KeyboardEvent) => void;
    oninvalid?: (event: Event) => void;
  }

  let {
    id,
    name,
    type = 'text',
    value = $bindable(''),
    placeholder,
    disabled = false,
    readonly = false,
    required = false,
    error = null,
    class: extraClass = '',
    min,
    max,
    step,
    autocomplete,
    ariaDescribedby,
    ariaInvalid,
    ariaBusy,
    oninput,
    onchange,
    onblur,
    onkeydown,
    oninvalid
  }: Props = $props();

  const invalid = $derived(ariaInvalid ?? Boolean(error));
  const cls = $derived(`ui-field ${extraClass}`);
</script>

<input
  {id}
  {name}
  {type}
  bind:value
  {placeholder}
  {disabled}
  {readonly}
  {required}
  {min}
  {max}
  {step}
  {autocomplete}
  class={cls}
  aria-invalid={invalid ? 'true' : undefined}
  aria-describedby={ariaDescribedby ?? (error ? `${id}-error` : undefined)}
  aria-busy={ariaBusy ? 'true' : undefined}
  {oninput}
  {onchange}
  {onblur}
  {onkeydown}
  {oninvalid}
/>
{#if error}
  <p id="{id}-error" class="ui-help ui-help-error mt-1">{error}</p>
{/if}
