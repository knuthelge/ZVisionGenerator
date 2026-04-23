<script lang="ts">
  interface Props {
    id?: string;
    name?: string;
    value?: string;
    placeholder?: string;
    rows?: number;
    disabled?: boolean;
    readonly?: boolean;
    error?: string | null;
    mono?: boolean;
    class?: string;
    oninput?: (event: Event) => void;
    onchange?: (event: Event) => void;
    onblur?: (event: FocusEvent) => void;
    onkeydown?: (event: KeyboardEvent) => void;
  }

  let {
    id,
    name,
    value = $bindable(''),
    placeholder,
    rows = 4,
    disabled = false,
    readonly = false,
    error = null,
    mono = false,
    class: extraClass = '',
    oninput,
    onchange,
    onblur,
    onkeydown
  }: Props = $props();

  const baseCls = 'surface-textarea w-full transition-colors';
  const errorCls = 'border-red-500';
  const monoCls = 'font-mono';
  const cls = $derived(`${baseCls} ${error ? errorCls : ''} ${mono ? monoCls : ''} ${extraClass}`);
</script>

<textarea
  {id}
  {name}
  bind:value
  {placeholder}
  {rows}
  {disabled}
  {readonly}
  class={cls}
  aria-invalid={error ? 'true' : undefined}
  aria-describedby={error ? `${id}-error` : undefined}
  {oninput}
  {onchange}
  {onblur}
  {onkeydown}
></textarea>
{#if error}
  <p id="{id}-error" class="mt-1 text-xs text-red-400">{error}</p>
{/if}
