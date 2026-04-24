<script lang="ts">
  import type { Snippet } from 'svelte';

  interface Props {
    id?: string;
    name: string;
    value?: string;
    testId?: string;
    disabled?: boolean;
    class?: string;
    ariaLabel?: string;
    children?: Snippet;
    options?: Snippet;
    onchange?: (event: Event) => void;
  }

  let {
    id,
    name,
    value = $bindable(''),
    testId,
    disabled = false,
    class: extraClass = '',
    ariaLabel,
    children,
    options,
    onchange,
  }: Props = $props();

  let hovered = $state(false);
  let focused = $state(false);

  const stateClass = $derived(
    focused
      ? 'bg-bg-surface border-primary-main ring-4 ring-primary-main'
      : hovered
        ? 'bg-bg-surface-hover border-border-subtle'
        : 'bg-bg-surface border-border-subtle'
  );

  const shellClass = $derived(
    `relative flex items-center justify-between rounded-md border px-3 py-2 text-sm transition-colors text-text-primary ${stateClass} ${extraClass}`
  );

  function handlePointerEnter(): void {
    hovered = true;
  }

  function handlePointerLeave(): void {
    hovered = false;
  }

  function handleFocus(): void {
    focused = true;
  }

  function handleBlur(): void {
    focused = false;
  }
</script>

<div
  data-testid={testId}
  data-hovered={hovered ? 'true' : 'false'}
  data-focused={focused ? 'true' : 'false'}
  class={shellClass}
>
  {@render children?.()}
  <svg class="w-4 h-4 text-zinc-500 pointer-events-none shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path>
  </svg>
  <select
    {id}
    {name}
    bind:value
    {disabled}
    aria-label={ariaLabel}
    class="absolute inset-0 appearance-none opacity-0 cursor-pointer focus:outline-none"
    onmouseenter={handlePointerEnter}
    onmouseleave={handlePointerLeave}
    onfocus={handleFocus}
    onblur={handleBlur}
    {onchange}
  >
    {@render options?.()}
  </select>
</div>