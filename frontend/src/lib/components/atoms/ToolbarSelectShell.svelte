<script lang="ts">
  import type { Snippet } from 'svelte';
  import Icon from './Icon.svelte';

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
      ? 'bg-bg-base border-border-strong outline-2 outline-offset-2 outline-primary-main'
      : hovered
        ? 'bg-bg-surface-hover border-border-strong'
        : 'bg-bg-base border-border-strong'
  );

  const shellClass = $derived(
    `relative flex h-control items-center justify-between gap-2 rounded-sm border px-2 text-ui transition-colors text-text-primary ${stateClass} ${extraClass}`
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
  <Icon name="chevdown" size={14} class="pointer-events-none shrink-0 text-text-muted" />
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