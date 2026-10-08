<script lang="ts">
  import type { Snippet } from 'svelte';
  import type { HTMLButtonAttributes } from 'svelte/elements';
  import Spinner from './Spinner.svelte';

  interface Props extends Omit<HTMLButtonAttributes, 'class' | 'type' | 'disabled' | 'onclick' | 'children'> {
    variant?: 'primary' | 'secondary' | 'quiet' | 'danger' | 'add';
    /** `sm` is 22px for inline actions; inside an ActionBar every button is 36px regardless. */
    size?: 'sm' | 'md';
    /** Square, icon-only button; give it an `aria-label`. */
    icon?: boolean;
    /** The one main action of the screen: takes the remaining width of its ActionBar. */
    main?: boolean;
    disabled?: boolean;
    loading?: boolean;
    type?: 'button' | 'submit' | 'reset';
    class?: string;
    onclick?: (event: MouseEvent) => void;
    children?: Snippet;
  }

  let {
    variant = 'secondary',
    size = 'md',
    icon = false,
    main = false,
    disabled = false,
    loading = false,
    type = 'button',
    class: extraClass = '',
    onclick,
    children,
    ...rest
  }: Props = $props();

  const cls = $derived(
    [
      'ui-btn',
      variant !== 'secondary' && `ui-btn-${variant}`,
      size === 'sm' && 'ui-btn-sm',
      icon && 'ui-btn-icon',
      main && 'ui-btn-main',
      extraClass,
    ].filter(Boolean).join(' ')
  );
</script>

<button
  {...rest}
  {type}
  class={cls}
  disabled={disabled || loading}
  aria-busy={loading ? 'true' : undefined}
  {onclick}
>
  {#if loading}
    <Spinner size="sm" />
  {/if}
  {@render children?.()}
</button>
