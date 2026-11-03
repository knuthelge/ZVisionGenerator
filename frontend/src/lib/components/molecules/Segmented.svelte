<script lang="ts" generics="T extends string">
  import { Icon } from '$lib/components/atoms';
  import type { IconName } from '$lib/components/atoms/Icon.svelte';

  interface Option {
    value: T;
    label: string;
    /** A small glyph before the label. */
    icon?: IconName;
    disabled?: boolean;
    title?: string;
  }

  interface Props {
    options: Option[];
    value: T;
    /** Accessible name of the group. */
    label: string;
    size?: 'sm' | 'md';
    /** Monospace labels, e.g. ratios and sizes. */
    mono?: boolean;
    disabled?: boolean;
    testId?: string;
    class?: string;
    onchange: (value: T) => void;
  }

  let { options, value, label, size = 'md', mono = false, disabled = false, testId, class: extraClass = '', onchange }: Props = $props();
</script>

<div
  class="ui-segmented {size === 'sm' ? 'ui-segmented-sm' : ''} {mono ? 'ui-segmented-mono' : ''} {extraClass}"
  role="group"
  aria-label={label}
  data-testid={testId}
>
  {#each options as option (option.value)}
    <button
      type="button"
      aria-pressed={option.value === value}
      disabled={disabled || option.disabled}
      title={option.title}
      data-value={option.value}
      onclick={() => { if (option.value !== value) onchange(option.value); }}
    >{#if option.icon}<Icon name={option.icon} size={14} />{/if}{option.label}</button>
  {/each}
</div>
