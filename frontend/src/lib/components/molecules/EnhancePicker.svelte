<script lang="ts" module>
  import type { EnhanceOptionInfo } from '$lib/types';

  /** The option an axis means by "don't impose one"; unset pickers show it grey. */
  export const KEEP = 'keep';

  /** Group *options* in their listed order: an option without a group (Keep) gets a group of its own. */
  export function groupOptions(options: EnhanceOptionInfo[]): { name: string | null; options: EnhanceOptionInfo[] }[] {
    const groups: { name: string | null; options: EnhanceOptionInfo[] }[] = [];
    for (const option of options) {
      const last = groups.at(-1);
      if (last && last.name === option.group) last.options.push(option);
      else groups.push({ name: option.group, options: [option] });
    }
    return groups;
  }
</script>

<script lang="ts">
  import { tick } from 'svelte';
  import { popover } from '$lib/actions/popover';
  import { Icon } from '$lib/components/atoms';
  import type { EnhanceAxis } from '$lib/types';

  interface Props {
    axis: EnhanceAxis;
    /** The chosen option's slug. */
    value: string;
    /** Id of the trigger button, which a row label points at. */
    id: string;
    disabled?: boolean;
    onchange: (slug: string) => void;
  }

  let { axis, value, id, disabled = false, onchange }: Props = $props();

  const current = $derived(axis.options.find((option) => option.slug === value) ?? axis.options[0]);
  const fallback = $derived(axis.default[0] ?? KEEP);
  // A picker that differs from its default is "set": it turns teal, and one with a Keep option gets a clear button.
  const isSet = $derived(value !== fallback);
  const clearable = $derived(isSet && axis.options.some((option) => option.slug === KEEP));
  const groups = $derived(groupOptions(axis.options));

  let open = $state(false);
  let trigger = $state<HTMLButtonElement | null>(null);
  let list = $state<HTMLDivElement | null>(null);

  function options(): HTMLButtonElement[] {
    return Array.from(list?.querySelectorAll<HTMLButtonElement>('[role="option"]') ?? []);
  }

  async function toggle(): Promise<void> {
    open = !open;
    if (!open) return;
    await tick();
    (options().find((option) => option.getAttribute('aria-selected') === 'true') ?? options()[0])?.focus();
  }

  function close(): void {
    open = false;
  }

  function choose(slug: string): void {
    open = false;
    if (slug !== value) onchange(slug);
    trigger?.focus();
  }

  function onListKeydown(event: KeyboardEvent): void {
    const items = options();
    const index = items.indexOf(document.activeElement as HTMLButtonElement);
    const move = (next: number) => { event.preventDefault(); items[(next + items.length) % items.length]?.focus(); };
    if (event.key === 'ArrowDown') move(index + 1);
    else if (event.key === 'ArrowUp') move(index - 1);
    else if (event.key === 'Home') move(0);
    else if (event.key === 'End') move(items.length - 1);
    else if (event.key === 'Tab') close();
  }
</script>

<div class="picker-wrap">
  <button
    bind:this={trigger}
    {id}
    type="button"
    class="picker ui-field"
    class:set={isSet}
    aria-haspopup="listbox"
    aria-expanded={open}
    data-value={value}
    {disabled}
    onclick={toggle}
  >
    <span class="picker-value" class:keep={value === KEEP}>{current?.label ?? value}</span>
    {#if !clearable}<Icon name="chevdown" size={12} />{/if}
  </button>
  {#if clearable}
    <button type="button" class="picker-clear" aria-label="{axis.label}: back to Keep" title="Back to Keep" {disabled} onclick={() => choose(KEEP)}>
      <Icon name="close" size={12} />
    </button>
  {/if}
</div>

{#if open}
  <!-- Moved to <body> by the popover action; data-popover-nested keeps the enhancer panel open while it is used. -->
  <div
    bind:this={list}
    use:popover={{ anchor: trigger, align: 'start', onclose: close }}
    class="picker-menu ui-overlay"
    role="listbox"
    aria-label={axis.label}
    data-popover-nested
    tabindex="-1"
    onkeydown={onListKeydown}
  >
    {#each groups as group, index (index)}
      {#if group.name}<p class="picker-group ui-area-label" role="presentation">{group.name}</p>{/if}
      {#each group.options as option (option.slug)}
        {@const selected = option.slug === value}
        <button type="button" class="picker-option" role="option" aria-selected={selected} data-slug={option.slug} onclick={() => choose(option.slug)}>
          <span class="picker-tick">{#if selected}<Icon name="check" size={14} />{/if}</span>
          <span class="picker-name">{option.label}</span>
          {#if option.description}<span class="picker-desc">{option.description}</span>{/if}
        </button>
      {/each}
    {/each}
  </div>
{/if}

<style>
  .picker-wrap { position: relative; display: flex; min-width: 0; flex: 1; }
  .picker { display: flex; align-items: center; gap: 6px; width: 100%; text-align: left; cursor: pointer; color: var(--color-text-muted); }
  @media (hover: hover) { .picker:hover:not(:disabled) { color: var(--color-text-secondary); } }
  .picker-value { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--color-text-primary); }
  .picker-value.keep { color: var(--color-text-muted); }
  .picker.set { background: color-mix(in srgb, var(--color-primary-main) 12%, var(--color-bg-base)); padding-right: 28px; }
  .picker.set .picker-value { color: var(--color-primary-main); font-weight: 600; }
  .picker-clear { position: absolute; top: 50%; right: 4px; display: inline-grid; place-items: center; width: 20px; height: 20px; border-radius: 4px; color: var(--color-text-muted); transform: translateY(-50%); }
  @media (hover: hover) { .picker-clear:hover:not(:disabled) { background: var(--color-bg-surface); color: var(--color-text-primary); } }
  /* Above the bottom sheet the options sit in on a phone. */
  .picker-menu { z-index: var(--z-menu); width: min(320px, calc(100vw - 16px)); max-height: min(420px, calc(100dvh - 16px)); overflow-y: auto; padding: 4px; }
  .picker-group { margin: 0; padding: 8px 8px 4px; }
  .picker-option { display: grid; grid-template-columns: 16px minmax(0, 1fr); column-gap: 8px; width: 100%; padding: 6px 8px; border-radius: var(--radius-sm); text-align: left; cursor: pointer; }
  .picker-option:focus-visible, .picker-option[aria-selected='true'] { background: var(--color-bg-surface-hover); outline: none; }
  @media (hover: hover) { .picker-option:hover { background: var(--color-bg-surface-hover); } }
  .picker-tick { display: grid; place-items: start center; padding-top: 2px; color: var(--color-primary-main); }
  .picker-name { font-size: var(--text-ui); font-weight: 600; line-height: 18px; color: var(--color-text-primary); }
  .picker-desc { grid-column: 2; font-size: var(--text-meta); line-height: 1.35; color: var(--color-text-muted); }
</style>
