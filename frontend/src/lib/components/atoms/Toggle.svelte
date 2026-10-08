<script lang="ts">
  import type { Snippet } from 'svelte';

  interface Props {
    id?: string;
    name?: string;
    checked?: boolean;
    disabled?: boolean;
    label?: string;
    /** Accessible name when there is no visible label. */
    ariaLabel?: string;
    labelSnippet?: Snippet;
    class?: string;
    onchange?: (event: Event) => void;
  }

  let {
    id,
    name,
    checked = $bindable(false),
    disabled = false,
    label,
    ariaLabel,
    labelSnippet,
    class: extraClass = '',
    onchange
  }: Props = $props();
</script>

<label class="toggle {extraClass}" class:toggle-disabled={disabled}>
  <input
    {id}
    {name}
    type="checkbox"
    role="switch"
    bind:checked
    {disabled}
    aria-label={ariaLabel}
    class="sr-only"
    {onchange}
  />
  <span class="track" aria-hidden="true"></span>
  {#if label}
    <span class="toggle-label">{label}</span>
  {:else if labelSnippet}
    {@render labelSnippet()}
  {/if}
</label>

<style>
  /* Relative, so the visually hidden input sits at the switch and focusing it never scrolls elsewhere. */
  .toggle { position: relative; display: inline-flex; align-items: center; gap: 8px; cursor: pointer; }
  .toggle-disabled { opacity: 0.4; cursor: not-allowed; }
  .track { position: relative; flex-shrink: 0; width: 26px; height: 15px; border-radius: 9999px; background: var(--color-zinc-700); transition: background-color 0.12s ease; }
  .track::after { content: ''; position: absolute; top: 2px; left: 2px; width: 11px; height: 11px; border-radius: 50%; background: var(--color-zinc-500); transition: transform 0.12s ease, background-color 0.12s ease; }
  input:checked + .track { background: var(--color-primary-main); }
  input:checked + .track::after { transform: translateX(11px); background: var(--color-primary-ink); }
  input:focus-visible + .track { outline: 2px solid var(--color-primary-main); outline-offset: 2px; }
  .toggle-label { font-size: var(--text-ui); color: var(--color-text-secondary); user-select: none; }
</style>
