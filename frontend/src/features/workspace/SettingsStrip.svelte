<script lang="ts">
  import { Icon } from '$lib/components/atoms';
  import type { ChipKey, SettingChip } from './settingsChips';

  interface Props {
    chips: SettingChip[];
    /** The chip whose popover is open. */
    activeKey: ChipKey | null;
    /** Settings that differ from the defaults, shown on All. */
    changedCount: number;
    onopen: (key: ChipKey) => void;
    onall: () => void;
  }

  let { chips, activeKey, changedCount, onopen, onall }: Props = $props();

  let scroller = $state<HTMLDivElement | null>(null);

  // Bring the chip that opened into view, so its popover and its chip line up.
  $effect(() => {
    if (!activeKey || !scroller) return;
    scroller.querySelector<HTMLElement>(`[data-chip="${activeKey}"]`)?.scrollIntoView?.({ inline: 'nearest', block: 'nearest', behavior: 'smooth' });
  });
</script>

<!-- One chip per setting, its value on it and a dot when changed. The row scrolls; All stays pinned at the right. -->
<div class="settings-strip" role="toolbar" aria-label="Settings" data-testid="settings-strip">
  <div bind:this={scroller} class="strip-chips">
    {#each chips as chip (chip.key)}
      <button
        type="button"
        class="setchip"
        class:warn={chip.warn}
        data-chip={chip.key}
        data-changed={chip.changed}
        aria-expanded={activeKey === chip.key}
        aria-label="{chip.label || chip.key}: {chip.value}"
        onclick={() => onopen(chip.key)}
      >
        {#if chip.key === 'model'}<Icon name="cube" size={14} />{/if}
        {#if chip.key === 'reference'}<Icon name="reference" size={14} />{/if}
        {#if chip.key === 'source'}<Icon name="film" size={14} />{/if}
        {#if chip.label}<span class="setchip-k">{chip.label}</span>{/if}
        <span class="setchip-v" class:on={chip.on}>{chip.value}</span>
        {#if chip.warn}<Icon name="alert" size={12} class="setchip-warn" />{/if}
      </button>
    {/each}
  </div>
  <button type="button" class="setchip setchip-all" aria-label="All settings" onclick={onall}>
    <Icon name="sliders" size={14} />All{#if changedCount > 0}<span class="setchip-n">{changedCount}</span>{/if}
  </button>
</div>

<style>
  .settings-strip { display: flex; flex: none; align-items: center; padding: 8px max(12px, env(safe-area-inset-right, 0px)) 8px 0; background: var(--color-bg-base); border-top: 1px solid var(--color-border-subtle); }
  .strip-chips { display: flex; flex: 1; min-width: 0; gap: 6px; overflow-x: auto; padding: 0 12px 0 max(12px, env(safe-area-inset-left, 0px)); scrollbar-width: none; mask-image: linear-gradient(to right, black calc(100% - 16px), transparent); -webkit-mask-image: linear-gradient(to right, black calc(100% - 16px), transparent); }
  .strip-chips::-webkit-scrollbar { display: none; }
  .setchip { position: relative; display: inline-flex; flex: none; align-items: center; gap: 6px; height: 32px; padding: 0 10px; border-radius: var(--radius-sm); background: var(--color-bg-surface-hover); font-size: var(--text-ui); font-weight: 600; white-space: nowrap; color: var(--color-text-primary); transition: background-color 0.12s ease, color 0.12s ease; }
  .setchip-k { color: var(--color-text-muted); font-weight: 500; }
  .setchip-v { max-width: 36vw; overflow: hidden; text-overflow: ellipsis; font-family: var(--font-mono); font-weight: 500; }
  .setchip-v.on { color: var(--color-primary-main); }
  .setchip[data-changed='true']::after { content: ''; position: absolute; top: 5px; right: 5px; width: 6px; height: 6px; border-radius: 50%; background: var(--color-primary-main); }
  .setchip[aria-expanded='true'], .setchip[aria-expanded='true'] .setchip-k, .setchip[aria-expanded='true'] .setchip-v { background: color-mix(in srgb, var(--color-primary-main) 16%, transparent); color: var(--color-primary-main); }
  .setchip[aria-expanded='true'] .setchip-k, .setchip[aria-expanded='true'] .setchip-v { background: transparent; }
  .setchip :global(.setchip-warn) { color: var(--color-warning); }
  .setchip-all { margin-left: 2px; gap: 8px; }
  .setchip-n { font-family: var(--font-mono); font-size: var(--text-meta); color: var(--color-primary-main); }
</style>
