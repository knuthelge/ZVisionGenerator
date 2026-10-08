<script lang="ts">
  import type { ShortcutEntry } from '$lib/keyboard';

  interface Props {
    entries: readonly ShortcutEntry[];
  }

  let { entries }: Props = $props();
</script>

<dl class="shortcut-list">
  {#each entries as entry (entry.label)}
    <div>
      <dt>{#each entry.keys as key (key)}<kbd>{key}</kbd>{/each}</dt>
      <dd>{entry.label}</dd>
    </div>
  {/each}
</dl>

<style>
  .shortcut-list { display: flex; flex-direction: column; gap: 6px; margin: 0; }
  .shortcut-list div { display: flex; align-items: center; justify-content: space-between; gap: 12px; font-size: var(--text-ui); }
  .shortcut-list dt { display: flex; flex: none; gap: 4px; }
  .shortcut-list dd { margin: 0; text-align: right; color: var(--color-zinc-300); }
  .shortcut-list kbd { padding: 0 5px; border: 1px solid var(--color-border-strong); border-radius: 4px; font-family: var(--font-mono); font-size: var(--text-meta); line-height: 16px; color: var(--color-text-muted); white-space: nowrap; }
</style>
