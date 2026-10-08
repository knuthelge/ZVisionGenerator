<script lang="ts" module>
  export interface KeyValueItem {
    label: string;
    value: string;
    /** Monospace and breakable anywhere, for paths and identifiers. */
    mono?: boolean;
    /** One line of context under the value. */
    hint?: string;
    tone?: 'success' | 'muted';
  }
</script>

<script lang="ts">
  interface Props {
    items: KeyValueItem[];
    class?: string;
  }

  let { items, class: extraClass = '' }: Props = $props();
</script>

<!-- Read-only settings as label and value rows. -->
<dl class="kv {extraClass}">
  {#each items as item (item.label)}
    <div class="kv-row">
      <dt class="ui-label">{item.label}</dt>
      <dd class="min-w-0">
        <span class="kv-value" class:mono={item.mono} data-tone={item.tone}>{item.value}</span>
        {#if item.hint}<span class="ui-help block">{item.hint}</span>{/if}
      </dd>
    </div>
  {/each}
</dl>

<style>
  .kv { display: flex; flex-direction: column; }
  .kv-row { display: grid; grid-template-columns: minmax(120px, 200px) minmax(0, 1fr); gap: 4px 16px; padding: 8px 0; border-top: 1px solid var(--color-border-subtle); }
  .kv-row:first-child { border-top: 0; padding-top: 0; }
  .kv-row:last-child { padding-bottom: 0; }
  .kv-value { display: block; font-size: var(--text-ui); color: var(--color-text-primary); overflow-wrap: anywhere; }
  .kv-value.mono { font-family: var(--font-mono); }
  .kv-value[data-tone='success'] { color: var(--color-success); }
  .kv-value[data-tone='muted'] { color: var(--color-text-muted); }
  @media (max-width: 639px) {
    .kv-row { grid-template-columns: minmax(0, 1fr); }
  }
</style>
