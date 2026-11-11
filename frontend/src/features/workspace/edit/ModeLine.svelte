<script lang="ts">
  import { Icon } from '$lib/components/atoms';
  import type { ModeSummary } from './editView';

  interface Props {
    summary: ModeSummary;
  }

  let { summary }: Props = $props();
</script>

<!-- What the edit will send: the mode as a badge, then what it holds. Shared by the Image to edit section and the phone card. -->
<span class="mode-line">
  <span class="ui-badge ui-badge-info"><Icon name={summary.mode === 'paint' ? 'brush' : 'extend'} size={12} />{summary.label}</span>
  <span class="mode-detail" data-tone={summary.tone}>{summary.detail}</span>
</span>

<style>
  .mode-line { display: flex; min-width: 0; align-items: center; gap: 8px; }
  .mode-detail { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: var(--text-ui); color: var(--color-text-secondary); }
  .mode-detail[data-tone='muted'] { color: var(--color-text-muted); }
  .mode-detail[data-tone='warning'] { color: var(--color-warning); }
</style>
