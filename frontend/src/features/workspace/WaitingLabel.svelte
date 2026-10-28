<script lang="ts">
  import { Icon } from '$lib/components/atoms';
  import { waitText, type LoraBlock } from '$lib/state/loraDownloads';

  interface Props {
    waiting: NonNullable<LoraBlock['waiting']>;
    /** A phone: "Waiting · 40 %" without the name. */
    compact?: boolean;
  }

  let { waiting, compact = false }: Props = $props();

  const text = $derived(waitText(waiting, compact));
  // A screen reader hears each quarter of the way, not every percent.
  const spoken = $derived(waiting.percent === null ? text.full : `Waiting for ${waiting.name}, ${Math.floor(waiting.percent / 25) * 25} percent`);
</script>

<Icon name="download" size={16} />
<span class="ui-btn-wait-t">{text.lead}</span>
{#if text.percent}<span class="ui-btn-wait-p">· {text.percent}</span>{/if}
<span class="sr-only" role="status">{spoken}</span>
