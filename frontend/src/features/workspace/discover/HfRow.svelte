<script lang="ts">
  import { Icon } from '$lib/components/atoms';
  import type { HfItem } from '$lib/types';
  import { compactCount, hfRowFacts } from './discoverView';

  interface Props {
    item: HfItem;
    onopen: (item: HfItem, trigger: HTMLElement) => void;
  }

  let { item, onopen }: Props = $props();

  const facts = $derived(hfRowFacts(item));
  // A preview that fails to load leaves the placeholder, never a broken-image mark.
  let broken = $state(false);
</script>

<button type="button" class="disc-row" onclick={(event) => onopen(item, event.currentTarget)}>
  <span class="th">{#if item.thumbnail && !broken}<img src={item.thumbnail} alt="" loading="lazy" decoding="async" onerror={() => { broken = true; }}>{:else}<Icon name="file" size={20} />{/if}</span>
  <span class="tx">
    <span class="tl">
      <span class="disc-name">{item.name}</span>
      {#if item.curated_id}<span class="ui-badge ui-badge-info">Curated</span>{/if}
    </span>
    <span class="disc-meta">
      <span>{item.author}</span>
      <span><Icon name="download" size={11} />{compactCount(item.downloads)}</span>
      <span>♥ {compactCount(item.likes)}</span>
      {#if facts.files}<span>{item.files} files</span>{/if}
      {#if facts.noLicence}<span class="w">No licence</span>{/if}
    </span>
  </span>
  <Icon name="chevright" size={14} class="chev" />
</button>
