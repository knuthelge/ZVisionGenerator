<script lang="ts">
  import { Icon } from '$lib/components/atoms';
  import { compactCount, type TileView } from './discoverView';

  interface Props {
    tile: TileView;
    /** Every tile is a file that is here (the Installed tab), so "Installed" says nothing. */
    hideInstalled?: boolean;
    onopen: (key: string, trigger: HTMLElement) => void;
  }

  let { tile, hideInstalled = false, onopen }: Props = $props();

  // A picture that fails to load leaves the placeholder, never a broken-image mark.
  let broken = $state(false);
</script>

<button type="button" class="disc-card" onclick={(event) => onopen(tile.key, event.currentTarget)}>
  <span class="disc-img">
    {#if tile.image && !broken}<img src={tile.image} alt="" loading="lazy" decoding="async" onerror={() => { broken = true; }}>{:else}<span class="disc-noimg"><Icon name="file" size={22} /></span>{/if}
    <span class="disc-badges">
      {#if tile.status.inUse}<span class="disc-ib on"><Icon name="check" size={12} />In use</span>
      {:else if tile.status.file === 'installed' && !hideInstalled}<span class="disc-ib"><Icon name="check" size={12} />Installed</span>{/if}
      {#if tile.nonCommercial}<span class="disc-ib warn">Non-commercial</span>{/if}
    </span>
  </span>
  <span class="disc-cbody">
    <span class="disc-tl"><span class="disc-name">{tile.title}</span>{#if tile.tool}<span class="disc-tag">Tool</span>{/if}</span>
    {#if tile.line}<span class="disc-desc">{tile.line}</span>{/if}
    {#if tile.author || tile.downloads !== null}
      <span class="disc-meta">
        <span class="disc-author">{tile.author ?? ''}</span>
        {#if tile.downloads !== null}<span class="r"><Icon name="download" size={11} />{compactCount(tile.downloads)}</span>{/if}
      </span>
    {/if}
  </span>
</button>
