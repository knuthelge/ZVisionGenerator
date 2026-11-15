<script lang="ts">
  import { FileName, Icon } from '$lib/components/atoms';
  import { kindIcon } from '$lib/state/audioAsset';
  import type { GalleryAsset } from '$lib/types';

  interface Props {
    asset: GalleryAsset;
    /** The kind and size under the name: `Image · 1216 × 832`. */
    meta: string;
  }

  let { asset, meta }: Props = $props();

  // A take has no picture: its icon stands in.
  const thumbnail = $derived(asset.media_type === 'audio' ? null : (asset.thumbnail_url || asset.url));
</script>

<!-- The file the panel is about: its thumbnail, its name cut in the middle, and its kind and size. -->
<span class="panel-file">
  <span class="panel-thumb" aria-hidden="true">
    {#if thumbnail}<img src={thumbnail} alt="">{:else}<Icon name={kindIcon(asset)} size={20} />{/if}
  </span>
  <span class="panel-file-text">
    <FileName name={asset.filename} class="panel-file-name" />
    <small class="panel-file-meta">{meta}</small>
  </span>
</span>

<style>
  .panel-file { display: flex; min-width: 0; flex: 1 1 auto; align-items: center; gap: var(--sp-8); }
  .panel-thumb { display: grid; width: 40px; height: 40px; flex: none; place-items: center; overflow: hidden; border-radius: var(--radius-sm); background: var(--color-bg-surface-hover); color: var(--color-text-muted); }
  .panel-thumb img { width: 100%; height: 100%; object-fit: cover; }
  .panel-file-text { display: flex; min-width: 0; flex: 1; flex-direction: column; }
  .panel-file-text :global(.panel-file-name) { font-size: var(--text-content); font-weight: 600; line-height: 1.3; color: var(--color-text-primary); }
  .panel-file-meta { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: var(--text-meta); color: var(--color-text-muted); }
</style>
