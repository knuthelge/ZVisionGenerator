<script lang="ts">
  import { Icon } from '$lib/components/atoms';
  import type { AssetActionHandlers, ReferenceTarget } from '$lib/state/assetActions';
  import type { GalleryAsset } from '$lib/types';
  import ActionMenu from './ActionMenu.svelte';
  import { assetMenuEntries } from './assetMenu';

  interface Props extends AssetActionHandlers {
    asset: GalleryAsset;
    /** `compact`: filmstrip thumbnail. `card`: gallery card with footer. `stage`: full image, letterboxed. */
    density?: 'compact' | 'card' | 'stage';
    selected?: boolean;
    /** Shows a selection checkbox when provided. */
    onselect?: (asset: GalleryAsset, selected: boolean) => void;
    deleting?: boolean;
    /** Reference targets the current model can't use, with the reason. */
    referenceUnavailable?: Partial<Record<ReferenceTarget, string>>;
    eager?: boolean;
    onmediaload?: () => void;
    class?: string;
  }

  let {
    asset,
    density = 'card',
    selected = false,
    onselect,
    deleting = false,
    referenceUnavailable = {},
    eager = false,
    onmediaload,
    onpreview,
    onreuse,
    onreference,
    onupscale,
    ondelete,
    class: extraClass = '',
  }: Props = $props();

  let menuOpen = $state(false);
  let moreButton = $state<HTMLButtonElement | null>(null);

  const isVideo = $derived(asset.media_type === 'video');
  const canReuse = $derived(asset.has_reusable_config === true);
  const menuItems = $derived(assetMenuEntries(asset, { onreference, onupscale, ondelete }, deleting, referenceUnavailable));
  // Stage videos play inline, so previewing goes through the expand action instead of the media.
  const inlineVideo = $derived(density === 'stage' && isVideo);
  // The stage shows the full-resolution file; small tiles use the thumbnail.
  const src = $derived(density === 'stage' ? asset.url : (asset.thumbnail_url || asset.url));

  function preview(trigger: HTMLElement): void {
    onpreview?.(asset, trigger);
  }
</script>

<article
  class="asset-tile {extraClass}"
  data-density={density}
  data-selected={selected}
  data-menu-open={menuOpen}
  aria-busy={deleting}
  aria-label={asset.filename}
>
  {#snippet media()}
    {#if isVideo}
      <video
        {src}
        class="asset-tile-img"
        muted
        controls={inlineVideo}
        preload="metadata"
        onloadeddata={onmediaload}
      ></video>
    {:else}
      <img
        {src}
        alt={asset.prompt || asset.filename}
        class="asset-tile-img"
        loading={eager ? 'eager' : 'lazy'}
        onload={onmediaload}
        onerror={onmediaload}
      >
    {/if}
  {/snippet}

  {#if inlineVideo}
    <div class="asset-tile-media">{@render media()}</div>
  {:else}
    <button
      type="button"
      class="asset-tile-media"
      aria-label="View {asset.filename}"
      onclick={(event) => preview(event.currentTarget)}
    >{@render media()}</button>
  {/if}

  {#if onselect}
    <input
      type="checkbox"
      class="asset-tile-check accent-primary-main"
      checked={selected}
      aria-label="Select {asset.filename}"
      onchange={(event) => onselect(asset, (event.currentTarget as HTMLInputElement).checked)}
    >
  {/if}

  <div class="asset-tile-actions">
    {#if onpreview}
      <button
        type="button"
        class="asset-tile-action ui-media-action"
        title="View fullscreen"
        aria-label="View {asset.filename} fullscreen"
        onclick={(event) => preview(event.currentTarget)}
      ><Icon name="expand" size={14} /></button>
    {/if}
    {#if onreuse}
      <button
        type="button"
        class="asset-tile-action ui-media-action-primary"
        title={canReuse ? 'Reuse settings' : 'Reusable settings unavailable'}
        aria-label={canReuse ? `Reuse settings from ${asset.filename}` : 'Reusable settings unavailable'}
        disabled={!canReuse}
        onclick={() => onreuse(asset)}
      ><Icon name="reuse" size={14} /></button>
    {/if}
    <button
      type="button"
      bind:this={moreButton}
      class="asset-tile-action ui-media-action"
      title="More actions"
      aria-label="More actions for {asset.filename}"
      aria-haspopup="menu"
      aria-expanded={menuOpen}
      onclick={() => { menuOpen = !menuOpen; }}
    ><Icon name="more" size={14} /></button>
  </div>

  {#if density === 'card'}
    <div class="asset-tile-foot ui-media-foot">
      <p class="truncate text-ui font-medium text-text-secondary">{asset.filename}</p>
      <p class="mt-0.5 truncate text-meta text-text-muted">{asset.workflow} &middot; {new Date(asset.created_at).toLocaleDateString()}</p>
    </div>
  {/if}

  <ActionMenu
    open={menuOpen}
    anchor={moreButton}
    items={menuItems}
    label="Actions for {asset.filename}"
    onclose={() => { menuOpen = false; }}
  />
</article>

<style>
  .asset-tile {
    position: relative;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    isolation: isolate;
    border: 1px solid var(--color-border-strong);
    border-radius: var(--radius-md);
    background: var(--color-bg-surface);
    transition: border-color 0.12s ease;
  }
  .asset-tile:hover, .asset-tile:focus-within { border-color: color-mix(in srgb, var(--color-primary-main) 45%, var(--color-border-strong)); }
  .asset-tile[data-selected='true'] { border-color: var(--color-primary-main); box-shadow: 0 0 0 1px var(--color-primary-main); }
  .asset-tile[aria-busy='true'] { opacity: 0.6; }
  .asset-tile[data-density='stage'] { border-color: transparent; background: transparent; }

  .asset-tile-media { position: relative; display: block; flex: 1; min-height: 0; width: 100%; padding: 0; cursor: zoom-in; }
  .asset-tile-media:focus-visible { outline: 2px solid var(--color-primary-main); outline-offset: -2px; }
  .asset-tile-img { display: block; width: 100%; height: 100%; object-fit: cover; }
  .asset-tile[data-density='card'] .asset-tile-media { flex: none; aspect-ratio: 1; }
  /* Never enlarge: small outputs stay at their own size, large ones shrink to fit. */
  .asset-tile[data-density='stage'] .asset-tile-img { object-fit: scale-down; }

  /* Top gradient keeps the actions legible on bright images. */
  .asset-tile::after {
    content: '';
    position: absolute;
    inset: 0;
    z-index: 1;
    pointer-events: none;
    background: linear-gradient(180deg, var(--color-scrim), transparent 42%);
    opacity: 0;
    transition: opacity 0.12s ease;
  }
  .asset-tile[data-density='stage']::after { display: none; }

  .asset-tile-actions {
    position: absolute;
    top: 6px;
    right: 6px;
    z-index: 2;
    display: flex;
    gap: 4px;
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.12s ease;
  }
  .asset-tile:hover .asset-tile-actions,
  .asset-tile:focus-within .asset-tile-actions,
  .asset-tile[data-menu-open='true'] .asset-tile-actions { opacity: 1; pointer-events: auto; }
  .asset-tile:hover::after,
  .asset-tile:focus-within::after,
  .asset-tile[data-menu-open='true']::after { opacity: 1; }
  @media (hover: none) {
    .asset-tile-actions { opacity: 1; pointer-events: auto; }
  }

  .asset-tile-action { display: grid; place-items: center; width: 28px; height: 28px; border-radius: var(--radius-sm); }
  .asset-tile-action:focus-visible { outline: 2px solid var(--color-primary-main); outline-offset: 1px; }
  .asset-tile-action:disabled { opacity: 0.5; cursor: not-allowed; }
  .asset-tile[data-density='compact'] .asset-tile-actions { top: 4px; right: 4px; gap: 3px; }
  .asset-tile[data-density='compact'] .asset-tile-action { width: 24px; height: 24px; border-radius: var(--radius-sm); }

  .asset-tile-check { position: absolute; top: 8px; left: 8px; z-index: 2; width: 16px; height: 16px; cursor: pointer; border-radius: 4px; }
  .asset-tile-foot { padding: 8px 12px; }
</style>
