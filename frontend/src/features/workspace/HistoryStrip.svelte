<script lang="ts">
  import { draft } from '$lib/state/draft.svelte';
  import { router } from '$lib/state/router.svelte';
  import { assetAspect, type AssetActionHandlers, type ReferenceTarget } from '$lib/state/assetActions';
  import { Icon } from '$lib/components/atoms';
  import { AssetTile } from '$lib/components/molecules';
  import type { GalleryAsset } from '$lib/types';

  interface Props extends AssetActionHandlers {
    assets: GalleryAsset[];
    loading?: boolean;
    deletingIds?: ReadonlySet<string>;
    referenceUnavailable?: Partial<Record<ReferenceTarget, string>>;
  }

  let { assets, loading = false, deletingIds = new Set<string>(), referenceUnavailable = {}, ...handlers }: Props = $props();

  const collapsed = $derived(draft.state.historyCollapsed);
  const TILE_HEIGHT_PX = 96;

  // Left/right arrows move focus between tiles; Tab still walks through each tile's actions.
  function onKeydown(event: KeyboardEvent): void {
    if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return;
    const tiles = Array.from((event.currentTarget as HTMLElement).querySelectorAll<HTMLElement>('.asset-tile-media'));
    const current = tiles.findIndex((tile) => tile.closest('.asset-tile')?.contains(document.activeElement));
    if (current < 0) return;
    const next = tiles[current + (event.key === 'ArrowRight' ? 1 : -1)];
    if (!next) return;
    event.preventDefault();
    next.focus();
    next.scrollIntoView({ block: 'nearest', inline: 'nearest' });
  }
</script>

<section id="ws-history-shell" class="history-strip" aria-labelledby="ws-history-title" data-collapsed={collapsed}>
  <div class="strip-head">
    <button
      type="button"
      id="ws-history-toggle"
      class="strip-toggle ui-link"
      aria-expanded={!collapsed}
      aria-controls="ws-history-scroll"
      aria-label="{collapsed ? 'Expand' : 'Collapse'} history"
      onclick={() => draft.update('historyCollapsed', !collapsed)}
    >
      <Icon name="chevdown" size={14} class="strip-chev" />
      <span id="ws-history-title" class="ui-area-label">History</span>
      <span class="strip-count font-mono">{assets.length}</span>
    </button>
    <span class="flex-1"></span>
    {#if !collapsed && assets.length > 0}
      <span class="strip-hint">Click a tile to open the viewer</span>
    {/if}
    <button type="button" class="ui-btn ui-btn-sm ui-btn-quiet" onclick={() => router.navigate('gallery')}>Open Gallery</button>
  </div>

  {#if !collapsed}
    <!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
    <div
      id="ws-history-scroll"
      class="strip-scroll"
      role="list"
      aria-label="History, newest first"
      onkeydown={onKeydown}
    >
      {#if loading && assets.length === 0}
        <p class="strip-empty">Loading history…</p>
      {:else if assets.length === 0}
        <p class="strip-empty">No history yet. Generated assets will appear here.</p>
      {:else}
        {#each assets as asset (asset.id)}
          <div role="listitem" class="strip-item" style="height: {TILE_HEIGHT_PX}px; width: {Math.round(TILE_HEIGHT_PX * assetAspect(asset))}px">
            <AssetTile {asset} density="compact" deleting={deletingIds.has(asset.id)} {referenceUnavailable} class="h-full w-full" {...handlers} />
          </div>
        {/each}
      {/if}
    </div>
  {/if}
</section>

<style>
  .history-strip { flex: none; border-top: 1px solid var(--color-border-subtle); background: var(--color-bg-base); }
  .strip-head { display: flex; align-items: center; gap: 10px; height: 32px; padding: 0 12px; background: var(--color-bg-raised); }
  .strip-toggle { display: inline-flex; align-items: center; gap: 6px; padding: 3px 6px; border-radius: 4px; }
  .strip-toggle:focus-visible { outline: 2px solid var(--color-primary-main); outline-offset: 2px; }
  .strip-toggle :global(.strip-chev) { transition: transform 0.12s ease; }
  .history-strip[data-collapsed='true'] .strip-toggle :global(.strip-chev) { transform: rotate(180deg); }
  .strip-count { font-size: var(--text-ui); color: var(--color-text-muted); }
  .strip-hint { font-size: var(--text-meta); color: var(--color-text-muted); }
  .strip-scroll { display: flex; gap: 8px; overflow-x: auto; overflow-y: hidden; padding: 10px 12px; }
  .strip-item { flex: none; }
  .strip-empty { padding: 12px 0 18px; font-size: var(--text-ui); color: var(--color-text-muted); }
</style>
