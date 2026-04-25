<script lang="ts">
  import type { GalleryAsset } from '$lib/types';

  interface Props {
    asset: GalleryAsset;
    selected?: boolean;
    active?: boolean;
    onselect?: (asset: GalleryAsset, selected: boolean) => void;
    onactivate?: (asset: GalleryAsset) => void;
    onopenlightbox?: (asset: GalleryAsset) => void;
    ondownload?: (asset: GalleryAsset) => void;
    onreuse?: (asset: GalleryAsset) => void;
    ondelete?: (asset: GalleryAsset) => void;
  }

  let {
    asset,
    selected = false,
    active = false,
    onselect,
    onactivate,
    onopenlightbox,
    ondownload,
    onreuse,
    ondelete
  }: Props = $props();

  let hovered = $state(false);

  const cardCls = $derived(
    selected
      ? 'surface-gallery-card-selected'
      : 'surface-gallery-card'
  );
  const activeCls = $derived(active && !selected ? 'ring-2 ring-white/30' : '');
</script>

<div
  class="relative group overflow-hidden rounded-lg transition-all cursor-pointer {cardCls} {activeCls}"
  onmouseenter={() => hovered = true}
  onmouseleave={() => hovered = false}
  onclick={() => onactivate?.(asset)}
  role="button"
  tabindex="0"
  onkeydown={(e) => e.key === 'Enter' && onactivate?.(asset)}
  aria-label="Asset: {asset.filename}"
>
  <!-- Thumbnail -->
  <div class="surface-media-frame aspect-square overflow-hidden">
    {#if asset.media_type === 'video'}
      <video
        src={asset.thumbnail_url || asset.url}
        class="w-full h-full object-cover"
        muted
        preload="none"
      ></video>
    {:else}
      <img
        src={asset.thumbnail_url || asset.url}
        alt={asset.prompt || asset.filename}
        class="w-full h-full object-cover"
        loading="lazy"
      />
    {/if}
  </div>

  <!-- Selection checkbox (top-left) -->
  <div class="absolute top-2 left-2 z-10">
    <input
      type="checkbox"
      checked={selected}
      class="surface-checkbox h-4 w-4 cursor-pointer rounded focus-visible:focus-ring"
      aria-label="Select {asset.filename}"
      onclick={(e) => e.stopPropagation()}
      onchange={(e) => onselect?.(asset, (e.currentTarget as HTMLInputElement).checked)}
    />
  </div>

  <!-- Hover overlay with action buttons -->
  {#if hovered}
    <div class="absolute inset-0 bg-black/60 flex items-center justify-center gap-2 z-10 pointer-events-none">
      <!-- View / Fullscreen -->
      <button
        type="button"
        onclick={(e) => { e.stopPropagation(); onopenlightbox?.(asset); }}
        class="surface-overlay-action pointer-events-auto rounded-md p-2 focus-visible:focus-ring"
        title="View fullscreen"
        aria-label="View fullscreen"
      >
        <svg class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2" aria-hidden="true">
          <path stroke-linecap="round" stroke-linejoin="round" d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5v-4m0 4h-4m4 0l-5-5" />
        </svg>
      </button>
      <!-- Download -->
      <a
        href={asset.url}
        download={asset.filename}
        onclick={(e) => { e.stopPropagation(); ondownload?.(asset); }}
        class="surface-overlay-action pointer-events-auto rounded-md p-2 focus-visible:focus-ring"
        title="Download"
        aria-label="Download {asset.filename}"
      >
        <svg class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2" aria-hidden="true">
          <path stroke-linecap="round" stroke-linejoin="round" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
        </svg>
      </a>
      <!-- Reuse in workspace -->
      <a
        href={asset.reuse_workspace_url}
        onclick={(e) => { e.stopPropagation(); onreuse?.(asset); }}
        class="surface-overlay-action-primary pointer-events-auto rounded-md p-2 focus-visible:focus-ring"
        title="Reuse in workspace"
        aria-label="Reuse settings in workspace"
      >
        <svg class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2" aria-hidden="true">
          <path stroke-linecap="round" stroke-linejoin="round" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
        </svg>
      </a>
      <!-- Delete -->
      <button
        type="button"
        onclick={(e) => { e.stopPropagation(); ondelete?.(asset); }}
        class="surface-overlay-action-danger pointer-events-auto rounded-md p-2 focus-visible:focus-ring"
        title="Delete"
        aria-label="Delete {asset.filename}"
      >
        <svg class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2" aria-hidden="true">
          <path stroke-linecap="round" stroke-linejoin="round" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
        </svg>
      </button>
    </div>
  {/if}

  <!-- Footer -->
  <div class="surface-footer-strip px-3 py-2">
    <p class="text-xs text-zinc-300 truncate font-medium">{asset.filename}</p>
    <p class="text-xs text-zinc-500 truncate mt-0.5">
      {asset.workflow} &middot; {new Date(asset.created_at).toLocaleDateString()}
    </p>
  </div>
</div>
