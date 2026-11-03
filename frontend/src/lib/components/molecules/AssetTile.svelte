<script lang="ts">
  import { Icon, Waveform } from '$lib/components/atoms';
  import type { IconName } from '$lib/components/atoms/Icon.svelte';
  import type { AssetActionHandlers, UseAsState } from '$lib/state/assetActions';
  import { audioFoot, hasNoWords, lengthLabel, tileWords } from '$lib/state/audioAsset';
  import type { GalleryAsset } from '$lib/types';
  import { hold } from '$lib/actions/hold';
  import ActionMenu from './ActionMenu.svelte';
  import AudioPlayer from './AudioPlayer.svelte';
  import { assetMenuEntries } from './assetMenu';

  interface Props extends AssetActionHandlers {
    asset: GalleryAsset;
    /** `compact`: filmstrip thumbnail. `card`: gallery card with footer. `stage`: full image, letterboxed. */
    density?: 'compact' | 'card' | 'stage';
    selected?: boolean;
    /** Shows a selection checkbox when provided. */
    onselect?: (asset: GalleryAsset, selected: boolean) => void;
    deleting?: boolean;
    /** What the Use as items show: which targets are disabled, and their hints. */
    useAs?: UseAsState;
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
    useAs = {},
    eager = false,
    onmediaload,
    onpreview,
    onreuse,
    onreference,
    onextend,
    onupscale,
    onuseaudio,
    ondelete,
    class: extraClass = '',
  }: Props = $props();

  let menuOpen = $state(false);
  let moreButton = $state<HTMLButtonElement | null>(null);

  const isVideo = $derived(asset.media_type === 'video');
  const isAudio = $derived(asset.media_type === 'audio');
  // A take plays in the stage's panel; in a grid or strip it is a cover that opens the viewer, never a player.
  const audioPlayer = $derived(isAudio && density === 'stage');
  const words = $derived(isAudio ? tileWords(asset) : '');
  // A tile with nothing to read, and every strip tile, shows only its icon and waveform.
  const bare = $derived(isAudio && (density === 'compact' || hasNoWords(asset)));
  const kindIcon = $derived<IconName>(asset.kind === 'speech' ? 'speech' : asset.kind === 'sound' ? 'waveform' : 'music');
  const length = $derived(asset.duration != null ? lengthLabel(asset.duration) : null);
  const transparent = $derived(asset.transparent === true && !isVideo);
  const canReuse = $derived(asset.has_reusable_config === true);
  const menuItems = $derived(assetMenuEntries(asset, { onreuse, onreference, onextend, onupscale, onuseaudio, ondelete }, deleting, useAs));
  // Stage videos play inline, so previewing goes through the expand action instead of the media.
  const inlineVideo = $derived(density === 'stage' && isVideo);
  // The stage shows the full-resolution file; small tiles use the thumbnail.
  const src = $derived(density === 'stage' ? asset.url : (asset.thumbnail_url || asset.url));

  // A player has no image to wait for: the stage may show it as soon as it is there.
  $effect(() => {
    if (audioPlayer) onmediaload?.();
  });

  function preview(trigger: HTMLElement): void {
    onpreview?.(asset, trigger);
  }

  // A right-click opens the menu at the pointer: a 1px anchor there stands in for the ⋯ button.
  let contextAt = $state<{ x: number; y: number } | null>(null);
  let contextAnchorEl = $state<HTMLSpanElement | null>(null);
  function onContextMenu(event: MouseEvent): void {
    // A finger's hold already prevented the browser's menu and opened ours. A video keeps the browser's own menu
    // (loop, picture in picture), and the tile's buttons their defaults, as the hold leaves them alone too.
    if (event.defaultPrevented || (event.target as Element).closest('video, .asset-tile-actions, .asset-tile-check')) return;
    event.preventDefault();
    contextAt = { x: event.clientX, y: event.clientY };
    menuOpen = true;
  }
</script>

<!-- A finger's long press opens the same menu as ⋯; so does a right-click, at the pointer. -->
<article
  use:hold={{ onhold: () => { contextAt = null; menuOpen = true; }, ignore: '.asset-tile-actions, .asset-tile-check, video, [data-own-drag]' }}
  oncontextmenu={onContextMenu}
  class="asset-tile {extraClass}"
  data-density={density}
  data-asset-id={asset.id}
  data-video={isVideo}
  data-audio={isAudio}
  data-transparent={transparent}
  data-selected={selected}
  data-menu-open={menuOpen}
  aria-busy={deleting}
  aria-label={asset.filename}
>
  {#snippet audioCover()}
    <span class="audio-cover" class:bare data-bare={bare} data-testid="audio-cover">
      <span class="audio-icon" aria-hidden="true"><Icon name={kindIcon} size={32} /></span>
      {#if !bare}<span class="audio-words" data-testid="audio-words"><span class="audio-text">{words}</span></span>{/if}
      <Waveform class="audio-line" peaks={asset.peaks ?? []} />
    </span>
    {#if length}<span class="media-tag audio-length" data-testid="audio-length">{length}</span>{/if}
  {/snippet}

  {#snippet media()}
    {#if isAudio}
      {@render audioCover()}
    {:else if isVideo}
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
        class:ui-checkerboard={transparent}
        data-transparent={transparent}
        loading={eager ? 'eager' : 'lazy'}
        onload={onmediaload}
        onerror={onmediaload}
      >
    {/if}
  {/snippet}

  {#if audioPlayer}
    <div class="asset-tile-media audio-stage"><AudioPlayer src={asset.url} peaks={asset.peaks ?? []} filename={asset.filename} duration={asset.duration} size="stage" /></div>
  {:else if inlineVideo}
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
      class="asset-tile-check ui-checkbox"
      checked={selected}
      aria-label="Select {asset.filename}"
      onchange={(event) => onselect(asset, (event.currentTarget as HTMLInputElement).checked)}
    >
  {/if}

  <div class="asset-tile-actions">
    {#if onpreview && !audioPlayer}
      <button
        type="button"
        class="asset-tile-action asset-tile-expand ui-media-action"
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
      onclick={() => { contextAt = null; menuOpen = !menuOpen; }}
    ><Icon name="more" size={14} /></button>
  </div>

  {#if density === 'card'}
    <div class="asset-tile-foot ui-media-foot">
      <p class="truncate text-ui font-medium text-text-secondary">{asset.filename}</p>
      <p class="mt-0.5 truncate text-meta text-text-muted">{#if isAudio}{audioFoot(asset, new Date(asset.created_at).toLocaleDateString())}{:else}{asset.workflow} &middot; {new Date(asset.created_at).toLocaleDateString()}{/if}</p>
    </div>
  {/if}

  {#if contextAt}<span bind:this={contextAnchorEl} class="asset-tile-context-anchor" style:left="{contextAt.x}px" style:top="{contextAt.y}px" aria-hidden="true"></span>{/if}
  <ActionMenu
    open={menuOpen}
    anchor={contextAt ? contextAnchorEl : moreButton}
    items={menuItems}
    label="Actions for {asset.filename}"
    onclose={() => { menuOpen = false; contextAt = null; }}
  />
</article>

<style>
  .asset-tile {
    position: relative;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    isolation: isolate;
    border-radius: var(--radius-md);
    background: var(--color-bg-surface);
    transition: background-color 0.12s ease;
  }
  .asset-tile:focus-within { background: var(--color-bg-surface-hover); }
  @media (hover: hover) { .asset-tile:hover { background: var(--color-bg-surface-hover); } }
  /* The selected tile keeps an outline: it carries meaning. */
  .asset-tile[data-selected='true'] { box-shadow: 0 0 0 2px var(--color-primary-main); }
  .asset-tile[aria-busy='true'] { opacity: 0.6; }
  /* A long press sinks the tile as it fills; no callout or selection on the image meanwhile. */
  .asset-tile { -webkit-touch-callout: none; transition: background-color 0.12s ease, scale 0.2s ease; }
  .asset-tile:global([data-holding]) { scale: 0.96; transition: scale var(--hold-ms, 480ms) ease-out; }
  .asset-tile-img { -webkit-user-select: none; user-select: none; }
  .asset-tile[data-density='stage'] { background: transparent; }

  .asset-tile-media { position: relative; display: block; flex: 1; min-height: 0; width: 100%; padding: 0; cursor: zoom-in; }
  .asset-tile-media:focus-visible { outline: 2px solid var(--color-primary-main); outline-offset: -2px; }
  .asset-tile-img { display: block; width: 100%; height: 100%; object-fit: cover; }
  .asset-tile[data-density='card'] .asset-tile-media { flex: none; aspect-ratio: 1; }
  /* Never enlarge: small outputs stay at their own size, large ones shrink to fit. */
  .asset-tile[data-density='stage'] .asset-tile-img { object-fit: scale-down; }
  /* A transparent image carries its checkerboard on the <img>, so the <img> is sized to the picture: the same look as
     scale-down (at most its own size, centred) without painting the letterbox. */
  .asset-tile[data-density='stage'][data-transparent='true'] .asset-tile-media { display: flex; align-items: center; justify-content: center; }
  .asset-tile[data-density='stage'][data-transparent='true'] .asset-tile-img { width: auto; height: auto; max-width: 100%; max-height: 100%; object-fit: fill; }

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
  .asset-tile:focus-within .asset-tile-actions, .asset-tile[data-menu-open='true'] .asset-tile-actions { opacity: 1; pointer-events: auto; }
  @media (hover: hover) { .asset-tile:hover .asset-tile-actions { opacity: 1; pointer-events: auto; } }
  .asset-tile:focus-within::after, .asset-tile[data-menu-open='true']::after { opacity: 1; }
  @media (hover: hover) { .asset-tile:hover::after { opacity: 1; } }
  @media (hover: none) {
    .asset-tile-actions { opacity: 1; pointer-events: auto; }
    /* A grid or strip tile has room for one button under a finger: ⋯ stays (Reuse leads its menu), and a tap on
       the tile opens it fullscreen. The stage keeps its buttons; it is the screen. */
    .asset-tile[data-density='compact'] .asset-tile-action:not(:last-child),
    .asset-tile[data-density='card'] .asset-tile-action:not(:last-child) { display: none; }
    .asset-tile[data-density='stage'][data-video='false'] .asset-tile-expand { display: none; }
    .asset-tile[data-density='stage'] .asset-tile-actions { top: 8px; right: 8px; gap: 6px; }
    .asset-tile[data-density='stage'] .asset-tile-action { width: 36px; height: 36px; box-shadow: 0 1px 6px var(--color-scrim); }
  }

  .asset-tile-action { display: grid; place-items: center; width: 28px; height: 28px; border-radius: var(--radius-sm); }
  .asset-tile-action:focus-visible { outline: 2px solid var(--color-primary-main); outline-offset: 1px; }
  .asset-tile-action:disabled { opacity: 0.5; cursor: not-allowed; }
  .asset-tile[data-density='compact'] .asset-tile-actions { top: 4px; right: 4px; gap: 3px; }
  .asset-tile[data-density='compact'] .asset-tile-action { width: 24px; height: 24px; border-radius: var(--radius-sm); }

  /* Audio. The stage shows the take's player panel (max 880px), centred, with the corner actions at 10px of the panel. */
  .asset-tile[data-audio='true'][data-density='stage'] { flex: none; width: 100%; max-width: 880px; align-self: center; }
  .asset-tile[data-audio='true'][data-density='stage'] .asset-tile-media { flex: none; cursor: default; }
  .asset-tile[data-audio='true'][data-density='stage'] .asset-tile-actions { top: 10px; right: 10px; gap: 6px; opacity: 1; pointer-events: auto; }
  .audio-stage { display: block; }

  /* A grid tile (D4): the kind icon, the first lines of the words, then a thin waveform line pinned to the foot. */
  .audio-cover { display: flex; flex-direction: column; align-items: stretch; gap: 8px; width: 100%; height: 100%; padding: 12px 12px 10px; background: var(--color-bg-surface-hover); text-align: left; }
  .audio-icon { display: flex; flex: none; justify-content: center; color: var(--color-text-muted); }
  .audio-words { display: block; flex: 0 1 auto; min-height: 0; overflow: hidden; }
  /* The clamp sits on this inner block, not on the flex item, so the line count holds. */
  .audio-text { display: -webkit-box; overflow: hidden; -webkit-box-orient: vertical; -webkit-line-clamp: 4; line-clamp: 4; white-space: pre-line; overflow-wrap: anywhere; font-size: var(--text-ui); line-height: 1.45; color: var(--color-text-secondary); }
  .audio-cover :global(.audio-line) { flex: none; height: 16px; margin-top: auto; }
  /* Nothing to read, or a strip tile: the icon and a waveform, centred together. */
  .audio-cover.bare { justify-content: center; gap: 18px; }
  .audio-cover.bare .audio-icon :global(svg) { width: 44px; height: 44px; }
  .audio-cover.bare :global(.audio-line) { height: 28px; margin-top: 0; }
  .audio-length { position: absolute; top: 6px; right: 6px; z-index: 1; }
  .asset-tile[data-density='compact'] .audio-cover { gap: 10px; padding: 8px; }
  .asset-tile[data-density='compact'] .audio-cover.bare .audio-icon :global(svg) { width: 24px; height: 24px; }
  .asset-tile[data-density='compact'] .audio-cover.bare :global(.audio-line) { height: 20px; }
  .asset-tile[data-density='compact'] .audio-length { top: auto; bottom: 4px; right: 4px; height: 18px; line-height: 18px; font-size: 10px; }
  /* The tile's buttons take the corner while they show: the length steps aside (a finger's ⋯ never leaves, so it moves over). */
  @media (hover: hover) {
    .asset-tile[data-density='card']:hover .audio-length, .asset-tile[data-density='card']:focus-within .audio-length, .asset-tile[data-density='card'][data-menu-open='true'] .audio-length { opacity: 0; }
  }
  @media (hover: none) { .asset-tile[data-density='card'] .audio-length { right: 40px; } }
  @media (max-width: 639px) {
    .audio-cover { gap: 6px; padding: 10px 10px 8px; }
    .audio-icon :global(svg) { width: 28px; height: 28px; }
    .audio-text { -webkit-line-clamp: 3; line-clamp: 3; }
  }

  .asset-tile-check { position: absolute; top: 8px; left: 8px; z-index: 2; width: 16px; height: 16px; border-radius: 4px; }
  .asset-tile-context-anchor { position: fixed; width: 1px; height: 1px; pointer-events: none; }
  .asset-tile-foot { padding: 8px 12px; }
</style>
