<script lang="ts">
  import { tick } from 'svelte';
  import { on } from 'svelte/events';
  import { Icon, Spinner, Waveform } from '$lib/components/atoms';
  import type { IconName } from '$lib/components/atoms/Icon.svelte';
  import type { AssetActionHandlers, UseAsState } from '$lib/state/assetActions';
  import { audioFoot, hasNoWords, tileWords } from '$lib/state/audioAsset';
  import { audioPeaks } from '$lib/state/audioPeaks.svelte';
  import { formatClock } from '$lib/state/clock';
  import type { FramePoint } from '$lib/state/framePoint';
  import { busyFor } from '$lib/state/frameGrab.svelte';
  import type { GalleryAsset } from '$lib/types';
  import type { MenuLead } from '$lib/state/assetActions';
  import { hold } from '$lib/actions/hold';
  import ActionMenu from './ActionMenu.svelte';
  import MediaPlayer from './MediaPlayer.svelte';
  import { assetMenuEntries } from './assetMenu';
  import { frameMenuEntries } from './frameMenu';
  import MoveMenu from './MoveMenu.svelte';

  interface Props extends AssetActionHandlers {
    asset: GalleryAsset;
    /** `compact`: filmstrip thumbnail. `card`: gallery card with footer. `stage`: full image, letterboxed. */
    density?: 'compact' | 'card' | 'stage';
    selected?: boolean;
    /** Select mode is on: a click on the media toggles the tile instead of opening the viewer, and the checkbox shows on every tile. */
    selectMode?: boolean;
    /** Shows a selection checkbox when provided; ticking it toggles the tile. */
    onselect?: (asset: GalleryAsset, selected: boolean) => void;
    /** A click on the media in Select mode, with its event so the owner can tell a Shift-click. */
    onmediaclick?: (asset: GalleryAsset, event: MouseEvent) => void;
    /** Enter Select mode with this tile selected: the checkbox ticked outside the mode, and the menu's Select. */
    onstartselect?: (asset: GalleryAsset) => void;
    deleting?: boolean;
    /** What the Use as items show: which targets are disabled, and their hints. */
    useAs?: UseAsState;
    eager?: boolean;
    /** Card density: lead the meta line with the asset's project, when it has one, in place of the workflow word. */
    showProject?: boolean;
    onmediaload?: () => void;
    class?: string;
  }

  let {
    asset,
    density = 'card',
    selected = false,
    selectMode = false,
    onselect,
    onmediaclick,
    onstartselect,
    deleting = false,
    useAs = {},
    eager = false,
    showProject = false,
    onmediaload,
    onpreview,
    onreuse,
    onreference,
    onextend,
    onuseframe,
    onedit,
    onupscale,
    onuseaudio,
    ondelete,
    onmove,
    onstar,
    class: extraClass = '',
  }: Props = $props();

  let menuOpen = $state(false);
  // The Move list opens after the ⋯ menu has closed, anchored at the ⋯ button, which is there on every tile.
  let moveOpen = $state(false);
  let moreButton = $state<HTMLButtonElement | null>(null);
  let mediaButton = $state<HTMLButtonElement | null>(null);

  const isVideo = $derived(asset.media_type === 'video');
  const isAudio = $derived(asset.media_type === 'audio');
  // A take plays in the stage's panel; in a grid or strip it is a cover that opens the viewer, never a player.
  const audioPlayer = $derived(isAudio && density === 'stage');
  const words = $derived(isAudio ? tileWords(asset) : '');
  // The listing carries no waveform; the tile asks for it once, and the server keeps it.
  const peaks = $derived(audioPeaks.of(asset));
  $effect(() => {
    if (isAudio) audioPeaks.ensure(asset);
  });
  // A tile with nothing to read, and every strip tile, shows only its icon and waveform.
  const bare = $derived(isAudio && (density === 'compact' || hasNoWords(asset)));
  const kindIcon = $derived<IconName>(asset.kind === 'speech' ? 'speech' : asset.kind === 'sound' ? 'waveform' : 'music');
  const length = $derived(asset.duration != null ? formatClock(asset.duration) : null);
  const transparent = $derived(asset.transparent === true && !isVideo);
  const canReuse = $derived(asset.has_reusable_config === true);
  // The frame the stage's video is paused on, which its player reports; the time chip's menu and the ⋯ menu offer it as a first or last frame and extend from it.
  let point = $state<FramePoint | null>(null);
  const extracting = $derived(busyFor(asset.id) !== null);
  const frameActions = $derived(point ? frameMenuEntries(asset, point, { onextend, onuseframe }, useAs) : []);
  // Only a tile that can start Select mode has a lead: Select outside the mode, View inside it.
  const lead = $derived(onstartselect ? (selectMode ? 'view' : 'select') : null);
  const starred = $derived(asset.starred === true);
  const menuItems = $derived(assetMenuEntries(asset, { onreuse, onreference, onextend, onedit, onuseframe, onupscale, onuseaudio, ondelete, onstar, onlead: onLead }, deleting, useAs, onmove ? () => { moveOpen = true; } : undefined, point, lead));
  // Stage videos play inline, so previewing goes through the expand action instead of the media.
  const inlineVideo = $derived(density === 'stage' && isVideo);
  const thumbnail = $derived(asset.thumbnail_url || asset.url);
  // A thumbnail that failed to load: an image falls back to the file itself, a video to an icon.
  let failedThumbnail = $state<string | null>(null);
  const thumbnailFailed = $derived(failedThumbnail === thumbnail);
  // The stage shows the full-resolution file; small tiles use the thumbnail.
  const src = $derived(density === 'stage' || thumbnailFailed ? asset.url : thumbnail);

  function onImageError(): void {
    if (src === asset.url) onmediaload?.();
    else failedThumbnail = thumbnail;
  }

  function onVideoStillError(): void {
    failedThumbnail = thumbnail;
    onmediaload?.();
  }

  // A player has no image to wait for: the stage may show it as soon as it is there.
  $effect(() => {
    if (audioPlayer) onmediaload?.();
  });

  function preview(trigger: HTMLElement): void {
    onpreview?.(asset, trigger);
  }

  function onLead(picked: MenuLead): void {
    if (picked === 'view') {
      if (moreButton) preview(moreButton);
      return;
    }
    onstartselect?.(asset);
    // The menu's item had focus and goes with it: the tile's media takes it back.
    void tick().then(() => mediaButton?.focus());
  }

  // In Select mode the media toggles; outside it, it opens the viewer.
  function onMediaClick(event: MouseEvent & { currentTarget: HTMLButtonElement }): void {
    if (!selectMode) preview(event.currentTarget);
    else onmediaclick?.(asset, event);
  }

  // Enter always opens the viewer, in Select mode too, where a click would toggle.
  function onMediaKeydown(event: KeyboardEvent & { currentTarget: HTMLButtonElement }): void {
    if (event.key !== 'Enter' || event.shiftKey || event.metaKey || event.ctrlKey || event.altKey) return;
    event.preventDefault();
    preview(event.currentTarget);
  }

  // A Shift-click adds a range: the browser must not also select the text between the two tiles.
  function onMediaMousedown(event: MouseEvent): void {
    if (selectMode && event.shiftKey) event.preventDefault();
  }

  // Outside Select mode ticking the box starts it, with this tile selected.
  function onCheckChange(event: Event & { currentTarget: HTMLInputElement }): void {
    if (!selectMode && onstartselect) onstartselect(asset);
    else onselect?.(asset, event.currentTarget.checked);
  }

  // The star toggles the star alone: not the selection, and not the grid's Space and X, which select the tile.
  function onStarClick(event: MouseEvent): void {
    event.stopPropagation();
    onstar?.(asset, !starred);
  }

  // A native listener: the grid listens on the document, which a delegated handler would not stop.
  function keepSelectKeys(node: HTMLElement): { destroy: () => void } {
    const off = on(node, 'keydown', (event) => {
      if (event.key === ' ' || event.key.toLowerCase() === 'x') event.stopPropagation();
    });
    return { destroy: off };
  }

  // A right-click opens the menu at the pointer: a 1px anchor there stands in for the ⋯ button.
  let contextAt = $state<{ x: number; y: number } | null>(null);
  let contextAnchorEl = $state<HTMLSpanElement | null>(null);
  function onContextMenu(event: MouseEvent): void {
    // A finger's hold already prevented the browser's menu and opened ours. A video keeps the browser's own menu
    // (loop, picture in picture), and the tile's buttons their defaults, as the hold leaves them alone too.
    if (event.defaultPrevented || (event.target as Element).closest('video, .asset-tile-actions, .asset-tile-check, .asset-tile-star')) return;
    event.preventDefault();
    contextAt = { x: event.clientX, y: event.clientY };
    menuOpen = true;
  }
</script>

<!-- A finger's long press opens the same menu as ⋯; so does a right-click, at the pointer. -->
<article
  use:hold={{ onhold: () => { contextAt = null; menuOpen = true; }, ignore: '.asset-tile-actions, .asset-tile-check, .asset-tile-star, video, [data-own-drag]' }}
  oncontextmenu={onContextMenu}
  class="asset-tile {extraClass}"
  data-density={density}
  data-asset-id={asset.id}
  data-video={isVideo}
  data-audio={isAudio}
  data-transparent={transparent}
  data-selected={selected}
  data-select-mode={selectMode}
  data-menu-open={menuOpen || moveOpen}
  aria-busy={deleting}
  aria-label={asset.filename}
>
  {#snippet audioCover()}
    <span class="audio-cover" class:bare data-bare={bare} data-testid="audio-cover">
      <span class="audio-icon" aria-hidden="true"><Icon name={kindIcon} size={32} /></span>
      {#if !bare}<span class="audio-words" data-testid="audio-words"><span class="audio-text">{words}</span></span>{/if}
      <Waveform class="audio-line" {peaks} />
    </span>
    {#if length}<span class="media-tag audio-length" data-testid="audio-length">{length}</span>{/if}
  {/snippet}

  {#snippet media()}
    {#if isAudio}
      {@render audioCover()}
    {:else if inlineVideo}
      <!-- `loadeddata` does not bubble, so the tile listens for it on the way down to the player's video. -->
      <div class="player-host" onloadeddatacapture={onmediaload}>
        <!-- A new video is a new player, so the frame the last one was paused on is gone with it. -->
        {#key asset.id}
          <MediaPlayer kind="video" size="stage" naturalSize={asset.width && asset.height ? { width: asset.width, height: asset.height } : null} src={asset.url} label={asset.filename} poster={asset.thumbnail_url} filePath={asset.file_path} duration={asset.duration} muted actions={frameActions} onpoint={(next) => { point = next; }} />
        {/key}
      </div>
    {:else if isVideo && thumbnailFailed}
      <span class="video-fallback" data-testid="video-fallback"><Icon name="film" size={28} /></span>
    {:else if isVideo}
      <img
        src={thumbnail}
        alt={asset.prompt || asset.filename}
        class="asset-tile-img"
        loading={eager ? 'eager' : 'lazy'}
        onload={onmediaload}
        onerror={onVideoStillError}
      >
      <span class="play-badge" data-testid="play-badge" aria-hidden="true"><Icon name="play" size={14} /></span>
    {:else}
      <img
        {src}
        alt={asset.prompt || asset.filename}
        class="asset-tile-img"
        class:ui-checkerboard={transparent}
        data-transparent={transparent}
        loading={eager ? 'eager' : 'lazy'}
        onload={onmediaload}
        onerror={onImageError}
      >
    {/if}
  {/snippet}

  {#if audioPlayer}
    <div class="asset-tile-media audio-stage">
      {#key asset.url}
        <MediaPlayer kind="audio" src={asset.url} label={asset.filename} {peaks} duration={asset.duration} size="stage" />
      {/key}
    </div>
  {:else if inlineVideo}
    <div class="asset-tile-media">{@render media()}</div>
  {:else}
    <button
      type="button"
      bind:this={mediaButton}
      class="asset-tile-media"
      aria-label="{selectMode ? 'Select' : 'View'} {asset.filename}"
      aria-pressed={selectMode ? selected : undefined}
      onmousedown={onMediaMousedown}
      onclick={onMediaClick}
      onkeydown={onMediaKeydown}
    >{@render media()}</button>
  {/if}

  {#if onselect}
    <input
      type="checkbox"
      class="asset-tile-check ui-checkbox"
      checked={selected}
      aria-label="Select {asset.filename}"
      onchange={onCheckChange}
    >
  {/if}

  <!-- A strip tile has no caption and no room for a button: a starred file shows a mark, which is not a control. -->
  {#if density === 'compact' && starred}
    <span class="asset-tile-star-mark" role="img" aria-label="Starred" data-testid="starred-mark"><Icon name="star" size={12} solid /></span>
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
    {#if onstar && density === 'stage'}
      <button
        type="button"
        class="asset-tile-action asset-tile-star ui-media-action"
        data-starred={starred}
        title={starred ? 'Unstar' : 'Star'}
        aria-label="Star {asset.filename}"
        aria-pressed={starred}
        onclick={onStarClick}
        use:keepSelectKeys
      ><Icon name="star" size={14} solid={starred} /></button>
    {/if}
    <button
      type="button"
      bind:this={moreButton}
      class="asset-tile-action ui-media-action"
      title="More actions"
      aria-label="More actions for {asset.filename}"
      aria-haspopup="menu"
      aria-expanded={menuOpen}
      aria-busy={extracting && !menuOpen}
      onclick={() => { contextAt = null; menuOpen = !menuOpen; }}
    >{#if extracting && !menuOpen}<Spinner size="sm" />{:else}<Icon name="more" size={14} />{/if}</button>
  </div>

  {#if density === 'card'}
    <div class="asset-tile-foot ui-media-foot" class:has-star={onstar}>
      <div class="foot-text">
        <p class="truncate text-ui font-medium text-text-secondary">{asset.filename}</p>
        <p class="mt-0.5 truncate text-meta text-text-muted">{#if isAudio}{audioFoot(asset, new Date(asset.created_at).toLocaleDateString(), showProject)}{:else}{showProject && asset.project !== null ? asset.project : asset.workflow} &middot; {new Date(asset.created_at).toLocaleDateString()}{/if}</p>
      </div>
      {#if onstar}
        <button
          type="button"
          class="asset-tile-star"
          data-starred={starred}
          title={starred ? 'Unstar' : 'Star'}
          aria-label="Star {asset.filename}"
          aria-pressed={starred}
          onclick={onStarClick}
          use:keepSelectKeys
        ><Icon name="star" size={16} solid={starred} /></button>
      {/if}
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
  {#if onmove && moveOpen}
    <MoveMenu
      open={moveOpen}
      anchor={moreButton}
      align="end"
      current={asset.project}
      onmove={(project) => onmove(asset, project)}
      onclose={() => { moveOpen = false; }}
    />
  {/if}
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
  .player-host { width: 100%; height: 100%; }
  .asset-tile[data-density='card'] .asset-tile-media { flex: none; aspect-ratio: 1; }
  /* Never enlarge: small outputs stay at their own size, large ones shrink to fit. */
  .asset-tile[data-density='stage'] .asset-tile-img { object-fit: scale-down; }
  /* A transparent image carries its checkerboard on the <img>, so the <img> is sized to the picture: the same look as
     scale-down (at most its own size, centred) without painting the letterbox. */
  .asset-tile[data-density='stage'][data-transparent='true'] .asset-tile-media { display: flex; align-items: center; justify-content: center; }
  .asset-tile[data-density='stage'][data-transparent='true'] .asset-tile-img { width: auto; height: auto; max-width: 100%; max-height: 100%; object-fit: fill; }

  /* A video's still carries a play badge, drawn here so the picture stays a plain still. */
  .play-badge { position: absolute; right: 8px; bottom: 8px; display: grid; place-items: center; width: 28px; height: 28px; border-radius: 50%; background: color-mix(in srgb, var(--color-bg-base) 78%, transparent); color: var(--color-text-primary); pointer-events: none; }
  .asset-tile[data-density='compact'] .play-badge { right: 4px; bottom: 4px; width: 20px; height: 20px; }
  .video-fallback { display: grid; place-items: center; width: 100%; height: 100%; background: var(--color-bg-surface-hover); color: var(--color-text-muted); }

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
  /* A finger gets the touch checkbox size and a 44px target: the box sits in the top-left corner, so the extra reach only
     covers the corner of the media button, never its middle. */
  @media (pointer: coarse) {
    .asset-tile-check { width: 20px; height: 20px; }
    .asset-tile-check::before { content: ''; position: absolute; inset: -12px; }
  }
  /* The checkbox is a way into Select mode: with a mouse it shows on hover or focus (and while the mode is on), on touch
     always. Hidden it still takes focus, which shows it. */
  @media (hover: hover) {
    .asset-tile-check { opacity: 0; pointer-events: none; transition: opacity 0.12s ease; }
    .asset-tile:hover .asset-tile-check, .asset-tile:focus-within .asset-tile-check, .asset-tile[data-menu-open='true'] .asset-tile-check,
    .asset-tile[data-select-mode='true'] .asset-tile-check { opacity: 1; pointer-events: auto; }
  }

  /* In Select mode a click toggles: the cursor says so, and hovering a tile that is not selected previews the ring. */
  .asset-tile[data-select-mode='true'] .asset-tile-media { cursor: pointer; }
  @media (hover: hover) { .asset-tile[data-select-mode='true'][data-selected='false']:hover { box-shadow: 0 0 0 2px var(--color-primary-border); } }

  /* The star. In the caption row it sits at the right end, beside the filename and date; 28px (32px on touch) with a 44px
     hit area. With a mouse it shows on hover or focus and stays while starred; on touch only a starred one shows, and a
     tap unstars it. */
  .asset-tile-star { position: relative; }
  .asset-tile-foot.has-star { display: flex; align-items: center; gap: 6px; padding-right: 6px; }
  .foot-text { min-width: 0; flex: 1; }
  .asset-tile-foot > .asset-tile-star {
    display: grid; flex: none; place-items: center; width: 28px; height: 28px; border-radius: var(--radius-sm);
    color: var(--color-text-muted); opacity: 0; transition: opacity 0.12s ease, background-color 0.12s ease, color 0.12s ease;
  }
  .asset-tile-foot > .asset-tile-star:focus-visible { outline: 2px solid var(--color-primary-main); outline-offset: 1px; }
  .asset-tile:focus-within .asset-tile-foot > .asset-tile-star, .asset-tile[data-menu-open='true'] .asset-tile-foot > .asset-tile-star { opacity: 1; }
  /* Amber only for a starred star, and after the resting colour above so it wins at rest (the same specificity, so the
     order decides). Focus shows the star but leaves its colour alone. */
  .asset-tile-foot > .asset-tile-star[aria-pressed='true'] { opacity: 1; color: var(--color-star); }
  @media (hover: hover) {
    .asset-tile:hover .asset-tile-foot > .asset-tile-star { opacity: 1; }
    .asset-tile-foot > .asset-tile-star:hover { background: var(--color-bg-surface-hover); color: var(--color-text-primary); }
    .asset-tile-foot > .asset-tile-star[aria-pressed='true']:hover { color: var(--color-star); }
  }
  @media (hover: none) { .asset-tile-foot > .asset-tile-star:not([aria-pressed='true']) { display: none; } }
  @media (pointer: coarse) {
    .asset-tile-foot > .asset-tile-star::before { content: ''; position: absolute; inset: -6px; }
    .asset-tile-foot > .asset-tile-star { width: 32px; height: 32px; }
  }
  /* The stage keeps the star among its corner buttons, before ⋯. */
  .asset-tile-actions > .asset-tile-star[aria-pressed='true'] { color: var(--color-star); }
  /* A strip tile's starred mark: top-left, a mark and not a control. */
  .asset-tile-star-mark {
    position: absolute; top: 4px; left: 4px; z-index: 2; display: grid; place-items: center; width: 18px; height: 18px; border-radius: 50%;
    background: color-mix(in srgb, var(--color-bg-base) 78%, transparent); color: var(--color-star); pointer-events: none;
  }
  .asset-tile-context-anchor { position: fixed; width: 1px; height: 1px; pointer-events: none; }
  .asset-tile-foot { padding: 8px 12px; }
</style>
