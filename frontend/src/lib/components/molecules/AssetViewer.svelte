<script lang="ts" module>
  const DETAILS_KEY = 'ziv-viewer-details-v1';

  function loadDetailsOpen(): boolean {
    try {
      return localStorage.getItem(DETAILS_KEY) === 'true';
    } catch {
      return false;
    }
  }

  function saveDetailsOpen(open: boolean): void {
    try {
      localStorage.setItem(DETAILS_KEY, String(open));
    } catch {
      // storage unavailable — keep the in-memory choice only
    }
  }
</script>

<script lang="ts">
  import { FileName, Icon, ShortcutList, Waveform } from '$lib/components/atoms';
  import AdjustEditor from '$lib/components/organisms/adjust/AdjustEditor.svelte';
  import { inMenu, isPlainKey, isTyping } from '$lib/keyboard';
  import { canAdjust } from '$lib/state/adjustState';
  import { describeFallbackReason, offeredUpscaleFactors, plainUpscaleMethod, upscaleOption, type AssetActionHandlers, type UseAsState } from '$lib/state/assetActions';
  import { wordBlocks } from '$lib/state/audioAsset';
  import { audioPeaks } from '$lib/state/audioPeaks.svelte';
  import type { FramePoint } from '$lib/state/framePoint';
  import { addToast, explainOff } from '$lib/state/toasts.svelte';
  import type { AudioModelOption, AudioSettingsVocabulary, GalleryAsset, UpscaleFactor } from '$lib/types';
  import AssetPanelMenu from './AssetPanelMenu.svelte';
  import { assetDetailSections, fileName, type DetailFact } from './assetDetails';
  import { assetPanel, type PanelItem } from './assetMenu';
  import ViewerButton from './ViewerButton.svelte';
  import { chromeFor, phoneActs, viewerActs, type ViewerAct } from './viewerActions';
  import MoveMenu from './MoveMenu.svelte';
  import { frameMenuEntries } from './frameMenu';
  import MediaPlayer from './MediaPlayer.svelte';
  import { viewerShortcutsFor, stepUpscaleChord, viewerActionFor, type ViewerAction } from './viewerShortcuts';
  import { layout } from '$lib/state/layoutMode.svelte';
  import { sheetDrag } from '$lib/actions/sheetDrag';
  import { dockHeight } from '$lib/actions/dockHeight';
  import { pushEscapeLayer } from '$lib/actions/escapeStack';
  import { tick, untrack } from 'svelte';
  import { FLIGHT_EASING, FLIGHT_MS, fitRect, flightKeyframes, rectOf, usableRect, type Rect } from './viewerFlight';
  import { pagerOutcome, pullBackdropOpacity, pullOutcome, pullTransform, type PagerOutcome } from './viewerPager';
  import { lockAxis, releaseVelocity, rubberBand, type PanAxis, type PanSample } from '$lib/actions/pan';
  import { hold } from '$lib/actions/hold';
  import { FIT, actualPixelsScale, clampPan, fitOffset, settleZoom, stageReach, zoomAt, type Cover, type Zoom } from './viewerZoom';

  interface Props extends Omit<AssetActionHandlers, 'onpreview' | 'onadjust'> {
    assets: readonly GalleryAsset[];
    /** The id of the open asset: the viewer follows it, so assets landing before or after it never move the view. */
    currentId: string | null;
    open: boolean;
    /** Where these assets come from, e.g. "History"; shown under the filename. */
    setLabel?: string;
    deletingIds?: ReadonlySet<string>;
    /** What the Use as items show: which targets are disabled, and their hints. */
    useAs?: UseAsState;
    /** The audio models this machine runs, which name the model of a take; without them the id the file records shows. */
    audioModels?: readonly AudioModelOption[];
    /** The vocabulary of the audio settings, which names a take's key, delivery and sound type. */
    audioVocabulary?: AudioSettingsVocabulary;
    /** Called when the viewer nears the end of the list, so a paged source can load more. */
    onnearend?: () => void;
    onclose: () => void;
    onnavigate: (index: number) => void;
    /** Opens with the Adjust editor already on the shown image (a tile's Adjust item); read when the viewer opens. */
    adjustOnOpen?: boolean;
    /** Called after the editor saved: the saved asset, whether it replaced the original, and the id of the asset that was adjusted. */
    onadjusted?: (saved: GalleryAsset, replaced: boolean, sourceId: string) => void;
    /** The tile that opened the viewer: the image grows out of it and shrinks back into it. */
    origin?: HTMLElement | null;
  }

  let {
    assets,
    currentId,
    open,
    setLabel = '',
    deletingIds = new Set<string>(),
    useAs = {},
    audioModels,
    audioVocabulary,
    onnearend,
    onclose,
    onnavigate,
    adjustOnOpen = false,
    onadjusted,
    origin = null,
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
  }: Props = $props();

  // The open asset is found by id. When it leaves the list (a delete), the viewer stays on the place it had, which is
  // its nearest remaining neighbour, and tells the parent so the id follows.
  let lastIndex = 0;
  const foundIndex = $derived(currentId === null ? -1 : assets.findIndex((item) => item.id === currentId));
  const index = $derived(foundIndex >= 0 ? foundIndex : Math.min(lastIndex, assets.length - 1));
  const asset = $derived(index >= 0 ? assets[index] : null);
  const hasPrev = $derived(index > 0);
  const hasNext = $derived(index < assets.length - 1);
  const deleting = $derived(asset ? deletingIds.has(asset.id) : false);
  const fallbackReasons = $derived(asset?.reuse_state?.fallback_reasons ?? []);
  let filmEl = $state<HTMLDivElement | null>(null);
  const NEAR_END = 3;

  $effect(() => {
    if (open && assets.length > 0 && index >= assets.length - NEAR_END) onnearend?.();
  });

  $effect(() => {
    if (foundIndex >= 0) lastIndex = foundIndex;
    else if (open && currentId !== null && assets.length > 0) onnavigate(index);
  });

  // Keep the current thumbnail centred in the strip as the viewer moves.
  $effect(() => {
    if (!open || !filmEl) return;
    const current = filmEl.querySelector<HTMLElement>(`[data-film-index="${index}"]`);
    current?.scrollIntoView?.({ block: 'nearest', inline: 'center', behavior: 'smooth' });
  });

  let detailsOpen = $state(loadDetailsOpen());
  // The one popup open over the viewer, and the element it opens from, taken when it opens: a button's, the one a key names, or none.
  // Use as and Upscale open the panel's own choices, More the whole panel, Move the project list.
  let popup = $state<'useas' | 'upscale' | 'move' | 'more' | null>(null);
  let popupAnchor = $state<HTMLElement | null>(null);
  let viewerEl = $state<HTMLElement | null>(null);
  let closeButton = $state<HTMLButtonElement | null>(null);
  let player = $state<ReturnType<typeof MediaPlayer> | null>(null);
  let helpOpen = $state(false);
  // The shortcut list is a layer of its own: Escape closes it before the viewer.
  $effect(() => {
    if (!helpOpen) return;
    return pushEscapeLayer(() => { helpOpen = false; });
  });

  const isAudio = $derived(asset?.media_type === 'audio');
  // The listing carries no waveform; the shown take asks for it once (the strip and the neighbours draw what is already fetched).
  $effect(() => {
    if (asset && isAudio) audioPeaks.ensure(asset);
  });
  // The file panel's own list of actions: the toolbar, the menus and the keys all read it. Extend means from the end here, as the E key does.
  const panelHandlers = $derived({ onreuse, onreference, onextend, onedit, onadjust: () => void startAdjust(), onupscale, onuseaudio, ondelete, onstar });
  const panel = $derived(asset ? assetPanel(asset, panelHandlers, { deleting, useAs, openMove: onmove ? () => openPopup('move', buttonFor('move') ?? buttonFor('more')) : undefined }) : null);
  // Use as: an image's start image, reference, first and last frame; a speech take's voice and saved voice.
  const canUseAs = $derived(panel?.sections.some((section) => section.id === 'use-as') ?? false);
  /** One of the panel's actions, which the toolbar, the dock, the menus and the keys all run: the viewer keeps no state of its own about what is offered. */
  const panelItem = (id: string) => [...(panel?.make ?? []), ...(panel?.file ?? [])].find((item) => item.id === id) ?? null;
  const words = $derived(asset && isAudio ? wordBlocks(asset) : []);

  /** The keys that follow X for the factors this asset offers: "2 or 4" for an image, "2" for a video. */
  const upscaleKeys = $derived(asset ? offeredUpscaleFactors(asset).join(' or ') : '');

  // The Adjust editor is a layer of its own over the viewer. While it is open the viewer has no escape layer and no key
  // listeners, and its panel is inert, so no key, tap or gesture reaches the viewer's actions.
  let adjusting = $state(false);
  // The asset the editor was opened on: the editor closes when the viewer moves to another (it left the list).
  let adjustedId: string | null = null;
  // Set while the editor is open: when it closes, focus goes back to where Adjust is.
  let returnFocus = false;

  /** Open the Adjust editor on the shown image; false for anything else. */
  function startAdjust(): boolean {
    if (!asset || !canAdjust(asset)) return false;
    adjustedId = asset.id;
    returnFocus = true;
    helpOpen = false;
    adjusting = true;
    return true;
  }

  // The editor starts open when the viewer opens on a tile's Adjust, and never outlives the viewer.
  $effect(() => {
    if (!open) {
      adjusting = false;
      return;
    }
    untrack(() => {
      if (adjustOnOpen) startAdjust();
    });
  });

  $effect(() => {
    const id = asset?.id ?? null;
    if (untrack(() => adjusting) && id !== adjustedId) adjusting = false;
  });

  // The toolbar and the dock list the same actions, from the panel; the pointer decides which of them is drawn.
  const acts = $derived(asset && panel ? viewerActs({ asset, panel, upscaleKeys }) : null);
  const chrome = $derived(chromeFor(layout));
  const phoneSlots = $derived(acts ? phoneActs(acts) : []);

  // The frame the shown video is paused on, which the player reports; null while it plays, at its start, or for anything but a video.
  let point = $state<FramePoint | null>(null);
  // The rows of the paused frame's menu: Extend from there, and the first and last frame.
  const frameActions = $derived(asset && point ? frameMenuEntries(asset, point, { onextend, onuseframe }, useAs) : []);

  // A phone: More opens the file panel as a sheet, the details open as a sheet, and a swipe moves between assets or closes.
  const compact = $derived(layout.compact);

  // --- Touch and zoom ---
  // A finger, at any width: a swipe moves between assets or (down) closes, a pinch or a double tap zooms an image,
  // and a zoomed image pans under one finger. A mouse click shows an image at its real pixels; a drag then pans it.
  const touchLayer = $derived(compact || layout.coarse);
  let stageEl = $state<HTMLDivElement | null>(null);
  // The stage's whole width, padding included.
  let stageWidth = $state(0);
  let barHeight = $state(0);
  let footHeight = $state(0);
  let dockWidth = $state(0);
  // The stage runs under the bar, the foot and a phone's side rail, so a zoomed image fills the screen once a tap hides them;
  // its padding keeps the fitted image clear of them. A phone held sideways lays its bar over the picture on a scrim, so only its rail covers an edge.
  const cover = $derived<Cover>({ top: chrome === 'rail' ? 0 : barHeight, right: chrome === 'rail' ? dockWidth : 0, bottom: footHeight });
  let mediaEl = $state<HTMLImageElement | null>(null);
  let zoom = $state<Zoom>(FIT);
  let zoomAnimated = $state(false);
  const zoomed = $derived(zoom.scale > 1);
  const isImage = $derived(asset?.media_type === 'image');
  // Real pixels for the shown image; null until it has loaded, or when it already shows at full size.
  let actualScale = $state<number | null>(null);

  $effect(() => {
    void asset?.id;
    zoom = FIT;
    actualScale = null;
  });

  // --- Open and close from the tile ---
  // The image grows out of the tile that opened it and shrinks back into it: a copy of its thumbnail flies, fixed over
  // everything, between the tile's box and the image's box while the viewer itself fades. Until the full image has
  // loaded, the flown thumbnail stands in for it.
  type Flight = { rect: Rect; src: string; transparent: boolean };
  const reduceMotion = () => typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches;
  let flight = $state<Flight | null>(null);
  let flightEl = $state<HTMLImageElement | null>(null);
  let phase = $state<'opening' | 'open' | 'closing'>('open');
  let mediaLoaded = $state(false);

  const viewportSize = () => ({ width: window.innerWidth, height: window.innerHeight });

  /** The tile's box, while it is still on screen. */
  function originRect(): Rect | null {
    if (!origin?.isConnected) return null;
    const rect = rectOf(origin.getBoundingClientRect());
    return usableRect(rect, viewportSize()) ? rect : null;
  }

  /** The box of the shown asset's own tile, in the grid or strip the viewer opened from; null when it is off screen. */
  function shownTileRect(): Rect | null {
    if (!origin?.isConnected || !asset) return null;
    const scope = origin.closest('[role="list"], .latest-region, #gallery-view') ?? document;
    const id = asset.id.replace(/["\\]/g, '\\$&');
    const tile = scope.querySelector<HTMLElement>(`.asset-tile[data-asset-id="${id}"] .asset-tile-media`);
    if (!tile) return null;
    const rect = rectOf(tile.getBoundingClientRect());
    return usableRect(rect, viewportSize()) ? rect : null;
  }

  /** The shown media's box: the loaded image, else the thumbnail standing in for it. */
  function mediaRect(): Rect | null {
    const shown = stageEl?.querySelector(mediaLoaded ? '.viewer-media' : '.viewer-standin');
    const rect = shown ? rectOf(shown.getBoundingClientRect()) : null;
    return usableRect(rect, viewportSize()) ? rect : null;
  }

  /** The aspect the image will show at: the asset's own size, else the tile's media, else the tile. */
  function mediaAspect(): number {
    if (asset?.width && asset.height) return asset.width / asset.height;
    const media = origin?.querySelector<HTMLImageElement | HTMLVideoElement>('img, video');
    const width = media instanceof HTMLVideoElement ? media.videoWidth : media?.naturalWidth;
    const height = media instanceof HTMLVideoElement ? media.videoHeight : media?.naturalHeight;
    if (width && height) return width / height;
    const box = originRect();
    return box ? box.width / box.height : 1;
  }

  /** Where the fitted media lands: the stage's content box. */
  function stageContentRect(): Rect | null {
    if (!stageEl) return null;
    const box = stageEl.getBoundingClientRect();
    const style = getComputedStyle(stageEl);
    const [top, right, bottom, left] = [style.paddingTop, style.paddingRight, style.paddingBottom, style.paddingLeft].map((value) => parseFloat(value) || 0);
    const rect = { left: box.left + left, top: box.top + top, width: box.width - left - right, height: box.height - top - bottom };
    return usableRect(rect, viewportSize()) ? rect : null;
  }

  /** Fly the thumbnail from one box to another, then call `done` (at once where the browser cannot animate). */
  function fly(from: Rect, to: Rect, shown: GalleryAsset, done: () => void): void {
    flight = { rect: to, src: shown.thumbnail_url || shown.url, transparent: isTransparent(shown) };
    void tick().then(() => {
      const el = flightEl;
      if (!el || typeof el.animate !== 'function') { done(); return; }
      const animation = el.animate(flightKeyframes(from, to), { duration: FLIGHT_MS, easing: FLIGHT_EASING, fill: 'both' });
      animation.onfinish = done;
      animation.oncancel = done;
    });
  }

  // Opening: the thumbnail flies from the tile to where the image will sit.
  $effect(() => {
    if (!open) return;
    untrack(() => {
      const shown = asset;
      // A video has no still to fly (its thumbnail is the video itself), so it fades in and out instead.
      const from = reduceMotion() || adjustOnOpen || shown?.media_type === 'video' || shown?.media_type === 'audio' ? null : originRect();
      const stage = from ? stageContentRect() : null;
      if (!shown || !from || !stage) { phase = 'open'; return; }
      phase = 'opening';
      fly(from, fitRect(stage, mediaAspect()), shown, () => {
        phase = 'open';
        flight = null;
      });
    });
    return () => { flight = null; phase = 'open'; chromeHidden = false; drag = null; helpOpen = false; popup = null; };
  });

  // The stand-in thumbnail shows until the image's own load event, so the flag resets only when the source changes: a new
  // asset object for the same file (a star, a move) fires no new load and must keep the full image shown.
  const mediaUrl = $derived(asset?.url);
  $effect(() => {
    void mediaUrl;
    mediaLoaded = false;
  });

  function onMediaLoad(): void {
    mediaLoaded = true;
    measureActual();
  }

  /** Close the viewer: the image shrinks back into its own tile first, while that tile is on screen. */
  function close(): void {
    if (phase === 'closing' || adjusting) return;
    const to = reduceMotion() || phase === 'opening' || asset?.media_type === 'video' || asset?.media_type === 'audio' ? null : shownTileRect();
    const from = to && asset ? mediaRect() : null;
    if (!to || !from || !asset) { onclose(); return; }
    phase = 'closing';
    fly(from, to, asset, onclose);
  }

  function measureActual(): void {
    if (!mediaEl) return;
    const scale = actualPixelsScale({ width: mediaEl.naturalWidth, height: mediaEl.naturalHeight }, { width: mediaEl.clientWidth, height: mediaEl.clientHeight });
    actualScale = scale > 1.05 ? scale : null;
  }

  /** A point on screen, from the fitted image's centre. */
  function fromCentre(clientX: number, clientY: number): { x: number; y: number } {
    const box = stageEl?.getBoundingClientRect();
    if (!box) return { x: 0, y: 0 };
    const offset = fitOffset(cover);
    return { x: clientX - box.left - box.width / 2 - offset.x, y: clientY - box.top - box.height / 2 - offset.y };
  }

  function setZoom(next: Zoom, animated: boolean): void {
    zoomAnimated = animated;
    zoom = mediaEl && stageEl
      ? clampPan(next, { width: mediaEl.clientWidth, height: mediaEl.clientHeight }, stageReach({ width: stageEl.clientWidth, height: stageEl.clientHeight }, cover, chromeHidden))
      : next;
  }

  // Hiding the controls slides a zoomed image out to fill the screen; when they come back they cover its edges, and it stays where it is.
  $effect(() => {
    void chromeHidden;
    untrack(() => { if (zoomed) setZoom(zoom, true); });
  });

  /** Toggle between the fitted image and a closer look at the point (real pixels, or 2× for a small image). */
  function toggleZoomAt(clientX: number, clientY: number, minimum: number): void {
    if (zoomed) { setZoom(FIT, true); return; }
    const scale = Math.max(minimum, actualScale ?? 1);
    if (scale <= 1) return;
    const point = fromCentre(clientX, clientY);
    setZoom(zoomAt(FIT, scale, point.x, point.y), true);
  }

  type Gesture =
    | { kind: 'swipe'; x: number; y: number; at: number; axis: PanAxis | null; samples: PanSample[] }
    | { kind: 'pan'; x: number; y: number; at: number; from: Zoom }
    | { kind: 'pinch'; distance: number; mid: { x: number; y: number }; from: Zoom };
  let gesture: Gesture | null = null;
  // The finger went down on a video's player, which answers a tap itself: the bar and the strip stay as they are.
  let startedOnPlayer = false;
  let lastTap: { x: number; y: number; at: number } | null = null;
  let tapTimer: ReturnType<typeof setTimeout> | undefined;
  const DOUBLE_TAP_MS = 300;
  const TAP_SLOP_PX = 10;
  /** A finger has moved this far before the drag's direction is known. */
  const AXIS_LOCK_PX = 8;
  const SETTLE_MS = 220;

  // The pager and the pull. Dragged sideways, the panes follow the finger and the neighbour slides in when it lets
  // go; dragged down, the image follows and shrinks, and lets go into its tile. A pull up opens the details.
  let drag = $state<{ axis: PanAxis; offset: number } | null>(null);
  let trackEl = $state<HTMLDivElement | null>(null);
  let paneEl = $state<HTMLDivElement | null>(null);
  let settling = $state(false);
  // A tap hides the bar and the strip for a clean look at the image; another brings them back.
  let chromeHidden = $state(false);
  // The stage's padding: the neighbours sit that much further out, so they stay clear of the stage's edges at any width.
  // A shown side rail covers its band, so the next file waits just behind it; a hidden one leaves the band open, so the next file waits past it.
  let trackWidth = $state(0);
  const paneGap = $derived(Math.max(0, stageWidth - trackWidth - (chromeHidden ? 0 : cover.right)));
  const paneWidth = () => (trackWidth || window.innerWidth) + paneGap;
  const thumbOf = (item: GalleryAsset) => item.thumbnail_url || item.url;
  /** Whether an image asset has a cut-out background, so its picture sits on a checkerboard. */
  const isTransparent = (item: GalleryAsset) => item.transparent === true && item.media_type !== 'video';

  /** Animate an element's transform, then call `done` (at once where the browser cannot animate). */
  function slide(el: HTMLElement | null, from: string, to: string, done: () => void): void {
    if (!el || typeof el.animate !== 'function' || reduceMotion()) { done(); return; }
    settling = true;
    const animation = el.animate([{ transform: from }, { transform: to }], { duration: SETTLE_MS, easing: FLIGHT_EASING, fill: 'both' });
    animation.oncancel = () => { settling = false; };
    animation.onfinish = () => {
      animation.oncancel = null;
      settling = false;
      done();
      animation.cancel();
    };
  }

  /** Slide the track to the neighbour, then step the index: the new asset is in place when the index changes. */
  function settlePager(outcome: PagerOutcome, from: number): void {
    const target = outcome === 'next' ? -paneWidth() : outcome === 'prev' ? paneWidth() : 0;
    slide(trackEl, `translateX(${from}px)`, `translateX(${target}px)`, () => {
      drag = null;
      if (outcome === 'next') onnavigate(index + 1);
      else if (outcome === 'prev') onnavigate(index - 1);
    });
  }

  function pinchOf(touches: TouchList): { distance: number; mid: { x: number; y: number } } {
    const [a, b] = [touches[0], touches[1]];
    return { distance: Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY), mid: fromCentre((a.clientX + b.clientX) / 2, (a.clientY + b.clientY) / 2) };
  }

  function onStageTouchStart(event: TouchEvent): void {
    // The player's own controls take the finger: scrubbing must not step to another asset. Its picture does not: a swipe over it pages.
    if ((event.target as Element).closest('.viewer-details-scrim, [data-own-drag]') || settling || startsInScrollableWords(event.target as Element)) { gesture = null; return; }
    startedOnPlayer = (event.target as Element).closest('[data-media-player]') !== null;
    if (event.touches.length === 2 && isImage) {
      gesture = { kind: 'pinch', ...pinchOf(event.touches), from: zoom };
      return;
    }
    const touch = event.touches[0];
    if (!touch || event.touches.length > 1) { gesture = null; return; }
    const at = Date.now();
    gesture = zoomed
      ? { kind: 'pan', x: touch.clientX, y: touch.clientY, at, from: zoom }
      : { kind: 'swipe', x: touch.clientX, y: touch.clientY, at, axis: null, samples: [{ x: touch.clientX, y: touch.clientY, at }] };
  }

  /** Whether a touch begins on the words under an audio player that scroll: the finger scrolls them, it does not page or pull. */
  function startsInScrollableWords(target: Element): boolean {
    const scroller = target.closest<HTMLElement>('.viewer-audio');
    return scroller !== null && scroller.scrollHeight > scroller.clientHeight;
  }

  function onStageTouchMove(event: TouchEvent): void {
    if (!gesture) return;
    if (gesture.kind === 'pinch' && event.touches.length === 2) {
      const now = pinchOf(event.touches);
      const scaled = zoomAt(gesture.from, gesture.from.scale * (now.distance / Math.max(gesture.distance, 1)), gesture.mid.x, gesture.mid.y);
      setZoom({ ...scaled, x: scaled.x + now.mid.x - gesture.mid.x, y: scaled.y + now.mid.y - gesture.mid.y }, false);
    } else if (gesture.kind === 'pan') {
      const touch = event.touches[0];
      setZoom({ ...gesture.from, x: gesture.from.x + touch.clientX - gesture.x, y: gesture.from.y + touch.clientY - gesture.y }, false);
    } else if (gesture.kind === 'swipe') {
      const touch = event.touches[0];
      if (!touch) return;
      const dx = touch.clientX - gesture.x;
      const dy = touch.clientY - gesture.y;
      gesture.samples.push({ x: touch.clientX, y: touch.clientY, at: Date.now() });
      if (gesture.axis === null && Math.hypot(dx, dy) >= AXIS_LOCK_PX) gesture.axis = lockAxis(dx, dy);
      if (gesture.axis === 'x') {
        // Past either end the panes give a little and no further.
        const offset = dx > 0 && !hasPrev ? rubberBand(dx, 48) : dx < 0 && !hasNext ? -rubberBand(-dx, 48) : dx;
        drag = { axis: 'x', offset };
      } else if (gesture.axis === 'y') {
        drag = { axis: 'y', offset: dy };
      }
    }
  }

  function onStageTouchEnd(event: TouchEvent): void {
    const current = gesture;
    if (current?.kind === 'pinch') {
      // One finger may stay down after a pinch; it starts nothing until it lifts.
      gesture = null;
      setZoom(settleZoom(zoom), true);
      lastTap = null;
      return;
    }
    gesture = null;
    const touch = event.changedTouches[0];
    if (!current || !touch) { drag = null; return; }
    const dx = touch.clientX - current.x;
    const dy = touch.clientY - current.y;
    const now = Date.now();
    if (Math.abs(dx) < TAP_SLOP_PX && Math.abs(dy) < TAP_SLOP_PX && now - current.at < DOUBLE_TAP_MS) {
      drag = null;
      if (!startedOnPlayer) onTap(touch.clientX, touch.clientY, now);
      // The player fills the pane, so a tap on its picture is the only way back to a bar and a strip a tap hid on an earlier asset.
      else if (chromeHidden) chromeHidden = false;
      return;
    }
    lastTap = null;
    if (current.kind !== 'swipe') return;
    if (current.axis === 'x') {
      settlePager(pagerOutcome(dx, releaseVelocity(current.samples, 'x'), paneWidth(), { prev: hasPrev, next: hasNext }), drag?.offset ?? dx);
    } else if (current.axis === 'y') {
      const outcome = pullOutcome(dy, releaseVelocity(current.samples, 'y'), compact && !detailsOpen);
      if (outcome === 'close') { close(); drag = null; return; }
      const from = pullTransform(drag?.offset ?? dy);
      drag = null;
      if (outcome === 'details') toggleDetails();
      else if (dy > 0) slide(paneEl, from, 'translate(0px, 0px) scale(1)', () => undefined);
    } else {
      drag = null;
    }
  }

  /** One tap shows or hides the bar and the strip; two taps in a row zoom an image instead. */
  function onTap(x: number, y: number, at: number): void {
    if (isImage && lastTap && at - lastTap.at < DOUBLE_TAP_MS && Math.hypot(x - lastTap.x, y - lastTap.y) < 40) {
      clearTimeout(tapTimer);
      lastTap = null;
      toggleZoomAt(x, y, 2);
      return;
    }
    lastTap = { x, y, at };
    clearTimeout(tapTimer);
    tapTimer = setTimeout(() => {
      lastTap = null;
      if (!detailsOpen) chromeHidden = !chromeHidden;
    }, DOUBLE_TAP_MS);
  }

  // A hold on the image, or a right-click, opens the same menu as the ⋯ button, at the pointer on a desktop.
  let contextAt = $state<{ x: number; y: number } | null>(null);
  let contextAnchorEl = $state<HTMLSpanElement | null>(null);
  // Where the finger went down: a hold opens the menu there on a tablet, where the menu is a popover, not a sheet.
  let pressAt: { x: number; y: number } | null = null;
  function onStageContextMenu(event: MouseEvent): void {
    if (event.defaultPrevented || !asset) return;
    event.preventDefault();
    contextAt = { x: event.clientX, y: event.clientY };
    openPopup('more', null);
  }

  // A mouse: a click toggles real pixels at the cursor; a drag pans a zoomed image.
  let mouseDrag: { x: number; y: number; from: Zoom; moved: boolean } | null = null;

  function onMediaPointerDown(event: PointerEvent): void {
    if (event.pointerType !== 'mouse' || event.button !== 0) return;
    event.preventDefault();
    mouseDrag = { x: event.clientX, y: event.clientY, from: zoom, moved: false };
    (event.currentTarget as Element).setPointerCapture?.(event.pointerId);
  }

  function onMediaPointerMove(event: PointerEvent): void {
    if (!mouseDrag) return;
    const dx = event.clientX - mouseDrag.x;
    const dy = event.clientY - mouseDrag.y;
    if (!mouseDrag.moved && Math.hypot(dx, dy) < 4) return;
    mouseDrag.moved = true;
    if (zoomed) setZoom({ ...mouseDrag.from, x: mouseDrag.from.x + dx, y: mouseDrag.from.y + dy }, false);
  }

  function onMediaPointerUp(event: PointerEvent): void {
    const drag = mouseDrag;
    mouseDrag = null;
    if (drag && !drag.moved) toggleZoomAt(event.clientX, event.clientY, 1);
  }

  const sections = $derived(asset ? assetDetailSections(asset, audioModels, audioVocabulary) : null);
  const source = $derived(asset?.details?.source ?? null);
  // A scene-only video carries its scene as its prompt; show it once, and a separate Scene section only when it differs.
  const sceneOnly = $derived(Boolean(asset?.scene) && asset?.scene === asset?.prompt);
  const separateScene = $derived(asset?.scene && asset.scene !== asset.prompt ? asset.scene : null);
  const sourceIndex = $derived(source?.id ? assets.findIndex((item) => item.id === source.id) : -1);

  /** Upscale the shown image by *factor* from the keyboard with the plain method; a disallowed factor explains why instead, and a warned size starts with its warning. */
  function runUpscale(factor: UpscaleFactor): boolean {
    if (!asset || !onupscale) return false;
    const method = plainUpscaleMethod(asset);
    const option = upscaleOption(asset, factor, method);
    if (!option) return false;
    if (!option.allowed) {
      addToast(option.reason ?? `Upscaling ${factor}× is not available for this image.`, 'warning');
      return true;
    }
    if (option.warning) addToast(option.warning, 'warning');
    onupscale(asset, factor, method);
    return true;
  }

  /**
   * Run one of the panel's items, which every button and key goes through. An item that is off does not run: a finger, a pen or a key
   * is told why in a toast; a mouse reads the tooltip. A key passes a pointer type that is not a mouse's.
   */
  function runItem(item: PanelItem, pointerType = 'keyboard'): void {
    if (item.disabled) {
      explainOff(item.title, pointerType);
      return;
    }
    item.onselect?.();
  }

  /** Run the panel item behind a key; false where the shown file doesn't offer it, so the key is left alone. */
  function runKey(id: string): boolean {
    const item = panelItem(id);
    if (!item) return false;
    runItem(item);
    return true;
  }

  /** Whether the popup an action opens is open, or undefined for an action that opens none. */
  function expandedOf(act: ViewerAct): boolean | undefined {
    return act.menu ? popup === act.menu : act.id === 'more' ? popup === 'more' : undefined;
  }

  /** The button of an action in the toolbar or the dock, by its `data-action`; null where the chrome has none (a phone's dock has no Move). */
  function buttonFor(action: string): HTMLElement | null {
    return viewerEl?.querySelector<HTMLElement>(`[data-action="${action}"]`) ?? null;
  }

  /** Open a popup from *anchor*, or close it when it is the one open. The panel's own right-click and hold open More with no anchor. */
  function openPopup(kind: NonNullable<typeof popup>, anchor: HTMLElement | null): void {
    if (kind !== 'more') contextAt = null;
    popup = popup === kind && anchor !== null ? null : kind;
    popupAnchor = anchor;
  }

  /** Close the popup *kind*, unless another has opened since. */
  function closePopup(kind: NonNullable<typeof popup>): void {
    if (popup !== kind) return;
    popup = null;
    if (kind === 'more') contextAt = null;
  }

  /** Run a toolbar or dock action: open its choice from its own button, or run its item. A link (Download) acts on its own. */
  function onAct(act: ViewerAct, event: MouseEvent & { currentTarget: HTMLElement }): void {
    const pointerType = (event as MouseEvent & { pointerType?: string }).pointerType ?? 'touch';
    if (act.menu) openPopup(act.menu, event.currentTarget);
    else if (act.id === 'more') openPopup('more', event.currentTarget);
    else runItem(act.item, pointerType);
  }

  function toggleDetails(): void {
    detailsOpen = !detailsOpen;
    saveDetailsOpen(detailsOpen);
  }

  $effect(() => {
    if (open && assets.length === 0) onclose();
  });

  /** Run a keyboard action on the shown asset; return whether it applied. */
  function runAction(action: ViewerAction): boolean {
    if (!asset) return false;
    switch (action) {
      case 'close':
        if (helpOpen) helpOpen = false;
        else if (zoomed) setZoom(FIT, true);
        else close();
        return true;
      case 'prev':
        if (hasPrev) onnavigate(index - 1);
        return hasPrev;
      case 'next':
        if (hasNext) onnavigate(index + 1);
        return hasNext;
      case 'first':
        if (hasPrev) onnavigate(0);
        return hasPrev;
      case 'last':
        if (hasNext) onnavigate(assets.length - 1);
        return hasNext;
      case 'reference':
        if (asset.media_type === 'video') return runKey('extend');
        if (asset.media_type === 'audio') return runKey('use-audio-track');
        if (!canUseAs) return false;
        openPopup('useas', buttonFor('use-as'));
        return true;
      case 'edit':
        return runKey('edit');
      case 'adjust':
        return runKey('adjust');
      case 'copy':
        return runKey('copy-prompt');
      case 'details':
        toggleDetails();
        return true;
      case 'help':
        helpOpen = !helpOpen;
        return true;
      case 'reuse':
        return runKey('reuse');
      case 'download':
        {
          const link = buttonFor('download');
          link?.click();
          return link !== null;
        }
      case 'star':
        if (!onstar) return false;
        onstar(asset, asset.starred !== true);
        return true;
      case 'delete':
      case 'delete-now':
        // Ignore key repeat while the first delete is still in flight.
        if (!ondelete || deleting) return false;
        ondelete(asset, { confirm: action === 'delete' });
        return true;
    }
  }

  $effect(() => {
    if (!open || adjusting) return;
    const back = returnFocus ? (buttonFor('adjust') ?? buttonFor('more')) : null;
    returnFocus = false;
    queueMicrotask(() => (back ?? closeButton)?.focus());
    // On the overlay stack like every layer: Escape closes it, innermost first (the shortcut list before it).
    const release = pushEscapeLayer(close);
    // When X began waiting for its 2 or 4. Only that completing key is the viewer's; every other digit reaches the player and jumps.
    let upscaleWaitingSince: number | null = null;
    /** Space on the viewer's Close, where a video or take opens, plays it instead of closing the viewer (the focus moves to the player, so the key's release does not click Close); Enter, Esc and a click still close it. */
    function playsFromClose(e: KeyboardEvent): boolean {
      return e.key === ' ' && e.target === closeButton && player !== null && !(popup !== null || helpOpen);
    }
    /** Finish a waiting X with 2 or 4. It runs before the player hears the key and marks it handled, so the playhead stays; no other key is taken. */
    function handleChordKey(e: KeyboardEvent): void {
      if (upscaleWaitingSince === null || e.defaultPrevented || popup !== null || isTyping(e.target) || inMenu(e.target)) return;
      const chord = stepUpscaleChord(upscaleWaitingSince, e, Date.now());
      if (!chord.value) return;
      upscaleWaitingSince = null;
      if (runUpscale(chord.value)) e.preventDefault();
    }
    function handleKeydown(e: KeyboardEvent): void {
      if (playsFromClose(e) && !e.repeat && isPlainKey(e)) {
        e.preventDefault();
        player?.engage();
        return;
      }
      // While a menu is open, keys belong to it: the viewer's own, and one opened from the player (the frame menu).
      // A focused player keeps its own keys (arrows step or seek, Space plays): it handles them first and marks them, so they arrive here prevented.
      if (e.defaultPrevented || popup !== null || isTyping(e.target) || inMenu(e.target)) {
        upscaleWaitingSince = null;
        return;
      }
      // Any key but a completing 2 or 4 (taken in the capture phase) ends a waiting X; an X starts one.
      upscaleWaitingSince = null;
      if (onupscale && asset && offeredUpscaleFactors(asset).length > 0) {
        const chord = stepUpscaleChord(null, e, Date.now());
        upscaleWaitingSince = chord.waitingSince;
        if (chord.consumed) {
          e.preventDefault();
          return;
        }
      }
      const action = viewerActionFor(e);
      // A held Del must not open a second confirmation, and a held S must not toggle the star again and again.
      if ((action === 'delete' || action === 'delete-now' || action === 'star') && e.repeat) {
        e.preventDefault();
        return;
      }
      if (action && runAction(action)) e.preventDefault();
    }
    document.addEventListener('keydown', handleChordKey, true);
    document.addEventListener('keydown', handleKeydown);
    return () => {
      document.removeEventListener('keydown', handleChordKey, true);
      document.removeEventListener('keydown', handleKeydown);
      release();
    };
  });
</script>

{#if open && asset}
  <div
    bind:this={viewerEl}
    class="asset-viewer ui-viewport"
    role="dialog"
    aria-modal="true"
    aria-labelledby="asset-viewer-title"
    data-testid="asset-viewer"
    data-details={detailsOpen}
    data-compact={compact}
    data-dock={chrome}
    data-film={assets.length > 1}
    data-phase={phase}
    data-flight={flight !== null}
    data-chrome={chromeHidden ? 'hidden' : 'shown'}
    style:--viewer-cover-top="{cover.top}px"
    style:--viewer-cover-right="{cover.right}px"
    style:--viewer-cover-bottom="{cover.bottom}px"
    style:--pull={drag?.axis === 'y' ? pullBackdropOpacity(drag.offset) : undefined}
  >
    <!-- On a desktop or a tablet the viewer is a panel over the page; a click on the backdrop closes it. -->
    <button type="button" class="viewer-backdrop" aria-label="Close viewer" tabindex="-1" inert={adjusting} onclick={close}></button>
    <div class="viewer-panel" inert={adjusting}>
    <div class="viewer-bar" data-bar={chrome} bind:offsetHeight={barHeight}>
      <div class="vb">
        <div class="vb-head">
          <div class="vb-title">
            {#if onstar}
              <button
                type="button"
                class="viewer-btn viewer-star ui-btn ui-btn-icon"
                data-action="star"
                aria-label="Star"
                aria-pressed={asset.starred === true}
                title={asset.starred === true ? 'Unstar (S)' : 'Star (S)'}
                onclick={() => onstar(asset, asset.starred !== true)}
              ><Icon name="star" size={16} solid={asset.starred === true} /></button>
            {/if}
            <div class="vb-name">
              <b id="asset-viewer-title"><FileName name={asset.filename} /></b>
              <small>{index + 1} of {assets.length}{setLabel ? ` · ${setLabel}` : ''}</small>
            </div>
          </div>
          <button
            type="button"
            class="viewer-btn ui-btn ui-btn-quiet vb-details"
            class:ui-btn-icon={chrome === 'phone' || chrome === 'rail'}
            data-action="details"
            aria-expanded={detailsOpen}
            aria-controls="asset-viewer-details"
            aria-label={chrome === 'phone' || chrome === 'rail' ? 'Details' : undefined}
            title="Details (I)"
            onclick={toggleDetails}
          ><Icon name="info" size={chrome === 'phone' || chrome === 'rail' ? 16 : 14} />{#if chrome !== 'phone' && chrome !== 'rail'}<span class="vb-label vb-details-label">Details</span>{/if}</button>
          {#if chrome === 'dock'}
            <button
              type="button"
              class="viewer-btn ui-btn ui-btn-quiet ui-btn-icon"
              data-action="shortcuts"
              aria-label="Keyboard shortcuts"
              aria-expanded={helpOpen}
              aria-controls="asset-viewer-shortcuts"
              title="Keyboard shortcuts (?)"
              onclick={() => { helpOpen = !helpOpen; }}
            ><kbd class="viewer-kbd">?</kbd></button>
          {/if}
        </div>
        {#if chrome === 'toolbar' && acts}
          <span class="ui-bar-rule vb-rule-head" aria-hidden="true"></span>
          <div class="vb-actions">
            {#each acts.make as act (act.id)}
              <ViewerButton {act} variant="bar" expanded={expandedOf(act)} onact={onAct} />
            {/each}
            <span class="vb-spacer"></span>
            {#each acts.file as act (act.id)}
              <ViewerButton {act} variant="bar" expanded={expandedOf(act)} onact={onAct} />
            {/each}
            {#if acts.del}
              <span class="ui-bar-rule" aria-hidden="true"></span>
              <ViewerButton act={acts.del} variant="bar" onact={onAct} />
            {/if}
          </div>
        {/if}
        <div class="vb-close">
          {#if chrome === 'toolbar'}<span class="ui-bar-rule" aria-hidden="true"></span>{/if}
          <button
            type="button"
            bind:this={closeButton}
            class="viewer-btn ui-btn ui-btn-icon"
            aria-label="Close viewer"
            title="Close (Esc)"
            onclick={close}
          ><Icon name="close" size={chrome === 'toolbar' ? 16 : 18} /></button>
        </div>
      </div>
    </div>

    <div class="viewer-main">
      <!-- svelte-ignore a11y_no_static_element_interactions because the swipe and pinch duplicate the arrows, keys and click -->
      <div
        bind:this={stageEl}
        bind:clientWidth={stageWidth}
        class="viewer-stage"
        data-touch={touchLayer}
        use:hold={{ onhold: () => { contextAt = pressAt; openPopup('more', null); }, ignore: 'button, [data-own-drag]' }}
        onpointerdown={(event) => { pressAt = { x: event.clientX, y: event.clientY }; }}
        oncontextmenu={onStageContextMenu}
        ontouchstart={onStageTouchStart}
        ontouchmove={onStageTouchMove}
        ontouchend={onStageTouchEnd}
        ontouchcancel={() => { gesture = null; drag = null; }}
      >
        {#if compact && detailsOpen}
          <button type="button" class="viewer-details-scrim" aria-label="Close details" tabindex="-1" onclick={toggleDetails}></button>
        {/if}
        <button
          type="button"
          class="viewer-nav prev ui-btn ui-btn-icon"
          aria-label="Previous file"
          disabled={!hasPrev}
          onclick={() => hasPrev && onnavigate(index - 1)}
        ><Icon name="chevleft" size={20} /></button>
        <!-- The pager: the shown asset between its neighbours' thumbnails, which a finger drags into view. -->
        <div bind:this={trackEl} bind:clientWidth={trackWidth} class="viewer-track" style:--pane-gap="{paneGap}px" style:transform={drag?.axis === 'x' ? `translateX(${drag.offset}px)` : undefined}>
          {#if touchLayer && hasPrev}
            <div class="viewer-pane viewer-pane-prev" aria-hidden="true">{@render neighbour(assets[index - 1])}</div>
          {/if}
          <div bind:this={paneEl} class="viewer-pane viewer-pane-current" style:transform={drag?.axis === 'y' ? pullTransform(drag.offset) : undefined}>
            {#if asset.media_type === 'audio'}
              <div class="viewer-audio" data-testid="viewer-audio">
                {#key asset.url}
                  <MediaPlayer bind:this={player} kind="audio" src={asset.url} label={asset.filename} peaks={audioPeaks.of(asset)} duration={asset.duration} size="viewer" />
                {/key}
                {#if words.length > 0}
                  <div class="viewer-words" data-testid="viewer-words">
                    {#each words as block (block.label)}
                      <section>
                        <h4 class="viewer-h ui-area-label">{block.label}</h4>
                        <p class="viewer-prompt viewer-words-text">{block.text}</p>
                      </section>
                    {/each}
                  </div>
                {/if}
              </div>
            {:else}
            {#if !mediaLoaded && asset.media_type !== 'video'}
              <!-- The thumbnail stands in until the image has loaded. -->
              <img class="viewer-standin" class:ui-checkerboard={isTransparent(asset)} data-transparent={isTransparent(asset)} src={thumbOf(asset)} alt="" aria-hidden="true">
            {/if}
            {#if asset.media_type === 'video'}
              {#key asset.url}
                <MediaPlayer bind:this={player} kind="video" size="viewer" naturalSize={asset.width && asset.height ? { width: asset.width, height: asset.height } : null} src={asset.url} label={asset.filename} poster={thumbOf(asset)} filePath={asset.file_path} duration={asset.duration} actions={frameActions} onpoint={(next) => { point = next; }} />
              {/key}
            {:else}
              <!-- svelte-ignore a11y_no_noninteractive_element_interactions because the zoom is a closer look; the keys and buttons do the rest -->
              <img
                bind:this={mediaEl}
                src={asset.url}
                alt={asset.prompt || asset.filename}
                class="viewer-media"
                class:ui-checkerboard={isTransparent(asset)}
                data-transparent={isTransparent(asset)}
                class:zoomable={actualScale !== null}
                class:zoomed
                class:animated={zoomAnimated}
                style:transform={zoomed ? `translate(${zoom.x}px, ${zoom.y}px) scale(${zoom.scale})` : undefined}
                draggable="false"
                data-testid="viewer-image"
                data-zoom={zoom.scale.toFixed(2)}
                onload={onMediaLoad}
                onpointerdown={onMediaPointerDown}
                onpointermove={onMediaPointerMove}
                onpointerup={onMediaPointerUp}
                ontransitionend={() => { zoomAnimated = false; }}
              >
            {/if}
            {/if}
          </div>
          {#if touchLayer && hasNext}
            <div class="viewer-pane viewer-pane-next" aria-hidden="true">{@render neighbour(assets[index + 1])}</div>
          {/if}
        </div>
        <button
          type="button"
          class="viewer-nav next ui-btn ui-btn-icon"
          aria-label="Next file"
          disabled={!hasNext}
          onclick={() => hasNext && onnavigate(index + 1)}
        ><Icon name="chevright" size={20} /></button>
      </div>

      {#if detailsOpen}
        <aside id="asset-viewer-details" class="viewer-details" class:ui-sheet={compact} aria-label="File details" use:sheetDrag={{ enabled: compact, onclose: () => { if (detailsOpen) toggleDetails(); } }}>
          {#if compact}
            <div class="ui-sheet-grab" data-sheet-handle aria-hidden="true"></div>
            <div class="ui-sheet-head" data-sheet-handle>
              <h2 class="ui-area-label">Details</h2>
              <span class="ui-sheet-end">
                <button type="button" class="ui-btn ui-btn-icon" aria-label="Close details" onclick={toggleDetails}><Icon name="close" size={16} /></button>
              </span>
            </div>
          {/if}
          <div class="viewer-details-body">
          {#if !isAudio}
          {#if separateScene}
            <section data-testid="viewer-scene">
              <h4 class="viewer-h ui-area-label">Scene</h4>
              <p class="viewer-prompt">{separateScene}</p>
            </section>
          {/if}
          <section>
            <h4 class="viewer-h ui-area-label">{sceneOnly ? 'Scene' : 'Prompt'}</h4>
            <p class="viewer-prompt">{asset.prompt || 'No prompt recorded.'}</p>
          </section>
          {/if}
          {#if asset.enhance}
            <section data-testid="viewer-enhance">
              <h4 class="viewer-h ui-area-label">Enhanced with</h4>
              <p class="viewer-enhance"><Icon name="sparkle" size={13} /><span>{asset.enhance.summary}</span></p>
              {#if asset.enhance.scene_input && asset.enhance.scene_input !== asset.scene}
                <h4 class="viewer-h viewer-h-sub ui-area-label">Your scene</h4>
                <p class="viewer-prompt">{asset.enhance.scene_input}</p>
              {/if}
              {#if asset.enhance.prompt_input && asset.enhance.prompt_input !== asset.prompt}
                <h4 class="viewer-h viewer-h-sub ui-area-label">Your prompt</h4>
                <p class="viewer-prompt">{asset.enhance.prompt_input}</p>
              {/if}
            </section>
          {/if}
          {#if sections?.negativePrompt}
            <section>
              <h4 class="viewer-h ui-area-label">Negative prompt</h4>
              <p class="viewer-prompt">{sections.negativePrompt}</p>
            </section>
          {/if}
          {#if source}
            <section data-testid="viewer-source">
              <h4 class="viewer-h ui-area-label">Made from</h4>
              {#snippet sourceLabel()}
                <span class="truncate">{fileName(source.path)}</span>
                {#if source.width && source.height}<small>{source.width}×{source.height}</small>{/if}
              {/snippet}
              {#if sourceIndex >= 0}
                <button type="button" class="viewer-source" onclick={() => onnavigate(sourceIndex)}>{@render sourceLabel()}</button>
              {:else}
                <p class="viewer-source" title={source.path}>{@render sourceLabel()}</p>
              {/if}
            </section>
          {/if}
          {#if sections}
            {@render factSection('Generation', sections.generation)}
            {@render factSection('Post-processing', sections.postProcessing)}
            {@render factSection('File', sections.file)}
          {/if}
          {#if fallbackReasons.length > 0}
            <section class="ui-alert ui-alert-warning flex-col gap-1" role="note">
              <h4 class="viewer-h ui-area-label">Reuse notice</h4>
              <ul class="list-inside list-disc space-y-0.5 text-ui text-text-secondary">
                {#each fallbackReasons as reason (reason)}
                  <li>{describeFallbackReason(reason)}</li>
                {/each}
              </ul>
            </section>
          {/if}
          </div>
        </aside>
      {/if}
    </div>

    <!-- The foot, dock and film strip, publishes its height (`--viewer-foot`) so a toast sits above it, as the Workspace's controls do. -->
    <div class="viewer-foot" use:dockHeight={'--viewer-foot'} bind:offsetHeight={footHeight}>
      <!-- A finger gets the toolbar's actions in a dock under the picture: the full list on a tablet, four slots on a phone, a rail on the right when a phone is held sideways. -->
      {#if acts && (chrome === 'dock' || chrome === 'phone' || chrome === 'rail')}
        <div bind:offsetWidth={dockWidth} class="viewer-dock ui-tool-row" class:ui-tool-row-four={chrome !== 'dock'} class:ui-tool-row-wrap={chrome === 'dock'} data-dock={chrome} role="toolbar" aria-label="Actions">
          {#if chrome === 'dock'}
            {#each acts.make as act (act.id)}
              <ViewerButton {act} variant="cell" expanded={expandedOf(act)} onact={onAct} />
            {/each}
            <span class="ui-tool-spacer"></span>
            {#each acts.file as act (act.id)}
              <ViewerButton {act} variant="cell" expanded={expandedOf(act)} onact={onAct} />
            {/each}
            {#if acts.del}
              <span class="ui-tool-rule" aria-hidden="true"></span>
              <ViewerButton act={acts.del} variant="cell" onact={onAct} />
            {/if}
          {:else}
            {#each phoneSlots as act (act.id)}
              <ViewerButton {act} variant="cell" expanded={expandedOf(act)} onact={onAct} />
            {/each}
          {/if}
        </div>
      {/if}

      {#if assets.length > 1}
        <div class="viewer-film" aria-label="Files in this set" bind:this={filmEl}>
          {#each assets as item, itemIndex (item.id)}
            <button
              type="button"
              data-film-index={itemIndex}
              aria-label="Show {item.filename}"
              aria-current={itemIndex === index}
              onclick={() => onnavigate(itemIndex)}
            >
              {#if item.media_type === 'audio'}
                <span class="film-audio" aria-hidden="true"><Waveform peaks={audioPeaks.of(item)} /></span>
              {:else}
                <img src={thumbOf(item)} alt="" loading="lazy" class:ui-checkerboard={isTransparent(item)} data-transparent={isTransparent(item)}>
                {#if item.media_type === 'video'}
                  <span class="film-play" data-testid="film-play-badge" aria-hidden="true"><Icon name="play" size={8} /></span>
                {/if}
              {/if}
            </button>
          {/each}
        </div>
      {/if}
    </div>

    {#if contextAt}<span bind:this={contextAnchorEl} class="viewer-context-anchor" style:left="{contextAt.x}px" style:top="{contextAt.y}px" aria-hidden="true"></span>{/if}
    <AssetPanelMenu
      open={popup === 'more'}
      anchor={contextAt ? contextAnchorEl : popupAnchor}
      {asset}
      panel={panel!}
      metaSuffix="{index + 1} of {assets.length}"
      onclose={() => closePopup('more')}
    />

    {#if helpOpen}
      <div id="asset-viewer-shortcuts" class="viewer-shortcuts ui-overlay" role="note" aria-label="Keyboard shortcuts">
        <h4 class="viewer-h ui-area-label">Keyboard shortcuts</h4>
        <ShortcutList entries={viewerShortcutsFor(asset.media_type)} />
      </div>
    {/if}

    </div>
    <AssetPanelMenu
      open={popup === 'useas'}
      anchor={popupAnchor}
      {asset}
      panel={panel!}
      only="use-as"
      align="start"
      side={chrome === 'dock' ? 'above' : 'below'}
      onclose={() => closePopup('useas')}
    />
    {#if onmove && asset}
      <MoveMenu
        open={popup === 'move'}
        anchor={popupAnchor}
        align={compact ? 'end' : 'start'}
        current={asset.project}
        onmove={(project) => onmove(asset, project)}
        onclose={() => closePopup('move')}
      />
    {/if}
    <AssetPanelMenu
      open={popup === 'upscale'}
      anchor={popupAnchor}
      {asset}
      panel={panel!}
      only="upscale"
      align="start"
      side={chrome === 'dock' ? 'above' : 'below'}
      onclose={() => closePopup('upscale')}
    />
    {#if adjusting}
      {#key asset.id}
        <AdjustEditor {asset} onclose={() => { adjusting = false; }} onsaved={(saved, replaced) => onadjusted?.(saved, replaced, adjustedId ?? asset.id)} />
      {/key}
    {/if}
  </div>
{/if}

{#if flight}
  <!-- The thumbnail in flight between the tile and the stage; it stands in for the image until that has loaded. -->
  <img
    bind:this={flightEl}
    class="viewer-flight"
    src={flight.src}
    alt=""
    aria-hidden="true"
    class:ui-checkerboard={flight.transparent}
    data-transparent={flight.transparent}
    data-testid="viewer-flight"
    style:left="{flight.rect.left}px"
    style:top="{flight.rect.top}px"
    style:width="{flight.rect.width}px"
    style:height="{flight.rect.height}px"
  >
{/if}

{#snippet neighbour(item: GalleryAsset)}
  <!-- A neighbour waiting beside the shown asset: an image in full, so it is sharp once the swipe settles, a video as its still. -->
  {#if item.media_type === 'audio'}
    <span class="viewer-neighbour viewer-neighbour-audio" aria-hidden="true"><Waveform peaks={audioPeaks.of(item)} /></span>
  {:else}
    <img class="viewer-neighbour" class:ui-checkerboard={isTransparent(item)} data-transparent={isTransparent(item)} src={item.media_type === 'video' ? thumbOf(item) : item.url} alt="">
  {/if}
{/snippet}

{#snippet factSection(title: string, facts: DetailFact[])}
  {#if facts.length > 0}
    <section>
      <h4 class="viewer-h ui-area-label">{title}</h4>
      <dl class="viewer-facts">
        {#each facts as fact (fact.label)}
          <div class:wide={fact.wide} class:wrap={fact.wrap}>
            <dt>{fact.label}</dt>
            <dd title={fact.title ?? fact.value}>{fact.value}</dd>
          </div>
        {/each}
      </dl>
    </section>
  {/if}
{/snippet}

<style>
  .asset-viewer { z-index: var(--z-viewer); display: flex; flex-direction: column; }
  /* The backdrop and the chrome clear as the image is pulled down (--pull), and the chrome goes on a tap. */
  .viewer-backdrop { position: absolute; inset: 0; background: color-mix(in srgb, var(--color-bg-base) 70%, black); opacity: var(--pull, 1); }
  .viewer-panel { position: relative; display: flex; flex: 1; min-height: 0; flex-direction: column; }
  /* A desktop or a tablet: a panel over the whole page but a gutter of backdrop, which closes on click; the details sit beside the image. */
  .asset-viewer[data-compact='false'] { padding: 24px; }
  .asset-viewer[data-compact='false'] .viewer-backdrop { background: var(--color-scrim); cursor: default; }
  .asset-viewer[data-compact='false'] .viewer-panel { flex: none; width: 100%; height: 100%; overflow: hidden; border-radius: var(--radius-lg); background: var(--color-bg-base); box-shadow: 0 24px 64px var(--color-scrim); }
  /* A tap that hides the chrome on a tablet takes the panel to the screen's edges; the stage pads by as much (--viewer-edge), so the image stays put. */
  @media (min-width: 768px) {
    .asset-viewer[data-compact='false'][data-chrome='hidden'] { --viewer-edge: 24px; padding: 0; }
    .asset-viewer[data-compact='false'][data-chrome='hidden'] .viewer-panel { border-radius: 0; box-shadow: none; }
  }
  .viewer-foot { flex: none; }
  .viewer-bar, .viewer-main, .viewer-film { position: relative; }
  /* The bar and the strip lie over the stage, which runs under them. */
  .viewer-bar, .viewer-film { z-index: 3; opacity: var(--pull, 1); transition: opacity 160ms ease; }
  .viewer-film { background: var(--color-bg-base); }
  .asset-viewer[data-compact='true'] .viewer-film { background: color-mix(in srgb, var(--color-bg-base) 70%, black); }
  .asset-viewer[data-chrome='hidden'] .viewer-bar, .asset-viewer[data-chrome='hidden'] .viewer-film { opacity: 0; pointer-events: none; }
  /* The bar is the container its label steps are measured in: the actions drop their words by whole groups as it narrows. */
  .viewer-bar { container: viewer-bar / inline-size; flex: none; min-height: 52px; padding: var(--sp-8) var(--sp-12); background: var(--color-bg-surface); }
  /* Under 768px the viewer drops its inset, so a tablet's full dock fits from 640px. */
  @media (max-width: 767px) {
    .asset-viewer[data-compact='false'] { padding: 0; }
    .asset-viewer[data-compact='false'] .viewer-panel { border-radius: 0; }
    .viewer-bar { padding-top: calc(var(--sp-8) + env(safe-area-inset-top, 0px)); }
  }
  .vb { display: flex; align-items: center; gap: var(--sp-6); }
  .vb-head { display: flex; min-width: 0; flex: none; align-items: center; gap: var(--sp-6); }
  /* The star leads, the name follows 4px after it, in a block of one width, so what comes next starts in the same place for every file. */
  .vb-title { display: flex; width: var(--name-w, 240px); min-width: 0; flex: none; align-items: center; gap: var(--sp-4); }
  .vb-name { display: flex; min-width: 0; flex: 1; flex-direction: column; }
  .vb-name b { display: block; min-width: 0; font-size: var(--text-content); font-weight: 600; line-height: 1.3; }
  .vb-name small { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: var(--text-meta); color: var(--color-text-muted); }
  .vb-actions { display: flex; min-width: 0; flex: 1 1 auto; align-items: center; gap: var(--sp-6); }
  .vb-spacer { min-width: var(--sp-12); flex: 1 1 var(--sp-12); }
  .vb-close { display: flex; flex: none; align-items: center; gap: var(--sp-6); }
  .vb-actions :global(.vb-chev) { margin-left: calc(var(--sp-2) - var(--sp-6)); opacity: 0.7; }
  /* A starred file's glyph and colour change, never its box. */
  .viewer-star, .viewer-star[aria-pressed='true'] { background: transparent; color: var(--color-text-secondary); }
  .viewer-star[aria-pressed='true'] { color: var(--color-star); }
  @media (hover: hover) {
    .viewer-star:hover:not(:disabled), .viewer-star[aria-pressed='true']:hover:not(:disabled) { background: var(--color-bg-control-hover); }
  }
  @media (hover: none) { .viewer-star:active:not(:disabled), .viewer-star[aria-pressed='true']:active:not(:disabled) { background: var(--color-bg-control-hover); } }

  /* The label steps, measured on the bar's own width (the window minus the viewer's 48px inset and the bar's 24px of padding):
     Details, then the File group, then the Make group after Reuse, then Reuse's second word, then two rows. Delete keeps its word. */
  @container viewer-bar (max-width: 1344px) {
    .viewer-bar[data-bar='toolbar'] .vb-title { --name-w: 150px; }
    .viewer-bar[data-bar='toolbar'] .vb-details-label { position: absolute; width: 1px; height: 1px; overflow: hidden; clip-path: inset(50%); white-space: nowrap; }
    .viewer-bar[data-bar='toolbar'] .vb-details { width: var(--spacing-control); padding: 0; }
  }
  @container viewer-bar (max-width: 1199px) {
    .viewer-bar[data-bar='toolbar'] .vb-actions :global([data-group='file'] .vb-label) { position: absolute; width: 1px; height: 1px; overflow: hidden; clip-path: inset(50%); white-space: nowrap; }
    .viewer-bar[data-bar='toolbar'] .vb-actions :global([data-group='file']:not([data-menu])) { width: var(--spacing-control); padding: 0; }
    .viewer-bar[data-bar='toolbar'] .vb-actions :global([data-group='file'][data-menu]) { padding: 0 var(--sp-6); }
  }
  @container viewer-bar (max-width: 987px) {
    .viewer-bar[data-bar='toolbar'] .vb-actions :global([data-group='make'] .vb-label) { position: absolute; width: 1px; height: 1px; overflow: hidden; clip-path: inset(50%); white-space: nowrap; }
    .viewer-bar[data-bar='toolbar'] .vb-actions :global([data-group='make']:not([data-menu])) { width: var(--spacing-control); padding: 0; }
    .viewer-bar[data-bar='toolbar'] .vb-actions :global([data-group='make'][data-menu]) { padding: 0 var(--sp-6); }
  }
  @container viewer-bar (max-width: 787px) {
    .viewer-bar[data-bar='toolbar'] .vb-actions :global(.vb-tail) { position: absolute; width: 1px; height: 1px; overflow: hidden; clip-path: inset(50%); white-space: nowrap; }
  }
  /* Two rows: the name, Details, ? and ✕ (still at the top right), then the actions. */
  @container viewer-bar (max-width: 737px) {
    .viewer-bar[data-bar='toolbar'] .vb { flex-wrap: wrap; row-gap: var(--sp-6); }
    .viewer-bar[data-bar='toolbar'] .vb-title { --name-w: 200px; }
    .viewer-bar[data-bar='toolbar'] .vb-head { order: 0; flex: 1 1 0; }
    .viewer-bar[data-bar='toolbar'] .vb-close { order: 1; }
    .viewer-bar[data-bar='toolbar'] .vb-actions { order: 2; flex: 1 0 100%; }
    .viewer-bar[data-bar='toolbar'] .vb-rule-head, .viewer-bar[data-bar='toolbar'] .vb-close .ui-bar-rule { display: none; }
  }

  /* A finger: a top bar with the star, the name, Details and ✕ alone at the right; the actions are in the dock. */
  .viewer-bar:not([data-bar='toolbar']) .vb-title { width: 300px; flex: 0 1 auto; }
  .viewer-bar:is([data-bar='phone'], [data-bar='rail']) .vb-title { width: 210px; }
  .viewer-bar:not([data-bar='toolbar']) .vb-close { margin-left: auto; }
  .viewer-bar:not([data-bar='toolbar']) .vb-head { flex: 0 1 auto; }

  /* The dock hides and fades with the bar and the strip (a tap, a pull), and goes while the details are a sheet over the picture. */
  .viewer-dock { position: relative; z-index: 3; background: var(--color-bg-surface); opacity: var(--pull, 1); transition: opacity 160ms ease; }
  .asset-viewer[data-chrome='hidden'] .viewer-dock { opacity: 0; pointer-events: none; }
  .asset-viewer[data-film='false'] .viewer-dock { padding-bottom: calc(var(--sp-8) + env(safe-area-inset-bottom, 0px)); }
  /* The rail stays beside the details sheet: it is out of the picture's way at the right, and its band is already reserved. */
  .asset-viewer[data-compact='true'][data-details='true'] .viewer-dock:not([data-dock='rail']) { display: none; }
  @media (max-width: 767px) { .asset-viewer[data-details='true'] .viewer-dock:not([data-dock='rail']) { display: none; } }
  /* Toasts clear the foot, whatever its height (`dockHeight` measures it); a hidden dock measures nothing, so a reason shown with the details open stays on screen. */
  :global(:root:has(.asset-viewer .viewer-panel .viewer-foot)) { --toast-bottom: calc(var(--viewer-foot, 0px) + 12px); }
  /* A phone held sideways: the dock is a rail on the right, padded by the safe area; the top bar lies over the picture on a scrim, and the strip stays. */
  .asset-viewer[data-dock='rail'] .viewer-panel { padding-right: calc(76px + env(safe-area-inset-right, 0px)); }
  .viewer-dock[data-dock='rail'] {
    position: absolute; top: 0; right: 0; bottom: 0; z-index: 4; width: calc(76px + env(safe-area-inset-right, 0px)); flex-direction: column; justify-content: center;
    padding: var(--sp-8) calc(var(--sp-8) + env(safe-area-inset-right, 0px)) var(--sp-8) var(--sp-8);
  }
  .viewer-dock[data-dock='rail'] > :global(.ui-tool) { flex: none; height: 60px; }
  /* The bar pads for the status bar; set before the rail's own padding-bottom, which it must not override. */
  .asset-viewer[data-compact='true'] .viewer-bar { padding: calc(6px + env(safe-area-inset-top, 0px)) max(8px, env(safe-area-inset-right, 0px)) 6px max(12px, env(safe-area-inset-left, 0px)); }
  .asset-viewer[data-dock='rail'] .viewer-bar {
    position: absolute; top: 0; left: 0; right: calc(76px + env(safe-area-inset-right, 0px)); z-index: 3; padding-bottom: 28px; background: var(--viewer-top-scrim);
  }
  .asset-viewer[data-dock='rail'] .viewer-film { padding: 4px max(8px, env(safe-area-inset-right, 0px)) calc(8px + env(safe-area-inset-bottom, 0px)) max(8px, env(safe-area-inset-left, 0px)); }
  .viewer-kbd { padding: 0 4px; border-radius: var(--radius-xs); background: var(--color-bg-surface-hover); font-family: var(--font-mono); font-size: var(--text-meta); line-height: 15px; color: var(--color-text-muted); }
  .viewer-shortcuts { position: absolute; top: 52px; right: 12px; z-index: 3; width: 280px; padding: 12px; }
  .viewer-main { display: flex; flex: 1; min-height: 0; }
  /* The stage runs under the bar, the foot and a side rail (--viewer-cover-*), and pads by as much, so the fitted image clears them. */
  .viewer-stage {
    --stage-pad-y: 20px; --stage-pad-x: 64px;
    position: relative; display: flex; flex: 1; min-width: 0; align-items: center; justify-content: center;
    margin: calc(-1 * var(--viewer-cover-top, 0px)) calc(-1 * var(--viewer-cover-right, 0px)) calc(-1 * var(--viewer-cover-bottom, 0px)) 0;
    padding:
      calc(var(--stage-pad-y) + var(--viewer-edge, 0px) + var(--viewer-cover-top, 0px))
      calc(var(--stage-pad-x) + var(--viewer-edge, 0px) + var(--viewer-cover-right, 0px))
      calc(var(--stage-pad-y) + var(--viewer-edge, 0px) + var(--viewer-cover-bottom, 0px))
      calc(var(--stage-pad-x) + var(--viewer-edge, 0px));
  }
  .viewer-media { max-width: 100%; max-height: 100%; object-fit: contain; border-radius: var(--radius-xs); transform-origin: center; user-select: none; }
  /* A finger's hold opens our menu, not iOS's image callout (Copy, Save Image); the whole stage opts out of it. */
  .viewer-stage, .viewer-stage img { -webkit-touch-callout: none; -webkit-user-select: none; user-select: none; }
  .viewer-media.zoomable { cursor: zoom-in; }
  .viewer-media.zoomed { position: relative; z-index: 1; cursor: grab; }
  .viewer-media.animated { transition: transform 0.18s ease-out; }
  /* The stage clips a zoomed image at the panel's edges; the bar, the dock and the strip stay on top of it. */
  .viewer-stage { overflow: hidden; }
  /* A finger owns every gesture on the stage, so the browser neither pans nor zooms the page: only the image zooms. */
  .viewer-stage[data-touch='true'] { touch-action: none; }
  .viewer-nav { position: absolute; top: calc(50% + (var(--viewer-cover-top, 0px) - var(--viewer-cover-bottom, 0px)) / 2); z-index: 2; width: var(--spacing-bar); height: var(--spacing-bar); transform: translateY(-50%); }
  .viewer-nav.prev { left: 14px; }
  .viewer-nav.next { right: 14px; }
  .viewer-details { display: flex; flex: none; flex-direction: column; width: 340px; min-height: 0; background: var(--color-bg-surface); }
  .viewer-details-body { display: flex; flex: 1; min-height: 0; flex-direction: column; gap: 16px; overflow-y: auto; overscroll-behavior: contain; padding: 16px; }
  .viewer-h { margin: 0 0 6px; }
  .viewer-h-sub { margin-top: 10px; }
  .viewer-enhance { display: flex; align-items: baseline; gap: 6px; font-size: var(--text-ui); line-height: 1.45; color: var(--color-primary-main); }
  .viewer-enhance :global(svg) { flex: none; transform: translateY(2px); }
  .viewer-prompt { padding: 8px 10px; border-radius: var(--radius-md); background: var(--color-bg-base); font-size: var(--text-content); line-height: 1.55; color: var(--color-text-secondary); white-space: pre-wrap; user-select: text; }
  .viewer-source { display: flex; width: 100%; align-items: baseline; justify-content: space-between; gap: 8px; padding: 8px 10px; border-radius: var(--radius-md); background: var(--color-bg-base); font-size: var(--text-ui); color: var(--color-text-primary); text-align: left; }
  .viewer-source small { flex: none; font-family: var(--font-mono); font-size: var(--text-meta); color: var(--color-text-muted); }
  @media (hover: hover) { button.viewer-source:hover { background: var(--color-bg-surface-hover); } }
  button.viewer-source:focus-visible, .viewer-film button:focus-visible { outline: 2px solid var(--color-primary-main); outline-offset: 2px; }
  .viewer-facts { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }
  .viewer-facts div { min-width: 0; padding: 6px 8px; border-radius: var(--radius-sm); background: var(--color-bg-base); }
  .viewer-facts .wide { grid-column: span 2; }
  .viewer-facts .wrap dd { white-space: normal; overflow-wrap: anywhere; }
  .viewer-facts dt { font-size: var(--text-meta); color: var(--color-text-muted); }
  .viewer-facts dd { margin: 2px 0 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-family: var(--font-mono); font-size: var(--text-ui); color: var(--color-text-primary); }
  .viewer-film { display: flex; flex: none; gap: 6px; overflow-x: auto; padding: 8px 12px 12px; }
  /* Auto margins centre a short strip but keep a long one scrollable from its first item. */
  .viewer-film > :first-child { margin-left: auto; }
  .viewer-film > :last-child { margin-right: auto; }
  .viewer-film button { position: relative; flex: none; width: 44px; height: 44px; overflow: hidden; padding: 0; border: 2px solid transparent; border-radius: var(--radius-sm); opacity: 0.55; transition: opacity 0.12s ease; }
  @media (hover: hover) { .viewer-film button:hover { opacity: 0.9; } }
  .viewer-film button[aria-current='true'] { border-color: var(--color-primary-main); opacity: 1; }
  /* An audio take in the film strip: the no-content tile at 44px, its waveform in the middle. */
  .film-audio { display: grid; width: 100%; height: 100%; place-items: center; padding: 14px 6px; background: var(--color-bg-surface-hover); }
  .film-play { position: absolute; right: 2px; bottom: 2px; display: flex; align-items: center; justify-content: center; width: 14px; height: 14px; border-radius: 50%; background: color-mix(in srgb, var(--color-bg-base) 78%, transparent); color: var(--color-text-primary); }
  .viewer-film img { display: block; width: 100%; height: 100%; object-fit: cover; }
  @media (max-width: 767px) {
    .viewer-stage { --stage-pad-y: 12px; --stage-pad-x: 52px; }
    .viewer-details { position: absolute; inset: auto 0 0 0; z-index: 3; width: auto; max-height: 50%; }
  }
  /* Phone: no arrows (a swipe moves), the details rise as a sheet. */
  .asset-viewer[data-compact='true'] .viewer-nav { display: none; }
  .viewer-details-scrim { position: absolute; inset: 0; z-index: 2; background: transparent; cursor: default; }
  .asset-viewer[data-compact='true'] .viewer-stage { --stage-pad-y: 8px; --stage-pad-x: 12px; }
  /* The details rise as a half sheet under the image, which moves up to make room; the strip goes meanwhile. */
  .asset-viewer[data-compact='true'] .viewer-main { flex-direction: column; }
  .asset-viewer[data-compact='true'] .viewer-details { position: static; z-index: 3; width: auto; max-height: 50%; flex: none; }
  .asset-viewer[data-compact='true'][data-details='true'] .viewer-film { display: none; }
  .asset-viewer[data-compact='true'] .viewer-details-body { padding: 4px max(16px, env(safe-area-inset-right, 0px)) calc(16px + env(safe-area-inset-bottom, 0px)) max(16px, env(safe-area-inset-left, 0px)); }
  .asset-viewer[data-compact='true'] .viewer-film { padding-bottom: calc(12px + env(safe-area-inset-bottom, 0px)); }

  /* The pager: the shown asset's pane between its neighbours, one pane to each side, moved by transform alone. */
  .viewer-track { position: relative; display: flex; width: 100%; height: 100%; will-change: transform; }
  .viewer-pane { position: relative; display: flex; flex: none; width: 100%; height: 100%; align-items: center; justify-content: center; }
  .viewer-pane-prev, .viewer-pane-next { position: absolute; top: 0; }
  .viewer-pane-prev { left: calc(-100% - var(--pane-gap, 0px)); }
  .viewer-pane-next { left: calc(100% + var(--pane-gap, 0px)); }
  .viewer-neighbour { width: 100%; height: 100%; object-fit: contain; border-radius: var(--radius-xs); }
  /* A transparent image carries its checkerboard on the <img>, so the <img> is sized to the picture, not the pane. */
  .viewer-neighbour[data-transparent='true'] { width: auto; height: auto; max-width: 100%; max-height: 100%; object-fit: fill; }
  .viewer-standin[data-transparent='true'] { width: auto; height: auto; max-width: 100%; max-height: 100%; margin: auto; object-fit: fill; }
  .viewer-standin { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: contain; border-radius: var(--radius-xs); pointer-events: none; }
  .viewer-neighbour-audio { display: block; max-width: 640px; height: 120px; margin: auto; padding: 12px; background: var(--color-bg-surface); border-radius: var(--radius-md); opacity: 0.6; }
  /* An audio take: the player (200px waveform, max 960) with its words under it, which scroll with the stage. */
  .viewer-audio { display: flex; flex-direction: column; gap: 16px; width: 100%; max-width: 960px; max-height: 100%; overflow-y: auto; overscroll-behavior: contain; }
  .viewer-words { display: flex; flex-direction: column; gap: 12px; padding: 0 4px 8px; }
  .viewer-words .viewer-prompt { padding: 0; background: none; font-size: var(--text-title); line-height: 1.6; }
  .asset-viewer[data-compact='true'] .viewer-words .viewer-prompt { font-size: 15px; }
  .viewer-context-anchor { position: fixed; width: 1px; height: 1px; pointer-events: none; }
  /* The flight: the thumbnail fixed over everything, moved by transform alone; the viewer fades in or out under it. */
  .viewer-flight { position: fixed; z-index: calc(var(--z-viewer) + 1); object-fit: cover; border-radius: var(--radius-xs); transform-origin: top left; pointer-events: none; will-change: transform; }
  .asset-viewer[data-flight='true'] .viewer-media, .asset-viewer[data-flight='true'] .viewer-standin { opacity: 0; }
  .asset-viewer[data-phase='opening'] { animation: viewer-fade-in 240ms ease-out both; }
  .asset-viewer[data-phase='closing'] { animation: viewer-fade-out 240ms ease-in both; pointer-events: none; }
  @keyframes viewer-fade-in { from { opacity: 0; } }
  @keyframes viewer-fade-out { to { opacity: 0; } }
  @media (prefers-reduced-motion: reduce) { .asset-viewer[data-phase] { animation: none; } }
</style>
