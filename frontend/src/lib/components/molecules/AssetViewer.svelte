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
  import { Icon, ShortcutList } from '$lib/components/atoms';
  import { isTyping } from '$lib/keyboard';
  import { canExtend, canUseAsReference, describeFallbackReason, extendReason, offeredUpscaleFactors, plainUpscaleMethod, upscaleOption, type AssetActionHandlers, type ReferenceTarget } from '$lib/state/assetActions';
  import { addToast } from '$lib/state/toasts.svelte';
  import type { GalleryAsset, UpscaleFactor } from '$lib/types';
  import ActionMenu, { type ActionMenuEntry } from './ActionMenu.svelte';
  import { assetDetailSections, fileName, type DetailFact } from './assetDetails';
  import { assetMenuEntries, referenceEntries, upscaleEntries } from './assetMenu';
  import { VIEWER_SHORTCUTS, stepUpscaleChord, viewerActionFor, type ViewerAction } from './viewerShortcuts';
  import { layout } from '$lib/state/layoutMode.svelte';
  import { sheetDrag } from '$lib/actions/sheetDrag';
  import { pushEscapeLayer } from '$lib/actions/escapeStack';
  import { tick, untrack } from 'svelte';
  import { FLIGHT_EASING, FLIGHT_MS, fitRect, flightKeyframes, rectOf, usableRect, type Rect } from './viewerFlight';
  import { pagerOutcome, pullBackdropOpacity, pullOutcome, pullTransform, type PagerOutcome } from './viewerPager';
  import { lockAxis, releaseVelocity, rubberBand, type PanAxis, type PanSample } from '$lib/actions/pan';
  import { hold } from '$lib/actions/hold';
  import { FIT, actualPixelsScale, clampPan, settleZoom, zoomAt, type Zoom } from './viewerZoom';

  interface Props extends Omit<AssetActionHandlers, 'onpreview'> {
    assets: GalleryAsset[];
    currentIndex: number;
    open: boolean;
    /** Where these assets come from, e.g. "History"; shown under the filename. */
    setLabel?: string;
    deletingIds?: ReadonlySet<string>;
    /** Reference targets the current model can't use, with the reason. */
    referenceUnavailable?: Partial<Record<ReferenceTarget, string>>;
    /** The reference targets the menu offers; absent offers an image and a video. */
    referenceTargets?: readonly ReferenceTarget[];
    /** Called when the viewer nears the end of the list, so a paged source can load more. */
    onnearend?: () => void;
    onclose: () => void;
    onnavigate: (index: number) => void;
    /** The tile that opened the viewer: the image grows out of it and shrinks back into it. */
    origin?: HTMLElement | null;
  }

  let {
    assets,
    currentIndex,
    open,
    setLabel = '',
    deletingIds = new Set<string>(),
    referenceUnavailable = {},
    referenceTargets = undefined,
    onnearend,
    onclose,
    onnavigate,
    origin = null,
    onreuse,
    onreference,
    onextend,
    onupscale,
    ondelete,
  }: Props = $props();

  // A delete can shrink the list under the viewer; stay on the nearest remaining asset.
  const index = $derived(Math.min(currentIndex, assets.length - 1));
  const asset = $derived(index >= 0 ? assets[index] : null);
  const hasPrev = $derived(index > 0);
  const hasNext = $derived(index < assets.length - 1);
  const canReuse = $derived(asset?.has_reusable_config === true);
  const deleting = $derived(asset ? deletingIds.has(asset.id) : false);
  const fallbackReasons = $derived(asset?.reuse_state?.fallback_reasons ?? []);
  let filmEl = $state<HTMLDivElement | null>(null);
  const NEAR_END = 3;
  // Videos have no lazy loading; only those near the current asset load their first frame.
  const FILM_VIDEO_RADIUS = 4;

  $effect(() => {
    if (open && assets.length > 0 && index >= assets.length - NEAR_END) onnearend?.();
  });

  // Keep the current thumbnail centred in the strip as the viewer moves.
  $effect(() => {
    if (!open || !filmEl) return;
    const current = filmEl.querySelector<HTMLElement>(`[data-film-index="${index}"]`);
    current?.scrollIntoView?.({ block: 'nearest', inline: 'center', behavior: 'smooth' });
  });

  let detailsOpen = $state(loadDetailsOpen());
  let referenceOpen = $state(false);
  let referenceButton = $state<HTMLButtonElement | null>(null);
  let upscaleOpen = $state(false);
  let upscaleButton = $state<HTMLButtonElement | null>(null);
  let closeButton = $state<HTMLButtonElement | null>(null);
  let downloadLink = $state<HTMLAnchorElement | null>(null);
  let helpOpen = $state(false);
  // The shortcut list is a layer of its own: Escape closes it before the viewer.
  $effect(() => {
    if (!helpOpen) return;
    return pushEscapeLayer(() => { helpOpen = false; });
  });

  const referenceItems = $derived<ActionMenuEntry[]>(
    asset && onreference ? referenceEntries(asset, onreference, referenceUnavailable, referenceTargets) : []
  );

  const upscaleItems = $derived<ActionMenuEntry[]>(
    asset && onupscale ? upscaleEntries(asset, onupscale) : []
  );
  /** The keys that follow X for the factors this asset offers: "2 or 4" for an image, "2" for a video. */
  const upscaleKeys = $derived(asset ? offeredUpscaleFactors(asset).join(' or ') : '');

  // A video offers Extend where an image offers Use as reference; the button is absent without the capability.
  const showExtend = $derived(Boolean(onextend && asset?.media_type === 'video' && asset.extend));
  const extendBlocked = $derived(asset ? extendReason(asset) : null);

  // A phone: one ⋯ menu holds every action (the same entries as a tile's menu, plus Reuse and Copy the prompt),
  // the details open as a sheet, and a swipe moves between assets or closes.
  const compact = $derived(layout.compact);
  let moreOpen = $state(false);
  let moreButton = $state<HTMLButtonElement | null>(null);
  const moreItems = $derived.by<ActionMenuEntry[]>(() => {
    if (!asset) return [];
    const entries = assetMenuEntries(asset, { onreuse, onreference, onextend, onupscale, ondelete }, deleting, referenceUnavailable, referenceTargets);
    if (!asset.prompt) return entries;
    const copy: ActionMenuEntry = { kind: 'item', id: 'copy-prompt', label: 'Copy the prompt', onselect: () => void copyPrompt(asset.prompt!) };
    // Copy the prompt sits with Reuse at the top, before the first separator.
    if (onreuse) return [entries[0], copy, ...entries.slice(1)];
    return [copy, { kind: 'separator' }, ...entries];
  });
  // --- Touch and zoom ---
  // A finger, at any width: a swipe moves between assets or (down) closes, a pinch or a double tap zooms an image,
  // and a zoomed image pans under one finger. A mouse click shows an image at its real pixels; a drag then pans it.
  const touchLayer = $derived(compact || layout.coarse);
  let stageEl = $state<HTMLDivElement | null>(null);
  let mediaEl = $state<HTMLImageElement | null>(null);
  let zoom = $state<Zoom>(FIT);
  let zoomAnimated = $state(false);
  const zoomed = $derived(zoom.scale > 1);
  const isImage = $derived(asset?.media_type !== 'video');
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
  type Flight = { rect: Rect; src: string };
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

  /** The shown media's box: the loaded image or the video, else the thumbnail standing in for the image. */
  function mediaRect(): Rect | null {
    const shown = stageEl?.querySelector(mediaLoaded || asset?.media_type === 'video' ? '.viewer-media' : '.viewer-standin');
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
  function fly(from: Rect, to: Rect, src: string, done: () => void): void {
    flight = { rect: to, src };
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
    // The neighbours sit a stage padding further out than the track, so they stay clear of its edges at any width.
    const measureGap = () => { paneGap = Math.max(0, (stageEl?.clientWidth ?? 0) - (trackEl?.clientWidth ?? 0)); };
    window.addEventListener('resize', measureGap);
    untrack(() => {
      measureGap();
      const shown = asset;
      // A video has no still to fly (its thumbnail is the video itself), so it fades in and out instead.
      const from = reduceMotion() || shown?.media_type === 'video' ? null : originRect();
      const stage = from ? stageContentRect() : null;
      if (!shown || !from || !stage) { phase = 'open'; return; }
      phase = 'opening';
      fly(from, fitRect(stage, mediaAspect()), shown.thumbnail_url || shown.url, () => {
        phase = 'open';
        flight = null;
      });
    });
    return () => { window.removeEventListener('resize', measureGap); flight = null; phase = 'open'; chromeHidden = false; drag = null; helpOpen = false; };
  });

  $effect(() => {
    void asset?.id;
    mediaLoaded = false;
  });

  function onMediaLoad(): void {
    mediaLoaded = true;
    measureActual();
  }

  /** Close the viewer: the image shrinks back into its own tile first, while that tile is on screen. */
  function close(): void {
    if (phase === 'closing') return;
    const to = reduceMotion() || phase === 'opening' || asset?.media_type === 'video' ? null : shownTileRect();
    const from = to && asset ? mediaRect() : null;
    if (!to || !from || !asset) { onclose(); return; }
    phase = 'closing';
    fly(from, to, asset.thumbnail_url || asset.url, onclose);
  }

  function measureActual(): void {
    if (!mediaEl) return;
    const scale = actualPixelsScale({ width: mediaEl.naturalWidth, height: mediaEl.naturalHeight }, { width: mediaEl.clientWidth, height: mediaEl.clientHeight });
    actualScale = scale > 1.05 ? scale : null;
  }

  /** A point on screen, from the stage's centre (where the fitted image's centre sits). */
  function fromCentre(clientX: number, clientY: number): { x: number; y: number } {
    const box = stageEl?.getBoundingClientRect();
    return box ? { x: clientX - box.left - box.width / 2, y: clientY - box.top - box.height / 2 } : { x: 0, y: 0 };
  }

  function setZoom(next: Zoom, animated: boolean): void {
    zoomAnimated = animated;
    zoom = mediaEl && stageEl
      ? clampPan(next, { width: mediaEl.clientWidth, height: mediaEl.clientHeight }, { width: stageEl.clientWidth, height: stageEl.clientHeight })
      : next;
  }

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
  // The stage's padding: the neighbours sit that much further out, so they stay clear of the stage's edges.
  let paneGap = $state(0);
  // A tap hides the bar and the strip for a clean look at the image; another brings them back.
  let chromeHidden = $state(false);
  const paneWidth = () => (trackEl?.clientWidth || stageEl?.clientWidth || window.innerWidth) + paneGap;
  const thumbOf = (item: GalleryAsset) => item.thumbnail_url || item.url;

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
    // A video's own controls take the finger: scrubbing must not step to another asset.
    if ((event.target as Element).closest('video, .viewer-details-scrim') || settling) { gesture = null; return; }
    if (event.touches.length === 2 && isImage) {
      gesture = { kind: 'pinch', ...pinchOf(event.touches), from: zoom };
      return;
    }
    const touch = event.touches[0];
    if (!touch || event.touches.length > 1) { gesture = null; return; }
    const at = Date.now();
    paneGap = Math.max(0, (stageEl?.clientWidth ?? 0) - (trackEl?.clientWidth ?? 0));
    gesture = zoomed
      ? { kind: 'pan', x: touch.clientX, y: touch.clientY, at, from: zoom }
      : { kind: 'swipe', x: touch.clientX, y: touch.clientY, at, axis: null, samples: [{ x: touch.clientX, y: touch.clientY, at }] };
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
      onTap(touch.clientX, touch.clientY, now);
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
    moreOpen = true;
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

  const sections = $derived(asset ? assetDetailSections(asset) : null);
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

  /** Extend the shown video from the keyboard; when it can't be extended, say why instead. */
  function runExtend(): boolean {
    if (!asset || !onextend || !showExtend) return false;
    if (!canExtend(asset)) {
      addToast(extendBlocked ?? 'This video cannot be extended.', 'warning');
      return true;
    }
    onextend(asset);
    return true;
  }

  function toggleDetails(): void {
    detailsOpen = !detailsOpen;
    saveDetailsOpen(detailsOpen);
  }

  async function copyPrompt(prompt: string): Promise<void> {
    try {
      await navigator.clipboard.writeText(prompt);
      addToast('Prompt copied', 'success');
    } catch {
      addToast('Could not copy the prompt', 'error');
    }
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
        if (asset.media_type === 'video') return runExtend();
        if (referenceItems.length === 0) return false;
        referenceOpen = true;
        return true;
      case 'copy':
        if (!asset.prompt) return false;
        void copyPrompt(asset.prompt);
        return true;
      case 'details':
        toggleDetails();
        return true;
      case 'help':
        helpOpen = !helpOpen;
        return true;
      case 'reuse':
        if (!onreuse || !canReuse) return false;
        onreuse(asset);
        return true;
      case 'download':
        downloadLink?.click();
        return downloadLink !== null;
      case 'delete':
      case 'delete-now':
        // Ignore key repeat while the first delete is still in flight.
        if (!ondelete || deleting) return false;
        ondelete(asset, { confirm: action === 'delete' });
        return true;
    }
  }

  $effect(() => {
    if (!open) return;
    queueMicrotask(() => closeButton?.focus());
    // On the overlay stack like every layer: Escape closes it, innermost first (the shortcut list before it).
    const release = pushEscapeLayer(close);
    let upscaleWaitingSince: number | null = null;
    function handleKeydown(e: KeyboardEvent): void {
      // While a menu is open, keys belong to it.
      // A focused video keeps its own keys: arrows seek, Space plays.
      const mediaKey = e.target instanceof HTMLMediaElement && (e.key.startsWith('Arrow') || e.key === ' ');
      if (e.defaultPrevented || referenceOpen || upscaleOpen || moreOpen || isTyping(e.target) || mediaKey) {
        upscaleWaitingSince = null;
        return;
      }
      if (onupscale && asset && offeredUpscaleFactors(asset).length > 0) {
        const chord = stepUpscaleChord(upscaleWaitingSince, e, Date.now());
        upscaleWaitingSince = chord.waitingSince;
        if (chord.value) {
          if (runUpscale(chord.value)) e.preventDefault();
          return;
        }
        if (chord.consumed) {
          e.preventDefault();
          return;
        }
      }
      const action = viewerActionFor(e);
      // A held Del must not open a second confirmation.
      if ((action === 'delete' || action === 'delete-now') && e.repeat) {
        e.preventDefault();
        return;
      }
      if (action && runAction(action)) e.preventDefault();
    }
    document.addEventListener('keydown', handleKeydown);
    return () => {
      document.removeEventListener('keydown', handleKeydown);
      release();
    };
  });
</script>

{#if open && asset}
  <div
    class="asset-viewer ui-viewport"
    role="dialog"
    aria-modal="true"
    aria-labelledby="asset-viewer-title"
    data-testid="asset-viewer"
    data-details={detailsOpen}
    data-compact={compact}
    data-phase={phase}
    data-flight={flight !== null}
    data-chrome={chromeHidden ? 'hidden' : 'shown'}
    style:--pull={drag?.axis === 'y' ? pullBackdropOpacity(drag.offset) : undefined}
  >
    <!-- On a desktop or a tablet the viewer is a panel over the page; a click on the backdrop closes it. -->
    <button type="button" class="viewer-backdrop" aria-label="Close viewer" tabindex="-1" onclick={close}></button>
    <div class="viewer-panel">
    <div class="viewer-bar">
      <div class="viewer-title">
        <b id="asset-viewer-title" class="truncate">{asset.filename}</b>
        <small>{index + 1} of {assets.length}{setLabel ? ` · ${setLabel}` : ''}</small>
      </div>
      {#if compact}
        <button type="button" class="viewer-btn ui-btn ui-btn-icon" data-action="details" aria-label="Details" aria-expanded={detailsOpen} aria-controls="asset-viewer-details" onclick={toggleDetails}><Icon name="info" size={16} /></button>
        <button type="button" bind:this={moreButton} class="viewer-btn ui-btn ui-btn-icon" data-action="more" aria-label="Actions" aria-haspopup="menu" aria-expanded={moreOpen} onclick={() => { contextAt = null; moreOpen = !moreOpen; }}><Icon name="more" size={16} /></button>
        <button type="button" bind:this={closeButton} class="viewer-btn ui-btn ui-btn-icon" aria-label="Close viewer" onclick={close}><Icon name="close" size={16} /></button>
      {:else}
      {#if onreuse}
        <button
          type="button"
          class="viewer-btn ui-btn ui-btn-primary"
          data-action="reuse"
          disabled={!canReuse}
          title={canReuse ? 'Load these settings into the workspace (R)' : 'Reusable settings unavailable'}
          onclick={() => onreuse(asset)}
        ><Icon name="reuse" size={14} /><span class="viewer-btn-label">Reuse settings</span></button>
      {/if}
      {#if onreference && canUseAsReference(asset)}
        <button
          type="button"
          bind:this={referenceButton}
          class="viewer-btn ui-btn"
          data-action="reference"
          aria-haspopup="menu"
          aria-expanded={referenceOpen}
          title="Use as reference (E)"
          onclick={() => { referenceOpen = !referenceOpen; }}
        ><Icon name="reference" size={14} /><span class="viewer-btn-label">Use as reference</span><Icon name="chevdown" size={12} class="opacity-70" /></button>
      {/if}
      {#if showExtend}
        <button
          type="button"
          class="viewer-btn ui-btn"
          data-action="extend"
          disabled={!canExtend(asset)}
          title={extendBlocked ?? 'Extend this video (E)'}
          onclick={() => onextend?.(asset)}
        ><Icon name="extend" size={14} /><span class="viewer-btn-label">Extend</span></button>
      {/if}
      {#if upscaleItems.length > 0}
        <button
          type="button"
          bind:this={upscaleButton}
          class="viewer-btn ui-btn"
          data-action="upscale"
          aria-haspopup="menu"
          aria-expanded={upscaleOpen}
          title="Upscale (X then {upscaleKeys})"
          onclick={() => { upscaleOpen = !upscaleOpen; }}
        ><Icon name="expand" size={14} /><span class="viewer-btn-label">Upscale</span><Icon name="chevdown" size={12} class="opacity-70" /></button>
      {/if}
      <a
        bind:this={downloadLink}
        class="viewer-btn ui-btn"
        data-action="download"
        href={asset.url}
        download={asset.filename}
        title="Download (D)"
      >
        <Icon name="download" size={14} /><span class="viewer-btn-label">Download</span>
      </a>
      {#if ondelete}
        <button
          type="button"
          class="viewer-btn ui-btn ui-btn-danger"
          data-action="delete"
          disabled={deleting}
          title="Delete (Del)"
          onclick={() => ondelete(asset)}
        ><Icon name="trash" size={14} /><span class="viewer-btn-label">{deleting ? 'Deleting…' : 'Delete'}</span></button>
      {/if}
      <span class="viewer-sep" aria-hidden="true"></span>
      <button
        type="button"
        class="viewer-btn ui-btn"
        data-action="details"
        aria-expanded={detailsOpen}
        aria-controls="asset-viewer-details"
        title="Details (I)"
        onclick={toggleDetails}
      ><Icon name="info" size={14} /><span class="viewer-btn-label">Details</span> <kbd class="viewer-kbd">I</kbd></button>
      <button
        type="button"
        class="viewer-btn ui-btn ui-btn-icon"
        data-action="shortcuts"
        aria-label="Keyboard shortcuts"
        aria-expanded={helpOpen}
        aria-controls="asset-viewer-shortcuts"
        title="Keyboard shortcuts (?)"
        onclick={() => { helpOpen = !helpOpen; }}
      ><kbd class="viewer-kbd">?</kbd></button>
      <button
        type="button"
        bind:this={closeButton}
        class="viewer-btn ui-btn ui-btn-icon"
        aria-label="Close viewer"
        title="Close (Esc)"
        onclick={close}
      ><Icon name="close" size={16} /></button>
      {/if}
    </div>

    <div class="viewer-main">
      <!-- svelte-ignore a11y_no_static_element_interactions because the swipe and pinch duplicate the arrows, keys and click -->
      <div
        bind:this={stageEl}
        class="viewer-stage"
        data-touch={touchLayer}
        use:hold={{ onhold: () => { contextAt = pressAt; moreOpen = true; }, ignore: 'video, button' }}
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
          aria-label="Previous asset"
          disabled={!hasPrev}
          onclick={() => hasPrev && onnavigate(index - 1)}
        ><Icon name="chevleft" size={20} /></button>
        <!-- The pager: the shown asset between its neighbours' thumbnails, which a finger drags into view. -->
        <div bind:this={trackEl} class="viewer-track" style:--pane-gap="{paneGap}px" style:transform={drag?.axis === 'x' ? `translateX(${drag.offset}px)` : undefined}>
          {#if touchLayer && hasPrev}
            <div class="viewer-pane viewer-pane-prev" aria-hidden="true">{@render neighbour(assets[index - 1])}</div>
          {/if}
          <div bind:this={paneEl} class="viewer-pane viewer-pane-current" style:transform={drag?.axis === 'y' ? pullTransform(drag.offset) : undefined}>
            {#if !mediaLoaded && asset.media_type !== 'video'}
              <!-- The thumbnail stands in until the image has loaded. -->
              <img class="viewer-standin" src={thumbOf(asset)} alt="" aria-hidden="true">
            {/if}
            {#if asset.media_type === 'video'}
              <!-- svelte-ignore a11y_media_has_caption because generated videos have no caption tracks -->
              <video src={asset.url} controls class="viewer-media"></video>
            {:else}
              <!-- svelte-ignore a11y_no_noninteractive_element_interactions because the zoom is a closer look; the keys and buttons do the rest -->
              <img
                bind:this={mediaEl}
                src={asset.url}
                alt={asset.prompt || asset.filename}
                class="viewer-media"
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
          </div>
          {#if touchLayer && hasNext}
            <div class="viewer-pane viewer-pane-next" aria-hidden="true">{@render neighbour(assets[index + 1])}</div>
          {/if}
        </div>
        <button
          type="button"
          class="viewer-nav next ui-btn ui-btn-icon"
          aria-label="Next asset"
          disabled={!hasNext}
          onclick={() => hasNext && onnavigate(index + 1)}
        ><Icon name="chevright" size={20} /></button>
      </div>

      {#if detailsOpen}
        <aside id="asset-viewer-details" class="viewer-details" class:ui-sheet={compact} aria-label="Asset details" use:sheetDrag={{ enabled: compact, onclose: () => { if (detailsOpen) toggleDetails(); } }}>
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
              <h4 class="viewer-h ui-area-label">Upscaled from</h4>
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

    {#if assets.length > 1}
      <div class="viewer-film" aria-label="Assets in this set" bind:this={filmEl}>
        {#each assets as item, itemIndex (item.id)}
          <button
            type="button"
            data-film-index={itemIndex}
            aria-label="Show {item.filename}"
            aria-current={itemIndex === index}
            onclick={() => onnavigate(itemIndex)}
          >
            {#if item.media_type === 'video' && Math.abs(itemIndex - index) > FILM_VIDEO_RADIUS}
              <span class="film-video-placeholder" aria-hidden="true">▶</span>
            {:else if item.media_type === 'video'}
              <video src={item.thumbnail_url || item.url} muted preload="metadata"></video>
            {:else}
              <img src={item.thumbnail_url || item.url} alt="" loading="lazy">
            {/if}
          </button>
        {/each}
      </div>
    {/if}

    {#if contextAt}<span bind:this={contextAnchorEl} class="viewer-context-anchor" style:left="{contextAt.x}px" style:top="{contextAt.y}px" aria-hidden="true"></span>{/if}
    <ActionMenu open={moreOpen} anchor={contextAt ? contextAnchorEl : moreButton} items={moreItems} label={asset.filename} onclose={() => { moreOpen = false; contextAt = null; }} />

    {#if helpOpen}
      <div id="asset-viewer-shortcuts" class="viewer-shortcuts ui-overlay" role="note" aria-label="Keyboard shortcuts">
        <h4 class="viewer-h ui-area-label">Keyboard shortcuts</h4>
        <ShortcutList entries={VIEWER_SHORTCUTS} />
      </div>
    {/if}

    </div>
    <ActionMenu
      open={referenceOpen}
      anchor={referenceButton}
      items={referenceItems}
      label="Use as reference"
      onclose={() => { referenceOpen = false; }}
    />
    <ActionMenu
      open={upscaleOpen}
      anchor={upscaleButton}
      items={upscaleItems}
      label="Upscale"
      onclose={() => { upscaleOpen = false; }}
    />
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
    data-testid="viewer-flight"
    style:left="{flight.rect.left}px"
    style:top="{flight.rect.top}px"
    style:width="{flight.rect.width}px"
    style:height="{flight.rect.height}px"
  >
{/if}

{#snippet neighbour(item: GalleryAsset)}
  <!-- A neighbour waiting beside the shown asset: its thumbnail, or a video's first frame. -->
  {#if item.media_type === 'video'}
    <video class="viewer-neighbour" src={thumbOf(item)} muted preload="metadata"></video>
  {:else}
    <img class="viewer-neighbour" src={thumbOf(item)} alt="">
  {/if}
{/snippet}

{#snippet factSection(title: string, facts: DetailFact[])}
  {#if facts.length > 0}
    <section>
      <h4 class="viewer-h ui-area-label">{title}</h4>
      <dl class="viewer-facts">
        {#each facts as fact (fact.label)}
          <div class:wide={fact.wide}>
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
  .viewer-bar, .viewer-main, .viewer-film { position: relative; }
  .viewer-bar, .viewer-film { opacity: var(--pull, 1); transition: opacity 160ms ease; }
  .asset-viewer[data-chrome='hidden'] .viewer-bar, .asset-viewer[data-chrome='hidden'] .viewer-film { opacity: 0; pointer-events: none; }
  .viewer-bar { display: flex; flex: none; align-items: center; gap: 6px; min-height: 44px; padding: 6px 12px 6px 16px; background: var(--color-bg-surface); }
  .viewer-title { display: flex; flex: 1 1 auto; min-width: 7rem; flex-direction: column; }
  .viewer-title b { font-size: var(--text-content); font-weight: 600; }
  .viewer-title small { font-size: var(--text-meta); color: var(--color-text-muted); }
  .viewer-sep { width: 1px; height: 20px; margin: 0 2px; background: var(--color-border-subtle); }
  .viewer-kbd { padding: 0 4px; border-radius: var(--radius-xs); background: var(--color-bg-surface-hover); font-family: var(--font-mono); font-size: var(--text-meta); line-height: 15px; color: var(--color-text-muted); }
  .viewer-shortcuts { position: absolute; top: 52px; right: 12px; z-index: 3; width: 280px; padding: 12px; }
  .viewer-main { display: flex; flex: 1; min-height: 0; }
  .viewer-stage { position: relative; display: flex; flex: 1; min-width: 0; align-items: center; justify-content: center; padding: 20px 64px; }
  .viewer-media { max-width: 100%; max-height: 100%; object-fit: contain; border-radius: var(--radius-xs); transform-origin: center; user-select: none; }
  /* A finger's hold opens our menu, not iOS's image callout (Copy, Save Image); the whole stage opts out of it. */
  .viewer-stage, .viewer-stage img, .viewer-stage video { -webkit-touch-callout: none; -webkit-user-select: none; user-select: none; }
  .viewer-media.zoomable { cursor: zoom-in; }
  .viewer-media.zoomed { position: relative; z-index: 1; cursor: grab; }
  .viewer-media.animated { transition: transform 0.18s ease-out; }
  /* The stage clips a zoomed image; the bar and the strip stay on top of it. */
  .viewer-stage { overflow: hidden; }
  /* A finger owns every gesture on the stage, so the browser neither pans nor zooms the page: only the image zooms. */
  .viewer-stage[data-touch='true'] { touch-action: none; }
  .viewer-nav { position: absolute; top: 50%; z-index: 2; width: var(--spacing-bar); height: var(--spacing-bar); transform: translateY(-50%); }
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
  .viewer-facts dt { font-size: var(--text-meta); color: var(--color-text-muted); }
  .viewer-facts dd { margin: 2px 0 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-family: var(--font-mono); font-size: var(--text-ui); color: var(--color-text-primary); }
  .viewer-film { display: flex; flex: none; gap: 6px; overflow-x: auto; padding: 8px 12px 12px; }
  /* Auto margins centre a short strip but keep a long one scrollable from its first item. */
  .viewer-film > :first-child { margin-left: auto; }
  .viewer-film > :last-child { margin-right: auto; }
  .viewer-film button { flex: none; width: 44px; height: 44px; overflow: hidden; padding: 0; border: 2px solid transparent; border-radius: var(--radius-sm); opacity: 0.55; transition: opacity 0.12s ease; }
  @media (hover: hover) { .viewer-film button:hover { opacity: 0.9; } }
  .viewer-film button[aria-current='true'] { border-color: var(--color-primary-main); opacity: 1; }
  .film-video-placeholder { display: grid; width: 100%; height: 100%; place-items: center; background: var(--color-bg-surface); font-size: var(--text-ui); color: var(--color-text-muted); }
  .viewer-film img, .viewer-film video { display: block; width: 100%; height: 100%; object-fit: cover; }
  @media (max-width: 767px) {
    .viewer-bar { flex-wrap: wrap; padding: 8px; }
    .viewer-stage { padding: 12px 52px; }
    .viewer-details { position: absolute; inset: auto 0 0 0; z-index: 3; width: auto; max-height: 50%; }
  }
  /* Phone: no arrows (a swipe moves), the bar pads for the status bar, the details rise as a sheet. */
  .asset-viewer[data-compact='true'] .viewer-bar { flex-wrap: nowrap; padding: calc(6px + env(safe-area-inset-top, 0px)) max(8px, env(safe-area-inset-right, 0px)) 6px max(12px, env(safe-area-inset-left, 0px)); }
  .asset-viewer[data-compact='true'] .viewer-nav { display: none; }
  .viewer-details-scrim { position: absolute; inset: 0; z-index: 2; background: transparent; cursor: default; }
  .asset-viewer[data-compact='true'] .viewer-stage { padding: 8px 12px; }
  /* The details rise as a half sheet under the image, which moves up to make room; the strip goes meanwhile. */
  .asset-viewer[data-compact='true'] .viewer-main { flex-direction: column; }
  .asset-viewer[data-compact='true'] .viewer-details { position: static; z-index: 3; width: auto; max-height: 50%; flex: none; }
  .asset-viewer[data-compact='true'][data-details='true'] .viewer-film { display: none; }
  .asset-viewer[data-compact='true'] .viewer-details-body { padding: 4px max(16px, env(safe-area-inset-right, 0px)) calc(16px + env(safe-area-inset-bottom, 0px)) max(16px, env(safe-area-inset-left, 0px)); }
  /* Below a full desktop the labelled buttons show their icons only, so the title keeps its room; the tooltip names them. */
  @media (max-width: 1023px) {
    .viewer-btn-label { position: absolute; width: 1px; height: 1px; overflow: hidden; clip-path: inset(50%); white-space: nowrap; }
  }
  .asset-viewer[data-compact='true'] .viewer-film { padding-bottom: calc(12px + env(safe-area-inset-bottom, 0px)); }

  /* The pager: the shown asset's pane between its neighbours, one pane to each side, moved by transform alone. */
  .viewer-track { position: relative; display: flex; width: 100%; height: 100%; will-change: transform; }
  .viewer-pane { position: relative; display: flex; flex: none; width: 100%; height: 100%; align-items: center; justify-content: center; }
  .viewer-pane-prev, .viewer-pane-next { position: absolute; top: 0; }
  .viewer-pane-prev { left: calc(-100% - var(--pane-gap, 0px)); }
  .viewer-pane-next { left: calc(100% + var(--pane-gap, 0px)); }
  .viewer-neighbour { width: 100%; height: 100%; object-fit: contain; border-radius: var(--radius-xs); }
  .viewer-standin { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: contain; border-radius: var(--radius-xs); pointer-events: none; }
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
