<script lang="ts">
  import { onMount, tick, untrack } from 'svelte';
  import { Icon, Spinner } from '$lib/components/atoms';
  import { Segmented, requestConfirm } from '$lib/components/molecules';
  import { pushEscapeLayer } from '$lib/actions/escapeStack';
  import { portal } from '$lib/actions/portal';
  import { isPlainKey, isCommandKey } from '$lib/keyboard';
  import { draft } from '$lib/state/draft.svelte';
  import { EXTEND_SIDES } from '$lib/state/editSlots';
  import { heldUploads } from '$lib/state/heldUploads.svelte';
  import { layout } from '$lib/state/layoutMode.svelte';
  import type { EditMode, ExtendSide } from '$lib/types';
  import {
    BRUSH_MAX, BRUSH_MIN, BRUSH_STEP, EXTEND_MAX, EXTEND_MIN, EXTEND_STEP,
    contentRect, extensionRect, sameOps, type Stroke,
  } from './editOps';
  import { imageEdit } from './editStore.svelte';
  import { sizeLine } from './editView';
  import { decodeMask, drawOverlay, drawStroke, drawStrokes, makeScratch, maskScale, maskUpload, readPalette, type Palette } from './paintCanvas';
  import {
    ZOOM_MAX, ZOOM_MIN, clampZoom, fitScale, fitView, panBy, pinchView, ringDiameter, scaleOf, stepZoom, toImage, toScreen, zoomAbout, zoomLabel,
    type PinchStart, type Point, type View,
  } from './paintView';

  interface Props {
    /** Called when the painter closes: with `true` after Done, with `false` after Cancel. */
    onclose: () => void;
  }

  let { onclose }: Props = $props();

  type Tool = 'brush' | 'erase';

  const source = $derived(imageEdit.source);
  const width = $derived(source?.width ?? null);
  const height = $derived(source?.height ?? null);
  const extension = $derived(imageEdit.extension);
  const cutout = $derived(source?.cutout === true);

  // The layer is one component with three arrangements: a floating bar on wide and tall screens, docked rows on a phone held upright,
  // and a single docked row on a phone held sideways, where there is no height for a top bar.
  const variant = $derived<'wide' | 'phone' | 'landscape'>(layout.short ? 'landscape' : layout.compact ? 'phone' : 'wide');

  // The mode is the history's: a switch is one of its steps, so Undo and Cancel bring the other mode back with its work.
  const mode = $derived(imageEdit.mode);
  let tool = $state<Tool>('brush');
  // With no history yet, the paint the draft holds (a mask restored after a reload) is what the strokes sit on.
  imageEdit.begin(draft.state.editMaskUpload !== null ? heldUploads.editMask : null);
  // The history when the painter opened: Cancel goes back to it.
  const opened = imageEdit.snapshot();
  let confirming = false;
  let announcement = $state('');

  // --- The canvas area and the view ------------------------------------------------------------------------------------
  let rootEl = $state<HTMLElement | null>(null);
  let surfaceEl = $state<HTMLElement | null>(null);
  let overlayEl = $state<HTMLCanvasElement | null>(null);
  let areaW = $state(0);
  let areaH = $state(0);
  const area = $derived({ w: areaW, h: areaH });
  let loaded = $state(false);
  let failed = $state(false);
  // The picture is drawn once it has loaded and the store knows its size (learnt in the background when the source is shown).
  const ready = $derived(loaded && width !== null && height !== null && areaW > 0 && areaH > 0);

  // Paint shows the picture alone, at the zoom and place the user left it; Extend shows the whole canvas, always at Fit.
  let paintView = $state<View>({ z: ZOOM_MIN, cx: 0, cy: 0 });
  const content = $derived(
    mode === 'extend' && width !== null && height !== null
      ? contentRect(width, height, extension)
      : { x: 0, y: 0, w: width ?? 1, h: height ?? 1 },
  );
  const fit = $derived(fitScale(area, content));
  // Zoom 1 is Fit, whatever the stored centre says, so the picture is always centred at Fit.
  const view = $derived<View>(mode === 'paint' && paintView.z > ZOOM_MIN ? paintView : fitView(content));
  const scale = $derived(scaleOf(view, fit));
  const imageBox = $derived.by(() => {
    if (width === null || height === null) return { x: 0, y: 0, w: 0, h: 0 };
    const origin = toScreen(view, fit, area, { x: 0, y: 0 });
    return { x: origin.x, y: origin.y, w: width * scale, h: height * scale };
  });
  const newArea = $derived.by(() => {
    // Paint shows the picture alone; the new area belongs to Extend.
    if (mode !== 'extend' || !extension || width === null || height === null) return null;
    const rect = extensionRect(width, height, extension);
    const origin = toScreen(view, fit, area, { x: rect.x, y: rect.y });
    return { x: origin.x, y: origin.y, w: rect.w * scale, h: rect.h * scale };
  });

  const reducedMotion = (): boolean => window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  let tween = 0;
  // Where a zoom button's animation is heading, so a second press during it steps from there and not from halfway.
  let heading: View | null = null;

  /** Go to a view: at once for a pinch, a drag or reduced motion, else over 120 ms (a zoom button). */
  function setView(next: View, animated = false): void {
    cancelAnimationFrame(tween);
    heading = null;
    if (!animated || reducedMotion()) {
      paintView = next;
      return;
    }
    const from = view;
    const started = performance.now();
    heading = next;
    const step = (now: number): void => {
      const t = Math.min(1, (now - started) / 120);
      const e = 1 - (1 - t) * (1 - t);
      paintView = { z: from.z + (next.z - from.z) * e, cx: from.cx + (next.cx - from.cx) * e, cy: from.cy + (next.cy - from.cy) * e };
      if (t < 1) tween = requestAnimationFrame(step);
      else heading = null;
    };
    tween = requestAnimationFrame(step);
  }

  function zoomTo(z: number, animated = true): void {
    if (mode !== 'paint' || width === null || height === null) return;
    const anchor = { x: area.w / 2, y: area.h / 2 };
    setView(zoomAbout(heading ?? view, fit, area, content, anchor, z), animated);
  }

  const zoomBy = (direction: 1 | -1): void => zoomTo(stepZoom((heading ?? view).z, direction));
  const zoomFit = (): void => {
    if (width !== null && height !== null) setView(fitView({ x: 0, y: 0, w: width, h: height }), true);
  };

  // --- The mask --------------------------------------------------------------------------------------------------------
  let maskEl: HTMLCanvasElement | null = null;
  let maskFactor = 1;
  // The history the mask was last drawn from, so a stroke just committed does not redraw what is already on it.
  let drawnOps: readonly unknown[] | null = null;
  let drawnBase: ImageBitmap | null = null;
  let maskTick = $state(0);

  function ensureMask(): HTMLCanvasElement | null {
    if (width === null || height === null) return null;
    const factor = maskScale(width, height);
    const mw = Math.max(1, Math.round(width * factor));
    const mh = Math.max(1, Math.round(height * factor));
    if (!maskEl || maskEl.width !== mw || maskEl.height !== mh) {
      maskEl = document.createElement('canvas');
      maskEl.width = mw;
      maskEl.height = mh;
      maskFactor = factor;
      drawnOps = null;
    }
    return maskEl;
  }

  // The picture of the paint the strokes sit on, once it is decoded; null while it loads or when there is none.
  let baseEl = $state.raw<ImageBitmap | null>(null);

  $effect(() => {
    const file = imageEdit.base;
    let current = true;
    void decodeMask(file).then((bitmap) => {
      if (current) baseEl = bitmap;
    });
    return () => { current = false; };
  });

  function rebuildMask(): void {
    const canvas = ensureMask();
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;
    drawStrokes(ctx, imageEdit.strokes, maskFactor, canvas.width, canvas.height, imageEdit.baseShown ? baseEl : null);
    drawnOps = imageEdit.ops;
    drawnBase = baseEl;
    maskTick += 1;
  }

  // Undo, Clear and Cancel change the history from outside the pointer, and the paint under it arrives once decoded: draw the mask again.
  $effect(() => {
    const ops = imageEdit.ops;
    const base = baseEl;
    void width;
    void height;
    if (width === null || height === null || (ops === drawnOps && base === drawnBase)) return;
    untrack(rebuildMask);
  });

  // --- Drawing the overlay ---------------------------------------------------------------------------------------------
  const scratch = makeScratch();
  let palette: Palette | null = null;
  let frame = 0;

  function draw(): void {
    frame = 0;
    if (!overlayEl || !palette || width === null || height === null) return;
    drawOverlay({
      canvas: overlayEl,
      scratch,
      mask: maskEl,
      image: { w: width, h: height },
      view,
      fit,
      area,
      dpr: Math.min(2, window.devicePixelRatio || 1),
      palette,
      join: newArea && extension ? { rect: extensionRect(width, height, extension), side: extension.side } : null,
    });
  }

  function schedule(): void {
    if (!frame) frame = requestAnimationFrame(draw);
  }

  $effect(() => {
    void [view.z, view.cx, view.cy, areaW, areaH, maskTick, mode, extension?.side, extension?.amount, ready, width, height];
    untrack(schedule);
  });

  // --- Loading the picture ---------------------------------------------------------------------------------------------
  onMount(() => {
    if (rootEl) palette = readPalette(rootEl);
    return () => {
      cancelAnimationFrame(frame);
      cancelAnimationFrame(tween);
    };
  });

  // --- Painting --------------------------------------------------------------------------------------------------------
  const pointers = new Map<number, Point>();
  let stroke: Stroke | null = null;
  let strokePointer: number | null = null;
  let pinch: PinchStart | null = null;
  let dragging: { last: Point } | null = null;
  let spaceDown = false;
  let overSurface = false;
  let ring = $state<{ x: number; y: number; touch: boolean } | null>(null);
  let panCursor = $state(false);

  function local(event: PointerEvent): Point {
    const rect = surfaceEl?.getBoundingClientRect();
    return { x: event.clientX - (rect?.left ?? 0), y: event.clientY - (rect?.top ?? 0) };
  }

  function imagePoint(point: Point): Point {
    return toImage(view, fit, area, point);
  }

  function startStroke(event: PointerEvent, point: Point): void {
    const at = imagePoint(point);
    stroke = { erase: tool === 'erase', size: imageEdit.brush, points: [at.x, at.y] };
    strokePointer = event.pointerId;
    const ctx = ensureMask()?.getContext('2d');
    if (ctx) drawStroke(ctx, stroke, maskFactor, 0);
    maskTick += 1;
  }

  function extendStroke(event: PointerEvent): void {
    if (!stroke) return;
    const before = stroke.points.length;
    const events = typeof event.getCoalescedEvents === 'function' ? event.getCoalescedEvents() : [];
    for (const item of events.length > 0 ? events : [event]) {
      const at = imagePoint(local(item));
      stroke.points.push(at.x, at.y);
    }
    const ctx = ensureMask()?.getContext('2d');
    if (ctx) drawStroke(ctx, stroke, maskFactor, before);
    maskTick += 1;
  }

  function endStroke(): void {
    if (stroke && strokePointer !== null) {
      imageEdit.addStroke(stroke);
      drawnOps = imageEdit.ops;
    }
    stroke = null;
    strokePointer = null;
  }

  /** A second finger lands: the stroke the first began was a pinch, not paint. Take it back off the mask. */
  function cancelStroke(): void {
    if (!stroke) return;
    stroke = null;
    strokePointer = null;
    rebuildMask();
  }

  function onpointerdown(event: PointerEvent): void {
    if (mode !== 'paint' || !ready || (event.pointerType === 'mouse' && event.button > 1)) return;
    surfaceEl?.setPointerCapture(event.pointerId);
    const point = local(event);
    pointers.set(event.pointerId, point);
    if (pointers.size === 1) {
      if (event.pointerType === 'mouse' && (event.button === 1 || spaceDown)) {
        dragging = { last: point };
        panCursor = true;
        event.preventDefault();
      } else if (event.button <= 0) {
        startStroke(event, point);
        ring = { ...point, touch: event.pointerType !== 'mouse' };
      }
    } else if (pointers.size === 2) {
      cancelStroke();
      dragging = null;
      ring = null;
      const [a, b] = [...pointers.values()];
      pinch = { view, mid: { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 }, distance: Math.hypot(a.x - b.x, a.y - b.y) };
    }
  }

  function onpointermove(event: PointerEvent): void {
    const point = local(event);
    overSurface = true;
    if (mode === 'paint' && !pinch && !dragging && (event.pointerType === 'mouse' || pointers.has(event.pointerId))) {
      ring = { ...point, touch: event.pointerType !== 'mouse' };
    }
    if (!pointers.has(event.pointerId)) return;
    pointers.set(event.pointerId, point);
    if (pinch && pointers.size === 2) {
      const [a, b] = [...pointers.values()];
      setView(pinchView(pinch, { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 }, Math.hypot(a.x - b.x, a.y - b.y), fit, area, content));
    } else if (dragging) {
      setView(panBy(view, fit, content, point.x - dragging.last.x, point.y - dragging.last.y));
      dragging.last = point;
    } else if (stroke && event.pointerId === strokePointer) {
      extendStroke(event);
    }
  }

  function onpointerup(event: PointerEvent): void {
    if (!pointers.has(event.pointerId)) return;
    if (event.pointerId === strokePointer) {
      if (event.type === 'pointercancel') cancelStroke();
      else endStroke();
    }
    pointers.delete(event.pointerId);
    if (pointers.size < 2) pinch = null;
    if (pointers.size === 0) {
      dragging = null;
      panCursor = spaceDown && overSurface;
      if (event.pointerType !== 'mouse') ring = null;
    }
  }

  function onpointerleave(event: PointerEvent): void {
    overSurface = false;
    if (event.pointerType === 'mouse' && pointers.size === 0) ring = null;
  }

  function onwheel(event: WheelEvent): void {
    if (mode !== 'paint' || !ready) return;
    event.preventDefault();
    const rect = surfaceEl?.getBoundingClientRect();
    const anchor = { x: event.clientX - (rect?.left ?? 0), y: event.clientY - (rect?.top ?? 0) };
    setView(zoomAbout(view, fit, area, content, anchor, clampZoom(view.z * Math.exp(-event.deltaY * (event.ctrlKey ? 0.01 : 0.0015)))));
  }

  // Not a Svelte handler: a wheel listener must be able to cancel the page's scroll.
  $effect(() => {
    const surface = surfaceEl;
    if (!surface) return;
    surface.addEventListener('wheel', onwheel, { passive: false });
    return () => surface.removeEventListener('wheel', onwheel);
  });

  // --- Keys ------------------------------------------------------------------------------------------------------------
  const FOCUSABLE = 'button:not([disabled]), input:not([disabled]), [href], [tabindex]:not([tabindex="-1"])';

  function typingElsewhere(target: EventTarget | null): boolean {
    if (!(target instanceof HTMLElement)) return false;
    if (target instanceof HTMLInputElement) return !['range', 'checkbox', 'radio', 'button'].includes(target.type);
    return target.isContentEditable || ['TEXTAREA', 'SELECT'].includes(target.tagName);
  }

  function nudgeBrush(direction: 1 | -1): void {
    imageEdit.setBrush(imageEdit.brush + direction * BRUSH_STEP * 2);
    announcement = `Brush ${imageEdit.brush} pixels`;
  }

  function setTool(next: Tool): void {
    tool = next;
    announcement = next === 'brush' ? 'Brush' : 'Eraser';
  }

  const NO_EXTEND_REASON = 'Extend is off: this picture has transparent parts.';

  /** Whether a mode can be entered, with the reason when it cannot: a cut-out cannot be extended. */
  function refusal(next: EditMode): string | null {
    return next === 'extend' && cutout ? NO_EXTEND_REASON : null;
  }

  /** What the live region says once the mode has changed, by a switch or by Undo. */
  function modeAnnouncement(current: EditMode): string {
    if (current === 'paint') return 'Paint';
    const now = imageEdit.extension;
    return now && width !== null && height !== null ? `Extend. Extending ${now.side} by ${now.amount} pixels, ${sizeLine(width, height, now).replace('→', 'to')}` : 'Extend. Choose a side to extend.';
  }

  let announcedMode = untrack(() => mode);

  // A half-drawn stroke or gesture does not carry over to the other mode, however the mode changed.
  $effect(() => {
    const current = mode;
    untrack(() => {
      if (current === announcedMode) return;
      announcedMode = current;
      cancelStroke();
      pointers.clear();
      pinch = null;
      dragging = null;
      ring = null;
      announcement = modeAnnouncement(current);
    });
  });

  /**
   * Switch mode: the other mode's work goes, so with work to lose it asks first (Undo brings the work back). Keys 1 and 2 and the
   * switch come here. A mode that cannot be entered says why instead.
   */
  async function chooseMode(next: EditMode): Promise<void> {
    if (next === mode || confirming) return;
    const reason = refusal(next);
    if (reason) {
      announcement = reason;
      return;
    }
    cancelStroke();
    pointers.clear();
    pinch = null;
    dragging = null;
    ring = null;
    if (imageEdit.hasWork) {
      const losing = mode === 'paint' ? 'paint' : 'extension';
      confirming = true;
      const approved = await requestConfirm({
        question: `Remove the ${losing} and switch to ${next === 'extend' ? 'Extend' : 'Paint'}?`,
        info: `One kind of change is sent at a time. Undo brings the ${losing} back.`,
        confirmLabel: mode === 'paint' ? 'Remove paint' : 'Remove extension',
        cancelLabel: mode === 'paint' ? 'Keep painting' : 'Keep extending',
      });
      confirming = false;
      if (!approved) {
        void tick().then(() => rootEl?.querySelector<HTMLElement>('[data-testid="painter-mode"] [aria-pressed="true"]')?.focus());
        return;
      }
    }
    imageEdit.setMode(next);
  }

  function handleKeydown(event: KeyboardEvent): void {
    const root = rootEl;
    if (!root || event.defaultPrevented) return;
    if (event.key === 'Tab') {
      if (!root.contains(event.target as Node)) return;
      const focusable = [...root.querySelectorAll<HTMLElement>(FOCUSABLE)].filter((el) => el.offsetParent !== null);
      if (focusable.length === 0) {
        event.preventDefault();
        return;
      }
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      const at = document.activeElement;
      if (event.shiftKey && (at === first || at === root)) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && (at === last || at === root)) {
        event.preventDefault();
        first.focus();
      }
      return;
    }
    if (typingElsewhere(event.target)) return;
    if (isCommandKey(event) && !event.altKey && event.key.toLowerCase() === 'z' && !event.shiftKey) {
      event.preventDefault();
      imageEdit.undo();
      return;
    }
    if (!isPlainKey(event)) return;
    // Space pans while the pointer is over the picture; anywhere else it keeps pressing the focused button.
    if (event.key === ' ' && mode === 'paint' && overSurface) {
      event.preventDefault();
      if (!event.repeat) {
        spaceDown = true;
        panCursor = true;
        ring = null;
      }
      return;
    }
    const key = event.key.toLowerCase();
    if (key === '1' || key === '2') {
      event.preventDefault();
      void chooseMode(key === '1' ? 'paint' : 'extend');
      return;
    }
    if (mode === 'paint') {
      if (key === 'b') setTool('brush');
      else if (key === 'e') setTool('erase');
      else if (event.key === ',') nudgeBrush(-1);
      else if (event.key === '.') nudgeBrush(1);
      else if (event.key === '+' || event.key === '=') zoomBy(1);
      else if (event.key === '-') zoomBy(-1);
      else if (event.key === '0') zoomFit();
      else return;
      event.preventDefault();
    }
  }

  function handleKeyup(event: KeyboardEvent): void {
    if (event.key !== ' ') return;
    spaceDown = false;
    panCursor = dragging !== null;
  }

  // --- Closing ---------------------------------------------------------------------------------------------------------
  const changed = $derived(!sameOps(imageEdit.ops, opened));

  let committing = false;

  /**
   * Done keeps what was painted: the mask is encoded as a PNG at the size it is drawn at (the server scales it to the picture), and the
   * draft takes it with its share and the extension. Nothing painted keeps no mask.
   */
  async function done(): Promise<void> {
    if (committing) return;
    committing = true;
    try {
      // A picture whose size is not known has nothing to measure the paint on: nothing to keep or lose.
      if (width === null || height === null) {
        onclose();
        return;
      }
      // The mask is drawn again from the history and the paint under it, which may still be decoding, so what is kept is what the history says.
      if (imageEdit.baseShown && baseEl === null) baseEl = await decodeMask(imageEdit.base);
      rebuildMask();
      const canvas = ensureMask();
      const painted = canvas ? await maskUpload(canvas) : null;
      const extension = imageEdit.extension;
      draft.commitEdit({ mode: imageEdit.mode, mask: painted?.file ?? null, share: painted?.share ?? 0, extension, amount: extension?.amount ?? imageEdit.extendAmount });
      onclose();
    } finally {
      committing = false;
    }
  }

  /** Cancel (and Esc) puts the paint back as it was when the painter opened, asking first only if something changed. */
  async function cancel(): Promise<void> {
    if (confirming) return;
    if (changed) {
      confirming = true;
      const approved = await requestConfirm({
        question: 'Discard changes?',
        info: 'The paint and the extension go back to how they were when you opened this.',
        confirmLabel: 'Discard',
        cancelLabel: 'Keep editing',
      });
      confirming = false;
      if (!approved) return;
      imageEdit.restore(opened);
    }
    onclose();
  }

  onMount(() => {
    const release = pushEscapeLayer(() => void cancel());
    document.addEventListener('keydown', handleKeydown);
    document.addEventListener('keyup', handleKeyup);
    void tick().then(() => rootEl?.focus({ preventScroll: true }));
    return () => {
      release();
      document.removeEventListener('keydown', handleKeydown);
      document.removeEventListener('keyup', handleKeyup);
    };
  });

  // --- What the controls show ------------------------------------------------------------------------------------------
  const modeOptions = $derived([
    { value: 'paint' as const, label: 'Paint', icon: variant === 'wide' ? ('brush' as const) : undefined, title: 'Paint the area to change (1)' },
    { value: 'extend' as const, label: 'Extend', icon: variant === 'wide' ? ('extend' as const) : undefined, title: cutout ? NO_EXTEND_REASON : 'Give the picture more room on one side (2)', disabled: cutout },
  ]);
  const toolOptions = $derived([
    { value: 'brush' as const, label: variant === 'landscape' ? '' : 'Brush', icon: 'brush' as const, title: 'Brush (B)' },
    { value: 'erase' as const, label: variant === 'landscape' ? '' : 'Erase', icon: 'eraser' as const, title: 'Erase (E)' },
  ]);
  const SIDE_LABEL: Record<ExtendSide, string> = { top: '+ Top', right: '+ Right', bottom: '+ Bottom', left: '+ Left' };
  const fillOf = (value: number, min: number, max: number): string => `${((value - min) / (max - min)) * 100}%`;
  const brushDot = $derived(6 + ((imageEdit.brush - BRUSH_MIN) / (BRUSH_MAX - BRUSH_MIN)) * 16);
  const dimensions = $derived(width !== null && height !== null && !cutout ? sizeLine(width, height, extension) : '');
  const canClear = $derived(imageEdit.hasWork);
  const pill = $derived.by(() => {
    const box = imageBox;
    return {
      top: { left: box.x + box.w / 2, top: box.y + 8, shift: '-50% 0' },
      right: { left: box.x + box.w - 8, top: box.y + box.h / 2, shift: '-100% -50%' },
      bottom: { left: box.x + box.w / 2, top: box.y + box.h - 8, shift: '-50% -100%' },
      left: { left: box.x + 8, top: box.y + box.h / 2, shift: '0 -50%' },
    } satisfies Record<ExtendSide, { left: number; top: number; shift: string }>;
  });

  function clear(): void {
    if (mode === 'paint') imageEdit.clearPaint();
    else imageEdit.clearExtension();
  }

  function pickSide(side: ExtendSide): void {
    imageEdit.toggleSide(side);
    const now = imageEdit.extension;
    announcement = now && width !== null && height !== null ? `Extending ${now.side} by ${now.amount} pixels. ${sizeLine(width, height, now)}` : 'No extension';
  }

  function setAmount(value: number): void {
    imageEdit.setExtendAmount(value);
    const now = imageEdit.extension;
    announcement = `Extend by ${imageEdit.extendAmount} pixels${now && width !== null && height !== null ? `. ${sizeLine(width, height, now)}` : ''}`;
  }

  const ringSize = $derived(ringDiameter(imageEdit.brush, view, fit));
</script>

{#snippet modeSwitch()}
  <Segmented label="Mode" testId="painter-mode" value={mode} options={modeOptions} onchange={(next) => void chooseMode(next)} />
{/snippet}

{#snippet toolSwitch()}
  <Segmented label="Tool" value={tool} options={toolOptions} onchange={setTool} />
{/snippet}

{#snippet brushSize()}
  <span class="size-dot" aria-hidden="true"><span style:width="{brushDot}px" style:height="{brushDot}px"></span></span>
  <input
    class="ui-range painter-range"
    type="range"
    min={BRUSH_MIN}
    max={BRUSH_MAX}
    step={BRUSH_STEP}
    value={imageEdit.brush}
    aria-label="Brush size"
    aria-valuetext="{imageEdit.brush} pixels"
    style:--fill={fillOf(imageEdit.brush, BRUSH_MIN, BRUSH_MAX)}
    oninput={(e) => imageEdit.setBrush(Number(e.currentTarget.value))}
  >
  <span class="painter-value">{imageEdit.brush} px</span>
{/snippet}

{#snippet extendBy()}
  <label class="painter-label" for="painter-extend-by">Extend by</label>
  <input
    id="painter-extend-by"
    class="ui-range painter-range"
    type="range"
    min={EXTEND_MIN}
    max={EXTEND_MAX}
    step={EXTEND_STEP}
    value={imageEdit.extendAmount}
    aria-valuetext="{imageEdit.extendAmount} pixels"
    style:--fill={fillOf(imageEdit.extendAmount, EXTEND_MIN, EXTEND_MAX)}
    oninput={(e) => setAmount(Number(e.currentTarget.value))}
  >
  <span class="painter-value">{imageEdit.extendAmount} px</span>
{/snippet}

{#snippet history()}
  <button type="button" class="ui-btn" aria-label="Undo" title="Undo (Ctrl+Z)" disabled={!imageEdit.canUndo} onclick={() => imageEdit.undo()}><Icon name="undo" size={14} />{#if variant !== 'landscape'}<span>Undo</span>{/if}</button>
  <button type="button" class="ui-btn" aria-label="Clear" title={mode === 'paint' ? 'Clear the paint' : 'Remove the extension'} disabled={!canClear} onclick={clear}><Icon name="trash" size={14} />{#if variant !== 'landscape'}<span>Clear</span>{/if}</button>
{/snippet}

{#snippet zoom()}
  <div class="painter-zoom" role="group" aria-label="Zoom">
    <button type="button" class="ui-btn ui-btn-icon" aria-label="Zoom out" title="Zoom out (-)" disabled={view.z <= ZOOM_MIN} onclick={() => zoomBy(-1)}><Icon name="minus" size={14} /></button>
    <span class="painter-value painter-zoom-value">{zoomLabel(view.z)}</span>
    <button type="button" class="ui-btn ui-btn-icon" aria-label="Zoom in" title="Zoom in (+)" disabled={view.z >= ZOOM_MAX - 1e-6} onclick={() => zoomBy(1)}><Icon name="plus" size={14} /></button>
    <button type="button" class="ui-btn" title="Show the whole picture (0)" disabled={view.z <= ZOOM_MIN} onclick={zoomFit}>Fit</button>
  </div>
{/snippet}

<div
  bind:this={rootEl}
  use:portal
  class="painter ui-viewport"
  role="dialog"
  aria-modal="true"
  aria-label="Edit the picture"
  tabindex="-1"
>
  {#if variant === 'wide'}
    <header class="painter-top painter-top-wide">
      <div class="painter-name">
        <b class="truncate" title={source?.name}>{source?.name}</b>
        {#if width !== null && height !== null}<span class="painter-dims">{width} × {height}</span>{/if}
      </div>
      {@render modeSwitch()}
      <div class="painter-end">
        <button type="button" class="ui-btn" data-testid="painter-cancel" onclick={() => void cancel()}>Cancel</button>
        <button type="button" class="ui-btn ui-btn-primary" onclick={() => void done()}>Done</button>
      </div>
    </header>
  {:else if variant === 'phone'}
    <header class="painter-top painter-top-phone">
      <button type="button" class="ui-btn ui-btn-icon" aria-label="Cancel" data-testid="painter-cancel" onclick={() => void cancel()}><Icon name="close" size={16} /></button>
      {@render modeSwitch()}
      <button type="button" class="ui-btn" title="Show the whole picture" disabled={mode !== 'paint' || view.z <= ZOOM_MIN} onclick={zoomFit}>Fit</button>
    </header>
  {/if}

  <div class="painter-area" bind:clientWidth={areaW} bind:clientHeight={areaH}>
    <div
      bind:this={surfaceEl}
      class="painter-surface"
      class:pan={panCursor}
      class:painting={mode === 'paint'}
      data-own-drag
     
      role="presentation"
      {onpointerdown}
      {onpointermove}
      {onpointerup}
      onpointercancel={onpointerup}
      {onpointerleave}
      oncontextmenu={(e) => e.preventDefault()}
    >
      {#if source}
        {#if newArea}
          <div class="painter-new ui-checkerboard" style:left="{newArea.x}px" style:top="{newArea.y}px" style:width="{newArea.w}px" style:height="{newArea.h}px"></div>
        {/if}
        <img
          class="painter-img"
          class:ui-checkerboard={cutout}
          src={source.url}
          alt=""
          draggable="false"
          style:left="{imageBox.x}px"
          style:top="{imageBox.y}px"
          style:width="{imageBox.w}px"
          style:height="{imageBox.h}px"
          style:visibility={ready ? 'visible' : 'hidden'}
          onload={() => { loaded = true; }}
          onerror={() => { failed = true; }}
        >
      {/if}
      <canvas bind:this={overlayEl} class="painter-overlay" style:width="{areaW}px" style:height="{areaH}px" aria-hidden="true"></canvas>
      {#if ring && mode === 'paint' && ready}
        <span class="painter-ring" style:width="{ringSize}px" style:height="{ringSize}px" style:left="{ring.x - ringSize / 2}px" style:top="{ring.y - ringSize / 2}px" aria-hidden="true"></span>
      {/if}
    </div>

    {#if !ready}
      <div class="painter-wait" role="status">
        {#if failed}
          <span>The image could not be loaded.</span>
        {:else}
          <Spinner size="lg" label="Loading the image" />
        {/if}
      </div>
    {/if}

    {#if ready && mode === 'extend'}
      {#each EXTEND_SIDES as side (side)}
        <button
          type="button"
          class="painter-pill"
          data-side={side}
          aria-pressed={extension?.side === side}
          style:left="{pill[side].left}px"
          style:top="{pill[side].top}px"
          style:translate={pill[side].shift}
          onclick={() => pickSide(side)}
        >{SIDE_LABEL[side]}</button>
      {/each}
    {/if}
  </div>

  {#if variant === 'wide'}
    <footer class="painter-foot painter-foot-wide">
      <div class="painter-bar ui-overlay" role="toolbar" aria-label={mode === 'paint' ? 'Paint tools' : 'Extend tools'}>
        {#if mode === 'paint'}
          {@render toolSwitch()}
          <div class="painter-size">{@render brushSize()}</div>
        {:else}
          <div class="painter-size">
            {@render extendBy()}
            <span class="painter-value painter-dimensions painter-dimensions-after">{dimensions}</span>
          </div>
        {/if}
        {@render history()}
        {#if mode === 'paint'}{@render zoom()}{/if}
      </div>
    </footer>
  {:else if variant === 'phone'}
    <footer class="painter-foot painter-foot-phone painter-dock">
      <div class="painter-row">
        {#if mode === 'paint'}
          {@render toolSwitch()}
        {:else}
          <span class="painter-value painter-dimensions">{dimensions}</span>
        {/if}
        <span class="painter-spacer"></span>
        {@render history()}
      </div>
      <div class="painter-row painter-size">
        {#if mode === 'paint'}{@render brushSize()}{:else}{@render extendBy()}{/if}
      </div>
      <button type="button" class="ui-btn ui-btn-primary ui-btn-main painter-done" onclick={() => void done()}>Done</button>
    </footer>
  {:else}
    <footer class="painter-foot painter-dock painter-foot-landscape">
      <button type="button" class="ui-btn ui-btn-icon" aria-label="Cancel" data-testid="painter-cancel" onclick={() => void cancel()}><Icon name="close" size={16} /></button>
      {@render modeSwitch()}
      {#if mode === 'paint'}
        {@render toolSwitch()}
        <div class="painter-size painter-size-grow">{@render brushSize()}</div>
      {:else}
        <div class="painter-size painter-size-grow">{@render extendBy()}</div>
        <span class="painter-value painter-dimensions painter-dimensions-after">{dimensions}</span>
      {/if}
      {@render history()}
      <button type="button" class="ui-btn ui-btn-primary" onclick={() => void done()}>Done</button>
    </footer>
  {/if}

  <p class="sr-only" aria-live="polite" data-testid="painter-live">{announcement}</p>
</div>

<style>
  .painter { z-index: var(--z-painter); display: flex; flex-direction: column; background: var(--color-bg-base); color: var(--color-text-primary); overscroll-behavior: none; outline: none; user-select: none; -webkit-user-select: none; }

  /* --- Top --- */
  .painter-top { display: flex; flex: none; align-items: center; gap: 12px; padding: 8px max(12px, env(safe-area-inset-right, 0px)) 8px max(12px, env(safe-area-inset-left, 0px)); padding-top: max(8px, env(safe-area-inset-top, 0px)); border-bottom: 1px solid var(--color-border-subtle); }
  .painter-top-wide { display: grid; grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr); }
  .painter-top-phone { justify-content: space-between; }
  .painter-name { display: flex; min-width: 0; align-items: baseline; gap: 10px; font-size: var(--text-content); }
  .painter-name b { min-width: 0; font-weight: 600; }
  .painter-dims, .painter-value { flex: none; font-family: var(--font-mono); font-size: var(--text-ui); color: var(--color-text-secondary); }
  .painter-end { display: flex; justify-content: flex-end; gap: 8px; }

  /* --- The canvas area --- */
  .painter-area { position: relative; flex: 1; min-height: 0; overflow: hidden; background: var(--color-bg-base); }
  .painter-surface { position: absolute; inset: 0; overflow: hidden; touch-action: none; user-select: none; -webkit-user-select: none; -webkit-touch-callout: none; overscroll-behavior: none; }
  .painter-surface.painting { cursor: none; }
  .painter-surface.pan { cursor: grab; }
  .painter-img, .painter-new { position: absolute; max-width: none; pointer-events: none; }
  .painter-overlay { position: absolute; left: 0; top: 0; pointer-events: none; }
  .painter-ring { position: absolute; border-radius: 50%; box-sizing: border-box; border: 1.5px solid var(--color-text-primary); box-shadow: 0 0 0 1.5px var(--color-bg-overlay), inset 0 0 0 1.5px var(--color-bg-overlay); pointer-events: none; }
  .painter-wait { position: absolute; inset: 0; display: grid; place-items: center; color: var(--color-text-secondary); font-size: var(--text-content); pointer-events: none; }

  /* The four sides of Extend: filled, so the label never sits on the picture's own pixels. */
  .painter-pill { position: absolute; z-index: 2; display: inline-flex; align-items: center; height: var(--spacing-bar); padding: 0 12px; border-radius: var(--radius-sm); background: var(--color-bg-overlay); font-size: var(--text-ui); font-weight: 600; white-space: nowrap; color: var(--color-text-primary); }
  .painter-pill[aria-pressed='true'] { background: var(--color-primary-main); color: var(--color-primary-ink); }
  @media (hover: hover) { .painter-pill:hover:not([aria-pressed='true']) { background: var(--color-bg-control-hover); } }
  @media (hover: none) { .painter-pill:active { background: var(--color-bg-control-hover); } .painter-pill[aria-pressed='true']:active { background: var(--color-primary-hover); } }

  /* --- The foot --- */
  .painter-foot { flex: none; }
  .painter-foot-wide { display: flex; justify-content: center; padding: 0 max(12px, env(safe-area-inset-right, 0px)) calc(16px + env(safe-area-inset-bottom, 0px)) max(12px, env(safe-area-inset-left, 0px)); }
  .painter-bar { display: flex; min-width: 0; max-width: 100%; align-items: center; gap: 8px; padding: 6px; }
  .painter-size { display: flex; min-width: 0; flex: 1; align-items: center; gap: 8px; }
  .painter-size-grow { flex: 1; }
  .painter-range { flex: 1; min-width: 120px; }
  .painter-label { flex: none; font-size: var(--text-ui); font-weight: 500; color: var(--color-text-secondary); }
  .painter-value { min-width: 4ch; text-align: right; white-space: nowrap; }
  .painter-dimensions { text-align: left; }
  /* After the amount: a divider keeps `256 px` and `832 × 1216 → 1088 × 1216` from running into one line. */
  .painter-dimensions-after { margin-left: 4px; padding-left: 12px; border-left: 1px solid var(--color-border-subtle); }
  .painter-zoom { display: flex; flex: none; align-items: center; gap: 4px; }
  .painter-zoom-value { min-width: 4ch; text-align: center; }
  .size-dot { display: grid; flex: none; place-items: center; width: 22px; height: 22px; }
  .size-dot > span { display: block; border-radius: 50%; background: var(--color-text-primary); }

  /* A phone: rows docked on a surface, with Done the one main action at the foot. */
  .painter-dock { display: flex; flex-direction: column; gap: 8px; padding: 8px max(12px, env(safe-area-inset-right, 0px)) calc(8px + env(safe-area-inset-bottom, 0px)) max(12px, env(safe-area-inset-left, 0px)); background: var(--color-bg-surface); }
  .painter-row { display: flex; align-items: center; gap: 8px; }
  .painter-spacer { flex: 1; }
  .painter-dock .ui-btn { height: var(--spacing-bar); }
  .painter-dock .ui-btn-icon { width: var(--spacing-bar); }
  .painter-foot-phone .painter-done { width: 100%; font-size: var(--text-content); }
  .painter-top-phone .ui-btn { height: var(--spacing-bar); }
  .painter-top-phone .ui-btn-icon { width: var(--spacing-bar); }
  .painter-foot-landscape { flex-direction: row; align-items: center; gap: 8px; padding-top: 6px; padding-bottom: calc(6px + env(safe-area-inset-bottom, 0px)); }
  .painter-foot-landscape .painter-range { min-width: 60px; }
</style>
