<script lang="ts">
  import { dragCrop, moveCrop, nudgeCrop, type Crop, type Handle, type Size } from '$lib/state/adjustGeometry';

  interface Props {
    /** The displayed frame's size, in source pixels. */
    frame: Size;
    /** The crop box, in source pixels of the frame; null is the whole frame. */
    crop: Crop | null;
    /** The locked width / height ratio, or null for Free (its edges move too). */
    ratio: number | null;
    /** CSS pixels per frame pixel. */
    scale: number;
    /** A finger: the handles' hit areas are 44px, not 24px. */
    coarse?: boolean;
    /** Nothing moves: a save is running. */
    disabled?: boolean;
    /** The id of the line that describes the frame. */
    describedBy?: string;
    /** Called with each new box while it is dragged or nudged. A cancelled drag calls it with the start box. */
    onchange: (crop: Crop) => void;
    /** Called when a drag ends or a nudge lands, so the size can be announced. */
    oncommit: () => void;
  }

  let { frame, crop, ratio, scale, coarse = false, disabled = false, describedBy, onchange, oncommit }: Props = $props();

  const CORNERS: Handle[] = ['nw', 'ne', 'sw', 'se'];
  const EDGES: Handle[] = ['n', 's', 'w', 'e'];

  const box = $derived(crop ?? { x: 0, y: 0, width: frame.width, height: frame.height });
  const px = $derived({ left: box.x * scale, top: box.y * scale, width: box.width * scale, height: box.height * scale });
  /** At least 16 source pixels, and enough that the handles keep a usable size on screen. */
  const minSize = $derived(Math.max(16, scale > 0 ? Math.ceil(44 / scale) : 0));

  interface Drag {
    pointerId: number;
    kind: 'move' | Handle;
    x: number;
    y: number;
    start: Crop;
  }
  let drag = $state<Drag | null>(null);

  function begin(event: PointerEvent, kind: 'move' | Handle): void {
    if (disabled || scale <= 0 || (event.pointerType === 'mouse' && event.button !== 0)) return;
    event.preventDefault();
    event.stopPropagation();
    try {
      (event.currentTarget as HTMLElement).setPointerCapture?.(event.pointerId);
    } catch {
      // The pointer is already gone; the drag ends on its own.
    }
    if (kind === 'move') (event.currentTarget as HTMLElement).focus({ preventScroll: true });
    drag = { pointerId: event.pointerId, kind, x: event.clientX, y: event.clientY, start: { ...box } };
  }

  function move(event: PointerEvent): void {
    if (!drag || event.pointerId !== drag.pointerId) return;
    const dx = (event.clientX - drag.x) / scale;
    const dy = (event.clientY - drag.y) / scale;
    onchange(drag.kind === 'move' ? moveCrop(drag.start, dx, dy, frame) : dragCrop(drag.start, drag.kind, dx, dy, ratio, frame, minSize));
  }

  function end(event: PointerEvent): void {
    if (!drag || event.pointerId !== drag.pointerId) return;
    drag = null;
    oncommit();
  }

  function cancel(event: PointerEvent): void {
    if (!drag || event.pointerId !== drag.pointerId) return;
    const { start } = drag;
    drag = null;
    onchange(start);
  }

  function onkeydown(event: KeyboardEvent): void {
    if (disabled || !event.key.startsWith('Arrow') || event.ctrlKey || event.metaKey || event.altKey) return;
    event.preventDefault();
    const next = nudgeCrop(box, event.key, event.shiftKey, frame);
    if (next.x === box.x && next.y === box.y) return;
    onchange(next);
    oncommit();
  }

  /** Where a handle sits, in CSS pixels from the picture's top-left. */
  function place(handle: Handle): string {
    const x = handle.includes('w') ? px.left : handle.includes('e') ? px.left + px.width : px.left + px.width / 2;
    const y = handle.includes('n') ? px.top : handle.includes('s') ? px.top + px.height : px.top + px.height / 2;
    return `left: ${x}px; top: ${y}px`;
  }

  const handles = $derived(ratio === null ? [...CORNERS, ...EDGES] : CORNERS);
</script>

<div class="crop-layer" data-coarse={coarse ? '' : undefined} inert={disabled}>
  <div class="crop-clip">
    <!-- svelte-ignore a11y_no_noninteractive_tabindex, a11y_no_noninteractive_element_interactions because the frame is a focusable group that the arrow keys move -->
    <div
      class="crop-box"
      role="group"
      aria-label="Crop frame"
      aria-describedby={describedBy}
      tabindex="0"
      data-crop-part
      data-testid="adjust-crop-box"
      data-dragging={drag ? '' : undefined}
      style="left: {px.left}px; top: {px.top}px; width: {px.width}px; height: {px.height}px"
      onpointerdown={(event) => begin(event, 'move')}
      onpointermove={move}
      onpointerup={end}
      onpointercancel={cancel}
      {onkeydown}
    >
      <span class="crop-third v" style="left: 33.333%"></span>
      <span class="crop-third v" style="left: 66.667%"></span>
      <span class="crop-third h" style="top: 33.333%"></span>
      <span class="crop-third h" style="top: 66.667%"></span>
    </div>
  </div>
  <!-- The handles overhang the picture, so they sit outside the clip. They are for a pointer; the keyboard moves the frame. -->
  <div class="crop-handles" aria-hidden="true">
    {#each handles as handle (handle)}
      <!-- svelte-ignore a11y_no_static_element_interactions because a handle is for a pointer only; the keyboard moves the frame -->
      <span
        class="crop-handle"
        data-h={handle}
        data-crop-part
        style={place(handle)}
        onpointerdown={(event) => begin(event, handle)}
        onpointermove={move}
        onpointerup={end}
        onpointercancel={cancel}
      >
        {#if handle.length === 2}
          <i class="arm-x"></i><i class="arm-y"></i>
        {:else}
          <i class="bar"></i>
        {/if}
      </span>
    {/each}
  </div>
</div>

<style>
  .crop-layer { position: absolute; inset: 0; --arm: 18px; pointer-events: none; }
  .crop-layer[data-coarse] { --arm: 22px; }
  .crop-clip { position: absolute; inset: 0; overflow: hidden; border-radius: var(--radius-xs); pointer-events: none; }
  .crop-box { position: absolute; z-index: 2; box-shadow: 0 0 0 100vmax var(--color-viewer-fade-low); outline: none; touch-action: none; cursor: move; pointer-events: auto; }
  .crop-box::before { content: ''; position: absolute; inset: 0; border: 1px solid var(--color-bloom-rest); box-shadow: inset 0 0 0 1px var(--color-hairline-shade), 0 0 0 1px var(--color-hairline-shade); pointer-events: none; }
  .crop-box:focus-visible::after { content: ''; position: absolute; inset: -4px; border: 2px solid var(--color-primary-main); border-radius: 2px; pointer-events: none; }
  .crop-third { position: absolute; background: var(--color-hairline); box-shadow: 0 0 0 0.5px var(--color-hairline-shade); pointer-events: none; transition: opacity 120ms var(--ease-standard); }
  .crop-third.v { top: 0; bottom: 0; width: 1px; }
  .crop-third.h { left: 0; right: 0; height: 1px; }
  .crop-box:not([data-dragging]) .crop-third { opacity: 0; }
  .crop-handles { position: absolute; inset: 0; z-index: 3; pointer-events: none; }
  /* The visible mark is 3px: an L at each corner and, in Free, a bar at each edge. The hit area is 24px with a mouse and
     44px on touch, centred on the frame's line. */
  .crop-handle { position: absolute; width: 24px; height: 24px; translate: -50% -50%; touch-action: none; pointer-events: auto; }
  .crop-layer[data-coarse] .crop-handle { width: 44px; height: 44px; }
  .crop-handle i { position: absolute; left: 50%; top: 50%; display: block; background: var(--color-text-primary); box-shadow: 0 0 0 1px var(--color-hairline-shade); }
  .crop-handle .arm-x { width: var(--arm); height: 3px; margin-top: -1.5px; }
  .crop-handle .arm-y { width: 3px; height: var(--arm); margin-left: -1.5px; }
  .crop-handle[data-h='nw'] .arm-x, .crop-handle[data-h='sw'] .arm-x { margin-left: -1.5px; }
  .crop-handle[data-h='ne'] .arm-x, .crop-handle[data-h='se'] .arm-x { margin-left: calc(1.5px - var(--arm)); }
  .crop-handle[data-h='nw'] .arm-y, .crop-handle[data-h='ne'] .arm-y { margin-top: -1.5px; }
  .crop-handle[data-h='sw'] .arm-y, .crop-handle[data-h='se'] .arm-y { margin-top: calc(1.5px - var(--arm)); }
  .crop-handle[data-h='n'] .bar, .crop-handle[data-h='s'] .bar { width: 20px; height: 3px; margin: -1.5px 0 0 -10px; }
  .crop-handle[data-h='e'] .bar, .crop-handle[data-h='w'] .bar { width: 3px; height: 20px; margin: -10px 0 0 -1.5px; }
  .crop-handle[data-h='nw'], .crop-handle[data-h='se'] { cursor: nwse-resize; }
  .crop-handle[data-h='ne'], .crop-handle[data-h='sw'] { cursor: nesw-resize; }
  .crop-handle[data-h='n'], .crop-handle[data-h='s'] { cursor: ns-resize; }
  .crop-handle[data-h='e'], .crop-handle[data-h='w'] { cursor: ew-resize; }
  @media (prefers-reduced-motion: reduce) { .crop-third { transition: none; } }
</style>
