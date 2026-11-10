<script lang="ts">
  import { CUT_STOP_REASON, extendCutView, extendFacts, extendFromAt, keepsLabel } from '$lib/state/extendCut';
  import { EXTEND_MIN_FRAMES } from '$lib/state/framePoint';
  import { coverTile, spriteTile, videoSprite } from '$lib/state/videoSprite.svelte';
  import type { ExtendSourceInfo } from '$lib/types';
  import { barCellCount, cellFrame } from './spriteCover';

  interface Props {
    /** The described source; the bar needs its frames, frame rate and first-frame time. */
    info: ExtendSourceInfo;
    /** The path the source was chosen by, which names its contact sheet. */
    path: string;
    /** Where the clip continues from, in seconds on the source's clock; null for the whole clip. */
    extendFrom: number | null;
    /** A submit is in flight: the bar shows its place and does not move. */
    busy?: boolean;
    /** The new place when a drag or a key ends: seconds on the source's clock, or null for the whole clip. */
    onchange: (extendFrom: number | null) => void;
    /** The frames the clip keeps while a drag or a held key is moving the handle, or null at rest. The thumb beside the bar shows that frame's sprite tile. */
    dragKeep?: number | null;
  }

  let { info, path, extendFrom, busy = false, onchange, dragKeep = $bindable(null) }: Props = $props();
  // The Source video section and its popover can both be on the page, so the reason's id is the bar's own.
  const uid = $props.id();
  const reasonId = `${uid}-reason`;

  const facts = $derived(extendFacts(info));
  const lastFrame = $derived(facts ? facts.frames - 1 : 0);
  const sprite = $derived(videoSprite.of(path));
  // What the bar shows is the place being dragged to, else the place the draft holds; the draft changes on release only.
  const view = $derived(facts ? extendCutView(info, dragKeep === null ? extendFrom : extendFromAt(dragKeep, facts)) : null);
  const keep = $derived(view?.keepFrames ?? 0);
  const atStop = $derived(view !== null && !view.whole && view.keepFrames <= EXTEND_MIN_FRAMES);

  // The first show of the bar asks for the contact sheet; until it arrives the cells are flat and everything else works.
  $effect(() => {
    videoSprite.ensure(path);
  });

  let width = $state(0);
  const cellCount = $derived(barCellCount(width));
  const cellWidth = $derived(width > 0 ? width / cellCount : 66);
  // The cells keep the clip's shape: as tall as one cell is wide at the source's shape, between 40 and 56 px.
  const shape = $derived(info.width && info.height ? info.height / info.width : 9 / 16);
  const height = $derived(Math.min(56, Math.max(40, cellWidth * shape)));
  const cells = $derived.by(() => {
    const frames = facts ? facts.frames : 1;
    return Array.from({ length: cellCount }, (_, index) => {
      if (!sprite) return { index, style: '' };
      const tile = spriteTile(cellFrame(index, cellCount, lastFrame), frames);
      const cover = coverTile(sprite, tile, { width: cellWidth, height });
      return {
        index,
        style: `background-image:url('${sprite.url}');background-size:${cover.sheetWidth}px ${cover.sheetHeight}px;background-position:${cover.offsetX}px ${cover.offsetY}px`,
      };
    });
  });

  const handleAt = $derived(lastFrame > 0 ? ((keep - 1) / lastFrame) * 100 : 100);
  const valueText = $derived(view === null ? '' : view.whole ? 'Whole clip' : `${view.keepSeconds.toFixed(1)} s of ${view.clipSeconds.toFixed(1)} s`);

  let bar: HTMLElement | undefined = $state();

  /** Return the frames kept when the handle is at *clientX*: the frame under it, at least the least a cut may keep. */
  function keepAt(clientX: number): number {
    if (!bar) return keep;
    const box = bar.getBoundingClientRect();
    const fraction = box.width > 0 ? Math.min(1, Math.max(0, (clientX - box.left) / box.width)) : 1;
    return Math.max(EXTEND_MIN_FRAMES, Math.round(fraction * lastFrame) + 1);
  }

  function commit(): void {
    if (dragKeep === null || !facts) return;
    const next = extendFromAt(dragKeep, facts);
    dragKeep = null;
    onchange(next);
  }

  // A mouse or pen press moves the handle at once. A finger waits for the first sideways movement or for the lift, so a
  // finger that starts scrolling the sidebar (the browser then cancels the touch) does not move it.
  let pressed: { id: number; x: number; moving: boolean } | null = null;

  function onPointerDown(event: PointerEvent): void {
    if (busy || !facts || (event.pointerType === 'mouse' && event.button !== 0)) return;
    const touch = event.pointerType === 'touch';
    pressed = { id: event.pointerId, x: event.clientX, moving: !touch };
    bar?.setPointerCapture?.(event.pointerId);
    bar?.focus({ preventScroll: true });
    if (!touch) dragKeep = keepAt(event.clientX);
  }

  function onPointerMove(event: PointerEvent): void {
    if (!pressed || pressed.id !== event.pointerId) return;
    if (!pressed.moving && Math.abs(event.clientX - pressed.x) < 4) return;
    pressed.moving = true;
    dragKeep = keepAt(event.clientX);
  }

  function onPointerUp(event: PointerEvent): void {
    if (!pressed || pressed.id !== event.pointerId) return;
    if (!pressed.moving) dragKeep = keepAt(event.clientX);
    pressed = null;
    commit();
  }

  function onPointerCancel(event: PointerEvent): void {
    if (!pressed || pressed.id !== event.pointerId) return;
    pressed = null;
    dragKeep = null;
  }

  // A held arrow moves the handle without a request; the key lifting (or the bar losing focus) settles it.
  const KEYS = new Set(['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Home', 'End']);

  function onKeydown(event: KeyboardEvent): void {
    if (busy || !facts || !KEYS.has(event.key) || event.altKey || event.ctrlKey || event.metaKey) return;
    event.preventDefault();
    const from = dragKeep ?? keep;
    const step = event.shiftKey ? 5 : 1;
    const next = event.key === 'Home' ? EXTEND_MIN_FRAMES
      : event.key === 'End' ? facts.frames
      : event.key === 'ArrowLeft' || event.key === 'ArrowDown' ? from - step
      : from + step;
    dragKeep = Math.min(facts.frames, Math.max(EXTEND_MIN_FRAMES, next));
  }

  function onKeyup(event: KeyboardEvent): void {
    if (KEYS.has(event.key) && !pressed) commit();
  }
</script>

{#if facts && view}
  <div class="cut" class:busy data-testid="extend-cut">
    <p class="keeps" data-testid="extend-keeps">{keepsLabel(view)}</p>
    <!-- svelte-ignore a11y_no_noninteractive_tabindex because a role="slider" is interactive -->
    <div
      bind:this={bar}
      bind:clientWidth={width}
      class="bar"
      role="slider"
      aria-label="Continue from"
      aria-valuemin={EXTEND_MIN_FRAMES}
      aria-valuemax={facts.frames}
      aria-valuenow={keep}
      aria-valuetext={valueText}
      aria-disabled={busy || undefined}
      aria-describedby={reasonId}
      tabindex={busy ? -1 : 0}
      data-own-drag
      data-testid="extend-cut-bar"
      style:height="{height}px"
      onpointerdown={onPointerDown}
      onpointermove={onPointerMove}
      onpointerup={onPointerUp}
      onpointercancel={onPointerCancel}
      onkeydown={onKeydown}
      onkeyup={onKeyup}
      onblur={commit}
    >
      <span class="cells" aria-hidden="true">
        {#each cells as cell (cell.index)}<span class="cell" style={cell.style}></span>{/each}
      </span>
      {#if !view.whole}<span class="dropped" style:left="{handleAt}%" data-testid="extend-cut-dropped" aria-hidden="true"></span>{/if}
      <span class="handle" style:left="{handleAt}%" aria-hidden="true"><span class="grip"></span></span>
    </div>
    <p id={reasonId} class="reason" data-testid="extend-cut-reason" aria-live="polite">{atStop ? CUT_STOP_REASON : ''}</p>
  </div>
{/if}

<style>
  .cut { display: flex; flex-direction: column; gap: 6px; min-width: 0; }
  .cut.busy { opacity: 0.4; pointer-events: none; }
  .keeps { margin: 0; font-size: var(--text-ui); color: var(--color-primary-main); }
  .bar { position: relative; width: 100%; overflow: hidden; border-radius: var(--radius-sm); background: var(--color-bg-surface-hover); cursor: ew-resize; touch-action: pan-y; -webkit-user-select: none; user-select: none; outline: none; }
  .bar:focus-visible { outline: 2px solid var(--color-primary-main); outline-offset: 2px; }
  @media (pointer: coarse) { .bar { min-height: 44px; } }
  .cells { position: absolute; inset: 0; display: flex; }
  .cell { flex: 1 1 0; min-width: 0; background-repeat: no-repeat; }
  /* From the handle to the right edge: the part the continued clip drops. */
  .dropped { position: absolute; top: 0; right: 0; bottom: 0; background: repeating-linear-gradient(135deg, var(--color-glass-strong) 0 6px, color-mix(in srgb, var(--color-bg-overlay) 60%, transparent) 6px 12px); pointer-events: none; }
  .handle { position: absolute; top: 0; bottom: 0; width: 3px; margin-left: -1.5px; background: var(--color-primary-main); box-shadow: 0 0 8px var(--color-primary-main); pointer-events: none; }
  .grip { position: absolute; top: 50%; left: 50%; width: 10px; height: 24px; translate: -50% -50%; border-radius: var(--radius-xs); background: var(--color-primary-main); }
  /* Two lines are reserved, so the reason appearing at the stop moves nothing. */
  .reason { margin: 0; min-height: 30px; font-size: var(--text-meta); line-height: 1.35; color: var(--color-warning); }
</style>
