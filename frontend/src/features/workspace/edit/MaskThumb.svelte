<script lang="ts">
  import { Icon } from '$lib/components/atoms';
  import { draft } from '$lib/state/draft.svelte';
  import { heldUploads } from '$lib/state/heldUploads.svelte';
  import { contentRect, extensionRect } from './editOps';
  import { imageEdit } from './editStore.svelte';
  import { decodeMask, drawThumbMask, readPalette } from './paintCanvas';

  interface Props {
    /** The side of the square thumbnail, in CSS pixels. */
    size?: number;
    /** The edit sends no region: the picture gets a frame. */
    whole?: boolean;
  }

  let { size = 40, whole = false }: Props = $props();

  const source = $derived(imageEdit.source);
  let canvas = $state<HTMLCanvasElement | null>(null);
  let failed = $state(false);

  // A picture that failed is not the next one's failure.
  $effect(() => {
    void source?.thumbUrl;
    failed = false;
  });

  const extending = $derived(draft.state.editMode === 'extend' && draft.state.editExtendSide !== null);

  // Paint: the committed paint is drawn over the picture, contained in the square like the picture itself.
  $effect(() => {
    const target = canvas;
    const file = draft.state.editMode === 'paint' && draft.state.editMaskUpload !== null ? heldUploads.editMask : null;
    if (!target) return;
    let current = true;
    void decodeMask(file).then((mask) => {
      if (current) drawThumbMask(target, mask, size, Math.min(2, window.devicePixelRatio || 1), readPalette(target).fill);
    });
    return () => { current = false; };
  });

  // Extend: the picture and its new area, contained in the square together, with the join marked.
  const layout = $derived.by(() => {
    const width = source?.width ?? null;
    const height = source?.height ?? null;
    const side = draft.state.editExtendSide;
    if (!extending || side === null || width === null || height === null) return null;
    const extension = { side, amount: draft.state.editExtendAmount };
    const canvasRect = contentRect(width, height, extension);
    const scale = size / Math.max(canvasRect.w, canvasRect.h);
    const place = (rect: { x: number; y: number; w: number; h: number }) => ({
      left: (size - canvasRect.w * scale) / 2 + (rect.x - canvasRect.x) * scale,
      top: (size - canvasRect.h * scale) / 2 + (rect.y - canvasRect.y) * scale,
      width: rect.w * scale,
      height: rect.h * scale,
    });
    return { picture: place({ x: 0, y: 0, w: width, h: height }), added: place(extensionRect(width, height, extension)), side };
  });

  // The picture's own box in the square, for the frame.
  const frameBox = $derived.by(() => {
    const width = source?.width ?? null;
    const height = source?.height ?? null;
    if (width === null || height === null) return { left: 0, top: 0, width: size, height: size };
    const scale = size / Math.max(width, height);
    return { left: (size - width * scale) / 2, top: (size - height * scale) / 2, width: width * scale, height: height * scale };
  });
  const box = (rect: { left: number; top: number; width: number; height: number }): string => `left:${rect.left}px;top:${rect.top}px;width:${rect.width}px;height:${rect.height}px`;
</script>

<span class="mask-thumb" style:width="{size}px" style:height="{size}px" data-testid="slot-thumb">
  {#if layout}
    <span class="thumb-new ui-checkerboard" data-side={layout.side} style={box(layout.added)}></span>
    {#if source && !failed}
      <img class="thumb-pic" src={source.thumbUrl} alt="" style={box(layout.picture)} onerror={() => { failed = true; }}>
    {/if}
  {:else}
    {#if source && !failed}
      <img src={source.thumbUrl} alt="" onerror={() => { failed = true; }}>
    {:else}
      <Icon name="reference" size={16} />
    {/if}
    <canvas bind:this={canvas} style:width="{size}px" style:height="{size}px" aria-hidden="true"></canvas>
    {#if whole}<span class="thumb-frame" style={box(frameBox)}></span>{/if}
  {/if}
</span>

<style>
  .mask-thumb { position: relative; display: grid; flex: none; place-items: center; overflow: hidden; border-radius: var(--radius-sm); background: var(--color-bg-surface-hover); color: var(--color-text-muted); }
  .mask-thumb img { width: 100%; height: 100%; object-fit: contain; }
  .mask-thumb canvas { position: absolute; inset: 0; pointer-events: none; }
  .thumb-pic, .thumb-new, .thumb-frame { position: absolute; box-sizing: border-box; }
  .mask-thumb .thumb-pic { width: auto; height: auto; object-fit: fill; }
  /* The join of the new area with the picture: the same 2 px ink line the painter draws. */
  .thumb-new[data-side='top'] { border-bottom: 2px solid var(--color-primary-ink); }
  .thumb-new[data-side='bottom'] { border-top: 2px solid var(--color-primary-ink); }
  .thumb-new[data-side='left'] { border-right: 2px solid var(--color-primary-ink); }
  .thumb-new[data-side='right'] { border-left: 2px solid var(--color-primary-ink); }
  /* The whole picture changes: a teal frame with an ink line inside it, readable on light and dark pictures. */
  .thumb-frame { box-shadow: inset 0 0 0 2px var(--color-primary-main), inset 0 0 0 3.5px var(--color-primary-ink); pointer-events: none; }
</style>
