<script lang="ts">
  import { Icon } from '$lib/components/atoms';
  import { draft } from '$lib/state/draft.svelte';
  import { heldUploads } from '$lib/state/heldUploads.svelte';
  import { imageEdit } from './editStore.svelte';
  import { decodeMask, drawThumbMask, readPalette } from './paintCanvas';

  interface Props {
    /** The side of the square thumbnail, in CSS pixels. */
    size?: number;
  }

  let { size = 40 }: Props = $props();

  const source = $derived(imageEdit.source);
  let canvas = $state<HTMLCanvasElement | null>(null);
  let failed = $state(false);

  // A picture that failed is not the next one's failure.
  $effect(() => {
    void source?.thumbUrl;
    failed = false;
  });

  // The committed paint is drawn over the picture, contained in the square like the picture itself.
  $effect(() => {
    const target = canvas;
    const file = draft.state.editMaskUpload !== null ? heldUploads.editMask : null;
    if (!target) return;
    let current = true;
    void decodeMask(file).then((mask) => {
      if (current) drawThumbMask(target, mask, size, Math.min(2, window.devicePixelRatio || 1), readPalette(target).fill);
    });
    return () => { current = false; };
  });
</script>

<span class="mask-thumb" style:width="{size}px" style:height="{size}px" data-testid="slot-thumb">
  {#if source && !failed}
    <img src={source.thumbUrl} alt="" onerror={() => { failed = true; }}>
  {:else}
    <Icon name="reference" size={16} />
  {/if}
  <canvas bind:this={canvas} style:width="{size}px" style:height="{size}px" aria-hidden="true"></canvas>
</span>

<style>
  .mask-thumb { position: relative; display: grid; flex: none; place-items: center; overflow: hidden; border-radius: var(--radius-sm); background: var(--color-bg-surface-hover); color: var(--color-text-muted); }
  .mask-thumb img { width: 100%; height: 100%; object-fit: contain; }
  .mask-thumb canvas { position: absolute; inset: 0; pointer-events: none; }
</style>
