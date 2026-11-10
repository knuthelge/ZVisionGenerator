<script lang="ts">
  import { FileThumb } from '$lib/components/molecules';
  import { extendCutView, extendFacts } from '$lib/state/extendCut';
  import { coverTile, spriteTile, videoSprite } from '$lib/state/videoSprite.svelte';
  import type { ExtendSourceInfo } from '$lib/types';

  interface Props {
    /** The path the source was chosen by. */
    path: string;
    /** The described source, or null before the server answers. */
    info: ExtendSourceInfo | null;
    /** Where the clip continues from, in seconds on the source's clock; null for the whole clip. */
    extendFrom: number | null;
    /** The frames the clip keeps while the handle is being dragged, or null at rest. */
    dragKeep?: number | null;
    iconSize?: number;
  }

  let { path, info, extendFrom, dragKeep = null, iconSize = 16 }: Props = $props();

  const facts = $derived(info ? extendFacts(info) : null);
  const sprite = $derived(videoSprite.of(path));
  const cut = $derived(info ? extendCutView(info, extendFrom) : null);
  // The second that names the frame at a real cut. Null shows the clip's last frame, by `frame="last"`: before the source is described and
  // while the whole clip is kept, which is the same picture, so describing the source never asks for it a second time.
  const exactAt = $derived(cut && !cut.whole ? cut.thumbAt : null);

  // The point whose exact frame last arrived. A drag shows the sheet's tile, which costs no request,
  // and the tile stays after the release until the exact frame has arrived, so the picture never steps back to an older frame.
  let loadedAt = $state<number | null | undefined>(undefined);
  const showTile = $derived(dragKeep !== null || loadedAt !== exactAt);
  const tileFrame = $derived(dragKeep !== null ? dragKeep - 1 : cut ? cut.keepFrames - 1 : 0);
  const tile = $derived(facts ? spriteTile(tileFrame, facts.frames) : 0);

  let box = $state({ width: 0, height: 0 });
  // Without a sheet the picture keeps what it showed until the release.
  const cover = $derived(showTile && sprite && box.width > 0 ? coverTile(sprite, tile, box) : null);
</script>

<!-- The exact frame's load is heard in the capture phase, because an image's load event does not bubble. -->
<span class="source-thumb" bind:clientWidth={box.width} bind:clientHeight={box.height} data-testid="extend-source-thumb" onloadcapture={() => { loadedAt = exactAt; }}>
  <FileThumb purpose="source_video" {path} kind="video" frame={exactAt === null ? 'last' : undefined} at={exactAt} {iconSize} fallbackIcon="film" />
  {#if cover && sprite}
    <span
      class="tile"
      data-testid="extend-thumb-tile"
      style:background-image="url('{sprite.url}')"
      style:background-size="{cover.sheetWidth}px {cover.sheetHeight}px"
      style:background-position="{cover.offsetX}px {cover.offsetY}px"
    ></span>
  {/if}
</span>

<style>
  .source-thumb { position: relative; display: block; width: 100%; height: 100%; overflow: hidden; }
  .tile { position: absolute; inset: 0; background-repeat: no-repeat; }
</style>
