<script lang="ts">
  import { Icon } from '$lib/components/atoms';
  import { pan, rubberBand, type PanEnd, type PanMove } from '$lib/actions/pan';
  import { pagerOutcome } from '$lib/components/molecules/viewerPager';

  interface Props {
    /** The preview image URLs. */
    images: string[];
    title: string;
    /** What the empty box says (the author added no images, or only videos); null for the bare box. */
    emptyNote?: string | null;
  }

  let { images, title, emptyNote = null }: Props = $props();

  let index = $state(0);
  // On touch the picture follows the finger sideways and settles on the neighbour when the drag is far or fast enough.
  let dragX = $state(0);
  let heroEl = $state<HTMLElement | null>(null);

  $effect(() => {
    void images;
    index = 0;
  });

  const last = $derived(images.length - 1);

  function step(delta: number): void {
    index = Math.min(last, Math.max(0, index + delta));
  }

  function onmove(move: PanMove): void {
    const open = move.offset < 0 ? index < last : index > 0;
    dragX = open ? move.offset : rubberBand(move.offset, 40);
  }

  function onend(end: PanEnd): void {
    dragX = 0;
    const outcome = pagerOutcome(end.offset, end.velocity, heroEl?.clientWidth ?? 1, { prev: index > 0, next: index < last });
    if (outcome === 'next') step(1);
    else if (outcome === 'prev') step(-1);
  }
</script>

{#if images.length === 0}
  <div class="disc-hero empty"><Icon name="file" size={22} />{#if emptyNote}&nbsp;{emptyNote}{/if}</div>
{:else}
  <div
    class="disc-hero"
    role="group"
    aria-roledescription="carousel"
    aria-label="Examples from {title}"
    bind:this={heroEl}
    use:pan={{ axes: ['x'], enabled: images.length > 1, onmove, onend, oncancel: () => { dragX = 0; } }}
  >
    <img src={images[index]} alt="Example {index + 1} of {images.length} from {title}" style:translate={dragX ? `${dragX}px 0` : undefined} draggable="false">
    {#if images.length > 1}
      <button type="button" class="nav l ui-media-action" aria-label="Previous image" disabled={index === 0} onclick={() => step(-1)}><Icon name="chevleft" size={16} /></button>
      <button type="button" class="nav r ui-media-action" aria-label="Next image" disabled={index === last} onclick={() => step(1)}><Icon name="chevright" size={16} /></button>
      <span class="disc-count" role="status">{index + 1} / {images.length}</span>
    {/if}
  </div>
  {#if images.length > 1}
    <div class="disc-thumbs">
      {#each images as image, i (image)}
        <button type="button" aria-label="Show example {i + 1}" aria-current={i === index} onclick={() => { index = i; }}><img src={image} alt="" loading="lazy"></button>
      {/each}
    </div>
    <div class="disc-dots" aria-hidden="true">{#each images as image, i (image)}<i class:on={i === index}></i>{/each}</div>
  {/if}
{/if}
