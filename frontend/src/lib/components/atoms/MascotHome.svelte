<script lang="ts">
  import type { MascotPlacement } from '$lib/state/mascot';
  import { BODY_PATH } from './mascotArt';
  import MascotSpot from './MascotSpot.svelte';

  interface Props {
    placement: MascotPlacement;
    /** The logo's size in px; the mascot's body fills it as the static icon does. */
    size: number;
    class?: string;
  }

  let { placement, size, class: extraClass = '' }: Props = $props();
</script>

<!-- The logo is the mascot's home. While he is out on a Workspace spot, his outline keeps his place. -->
<span class="mascot-home {extraClass}" style:--home={`${size}px`} data-away={placement.away ? '' : undefined} data-testid="mascot-home">
  {#if placement.away}
    <svg class="home-outline" viewBox="-14 -18 228 218" aria-hidden="true">
      <path d={BODY_PATH} />
      <g transform="rotate(-18 76 62)">
        <ellipse cx="76" cy="62" rx="36" ry="13" />
      </g>
    </svg>
  {:else}
    <MascotSpot mood={placement.mood} tool={placement.tool} size={size * 1.39} decorative class="home-spot" />
  {/if}
</span>

<style>
  /* The mascot's art has room around the body for its props and effects; the box (1.39 times the logo) overhangs it so the body lines up with the static icon. */
  .mascot-home { position: relative; display: inline-block; flex: none; width: var(--home); height: var(--home); }
  .mascot-home > :global(.home-spot), .home-outline {
    position: absolute;
    top: calc(var(--home) * -0.311);
    left: calc(var(--home) * -0.195);
    width: calc(var(--home) * 1.39);
    height: calc(var(--home) * 1.39);
  }
  .mascot-home > :global(.home-spot) { line-height: 0; }
  .home-outline { fill: none; stroke: var(--color-text-muted); stroke-width: 6; stroke-dasharray: 22 16; stroke-linecap: round; opacity: 0.7; pointer-events: none; }
</style>
