<script lang="ts">
  import type { MascotPlacement } from '$lib/state/mascot';
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
    <svg class="home-outline" viewBox="-20 -20 240 225" aria-hidden="true">
      <path d="M100,42 C150,42 172,90 170,130 C168,168 140,182 100,182 C60,182 32,168 30,130 C28,90 50,42 100,42 Z" />
      <ellipse cx="100" cy="46" rx="36" ry="10" />
      <circle cx="100" cy="35" r="5" />
    </svg>
  {:else}
    <MascotSpot mood={placement.mood} tool={placement.tool} size={size * 1.5} decorative class="home-spot" />
  {/if}
</span>

<style>
  /* The mascot's art has room around the body for its props; the box overhangs the logo so the body lines up with it. */
  .mascot-home { position: relative; display: inline-block; flex: none; width: var(--home); height: var(--home); }
  .mascot-home > :global(.home-spot), .home-outline {
    position: absolute;
    top: calc(var(--home) * -0.32);
    left: calc(var(--home) * -0.25);
    width: calc(var(--home) * 1.5);
    height: calc(var(--home) * 1.5);
  }
  .mascot-home > :global(.home-spot) { line-height: 0; }
  .home-outline { fill: none; stroke: var(--color-text-muted); stroke-width: 7; stroke-dasharray: 14 10; opacity: 0.7; pointer-events: none; }
</style>
