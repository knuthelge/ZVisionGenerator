<script lang="ts" module>
  import { backOut, cubicInOut } from 'svelte/easing';
  import type { TransitionConfig } from 'svelte/transition';

  // Where the mascot last left from, keyed so separate mascots never swap places.
  const departures = new Map<string, { rect: DOMRect; at: number }>();
  const HANDOFF_WINDOW_MS = 100;

  function motionAllowed(): boolean {
    if (typeof Element === 'undefined' || typeof Element.prototype.animate !== 'function') return false;
    return !window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
  }

  function isFresh(entry: { at: number } | undefined): boolean {
    return entry !== undefined && performance.now() - entry.at <= HANDOFF_WINDOW_MS;
  }

  /**
   * Record where every mascot currently sits. Call this before a DOM update that
   * swaps spots: once the new spot is inserted the old one can be pushed aside by
   * layout, so measuring during the outro would start the hop from the wrong place.
   */
  export function rememberMascotSpots(): void {
    document.querySelectorAll<HTMLElement>('[data-mascot-spot]').forEach((spot) => {
      const key = spot.dataset.mascotSpot;
      if (key) departures.set(key, { rect: spot.getBoundingClientRect(), at: performance.now() });
    });
  }

  /** Vanish; the next spot hops in from where this one was last remembered. */
  function depart(node: Element, key: string): TransitionConfig {
    if (!isFresh(departures.get(key))) {
      departures.set(key, { rect: node.getBoundingClientRect(), at: performance.now() });
    }
    return { duration: 0 };
  }

  /** Hop from the previous spot in an arc, or pop in when there is no previous spot. */
  function arrive(node: Element, key: string): () => TransitionConfig {
    // Deferred so a same-tick departure is recorded before this runs.
    return () => {
      if (!motionAllowed()) return { duration: 0 };
      const from = departures.get(key);
      departures.delete(key);
      if (!from || !isFresh(from)) {
        return {
          duration: 320,
          easing: backOut,
          css: (t) => `opacity: ${Math.min(1, t * 2)}; transform: scale(${t});`
        };
      }
      const to = node.getBoundingClientRect();
      if (to.width === 0 || to.height === 0) return { duration: 0 };
      const dx = from.rect.left - to.left;
      const dy = from.rect.top - to.top;
      const dw = from.rect.width / to.width;
      const dh = from.rect.height / to.height;
      const distance = Math.hypot(dx, dy);
      const hop = Math.min(80, 24 + distance * 0.15);
      return {
        duration: Math.min(750, 380 + distance * 0.4),
        easing: cubicInOut,
        css: (t, u) => `
          transform-origin: top left;
          transform: translate(${u * dx}px, ${u * dy - Math.sin(Math.PI * t) * hop}px) scale(${t + u * dw}, ${t + u * dh});
        `
      };
    };
  }
</script>

<script lang="ts">
  import Mascot, { type MascotMood } from './Mascot.svelte';

  interface Props {
    mood?: MascotMood;
    size?: number | string;
    /** Spots sharing a key hand the mascot over to each other with a hop. */
    travelKey?: string;
    class?: string;
  }

  let {
    mood = 'idle',
    size = 120,
    travelKey = 'mascot',
    class: extraClass = ''
  }: Props = $props();
</script>

<div
  class={extraClass}
  data-mascot-spot={travelKey}
  in:arrive|global={travelKey}
  out:depart|global={travelKey}
>
  <Mascot {mood} {size} />
</div>
