<script lang="ts" module>
  import { backOut, cubicInOut } from 'svelte/easing';
  import type { TransitionConfig } from 'svelte/transition';
  import { prefersReducedMotion } from './Mascot.svelte';

  // Where the mascot last left from, keyed so separate mascots never swap places.
  const departures = new Map<string, { rect: DOMRect; at: number }>();
  const HANDOFF_WINDOW_MS = 100;

  function motionAllowed(): boolean {
    if (typeof Element === 'undefined' || typeof Element.prototype.animate !== 'function') return false;
    return !prefersReducedMotion();
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

  /**
   * Fly a copy of the mascot above the page from one spot to the next, so the
   * hop is never clipped by the containers either spot sits in. The landing
   * spot is re-measured every frame, so a sheet or dialog still sliding in is
   * met where it ends up.
   */
  function fly(node: Element, from: DOMRect, duration: number): void {
    const flyer = node.cloneNode(true) as HTMLElement;
    flyer.removeAttribute('data-mascot-spot');
    flyer.setAttribute('aria-hidden', 'true');
    flyer.style.cssText = 'position: fixed; left: 0; top: 0; margin: 0; z-index: var(--z-mascot-flight); pointer-events: none; transform-origin: top left;';
    document.body.append(flyer);
    const start = performance.now();
    const frame = (now: number): void => {
      const to = node.getBoundingClientRect();
      const linear = Math.min(1, (now - start) / duration);
      const t = cubicInOut(linear);
      const u = 1 - t;
      if (linear >= 1 || !node.isConnected || to.width === 0) {
        flyer.remove();
        return;
      }
      const hop = Math.min(80, 24 + Math.hypot(from.left - to.left, from.top - to.top) * 0.15);
      const x = to.left + u * (from.left - to.left);
      const y = to.top + u * (from.top - to.top) - Math.sin(Math.PI * t) * hop;
      flyer.style.width = `${to.width}px`;
      flyer.style.height = `${to.height}px`;
      flyer.style.transform = `translate(${x}px, ${y}px) scale(${t + u * (from.width / to.width)}, ${t + u * (from.height / to.height)})`;
      requestAnimationFrame(frame);
    };
    requestAnimationFrame(frame);
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
      const duration = Math.min(750, 380 + Math.hypot(from.rect.left - to.left, from.rect.top - to.top) * 0.4);
      fly(node, from.rect, duration);
      // The spot itself keeps its place in the layout, hidden until the copy lands.
      return { duration, css: () => 'visibility: hidden;' };
    };
  }
</script>

<script lang="ts">
  import Mascot, { type MascotMood, type MascotTool } from './Mascot.svelte';

  interface Props {
    mood?: MascotMood;
    tool?: MascotTool;
    size?: number | string;
    /** Spots sharing a key hand the mascot over to each other with a hop. */
    travelKey?: string;
    /** Hidden from assistive technology, for a copy that only decorates (the logo). */
    decorative?: boolean;
    class?: string;
  }

  let {
    mood = 'idle',
    tool = 'brush',
    size = 120,
    travelKey = 'mascot',
    decorative = false,
    class: extraClass = ''
  }: Props = $props();
</script>

<div
  class={extraClass}
  data-mascot-spot={travelKey}
  in:arrive|global={travelKey}
  out:depart|global={travelKey}
>
  <Mascot {mood} {tool} {size} {decorative} />
</div>
