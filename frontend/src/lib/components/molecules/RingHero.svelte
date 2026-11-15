<script lang="ts" module>
  import type { Snippet } from 'svelte';

  /** Blob inside a ring: drawn at the size given, looking along `gaze` (degrees clockwise from up) when the ring has a working head. */
  export type RingMascot = Snippet<[number, number | undefined]>;
</script>

<script lang="ts">
  // The job ring with Blob inside it, the stamp at 6 o'clock and the status mark that takes the stamp's place when work
  // stops: the launcher's hero, sized for the job card (L 144, M 128) and the phone bar (S 70).
  import { Icon, ProgressRing } from '$lib/components/atoms';
  import type { IconName } from '$lib/components/atoms/Icon.svelte';
  import { ringGaze, type MarkKind, type RingView } from '$lib/state/jobRing';
  import { RING_SPECS, ghostPlace, markDrop, type RingSize } from '$lib/state/ringGeom';

  interface Props {
    size: Exclude<RingSize, 'XS'>;
    view: RingView;
    /** Blob, drawn at the size the snippet receives. Without it the ring stands empty (Blob is in the review overlay). */
    mascot?: RingMascot;
    /** The shade the stamp and the mark sit on, so they cut the track. */
    host?: string;
    /** How long the last step took (null until known: no ghost). */
    stepMs?: number | null;
    /** Changes with every step, so the ghost restarts. */
    stepKey?: string;
  }

  let { size, view, mascot, host = 'var(--color-bg-surface)', stepMs = null, stepKey = '' }: Props = $props();

  const MARKS: Record<MarkKind, { icon: IconName; color: string; label: string }> = {
    paused: { icon: 'pause', color: 'var(--color-warning)', label: 'Paused' },
    review: { icon: 'pause', color: 'var(--color-warning)', label: 'Waiting for review' },
    stale: { icon: 'offline', color: 'var(--color-warning)', label: 'Reconnecting' },
    completed: { icon: 'check', color: 'var(--color-success)', label: 'Completed' },
    failed: { icon: 'close', color: 'var(--color-error)', label: 'Failed' },
    stopped: { icon: 'stop', color: 'var(--color-text-muted)', label: 'Stopped' },
    unknown: { icon: 'minus', color: 'var(--color-text-muted)', label: 'Outcome unknown' },
  };

  const spec = $derived(RING_SPECS[size]);
  const mark = $derived(view.mark ? MARKS[view.mark] : null);
  const drop = $derived(markDrop(spec.box, spec.stroke, spec.mark));
  // One step of the working arc: on the ring while it is long enough to see, otherwise a foot line under the stamp.
  const place = $derived(view.ghost && stepMs ? ghostPlace(size, view.arcs.length, view.ghost.share, view.stamp !== null) : null);
  const ringGhost = $derived(place === 'ring' ? view.ghost : null);
  const stampGhost = $derived(place === 'stamp');
  // He looks at the dot: the spark, or the front of the ghost it rides to.
  const gaze = $derived(ringGaze(view, size, ringGhost));
</script>

<div class="hero size-{size}" data-size={size} style:--box="{spec.box}px" style:--blob="{spec.blob}px" style:--host={host} data-testid="ring-hero">
  {#if view.pool && size !== 'S'}<div class="pool pool-{view.pool}" data-pool={view.pool} aria-hidden="true"></div>{/if}
  <ProgressRing
    size={spec.box}
    stroke={spec.stroke}
    kind={view.kind}
    arcs={view.arcs}
    faded={view.faded}
    live={view.live}
    head={view.head}
    ghost={ringGhost}
    {stepMs}
    {stepKey}
    percent={view.percent}
    decorative={view.decorative}
    valuetext={view.valuetext}
  />
  {#if mascot}<div class="blob">{@render mascot(spec.blob, gaze)}</div>{/if}
  {#if view.stamp !== null}
    <span class="stamp" data-testid="ring-stamp" aria-hidden="true">{view.stamp}%{#if stampGhost}{#key stepKey}<span class="stamp-ghost" data-testid="stamp-ghost" style:--ms="{Math.round(stepMs ?? 0)}ms"></span>{/key}{/if}</span>
  {/if}
  {#if mark}
    <span class="mark" data-testid="ring-mark" data-mark={view.mark} role="img" aria-label={mark.label} style:--mark={mark.color} style:--msize="{spec.mark}px" style:translate="0 {drop}px">
      <Icon name={mark.icon} size={spec.icon} />
    </span>
  {/if}
</div>

<style>
  .hero { position: relative; display: grid; place-items: center; width: var(--box); height: var(--box); flex: none; isolation: isolate; }
  .hero > :global(*) { grid-area: 1 / 1; }
  .pool { position: absolute; left: 50%; top: 50%; z-index: -1; width: calc(var(--box) * 1.7); height: calc(var(--box) * 1.7); translate: -50% -50%; pointer-events: none; transition: opacity 600ms var(--ease-standard); }
  /* The launcher's weakest "rest" pool: mood only, the words carry the meaning. A failed job gets the grey one. */
  .pool-work { background: radial-gradient(closest-side, color-mix(in srgb, var(--color-primary-main) 14%, transparent), color-mix(in srgb, var(--color-primary-main) 4%, transparent) 55%, transparent); }
  .pool-dim { background: radial-gradient(closest-side, color-mix(in srgb, var(--color-text-muted) 10%, transparent), transparent); }
  /* His body, not his box, sits in the ring's centre (0.1 × size, as in the launcher); his props cross the ring at the top right. */
  .blob { width: var(--blob); height: var(--blob); translate: 0 calc(var(--blob) * -0.1); }
  .blob :global(svg) { display: block; width: var(--blob); height: var(--blob); overflow: visible; }
  .stamp {
    position: relative; overflow: hidden; align-self: end; justify-self: center; translate: 0 3px;
    padding: 0 7px; border-radius: 9999px; background: var(--host);
    font-family: var(--font-mono); font-size: var(--text-ui); font-weight: 600; font-variant-numeric: tabular-nums; line-height: 18px; color: var(--color-primary-main);
  }
  .size-S .stamp { font-size: var(--text-meta); line-height: 14px; padding: 0 4px; translate: 0 2px; }
  /* A step too short to see on the ring fills a 3 px foot line under the percentage instead, as .ui-btn-wait does. */
  .stamp-ghost { position: absolute; left: 7px; right: 7px; bottom: 1px; height: 3px; border-radius: 2px; background: var(--color-primary-main); opacity: 0.4; transform-origin: left; animation: stamp-ghost var(--ms) linear forwards; }
  @keyframes stamp-ghost { from { transform: scaleX(0); } }
  @media (prefers-reduced-motion: reduce) { .stamp-ghost { animation: none !important; } }
  .mark {
    align-self: end; justify-self: center;
    display: grid; place-items: center; width: var(--msize); height: var(--msize); border-radius: 50%;
    color: var(--mark); background: color-mix(in srgb, var(--mark) 14%, var(--host));
    box-shadow: 0 0 0 3px var(--host);
    animation: mark-in 200ms var(--ease-soft-out);
  }
  @keyframes mark-in { from { scale: 0; } }
  @media (forced-colors: active) { .pool { display: none; } }
</style>
