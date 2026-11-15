<script lang="ts">
  // The job's one progress shape: arcs round a ring, drawn from 12 o'clock with round caps. Two signs of work sit on the
  // arcs, so a slow job never looks stuck:
  // - the step ghost: the step in progress fills in at 40 % over the last step's time, then the solid fill snaps forward
  //   and a new ghost starts;
  // - the spark: a dot at the working head with a heartbeat ring every 1.6 s. It rides the ghost's front, waits at the
  //   fill's end between steps and leads the sweep when the length is unknown.
  import { arcLayout, headAngles, ringRadius } from '$lib/state/ringGeom';
  import type { RingArc, RingGhost, RingInk, RingKind } from '$lib/state/jobRing';

  interface Props {
    /** The box in px. */
    size: number;
    /** The arc's width; the track is half of it. */
    stroke?: number;
    /** `rest`: the track only. `busy`: a sweep, for a length that is not known. `arc`: the arcs. */
    kind: RingKind;
    arcs?: RingArc[];
    /** Every arc drawn muted: the stream is stale. */
    faded?: boolean;
    /** Work runs: show the spark. */
    live?: boolean;
    /** The arc whose fill end is the working head. */
    head?: number | null;
    /** The step ghost, drawn only when one step is long enough to see on the ring. */
    ghost?: RingGhost | null;
    /** How long the last step took; without it there is no ghost. */
    stepMs?: number | null;
    /** Changes with every step, so the ghost and the spark restart. */
    stepKey?: string;
    /** The whole job's percentage (floored). */
    percent?: number;
    /** Says nothing to assistive technology: a rest, ended or pill ring. */
    decorative?: boolean;
    /** Nothing moves: no sweep, no ghost, no spark. */
    still?: boolean;
    valuetext?: string;
  }

  let { size, stroke = 4, kind, arcs = [], faded = false, live = false, head = null, ghost = null, stepMs = null, stepKey = '', percent = 0, decorative = false, still = false, valuetext }: Props = $props();

  const INK: Record<RingInk, string> = {
    work: 'var(--color-primary-main)',
    paused: 'var(--color-warning)',
    done: 'var(--color-primary-main)',
    failed: 'var(--color-error)',
    stopped: 'color-mix(in srgb, var(--color-text-muted) 70%, transparent)',
  };

  const track = $derived(stroke / 2);
  const radius = $derived(ringRadius(size, stroke));
  const circumference = $derived(2 * Math.PI * radius);
  const centre = $derived(size / 2);
  const count = $derived(Math.max(1, arcs.length));
  const geometry = $derived(arcLayout(circumference, count, stroke, size));
  const placed = $derived(arcs.map((arc, index) => ({ ...arc, start: index * geometry.pitch + geometry.offset })));
  const announces = $derived(!decorative && (kind === 'arc' || kind === 'busy'));
  const showGhost = $derived(!still && !!ghost && !!stepMs && kind === 'arc' && !!placed[ghost.arc]);
  // The spark sits at the head's fill end; with a ghost it rides from there to the ghost's front over the step.
  const angles = $derived(headAngles(size, stroke, arcs, head, showGhost ? ghost : null));
  const from = $derived(angles?.from ?? 0);
  const to = $derived(angles?.to ?? 0);
  const sparkRadius = $derived(Math.max(2.5, stroke));
</script>

<svg
  class="pring"
  data-kind={kind}
  data-testid="progress-ring"
  class:faded
  class:dotted={kind === 'busy' && (still || !live)}
  class:has-ghost={showGhost}
  viewBox="0 0 {size} {size}"
  width={size}
  height={size}
  focusable="false"
  role={announces ? 'progressbar' : undefined}
  aria-label={announces ? 'Job progress' : undefined}
  aria-valuemin={announces && kind === 'arc' ? 0 : undefined}
  aria-valuemax={announces && kind === 'arc' ? 100 : undefined}
  aria-valuenow={announces && kind === 'arc' ? percent : undefined}
  aria-valuetext={announces ? valuetext : undefined}
  aria-busy={announces && kind === 'busy' ? 'true' : undefined}
  aria-hidden={announces ? undefined : 'true'}
  style:--stroke="{stroke}px"
  style:--track="{track}px"
>
  {#if kind !== 'arc' || arcs.length <= 1}
    <circle class="track" data-testid="ring-track" cx={centre} cy={centre} r={radius} />
  {/if}
  {#if kind === 'busy' && live && !still}
    <g class="spin" data-testid="ring-sweep" style:transform-origin="{centre}px {centre}px">
      <circle class="arc" cx={centre} cy={centre} r={radius} transform="rotate(-90 {centre} {centre})" style:stroke-dasharray="{circumference * 0.2} {circumference}" style:stroke="var(--color-primary-main)" />
      <g transform="rotate(72 {centre} {centre})" data-testid="ring-spark"><circle class="spark-halo" cx={centre} cy={centre - radius} r={sparkRadius} /><circle class="spark" cx={centre} cy={centre - radius} r={sparkRadius} /></g>
    </g>
    <!-- Reduced motion: the sweep is gone, the dotted track stays, and the spark breathes at 12 o'clock. -->
    <circle class="spark spark-still" data-testid="ring-spark" cx={centre} cy={centre - radius} r={sparkRadius} />
  {:else if kind === 'arc'}
    {#each placed as arc, index (index)}
      {#if arcs.length > 1}
        <circle class="track" data-testid="ring-track" cx={centre} cy={centre} r={radius} transform="rotate(-90 {centre} {centre})" style:stroke-dasharray="{geometry.dash} {circumference}" style:stroke-dashoffset={-arc.start} />
      {/if}
      <circle
        class="arc"
        data-testid="ring-arc"
        data-ink={arc.ink}
        cx={centre}
        cy={centre}
        r={radius}
        transform="rotate(-90 {centre} {centre})"
        style:stroke-dasharray="{Math.max(0, geometry.dash * arc.fill)} {circumference}"
        style:stroke-dashoffset={-arc.start}
        style:stroke={INK[arc.ink]}
        style:opacity={arc.fill > 0.001 ? 1 : 0}
      />
    {/each}
    {#if showGhost && ghost}
      {#key stepKey}
        <circle
          class="ghost"
          data-testid="ring-ghost"
          cx={centre}
          cy={centre}
          r={radius}
          transform="rotate(-90 {centre} {centre})"
          style:--glen="{geometry.dash * ghost.share}px"
          style:--gfull="{circumference}px"
          style:--ms="{Math.round(stepMs ?? 0)}ms"
          style:stroke-dashoffset={-(placed[ghost.arc].start + geometry.dash * ghost.from)}
        />
      {/key}
    {/if}
    {#if live && angles && !still}
      {#key stepKey}
        <g class="spark-ride" class:riding={showGhost} data-riding={showGhost ? '' : undefined} data-testid="ring-spark" style:--a0="{from}deg" style:--a1="{to}deg" style:--ms="{Math.round(stepMs ?? 0)}ms" style:transform-origin="{centre}px {centre}px">
          <circle class="spark-halo" cx={centre} cy={centre - radius} r={sparkRadius} />
          <circle class="spark" cx={centre} cy={centre - radius} r={sparkRadius} />
        </g>
      {/key}
    {/if}
  {/if}
</svg>

<style>
  .pring { display: block; overflow: visible; }
  .track { fill: none; stroke: var(--color-zinc-700); stroke-width: var(--track); stroke-linecap: round; transition: stroke 600ms var(--ease-standard); }
  .pring[data-kind='rest'] .track { stroke: var(--color-border-strong); }
  .arc {
    fill: none;
    stroke-width: var(--stroke);
    stroke-linecap: round;
    transition: stroke-dasharray 400ms var(--ease-standard), stroke 300ms var(--ease-standard), opacity 600ms var(--ease-standard);
  }
  /* While a ghost runs, the solid fill snaps to each finished step (as StepProgress does): easing it would open a gap
     between the fill and the ghost, which already starts at the new step. */
  .has-ghost .arc { transition: stroke 300ms var(--ease-standard), opacity 600ms var(--ease-standard); }
  .faded .arc { opacity: 0.45; stroke: var(--color-text-muted) !important; }

  /* The step ghost: the progress colour at 40 %, filling the step in progress over the last step's time, linear. */
  .ghost {
    fill: none;
    stroke: var(--color-primary-main);
    stroke-width: var(--stroke);
    stroke-linecap: round;
    opacity: 0.4;
    stroke-dasharray: var(--glen) var(--gfull);
    animation: ghost-fill var(--ms) linear forwards;
  }
  @keyframes ghost-fill { from { stroke-dasharray: 0px var(--gfull); } }

  /* The spark: the working head. A dot in the hover teal, with a ring that beats out from it every 1.6 s. */
  .spark { fill: var(--color-primary-hover); }
  .spark-halo { fill: none; stroke: var(--color-primary-main); stroke-width: 1.5; transform-box: fill-box; transform-origin: center; animation: spark-beat 1.6s var(--ease-soft-out) infinite; }
  @keyframes spark-beat { from { transform: scale(1); opacity: 0.7; } to { transform: scale(3.2); opacity: 0; } }
  .spark-ride { transform: rotate(var(--a0)); }
  .spark-ride.riding { animation: spark-ride var(--ms) linear forwards; }
  @keyframes spark-ride { from { transform: rotate(var(--a0)); } to { transform: rotate(var(--a1)); } }
  .spark-still { display: none; }

  .spin { animation: ring-sweep 1.6s linear infinite; }
  @keyframes ring-sweep { to { transform: rotate(360deg); } }

  /* A ring with no work moving (the status pill, or a lost connection) does not sweep: an unknown length is the dotted track. */
  .pring.dotted .track { stroke: var(--color-text-muted); stroke-dasharray: 2 6; }

  /* Reduced motion: nothing moves. No sweep, no riding, no beating ring, and the ghost stands still across the whole step
     (as StepProgress does). The sign of life is the spark breathing in place, opacity only: the one animation the
     app's global reduced-motion clamp lets through (`global.css` sets its duration and count in the clamp's own layer).
     An unknown length is a still dotted track with the breathing spark at 12 o'clock. */
  @media (prefers-reduced-motion: reduce) {
    .spin { display: none; }
    .pring[data-kind='busy'] .track { stroke: var(--color-text-muted); stroke-dasharray: 2 6; }
    .ghost { animation: none !important; }
    .spark-ride.riding { animation: none !important; transform: rotate(var(--a0)); }
    .spark-halo { display: none; }
    .spark-still { display: inline; }
    .spark { animation: spark-breathe 2.4s ease-in-out infinite alternate; }
  }
  @keyframes spark-breathe { from { opacity: 1; } to { opacity: 0.3; } }
  @media (forced-colors: active) {
    .arc, .ghost { stroke: Highlight !important; }
    .spark { fill: Highlight; }
    .track { stroke: GrayText; }
  }
</style>
