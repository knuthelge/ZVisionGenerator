<script lang="ts">
  import type { HaloKind } from './launcherView';

  interface Props {
    kind: HaloKind;
    /** 0 to 1: the arc of `progress`, and where an `error` froze. */
    fraction?: number;
    /** Accessible name when the ring is a progress bar. */
    label: string;
    /** Spoken value of a determinate bar, e.g. "41 percent, installing Z-Vision Generator". */
    valuetext?: string;
  }

  let { kind, fraction = 0, label, valuetext }: Props = $props();

  const RADIUS = 84;
  const CIRCUMFERENCE = 2 * Math.PI * RADIUS;
  const BUSY_SHARE = 0.2;

  const clamped = $derived(Math.min(1, Math.max(0, fraction)));
  const percent = $derived(Math.round(clamped * 100));
  const determinate = $derived(kind === 'progress');
  const busy = $derived(kind === 'busy');
  // A zero-length round cap would still draw a dot, so an empty arc is hidden instead.
  const arcShown = $derived(kind === 'busy' || kind === 'live' || ((kind === 'progress' || kind === 'error') && clamped > 0));
  const dash = $derived(busy ? `${CIRCUMFERENCE * BUSY_SHARE} ${CIRCUMFERENCE}` : `${CIRCUMFERENCE} ${CIRCUMFERENCE}`);
  const offset = $derived(kind === 'progress' || kind === 'error' ? CIRCUMFERENCE * (1 - clamped) : 0);
</script>

<svg
  class="halo"
  data-testid="halo"
  data-kind={kind}
  viewBox="0 0 176 176"
  width="176"
  height="176"
  focusable="false"
  role={determinate || busy ? 'progressbar' : undefined}
  aria-label={determinate || busy ? label : undefined}
  aria-valuemin={determinate ? 0 : undefined}
  aria-valuemax={determinate ? 100 : undefined}
  aria-valuenow={determinate ? percent : undefined}
  aria-valuetext={determinate ? (valuetext ?? `${percent} percent`) : undefined}
  aria-busy={busy ? 'true' : undefined}
  aria-hidden={determinate || busy ? undefined : 'true'}
>
  <circle class="halo-track" cx="88" cy="88" r={RADIUS} />
  <g class="halo-spin">
    <circle
      class="halo-arc"
      data-testid="halo-arc"
      cx="88"
      cy="88"
      r={RADIUS}
      transform="rotate(-90 88 88)"
      style:stroke-dasharray={dash}
      style:stroke-dashoffset={offset}
      style:opacity={arcShown ? 1 : 0}
    />
  </g>
</svg>

<style>
  .halo { display: block; overflow: visible; }
  .halo-track { fill: none; stroke: transparent; stroke-width: 2; transition: stroke 600ms var(--ease-standard); }
  .halo[data-kind='rest'] .halo-track,
  .halo[data-kind='error'] .halo-track { stroke: var(--color-border-strong); }
  .halo[data-kind='progress'] .halo-track,
  .halo[data-kind='busy'] .halo-track { stroke: var(--color-zinc-700); }
  .halo-arc {
    fill: none;
    stroke: var(--color-primary-main);
    stroke-width: 4;
    stroke-linecap: round;
    transition:
      stroke-dashoffset 400ms var(--ease-standard),
      stroke-dasharray 600ms var(--ease-standard),
      stroke 300ms var(--ease-standard),
      stroke-width 600ms var(--ease-standard),
      opacity 600ms var(--ease-standard);
  }
  .halo[data-kind='live'] .halo-arc {
    stroke-width: 2.5;
    filter: drop-shadow(0 0 6px color-mix(in srgb, var(--color-primary-main) 70%, transparent));
  }
  .halo[data-kind='error'] .halo-arc { stroke: var(--color-error); }
  .halo-spin { transform-origin: 88px 88px; }
  .halo[data-kind='busy'] .halo-spin { animation: halo-sweep 1.6s linear infinite; }
  @keyframes halo-sweep { to { transform: rotate(360deg); } }

  /* No sweep for reduced motion: a static dotted ring says "working" without looking like a 20 % fill. */
  @media (prefers-reduced-motion: reduce) {
    .halo[data-kind='busy'] .halo-arc { display: none; }
    .halo[data-kind='busy'] .halo-track { stroke: var(--color-text-muted); stroke-dasharray: 2 6; }
  }
  @media (forced-colors: active) {
    .halo-arc { stroke: Highlight; }
    .halo[data-kind='error'] .halo-arc { stroke: CanvasText; }
    .halo-track { stroke: GrayText; }
  }
</style>
