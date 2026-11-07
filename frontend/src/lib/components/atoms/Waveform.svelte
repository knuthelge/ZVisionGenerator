<script lang="ts" module>
  /** The number of bars a take's stored peaks hold. */
  export const PEAK_COUNT = 100;
  /** What a take whose peaks could not be read draws: a low, even line. */
  const FLAT_PEAK = 16;
  const BAR_SLOT = 4;
  /** A bar's width in the 400×100 box, and the radius of its rounded ends. */
  export const BAR_WIDTH = 2.6;
  export const BAR_RADIUS = 1.3;
  const HEIGHT = 100;
  const MIN_BAR = 3;
  const MAX_BAR = 96;

  /** One bar of the SVG: where it sits and how tall it is. */
  export interface WaveBar {
    x: number;
    y: number;
    height: number;
  }

  /** Lay out the bars for *peaks* (0 to 255 each) in a 400×100 box; an empty list gives a flat line, so a take still shows a waveform. */
  export function waveBars(peaks: readonly number[]): WaveBar[] {
    const values = peaks.length > 0 ? peaks : Array.from({ length: PEAK_COUNT }, () => FLAT_PEAK);
    return values.map((peak, index) => {
      const height = Math.max(MIN_BAR, (Math.min(255, Math.max(0, peak)) / 255) * MAX_BAR);
      return { x: index * BAR_SLOT + (BAR_SLOT - BAR_WIDTH) / 2, y: (HEIGHT - height) / 2, height };
    });
  }

  /** How many of *count* bars lie before *fraction* (0 to 1) of the take. */
  export function playedBars(count: number, fraction: number): number {
    return Math.max(0, Math.min(count, Math.floor(fraction * count)));
  }
</script>

<script lang="ts">
  interface Props {
    /** The take's peaks, 0 to 255 each (100 of them). */
    peaks: readonly number[];
    /** How much of the take has played (0 to 1); leave out for a still waveform with every bar the same colour. */
    played?: number | null;
    class?: string;
  }

  let { peaks, played = null, class: extraClass = '' }: Props = $props();

  const bars = $derived(waveBars(peaks));
  const width = $derived(bars.length * BAR_SLOT);
  const playedCount = $derived(played === null ? 0 : playedBars(bars.length, played));
</script>

<svg class="wave-svg {extraClass}" class:player={played !== null} viewBox="0 0 {width} {HEIGHT}" preserveAspectRatio="none" aria-hidden="true">
  {#each bars as bar, index (index)}
    <rect class="bar" data-testid="wave-bar" class:played={index < playedCount} x={bar.x} y={bar.y} width={BAR_WIDTH} height={bar.height} rx={BAR_RADIUS} />
  {/each}
</svg>

<style>
  .wave-svg { display: block; width: 100%; height: 100%; }
  .bar { fill: var(--color-zinc-500); }
  /* The player's own colours: unplayed bars zinc-600 (5.4:1 on the panel), played bars primary. */
  .player .bar { fill: var(--color-zinc-600); }
  .player .bar.played { fill: var(--color-primary-main); }
</style>
