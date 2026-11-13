<script lang="ts">
  import Mascot, { type MascotMood } from '$lib/components/atoms/Mascot.svelte';
  import ProgressHalo from './ProgressHalo.svelte';
  import type { HaloView, PoolStrength } from './launcherView';

  interface Props {
    mood: MascotMood;
    /** Share (0 to 1) of the work done; the Mascot's canvas follows it while he creates. */
    mascotProgress?: number;
    halo: HaloView;
    /** Shows the stamp on the ring: the percentage done. */
    percent?: number;
    /** An update is offered: the amber dot. */
    update?: boolean;
    /** The brand light's strength. */
    pool: PoolStrength;
  }

  let { mood, mascotProgress, halo, percent, update = false, pool }: Props = $props();

  const STRENGTHS: PoolStrength[] = ['live', 'work', 'rest', 'dim'];
</script>

<div class="hero">
  <!-- One layer per strength, so a change cross-fades instead of jumping. -->
  {#each STRENGTHS as level (level)}
    <div class="pool pool-{level}" class:on={pool === level} aria-hidden="true"></div>
  {/each}
  <ProgressHalo kind={halo.kind} fraction={halo.fraction} label={halo.label} valuetext={halo.valuetext} />
  <div class="mascot">
    <Mascot {mood} size={140} progress={mascotProgress} decorative />
  </div>
  {#if percent !== undefined}
    <span class="stamp" data-testid="halo-stamp">{percent}%</span>
  {/if}
  {#if update}
    <span class="update-dot" data-testid="update-dot" aria-hidden="true"></span>
  {/if}
</div>

<style>
  .hero { position: relative; flex: none; display: grid; place-items: center; height: 180px; }
  .hero > :global(*) { grid-area: 1 / 1; }
  .pool {
    position: absolute;
    left: 50%;
    top: 50%;
    z-index: -1;
    width: 420px;
    height: 240px;
    translate: -50% -50%;
    opacity: 0;
    pointer-events: none;
    transition: opacity 600ms var(--ease-standard);
  }
  .pool.on { opacity: 1; }
  .pool-live { background: radial-gradient(closest-side, color-mix(in srgb, var(--color-primary-main) 34%, transparent), color-mix(in srgb, var(--color-primary-main) 10%, transparent) 55%, transparent); }
  .pool-work { background: radial-gradient(closest-side, color-mix(in srgb, var(--color-primary-main) 22%, transparent), color-mix(in srgb, var(--color-primary-main) 7%, transparent) 55%, transparent); }
  .pool-rest { background: radial-gradient(closest-side, color-mix(in srgb, var(--color-primary-main) 14%, transparent), color-mix(in srgb, var(--color-primary-main) 4%, transparent) 55%, transparent); }
  .pool-dim { background: radial-gradient(closest-side, color-mix(in srgb, var(--color-text-muted) 10%, transparent), transparent); }
  /* The body, not the box, sits in the ring's centre: his body's centre is 0.1 x size below the box's. */
  .mascot { width: 140px; height: 140px; translate: 0 -14px; }
  .mascot :global(svg) { display: block; width: 140px; height: 140px; overflow: visible; }
  .stamp {
    align-self: end;
    justify-self: center;
    translate: 0 -2px;
    padding: 1px 8px;
    border-radius: 9999px;
    background: var(--color-bg-base);
    font-family: var(--font-mono);
    font-size: var(--text-ui);
    font-weight: 600;
    font-variant-numeric: tabular-nums;
    line-height: 18px;
    color: var(--color-primary-main);
  }
  .update-dot {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background: var(--color-accent-amber);
    box-shadow: 0 0 0 3px var(--color-bg-base);
    translate: 59px -59px;
    animation: dot-in 200ms var(--ease-soft-out);
  }
  @keyframes dot-in { from { scale: 0; } }
  @media (max-height: 499px) {
    .hero { height: 132px; }
    .pool { height: 180px; }
    .hero :global(.halo) { width: 128px; height: 128px; }
    .mascot, .mascot :global(svg) { width: 100px; height: 100px; }
    .mascot { translate: 0 -10px; }
    .update-dot { translate: 43px -43px; }
  }
  @media (forced-colors: active) {
    .pool, .update-dot { display: none; }
  }
</style>
