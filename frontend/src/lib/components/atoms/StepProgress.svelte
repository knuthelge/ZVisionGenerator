<script lang="ts">
  interface Props {
    /** Finished steps. */
    value: number;
    max: number;
    label: string;
    /** Work is running: show the ghost fill across the step in progress and the sweep along the track. */
    live?: boolean;
    /** How long the last step took, so the ghost fills over about one step. */
    stepMs?: number | null;
    /** Work runs between steps (getting ready, or finishing up after the last step): sweep without a ghost. */
    busy?: boolean;
    valuetext?: string;
    class?: string;
  }

  let { value, max, label, live = false, stepMs = null, busy = false, valuetext, class: extraClass = '' }: Props = $props();

  const fraction = $derived(max > 0 ? Math.min(1, Math.max(0, value / max)) : 0);
  // value counts finished steps, so the step being worked on is the next one.
  const showGhost = $derived(live && !!stepMs && max > 0 && value < max);
</script>

<div
  class="ui-progress step-progress {extraClass}"
  class:is-live={(live && value < max) || busy}
  class:has-ghost={showGhost}
  role="progressbar"
  aria-label={label}
  aria-valuemin="0"
  aria-valuemax={max}
  aria-valuenow={value}
  aria-valuetext={valuetext}
>
  <span class="ui-progress-fill" style:width="{fraction * 100}%"></span>
  {#if showGhost}
    {#key value}
      <span class="step-ghost" style:left="{fraction * 100}%" style:width="{100 / max}%" style:--step-ms="{Math.round(stepMs!)}ms"></span>
    {/key}
  {/if}
</div>

<style>
  .step-progress { isolation: isolate; }
  /* While the ghost runs, the solid fill snaps to each finished step: easing it would open a dark gap between the fill
     and the ghost, which already starts at the new step. */
  .has-ghost > :global(.ui-progress-fill) { transition: none; }
  /* The step in progress fills in over about as long as the last one took, so a slow step never looks stuck. */
  .step-ghost {
    position: absolute;
    top: 0;
    bottom: 0;
    background: var(--progress-color, var(--color-primary-main));
    opacity: 0.4;
    transform-origin: left;
    animation: step-ghost-fill var(--step-ms) linear forwards;
  }
  @keyframes step-ghost-fill {
    from { transform: scaleX(0); }
    to { transform: scaleX(1); }
  }
  /* A faint sweep along the track while work runs. */
  .is-live::after {
    content: '';
    position: absolute;
    inset: 0 auto 0 0;
    width: 35%;
    pointer-events: none;
    background: linear-gradient(90deg, transparent, var(--progress-color, var(--color-primary-main)) 50%, transparent);
    opacity: 0.3;
    transform: translateX(386%);
    animation: step-sweep 1.6s linear infinite;
  }
  @keyframes step-sweep {
    from { transform: translateX(386%); }
    to { transform: translateX(-100%); }
  }
  @media (prefers-reduced-motion: reduce) {
    .is-live::after { display: none; }
    .step-ghost { animation: none; }
  }
</style>
