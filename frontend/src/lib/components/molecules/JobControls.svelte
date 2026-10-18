<script lang="ts" module>
  import type { JobControlAction } from '$lib/types';

  export const CONTROL_KEYS: Record<JobControlAction, string> = { pause: 'P', resume: 'P', next: 'N', repeat: 'R', approve: 'Y', discard: 'N', quit: 'Esc' };
</script>

<script lang="ts">
  import { Icon } from '$lib/components/atoms';
  import type { IconName } from '$lib/components/atoms/Icon.svelte';

  interface Props {
    controls: JobControlAction[];
    /** Accessible name of the group, e.g. "Job controls" or "Current image". */
    label: string;
    /** Tooltip and accessible name per control, e.g. "Skip this image". */
    names: Partial<Record<JobControlAction, string>>;
    /** A control sent or queued that has not taken effect yet; it shows as on and the others wait. */
    pending?: JobControlAction | null;
    /** Everything is unavailable, e.g. while the connection is down. */
    disabled?: boolean;
    /** The first Esc armed Cancel: it shows as the danger button until pressed again. */
    cancelArmed?: boolean;
    onsend: (action: JobControlAction) => void;
  }

  let { controls, label, names, pending = null, disabled = false, cancelArmed = false, onsend }: Props = $props();

  const ICONS: Record<JobControlAction, IconName> = { pause: 'pause', resume: 'play', next: 'skip', repeat: 'reuse', approve: 'check', discard: 'close', quit: 'close' };
  const main = $derived(controls.filter((action) => action !== 'quit'));
  const hasQuit = $derived(controls.includes('quit'));

  function variant(action: JobControlAction): string {
    if (action === 'resume') return 'ui-btn-primary';
    if (action === 'quit') return cancelArmed ? 'ui-btn-danger' : 'ui-btn-quiet job-cancel';
    // Not combined with -quiet: its transparent background would win over -on.
    return pending === action ? 'ui-btn-on' : 'ui-btn-quiet';
  }

  function text(action: JobControlAction): string {
    if (action === 'resume') return 'Resume';
    if (action === 'quit' && cancelArmed) return 'Cancel job';
    if (action === 'pause' && pending === 'pause') return 'Pausing';
    return '';
  }

  // Cancel is never held up by another pending control.
  const isDisabled = (action: JobControlAction): boolean => disabled || (!!pending && pending !== action && action !== 'quit');
</script>

{#snippet button(action: JobControlAction)}
  <button
    type="button"
    class="ui-btn ui-btn-sm {variant(action)}"
    data-control={action === 'pause' || action === 'resume' ? 'pause-resume' : action}
    disabled={isDisabled(action)}
    title="{names[action]} ({CONTROL_KEYS[action]})"
    aria-label={names[action]}
    aria-busy={pending === action ? 'true' : undefined}
    onclick={() => onsend(action)}
  ><Icon name={ICONS[action]} size={14} />{text(action)}<kbd class="ui-kbd">{CONTROL_KEYS[action]}</kbd></button>
{/snippet}

<span class="job-controls" role="group" aria-label={label}>
  {#each main as action (action === 'resume' ? 'pause' : action)}{@render button(action)}{/each}
  {#if main.length && hasQuit}<span class="job-controls-divider" aria-hidden="true"></span>{/if}
  {#if hasQuit}{@render button('quit')}{/if}
</span>

<style>
  .job-controls { display: flex; align-items: center; gap: 2px; }
  .job-controls-divider { width: 1px; height: 16px; margin: 0 6px; background: var(--color-border-strong); }
  /* Quiet, but the red icon marks Cancel as the stop. */
  .job-cancel :global(svg) { color: var(--color-error); }
  /* A finger gets the touch layer's full control size, not the small one. */
  @media (pointer: coarse) {
    .job-controls :global(.ui-btn) { height: var(--spacing-control); min-width: var(--spacing-control); }
    .job-controls { gap: 4px; }
  }
</style>
