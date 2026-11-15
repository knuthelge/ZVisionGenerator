<script lang="ts" module>
  import type { JobControlAction } from '$lib/types';

  export const CONTROL_KEYS: Record<JobControlAction, string> = { pause: 'P', resume: 'P', next: 'N', repeat: 'R', approve: 'Y', discard: 'N', quit: 'Esc' };
</script>

<script lang="ts">
  import { Icon } from '$lib/components/atoms';
  import type { IconName } from '$lib/components/atoms/Icon.svelte';
  import type { JobControls } from '$lib/state/jobView';

  interface Props {
    /** What the job offers right now, split by scope. */
    controls: JobControls;
    /** What an item is called in the item group's name: "image", "take", "rewrite". */
    itemLabel: string;
    /** Tooltip and accessible name per control, e.g. "Skip this image". */
    names: Partial<Record<JobControlAction, string>>;
    /** A control sent or queued that has not taken effect yet; it shows as on and the others wait. */
    pending?: JobControlAction | null;
    /** Everything is unavailable, e.g. while the connection is down. */
    disabled?: boolean;
    /** The first Esc armed Cancel: it shows as the danger button until pressed again. */
    cancelArmed?: boolean;
    /** A finger: every control is an icon button with no key hint. */
    touch?: boolean;
    onsend: (action: JobControlAction) => void;
  }

  let { controls, itemLabel, names, pending = null, disabled = false, cancelArmed = false, touch = false, onsend }: Props = $props();

  const ICONS: Record<JobControlAction, IconName> = { pause: 'pause', resume: 'play', next: 'skip', repeat: 'reuse', approve: 'check', discard: 'close', quit: 'close' };
  const jobScope = $derived(controls.job.filter((action) => action !== 'quit'));
  const hasQuit = $derived(controls.job.includes('quit'));

  // Cancel is never held up by another pending control.
  const isDisabled = (action: JobControlAction): boolean => disabled || (!!pending && pending !== action && action !== 'quit');
</script>

{#snippet button(action: JobControlAction)}
  {#if action === 'resume'}
    <button type="button" class="ui-btn ui-btn-primary" class:ui-btn-sm={!touch} class:touch data-control="pause-resume" disabled={isDisabled(action)} title="{names[action]} ({CONTROL_KEYS[action]})" aria-label={names[action]} onclick={() => onsend(action)}>
      <Icon name="play" size={14} />Resume{#if !touch}<kbd>{CONTROL_KEYS[action]}</kbd>{/if}
    </button>
  {:else if action === 'quit' && cancelArmed && !touch}
    <button type="button" class="ui-btn ui-btn-sm ui-btn-danger" data-control="quit" disabled={isDisabled(action)} title="{names[action]} ({CONTROL_KEYS[action]})" aria-label={names[action]} onclick={() => onsend(action)}>
      <Icon name="close" size={14} />Cancel
    </button>
  {:else}
    <button
      type="button"
      class="ui-btn ui-btn-quiet"
      class:ui-btn-sm={!touch}
      class:ui-btn-icon={touch}
      class:ui-btn-on={pending === action}
      class:touch
      class:cancel={action === 'quit'}
      data-control={action === 'pause' ? 'pause-resume' : action}
      disabled={isDisabled(action)}
      title="{names[action]} ({CONTROL_KEYS[action]})"
      aria-label={names[action]}
      aria-busy={pending === action ? 'true' : undefined}
      onclick={() => onsend(action)}
    ><Icon name={ICONS[action]} size={touch ? 16 : 14} />{#if !touch}<kbd>{CONTROL_KEYS[action]}</kbd>{/if}</button>
  {/if}
{/snippet}

<div class="job-controls" class:touch class:armed={cancelArmed && !touch} data-testid="job-controls">
  {#if jobScope.length}
    <div role="group" aria-label="Job controls" class="group">
      {#each jobScope as action (action === 'resume' ? 'pause' : action)}{@render button(action)}{/each}
    </div>
  {/if}
  {#if controls.item.length}
    <div role="group" aria-label="Current {itemLabel}" class="group">
      {#each controls.item as action (action)}{@render button(action)}{/each}
    </div>
  {/if}
  {#if hasQuit}
    {#if jobScope.length || controls.item.length}<span class="divider" data-testid="controls-divider" aria-hidden="true"></span>{/if}
    {@render button('quit')}
  {/if}
</div>

<style>
  .job-controls { display: flex; align-items: center; gap: 2px; margin-left: -7px; }
  .job-controls.touch { gap: 4px; margin-left: -10px; }
  .group { display: flex; gap: inherit; }
  .ui-btn { gap: 5px; }
  .ui-btn.ui-btn-quiet { color: var(--color-text-secondary); }
  /* Quiet, but the red icon marks Cancel as the stop. */
  .ui-btn.cancel.ui-btn-quiet { color: var(--color-error); }
  .ui-btn.ui-btn-quiet.ui-btn-on { color: var(--color-primary-main); }
  /* The key hint is the app's bare in-button kbd: 11 px mono at 70 %, no chip. */
  .ui-btn.ui-btn-quiet kbd { color: var(--color-text-muted); opacity: 1; }
  .ui-btn.ui-btn-on kbd { color: inherit; opacity: 0.7; }
  .divider { width: 1px; height: 16px; margin: 0 4px; background: var(--color-border-strong); }
  /* A finger: Resume is a labelled primary at the touch layer's control size. */
  .ui-btn-primary.touch { height: var(--spacing-control); padding: 0 14px; }
</style>
