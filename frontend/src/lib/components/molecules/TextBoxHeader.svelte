<script lang="ts">
  import { tick, type Snippet } from 'svelte';
  import { Icon } from '$lib/components/atoms';
  import type { StepperState } from '$lib/state/versions';
  import { headerMarks } from '$lib/state/headerMarks';
  import RewriteStepper from './RewriteStepper.svelte';

  interface Props {
    /** The tab shown: your text, or the Enhanced text. */
    view?: 'mine' | 'enhanced';
    /** The own tab's name ("Prompt", "Style", "Text") and the Enhanced tab's ("Enhanced"; Speech's is "Written"). */
    label?: string;
    enhancedLabel?: string;
    /** The tablist's name for assistive technology. */
    tablistLabel?: string;
    /** The ids of the two tabs and of the panels they control (not needed for a heading). */
    mineId?: string;
    enhancedId?: string;
    mineControls?: string;
    enhancedControls?: string;
    /** A box with one view only (Lyrics, a JSON prompt, Enhance each image): a heading in place of the tabs, with no marks. */
    heading?: string;
    /** What follows the heading (Lyrics' ✦). */
    headingMark?: Snippet;
    /** Generate sends the Enhanced text, whichever tab is shown. */
    used?: boolean;
    /** The Enhanced text was made from other text of yours, or for another model, than there is now. */
    stale?: boolean;
    /** Enhance is running. The helpers that run from a panel or sheet have their own Stop and do not set this. */
    running?: boolean;
    /** The versions' stepper, shown last, in the Enhanced view (always, for a heading); null for none. */
    stepper?: StepperState | null;
    stepperLabel?: string;
    /** The row has no rule under it (Video's, over the Scene and Action boxes). */
    bare?: boolean;
    /** Controls in the right group, before the stepper (Lyrics' Edit as text). */
    right?: Snippet;
    /** Empty the Enhanced text; the header then shows your tab and focuses it. */
    onscratch?: () => void;
    /** Stop Enhance. */
    onstop?: () => void;
    onstep?: (index: number) => void;
  }

  let {
    view = $bindable('mine'),
    label = 'Prompt',
    enhancedLabel = 'Enhanced',
    tablistLabel = 'Prompt version',
    mineId = '',
    enhancedId = '',
    mineControls = '',
    enhancedControls = '',
    heading,
    headingMark,
    used = false,
    stale = false,
    running = false,
    stepper = null,
    stepperLabel = 'Enhanced versions',
    bare = false,
    right,
    onscratch,
    onstop,
    onstep,
  }: Props = $props();

  const open = $derived(view === 'enhanced');
  const marks = $derived(headerMarks({ open, used, stale, running }));
  const descriptionId = $derived(`${enhancedId}-state`);
  const showStepper = $derived(stepper !== null && (heading !== undefined || open));
  let mineTab = $state<HTMLButtonElement | null>(null);
  let enhancedTab = $state<HTMLButtonElement | null>(null);
  let ring = $state<HTMLButtonElement | null>(null);
  let focusEnhancedNext = false;

  const tabEl = (next: 'mine' | 'enhanced') => (next === 'mine' ? mineTab : enhancedTab);

  function select(next: 'mine' | 'enhanced', event: MouseEvent | KeyboardEvent): void {
    view = next;
    if (event instanceof KeyboardEvent) queueMicrotask(() => tabEl(next)?.focus());
  }

  function onTabKeydown(event: KeyboardEvent): void {
    if (event.key !== 'ArrowRight' && event.key !== 'ArrowLeft') return;
    event.preventDefault();
    select(open ? 'mine' : 'enhanced', event);
  }

  /** Empty the Enhanced text and show your own, with the focus on your tab as on the neighbour of a closed browser tab. */
  async function scratch(): Promise<void> {
    onscratch?.();
    view = 'mine';
    await tick();
    mineTab?.focus();
  }

  // When the ring goes (the run ended or was stopped) while it had the focus, the focus goes to the shown tab instead of the page: the Enhanced tab,
  // or your own after a failed run sent the box back to it.
  $effect.pre(() => {
    if (marks.after !== 'ring') return;
    return () => {
      focusEnhancedNext = document.activeElement === ring;
    };
  });
  $effect(() => {
    if (marks.after === 'ring' || !focusEnhancedNext) return;
    focusEnhancedNext = false;
    tabEl(view)?.focus();
  });
</script>

<div class="head" class:bare>
  <div class="left">
    {#if heading !== undefined}
      <span class="tab sel solo">{heading}{@render headingMark?.()}</span>
    {:else}
      <div class="tablist" role="tablist" aria-label={tablistLabel}>
        <button
          bind:this={mineTab}
          type="button"
          role="tab"
          id={mineId}
          class="tab"
          class:sel={!open}
          aria-selected={!open}
          aria-controls={mineControls}
          tabindex={open ? -1 : 0}
          onclick={(e) => select('mine', e)}
          onkeydown={onTabKeydown}
        >{label}</button>
        <button
          bind:this={enhancedTab}
          type="button"
          role="tab"
          id={enhancedId}
          class="tab"
          class:sel={open}
          class:has-after={marks.after !== null}
          aria-selected={open}
          aria-controls={enhancedControls}
          aria-describedby={marks.description ? descriptionId : undefined}
          tabindex={open ? 0 : -1}
          onclick={(e) => select('enhanced', e)}
          onkeydown={onTabKeydown}
        >
          {enhancedLabel}
          {#if marks.mark === 'spinner'}
            <span class="mark" data-mark="spinner" aria-hidden="true"><span class="spinner"></span></span>
          {:else if marks.mark === 'warn'}
            <span class="mark warn" data-mark="warn" aria-hidden="true" title="Generate sends this, but your text or the model changed since"><Icon name="alert" size={11} /></span>
          {:else if marks.mark === 'dot'}
            <span class="mark dot" data-mark="dot" aria-hidden="true" title="Generate sends this"></span>
          {/if}
        </button>
      </div>
      <span id={descriptionId} class="sr-only">{marks.description}</span>
      <!-- Siblings of the tablist, never inside a tab; drawn as one unit with the open Enhanced tab. -->
      {#if marks.after === 'ring'}
        <button bind:this={ring} type="button" class="stop-ring" aria-label="Stop enhancing" title="Stop enhancing" onclick={() => onstop?.()}><i></i></button>
      {:else if marks.after === 'scratch'}
        <button type="button" class="scratch" aria-label="Scratch the enhanced text" title="Scratch: Generate goes back to your text" onclick={scratch}><span class="glyph"><Icon name="close" size={11} /></span></button>
      {/if}
    {/if}
  </div>
  {#if right || showStepper}
    <div class="right">
      {@render right?.()}
      {#if showStepper && stepper}<RewriteStepper index={stepper.index} count={stepper.count} back={stepper.back} label={stepperLabel} onstep={(index) => onstep?.(index)} />{/if}
    </div>
  {/if}
</div>

<style>
  /* LEFT what you are looking at, RIGHT moving through it. Nothing reserves space: a control that appears moves its neighbours by its width. */
  .head { display: flex; align-items: flex-end; padding: 4px 4px 0 6px; border-bottom: 1px solid var(--color-border-subtle); }
  .head.bare { border-bottom: 0; padding-left: 2px; }
  .left { display: flex; align-items: stretch; align-self: stretch; min-width: 0; }
  .tablist { display: flex; align-items: flex-end; align-self: flex-end; gap: 4px; }
  .right { display: flex; align-items: center; align-self: center; gap: 4px; margin-left: auto; }

  /* An ordinary tab: 6px each side, no gap, no min-width. Only a tab that shows a mark grows. */
  .tab { position: relative; display: inline-flex; align-items: center; padding: 5px 6px 6px; white-space: nowrap; font-size: var(--text-ui); font-weight: 600; color: var(--color-text-muted); }
  .tab.solo { align-self: flex-end; }
  .tab:focus-visible { outline: 2px solid var(--color-primary-main); outline-offset: -2px; border-radius: var(--radius-xs); }
  @media (hover: hover) { .tab:hover { color: var(--color-text-primary); } }
  .tab.sel { color: var(--color-text-primary); }
  /* The underline's insets equal the tab's 6px side padding. */
  .tab.sel::after { content: ''; position: absolute; right: 6px; bottom: -1px; left: 6px; height: 2px; border-radius: 2px; background: var(--color-primary-main); }
  @media (pointer: coarse) { .tab { padding: 9px 6px 10px; } }

  /* The one mark a tab may carry. */
  .mark { display: inline-flex; align-items: center; margin-left: 5px; }
  .mark.dot { width: 6px; height: 6px; border-radius: 50%; background: var(--color-primary-main); }
  .mark.warn { color: var(--color-warning); }
  .spinner { display: inline-block; width: 9px; height: 9px; border: 1.5px solid var(--color-primary-main); border-right-color: transparent; border-radius: 50%; animation: turn 0.9s linear infinite; }
  @keyframes turn { to { transform: rotate(360deg); } }

  /* The open Enhanced tab and its ✕ (or stop ring) read as one tab: the tab drops its right padding, the glyph sits 4px (mouse) or 8px (touch)
     from the label, its hit area extends away from the label, and the underline runs under label, gap and glyph. */
  .tab.has-after { padding-right: 0; }
  .tab.has-after.sel::after { right: 0; border-radius: 2px 0 0 2px; }
  .scratch, .stop-ring { position: relative; display: flex; align-items: center; justify-content: flex-start; align-self: stretch; width: 18px; margin: 0 0 0 4px; padding: 0 0 1px; background: transparent; color: var(--color-text-secondary); }
  .scratch:focus-visible, .stop-ring:focus-visible { outline: 2px solid var(--color-primary-main); outline-offset: -2px; border-radius: var(--radius-xs); }
  @media (pointer: coarse) { .scratch, .stop-ring { width: 32px; margin-left: 8px; } }
  .glyph { display: grid; place-items: center; width: 11px; height: 11px; border-radius: 3px; }
  .scratch::after, .stop-ring::after { content: ''; position: absolute; left: -4px; bottom: -1px; width: 15px; height: 2px; border-radius: 0 2px 2px 0; background: var(--color-primary-main); }
  .stop-ring::after { width: 20px; }
  @media (pointer: coarse) {
    .scratch::after { left: -8px; width: 19px; }
    .stop-ring::after { left: -8px; width: 24px; }
  }
  /* Hover and press on either part brightens both; the glyph gets its own chip only under the pointer. */
  @media (hover: hover) {
    .left:has(.scratch:hover, .stop-ring:hover) .tab.sel { color: var(--color-text-primary); }
    .left:has(.tab.sel:hover) .scratch { color: var(--color-text-primary); }
    .scratch:hover .glyph { background: var(--color-bg-control-hover); box-shadow: 0 0 0 3px var(--color-bg-control-hover); }
  }
  /* The stop ring: a 16px ring with an 8px stop square, left-aligned in its hit area like the ✕ glyph. */
  .stop-ring::before { content: ''; position: absolute; left: 0; top: 50%; width: 16px; height: 16px; margin-top: -8.5px; border: 1.5px solid var(--color-primary-main); border-right-color: transparent; border-radius: 50%; box-sizing: border-box; animation: turn 0.9s linear infinite; }
  .stop-ring i { position: absolute; left: 4px; top: 50%; width: 8px; height: 8px; margin-top: -4.5px; border-radius: 1.5px; background: var(--color-text-secondary); }
  @media (hover: hover) { .stop-ring:hover i { background: var(--color-error); } }
  /* With reduced motion the ring and the spinner stand still; the streaming text and the placeholder carry the progress. */
  @media (prefers-reduced-motion: reduce) {
    .spinner, .stop-ring::before { animation: none; }
    .stop-ring::before { border-right-color: var(--color-primary-main); }
  }
</style>
