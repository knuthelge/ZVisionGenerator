<script lang="ts" module>
  import type { AudioRequest } from '$lib/api/promptEnhance';
  import type { WriterResult } from '$lib/state/writerRun.svelte';

  /** What a helper hands the panel to run: the request, how to tell the words it worked from are gone, and what to do with the answer. */
  export interface HelperJob {
    request: AudioRequest;
    /** The field of the answer the task writes. */
    field: 'lyrics' | 'script';
    /** Whether the words the helper worked from changed while it was working. */
    changed: () => boolean;
    /** What the panel says when they did, and nothing was applied. */
    changedNote: string;
    apply: (result: WriterResult) => void;
  }

  /** The panel's answer to a text it cannot work on: what it says, whether it carries the sparkle, and whether it offers Fill in from Describe it. */
  export interface HelperProblem {
    text: string;
    icon: boolean;
    fill: boolean;
  }
</script>

<script lang="ts">
  import type { Snippet } from 'svelte';
  import { Icon } from '$lib/components/atoms';
  import { refocusIfLost } from '$lib/focus';
  import { EnhanceChoice } from '$lib/components/molecules';
  import Sheet from '$lib/components/molecules/Sheet.svelte';
  import { popover } from '$lib/actions/popover';
  import { draft } from '$lib/state/draft.svelte';
  import { downloadedTiers } from '$lib/state/enhancerDownload.svelte';
  import { layout } from '$lib/state/layoutMode.svelte';
  import { downloadSizeOf, writerBlock, type WriterRun } from '$lib/state/writerRun.svelte';
  import type { PromptEnhancerContract } from '$lib/types';

  /**
   * The shell of a writing helper's panel (the lyrics and script helpers): a popover beside the box on a wide window and a sheet on a phone or
   * tablet, with the Enhancer choice and Run/Stop in the foot. The helper owns what it asks and what the options are; the shell owns the run.
   */
  interface Props {
    contract: PromptEnhancerContract;
    /** The helper's run, which the shell stops, disposes of and reads the progress of. */
    run: WriterRun;
    open: boolean;
    /** The element a popover opens beside (a sheet has none). */
    anchor: HTMLElement | null;
    /** Names the panel: the popover's heading and label, and the sheet's title. */
    title: string;
    /** Prefix of the panel's ids (`ws-<prefix>-run`) and test ids (`<prefix>-note`). */
    idPrefix: 'lyrics-helper' | 'script-helper';
    /** What the Smart enhancer is said to do in the Enhancer choice. */
    smartHelp: string;
    /** The run button's label. */
    runLabel: string;
    /** The text the helper works on can be run now (the shell adds the enhancer, the take and the run in progress). */
    ready: boolean;
    /** The panel says this in place of the options, and has no foot; null while the helper can work on its text. */
    problem: HelperProblem | null;
    /** The text the sheet quotes above the options: what the helper works on. */
    quote?: { label: string; text: string } | null;
    /** Why the written text is out of date, said beside Run; null while it is not. */
    staleReason?: string | null;
    /** Changes when a new session of the panel starts (a new target, or opening), which clears the last run's error and stop note. */
    session: unknown;
    /** A job is being submitted. */
    busy: boolean;
    /** A take is running or queued; the enhancer would need its memory. */
    jobsActive: boolean;
    onclose: () => void;
    /** Fill in from Describe it: offered where a problem says there is nothing to work on. */
    onfill?: () => void;
    /** What to run now; called only when the panel can run. */
    onrun: () => HelperJob;
    /** The helper's options: shown in the popover and the sheet. */
    body: Snippet;
  }

  let { contract, run, open, anchor, title, idPrefix, smartHelp, runLabel, ready, problem, quote = null, staleReason = null, session, busy, jobsActive, onclose, onfill, onrun, body }: Props = $props();

  const TIER_OPTIONS = $derived([
    { value: 'fast' as const, label: 'Fast', help: 'The default model.' },
    { value: 'smart' as const, label: 'Smart', help: smartHelp, unavailable: contract.smart?.unavailable ?? null },
  ]);

  let stopNote = $state<string | null>(null);

  /** Phones and tablets open the helper as a sheet; only the wide layout has room for a popover beside the box. */
  const sheet = $derived(layout.mode !== 'wide');
  const tier = $derived(run.tier(contract));
  const unavailable = $derived(writerBlock(contract, { jobsActive, busy }));
  const canRun = $derived(ready && unavailable === null && !run.running && !run.asking);
  const downloadSize = $derived(downloadSizeOf(contract, tier));
  const fastUndownloaded = $derived(tier === 'fast' && !contract.downloaded && !downloadedTiers.has('fast'));

  // A new session starts clean: no old error or stop note.
  $effect(() => {
    void session;
    run.clear();
    stopNote = null;
  });

  // Closing the panel while the helper works stops it: nothing is applied behind the user's back.
  $effect(() => {
    if (!open && run.running) run.stop();
  });

  $effect(() => () => run.dispose());

  // Run and Stop take the same place in the panel's foot; the focus goes from one to the other instead of falling to the page.
  let wasRunning = false;
  $effect(() => {
    const running = run.running;
    if (running === wasRunning) return;
    wasRunning = running;
    void refocusIfLost(() => document.getElementById(`ws-${idPrefix}-${running ? 'stop' : 'run'}`));
  });

  async function submit(): Promise<void> {
    if (!canRun) return;
    const job = onrun();
    stopNote = null;
    const result = await run.start(contract, job.request, job.field);
    if (result === null) return;
    // The words the helper worked from are gone: its answer would put them back over what the user typed.
    if (job.changed()) {
      run.errorText = job.changedNote;
      return;
    }
    job.apply(result);
    onclose();
    void refocusIfLost(() => anchor);
  }

  function stop(): void {
    stopNote = run.stop();
  }

  /**
   * The panel closes unless the helper is writing (only its Stop ends that) or the download dialog is in front of it (clicking that dialog is
   * outside the panel). While it writes, the focus stays on Stop: the popover's Escape would otherwise move it to the opener.
   */
  function dismiss(): void {
    if (run.running) {
      queueMicrotask(() => document.getElementById(`ws-${idPrefix}-stop`)?.focus());
      return;
    }
    if (!run.asking) onclose();
  }
</script>

{#snippet content()}
  {#if problem}
    <div class="hp-lane" role="status" data-testid="{idPrefix}-problem">
      {#if problem.icon}<Icon name="sparkle" size={14} />{/if}
      <p>{problem.text}</p>
    </div>
    {#if problem.fill && onfill}
      <button type="button" id="ws-{idPrefix}-fill" class="ui-btn ui-btn-primary hp-fill" disabled={unavailable !== null} onclick={() => { onclose(); onfill(); }}><Icon name="sparkle" size={13} />Fill in from Describe it</button>
      {#if unavailable}<p class="ui-help hp-reason" data-testid="{idPrefix}-fill-reason">{unavailable}</p>{/if}
    {/if}
  {:else}
    {@render body()}
  {/if}
{/snippet}

<!-- The Enhancer choice and the run button stay in view while the options scroll. -->
{#snippet foot()}
  <div class="hp-foot">
    {#if contract.smart}
      <EnhanceChoice label="Enhancer" options={TIER_OPTIONS} value={tier} disabled={run.running} idPrefix="ws-{idPrefix}-model" onchange={(value) => draft.update('enhanceModel', value)} />
    {/if}
    <span class="hp-spacer"></span>
    {#if run.running}
      <button type="button" id="ws-{idPrefix}-stop" class="ui-btn ui-btn-danger" onclick={stop}>Stop</button>
    {:else}
      <button type="button" id="ws-{idPrefix}-run" class="ui-btn ui-btn-primary" disabled={!canRun} onclick={submit}>{runLabel}</button>
    {/if}
    {#if staleReason && !run.running}
      <p class="ui-help hp-hint" data-testid="{idPrefix}-stale">{staleReason}</p>
    {/if}
    {#if run.running && run.statusText}
      <p class="ui-help hp-hint" role="status">{run.statusText}</p>
    {:else if run.errorText}
      <p class="ui-help ui-help-error hp-hint" role="alert">{run.errorText}</p>
    {:else if run.noteText}
      <p class="ui-help hp-hint" role="status" data-testid="{idPrefix}-note">{run.noteText}</p>
    {:else if stopNote}
      <p class="ui-help hp-hint">{stopNote}</p>
    {:else if unavailable}
      <p class="ui-help hp-hint">{unavailable}</p>
    {:else if downloadSize}
      <p class="ui-help hp-hint">First use downloads ≈{downloadSize}.</p>
    {:else if fastUndownloaded}
      <p class="ui-help hp-hint">First use downloads the model.</p>
    {/if}
  </div>
{/snippet}

{#if sheet}
  <!-- A phone or a tablet: the helper as a sheet, like every menu there; the text it works on is on top, and the run row is the footer. -->
  <!-- `open` is the panel's alone: the sheet's own Escape, backdrop and drag-down must not hide it while the helper writes. -->
  <Sheet bind:open={() => open, () => {}} {title} footer={problem === null ? foot : undefined} onclose={dismiss}>
    <div id="ws-{idPrefix}-panel" class="hp-sheet">
      {#if quote}
        <div class="hp-quote">
          <p class="ui-area-label">{quote.label}</p>
          <p class="hp-quote-text" data-testid="{idPrefix}-quote">{quote.text.trim()}</p>
        </div>
      {/if}
      {@render content()}
    </div>
  </Sheet>
{:else if open}
  <div use:popover={{ anchor, align: 'start', followViewport: true, onclose: dismiss }} id="ws-{idPrefix}-panel" class="hp-popover ui-overlay" role="dialog" aria-label={title}>
    <div class="hp-body">
      <h2 class="hp-heading">{title}</h2>
      {@render content()}
    </div>
    {#if problem === null}<div class="hp-popover-foot">{@render foot()}</div>{/if}
  </div>
{/if}

<style>
  /* Never taller or wider than the window: on small screens the options scroll inside the popover while the run row stays. */
  .hp-popover { z-index: 70; display: flex; flex-direction: column; width: min(368px, calc(100vw - 16px)); max-height: calc(100dvh - 16px); overflow: hidden; padding: 0; border-radius: var(--radius-md); }
  .hp-body { display: grid; flex: 1 1 auto; gap: 10px; min-height: 0; overflow-y: auto; overscroll-behavior: contain; padding: 12px; }
  .hp-popover-foot { flex: none; border-top: 1px solid var(--color-border-subtle); }
  .hp-heading { margin: 0; font-family: var(--font-heading); font-size: var(--text-title); font-weight: 800; }
  .hp-sheet { display: grid; gap: 10px; padding: 4px max(8px, env(safe-area-inset-right, 0px)) 12px max(8px, env(safe-area-inset-left, 0px)); }
  .hp-quote { display: grid; gap: 2px; padding: 8px 10px; border-radius: var(--radius-md); background: var(--color-bg-surface); }
  .hp-quote p { margin: 0; }
  .hp-quote-text { display: -webkit-box; -webkit-line-clamp: 4; line-clamp: 4; -webkit-box-orient: vertical; overflow: hidden; white-space: pre-line; font-size: var(--text-content); line-height: 1.45; color: var(--color-text-secondary); overflow-wrap: anywhere; }
  .hp-lane { display: flex; align-items: flex-start; gap: 8px; padding: 10px 12px; border-radius: var(--radius-md); background: var(--color-bg-surface); color: var(--color-text-secondary); }
  .hp-lane p { margin: 0; font-size: var(--text-content); line-height: 1.45; }
  .hp-lane :global(svg) { flex: none; margin-top: 2px; color: var(--color-primary-main); }
  .hp-fill { justify-self: start; }
  .hp-reason { margin: 0; }
  .hp-foot { display: flex; flex: 1 1 100%; flex-wrap: wrap; align-items: center; gap: 8px; min-width: 0; }
  .hp-popover-foot .hp-foot { padding: 10px 12px 12px; }
  .hp-hint { flex-basis: 100%; margin: 0; }
  .hp-spacer { flex: 1; }
</style>
