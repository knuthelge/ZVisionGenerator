<script lang="ts">
  import { tick } from 'svelte';
  import { autogrow } from '$lib/actions/autogrow';
  import { Icon } from '$lib/components/atoms';
  import { ActionMenu, HelpLine, PromptTools, type ActionMenuEntry } from '$lib/components/molecules';
  import { ClearUndo } from '$lib/state/clearUndo.svelte';
  import { layout } from '$lib/state/layoutMode.svelte';
  import {
    DESCRIBE_MAX_CHARS,
    applyFill,
    canUndoFill,
    describeFolded,
    fillContext,
    fillHint,
    fillRequest,
    fillSummary,
    fillTargets,
    fillable,
    keptFields,
    markedFields,
    undoFill,
  } from '$lib/state/audioFill';
  import { draft } from '$lib/state/draft.svelte';
  import { downloadedTiers } from '$lib/state/enhancerDownload.svelte';
  import { WriterRun, downloadSizeOf, writerBlock } from '$lib/state/writerRun.svelte';
  import type { AudioDescribe, AudioSettingsVocabulary, DescribeKind, FillField, PromptEnhancerContract } from '$lib/types';
  import type { AudioView } from './audioView';

  interface Props {
    /** The Audio tab's view of the open kind (Music or Speech): the model and the controls it shows decide what can be filled. */
    view: AudioView;
    kind: DescribeKind;
    contract: PromptEnhancerContract;
    /** The vocabulary of the take's settings: the Vibes a Fill in can pick, and the Delivery defaults that count as empty. */
    vocabulary?: AudioSettingsVocabulary;
    /** A job is being submitted. */
    busy: boolean;
    /** A take is running or queued; the enhancer would need its memory. */
    jobsActive?: boolean;
    /** A fill wrote these fields (the Style box shows your own text, where a filled Style lands). */
    onfilled?: (wrote: FillField[]) => void;
  }

  let { view, kind, contract, vocabulary, busy, jobsActive = false, onfilled }: Props = $props();

  const PLACEHOLDERS: Record<DescribeKind, string> = {
    music: 'Describe the song: what it is about, how it should sound, who sings…',
    speech: 'Describe it: what it is for, how long, how it should sound…',
  };

  const run = new WriterRun('describeTier');
  /** The Clear of the idea, with its Undo. */
  const cleared = new ClearUndo();
  const UNDO_ID = 'ws-describe-clear-undo';
  let tierButton = $state<HTMLButtonElement | null>(null);
  let tierOpen = $state(false);
  let textarea = $state<HTMLTextAreaElement | null>(null);
  let note = $state<string | null>(null);
  let stopNote = $state<string | null>(null);
  let seconds = $state(0);
  let timer: ReturnType<typeof setInterval> | null = null;

  const s = $derived(draft.state);
  const describe = $derived<AudioDescribe>(s.audioDescribe[kind]);
  const defaults = $derived(vocabulary?.delivery_defaults);
  /** The fields the open row could have filled (Stable Audio has no Lyrics, a recording no Vibe). */
  const requested = $derived(fillable(kind, view.shown, view.voice.recording));
  const targets = $derived(fillTargets(s, kind, requested, defaults));
  const kept = $derived(keptFields(s, kind, requested, defaults));
  const marked = $derived(markedFields(s, kind));
  const folded = $derived(describeFolded(describe, s, kind) && !run.running);
  const unavailable = $derived(writerBlock(contract, { jobsActive, busy }));
  const idea = $derived(describe.text.trim());
  const reason = $derived(unavailable ?? (idea === '' ? 'Describe it first.' : targets.length === 0 ? 'Every field is yours.' : null));
  /** The line under the box: why Fill in cannot run, or what it will write and keep. */
  const hint = $derived(unavailable ?? (idea === '' ? 'Describe it first.' : fillHint(targets, kept)));
  const canRun = $derived(reason === null && !run.running && !run.asking);
  const tier = $derived(run.tier(contract));
  const downloadSize = $derived(downloadSizeOf(contract, tier));
  const fastUndownloaded = $derived(tier === 'fast' && !contract.downloaded && !downloadedTiers.has('fast'));
  const canUndo = $derived(canUndoFill(s, kind));
  /** The line under the box: what a run is doing, what went wrong, or what the fill did. */
  const filledSummary = $derived(marked.length > 0 || canUndo ? fillSummary(marked) : null);

  /** Fill in's enhancer: Fast, or Smart where it is available here. */
  const tierItems = $derived.by((): ActionMenuEntry[] => {
    const smart = contract.smart;
    const smartSize = downloadSizeOf(contract, 'smart');
    return [
      // A sheet carries the name as its title; a popover has only the heading.
      ...(layout.compact ? [] : [{ kind: 'heading', label: 'Enhancer for Fill in' } as const]),
      { kind: 'item', id: 'describe-tier-fast', label: 'Fast', description: 'The default model.', checked: tier === 'fast', onselect: () => draft.update('describeTier', 'fast') },
      {
        kind: 'item',
        id: 'describe-tier-smart',
        label: 'Smart',
        description: smart?.unavailable ?? `Follows your idea more closely.${smartSize ? ` First use downloads ≈${smartSize}.` : ''}`,
        checked: tier === 'smart',
        disabled: Boolean(smart?.unavailable),
        onselect: () => draft.update('describeTier', 'smart'),
      },
    ];
  });

  function patch(next: Partial<AudioDescribe>): void {
    draft.update('audioDescribe', { ...draft.state.audioDescribe, [kind]: { ...draft.state.audioDescribe[kind], ...next } });
  }

  /** Empty the idea; what Fill in wrote stays, with its own Undo fill. Undo puts the idea back. */
  function clearIdea(): void {
    cleared.clear({ audioDescribe: { ...s.audioDescribe, [kind]: { ...describe, text: '' } } }, null);
    // A finger would raise the keyboard over the result: it gets the Undo, a mouse the empty idea (the open line when folded).
    void tick().then(() => document.getElementById(layout.coarse ? UNDO_ID : folded ? 'ws-describe-open' : 'ws-describe')?.focus());
  }

  function undoClear(): void {
    if (cleared.undo() !== null && !layout.coarse) void tick().then(() => document.getElementById(folded ? 'ws-describe-open' : 'ws-describe')?.focus());
  }

  // Another kind has its own box: what the last run said belongs to the one left.
  $effect(() => {
    void kind;
    note = null;
    stopNote = null;
    run.errorText = null;
    cleared.forget();
  });

  // The seconds a run has taken, so a wait on a model that loads is not a silence.
  $effect(() => {
    if (!run.running) {
      if (timer !== null) clearInterval(timer);
      timer = null;
      return;
    }
    seconds = 0;
    timer = setInterval(() => { seconds += 1; }, 1000);
    return () => {
      if (timer !== null) clearInterval(timer);
      timer = null;
    };
  });

  $effect(() => () => run.dispose());

  /** Write the empty fields (and the ones still marked ✦) from the idea. */
  export async function fill(): Promise<void> {
    if (!canRun) return;
    const started = kind;
    const asked = targets;
    const model = draft.state.model;
    note = null;
    stopNote = null;
    cleared.forget();
    const result = await run.start(contract, fillRequest(model, started, describe.text, asked, fillContext(draft.state, started, view.shown)), 'fill');
    if (result === null) return;
    // What the user typed while the helper worked is theirs: only the fields still empty (or still marked) are written.
    const now = draft.state;
    const still = fillTargets(now, started, asked, defaults).filter((field) => asked.includes(field));
    const applied = applyFill(now, started, still, { fields: result.fields, vocabulary });
    if (applied.wrote.length === 0) {
      note = result.warning ?? 'Nothing could be filled from that. Try describing it a little differently.';
      return;
    }
    draft.patch(applied.patch);
    note = result.warning;
    onfilled?.(applied.wrote);
  }

  /** Put the replaced fields back exactly as they were. */
  function undo(): void {
    const next = undoFill(draft.state, kind);
    if (next) draft.patch(next);
    note = null;
  }

  function stop(): void {
    stopNote = run.stop();
  }

  /** Open the box for editing and put the caret in it. */
  export async function open(): Promise<void> {
    patch({ folded: false });
    await tick();
    textarea?.focus();
  }

  function onkeydown(event: KeyboardEvent): void {
    if (event.key === 'Enter' && (event.metaKey || event.ctrlKey) && canRun) {
      event.preventDefault();
      void fill();
    }
  }
</script>

<div class="describe" data-testid="describe-it" data-folded={folded}>
  {#if folded}
    <div class="describe-fold">
      <span class="describe-lab">Describe it</span>
      <span class="describe-line" class:muted={idea === ''} data-testid="describe-line">{idea === '' ? 'Fill what is empty from an idea' : describe.text}</span>
      {#if idea !== ''}
        <button type="button" class="ui-btn ui-btn-sm ui-btn-icon clear-button" aria-label="Clear the idea" title="Clear" data-testid="clear-box" onclick={clearIdea}><Icon name="trash" size={14} /></button>
      {/if}
      <button
        type="button"
        class="ui-btn ui-btn-sm ui-btn-icon ui-btn-quiet"
        id="ws-describe-open"
        aria-label={idea === '' ? 'Open Describe it' : 'Edit the description'}
        aria-controls="ws-describe"
        onclick={open}
      ><Icon name={idea === '' ? 'chevdown' : 'edit'} size={14} /></button>
    </div>
  {:else}
    <div class="ui-prompt-box">
      <div class="describe-head">
        <label for="ws-describe" class="describe-tab">Describe it</label>
        <button
          type="button"
          class="ui-btn ui-btn-sm ui-btn-icon ui-btn-quiet"
          aria-label="Fold Describe it"
          title="Fold"
          aria-controls="ws-describe"
          disabled={run.running}
          onclick={() => patch({ folded: true })}
        ><Icon name="chevup" size={14} /></button>
      </div>
      <textarea
        id="ws-describe"
        class="describe-text"
        placeholder={PLACEHOLDERS[kind]}
        maxlength={DESCRIBE_MAX_CHARS}
        readonly={run.running}
        aria-busy={run.running}
        bind:this={textarea}
        value={describe.text}
        use:autogrow={describe.text}
        {onkeydown}
        oninput={(e) => { cleared.forget(); patch({ text: e.currentTarget.value }); }}
      ></textarea>
      <PromptTools onclear={idea !== '' && !run.running ? clearIdea : undefined} clearWhat="idea">
        {#if run.running}
          <button type="button" class="ui-btn ui-btn-sm ui-btn-danger" id="ws-describe-stop" onclick={stop}>Stop</button>
        {:else}
          <button
            type="button"
            id="ws-describe-fill"
            class="ui-btn ui-btn-sm"
            disabled={!canRun}
            aria-describedby="ws-describe-hint"
            onclick={fill}
          ><Icon name="sparkle" size={13} />{marked.length > 0 ? 'Fill in again' : 'Fill in'}</button>
        {/if}
        {#if contract.smart}
          <button
            bind:this={tierButton}
            type="button"
            id="ws-describe-tier"
            class="ui-btn ui-btn-sm ui-btn-quiet tier-button"
            aria-haspopup="menu"
            aria-expanded={tierOpen}
            aria-label="Enhancer for Fill in: {tier === 'smart' ? 'Smart' : 'Fast'}"
            title="Enhancer for Fill in"
            disabled={run.running}
            onclick={() => { tierOpen = !tierOpen; }}
          >{tier === 'smart' ? 'Smart' : 'Fast'}<Icon name="chevdown" size={12} /></button>
        {/if}
        {#if run.running}
          <span class="describe-time" role="status">{run.statusText ?? 'Filling in…'} · {seconds} s</span>
        {/if}
      </PromptTools>
    </div>
    {#if !run.running && !cleared.offered}
      <p id="ws-describe-hint" class="ui-help describe-hint" data-testid="describe-hint">
        {hint}
        {#if reason === null && downloadSize}First use downloads ≈{downloadSize}.{:else if reason === null && fastUndownloaded}First use downloads the model.{/if}
      </p>
    {/if}
  {/if}
  {#if !run.running && cleared.offered}
    <HelpLine undo={{ what: 'idea', id: UNDO_ID, onundo: undoClear }} />
  {/if}

  {#if run.errorText}
    <p class="ui-help ui-help-error" role="alert" data-testid="describe-error">{run.errorText}</p>
  {:else if stopNote}
    <p class="ui-help" role="status">{stopNote}</p>
  {/if}
  {#if !run.running && filledSummary !== null}
    <p class="describe-sum" role="status" data-testid="describe-summary">
      <Icon name="sparkle" size={12} />
      <span class="describe-sum-text">{filledSummary}</span>
      {#if canUndo}<button type="button" id="ws-describe-undo" class="describe-undo" onclick={undo}>Undo fill</button>{/if}
    </p>
  {/if}
  {#if note && !run.running}
    <p class="ui-help" role="status" data-testid="describe-note">{note}</p>
  {/if}
</div>
{#if contract.smart}
  <ActionMenu open={tierOpen} anchor={tierButton} items={tierItems} label="Enhancer for Fill in" align="start" onclose={() => { tierOpen = false; }} />
{/if}

<style>
  .describe { display: flex; flex-direction: column; gap: 4px; }
  .describe-head { display: flex; align-items: flex-end; gap: 4px; min-height: calc(var(--spacing-control-sm) + 6px); padding: 4px 4px 0 14px; border-bottom: 1px solid var(--color-border-subtle); }
  .describe-tab { position: relative; flex: 1; padding: 5px 0 6px; font-size: var(--text-ui); font-weight: 600; color: var(--color-text-primary); }
  .describe-head :global(.ui-btn) { align-self: center; margin-bottom: 2px; }
  .describe-text { display: block; width: 100%; min-height: 58px; max-height: 140px; overflow-y: auto; resize: none; padding: 9px 10px; border: 0; background: transparent; font-size: var(--text-content); line-height: 1.5; color: var(--color-text-primary); }
  .describe-text:focus, .describe-text:focus-visible { outline: none; }
  .describe-text::placeholder { color: var(--color-zinc-600); }
  /* Fill in's enhancer is the row's quiet button: no fill, secondary text. */
  .tier-button { padding: 0 6px; color: var(--color-text-secondary); }
  .tier-button[aria-expanded='true'] { background: var(--color-bg-surface-hover); color: var(--color-text-primary); }
  .clear-button { color: var(--color-text-muted); }
  .describe-time { font-family: var(--font-mono); font-size: var(--text-meta); color: var(--color-text-muted); }
  .describe-hint { margin: 0; padding: 0 4px; }
  /* Folded: the box is one line, the idea in the middle, the edit button at the end. */
  .describe-fold { display: flex; align-items: center; gap: 8px; min-height: var(--spacing-control); padding: 0 4px 0 12px; border-radius: var(--radius-md); background: var(--color-bg-fill); }
  .describe-lab { flex: none; font-size: var(--text-ui); font-weight: 600; color: var(--color-text-secondary); }
  .describe-line { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: var(--text-content); color: var(--color-text-primary); }
  .describe-line.muted { color: var(--color-text-muted); }
  .describe-sum { display: flex; align-items: center; gap: 6px; margin: 0; padding: 0 4px; min-height: var(--spacing-control-sm); font-size: var(--text-meta); color: var(--color-text-secondary); }
  .describe-sum :global(svg) { flex: none; color: var(--color-primary-main); }
  .describe-sum-text { min-width: 0; overflow-wrap: anywhere; }
  .describe-undo { flex: none; margin-left: auto; font-size: var(--text-meta); font-weight: 600; white-space: nowrap; color: var(--color-text-muted); }
  @media (hover: hover) { .describe-undo:hover { color: var(--color-text-primary); } }
  /* Touch: 16px text, below which iOS zooms into a field as it focuses. */
  @media (pointer: coarse) {
    .describe-text { font-size: 16px; }
    .describe-undo { min-height: 32px; padding: 0 6px; }
  }
</style>
