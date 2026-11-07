<script lang="ts">
  import { tick } from 'svelte';
  import { enhancePrompt } from '$lib/api/promptEnhance';
  import { Icon } from '$lib/components/atoms';
  import { EnhanceChoice, EnhanceOptions } from '$lib/components/molecules';
  import Sheet from '$lib/components/molecules/Sheet.svelte';
  import { popover } from '$lib/actions/popover';
  import { audioEnhanceContext, audioEnhanceRecord, audioSettingsPayload, axesOf, effectiveAudioSettings, enhanceBlock, enhanceTaskOf, isDefaultAudioSettings, taskSings } from '$lib/state/audioEnhance';
  import { pushRewrite } from '$lib/state/audioRewrites';
  import { draft } from '$lib/state/draft.svelte';
  import { confirmEnhancerDownload as confirmDownload, downloadedTiers, tierMissing } from '$lib/state/enhancerDownload.svelte';
  import { layout } from '$lib/state/layoutMode.svelte';
  import { effectiveTier, enhancePhaseMessage } from '$lib/state/promptEnhance';
  import { textWriting } from '$lib/state/textWriting.svelte';
  import type { AudioEnhanceSettings, AudioEnhanceTask, AudioSettingsVocabulary, EnhancerTier, PromptEnhancerContract } from '$lib/types';
  import type { AudioView } from './audioView';

  interface Props {
    contract: PromptEnhancerContract;
    /** The Audio tab's view of the open kind: the box, the model and the text the rewrite is made from. */
    view: AudioView;
    /** The vocabulary of the take's settings; a Sound's type and length go to the enhancer as context. */
    vocabulary?: AudioSettingsVocabulary;
    /** A job is being submitted. */
    busy: boolean;
    /** A take is running or queued; the enhancer would need its memory. */
    jobsActive?: boolean;
    /** Live state the box shows in its Enhanced tab. */
    enhancing?: boolean;
    streamingText?: string;
    errorText?: string | null;
    noteText?: string | null;
    statusText?: string | null;
    /** The Enhanced text is out of date: the button's icon is ⚠ and the panel opens on "Enhance again". */
    stale?: boolean;
    /** Why it is out of date, in a sentence; the panel says it beside "Enhance again". */
    staleReason?: string | null;
    /** Called when an enhancement starts, so the Enhanced tab can come forward. */
    onstart?: () => void;
    /** Called when a run ended in an error, so the box can show your own text again instead of an empty Enhanced tab. */
    onfail?: () => void;
  }

  let {
    contract,
    view,
    vocabulary,
    busy,
    jobsActive = false,
    enhancing = $bindable(false),
    streamingText = $bindable(''),
    errorText = $bindable(null),
    noteText = $bindable(null),
    statusText = $bindable(null),
    stale = false,
    staleReason = null,
    onstart,
    onfail,
  }: Props = $props();

  /** The field of a `text` or `done` frame each task writes. */
  const TASK_FIELDS: Record<AudioEnhanceTask, string> = { sound: 'prompt', style: 'style' };

  const task = $derived(view.kind === null ? null : enhanceTaskOf(view.kind));
  const matrix = $derived(contract.audio_matrix);
  /** The take has a singing voice; a style for a song that is instrumental, or for a model that sings nothing, has no Singer. */
  const singing = $derived(task === null || taskSings(draft.state, task, view));
  const axes = $derived(task === null ? [] : axesOf(matrix, task, singing));
  const settings = $derived(task === null ? matrix.defaults : effectiveAudioSettings(draft.state, task, matrix));
  const changed = $derived(task !== null && !isDefaultAudioSettings(settings, task, matrix, singing));
  const tier = $derived(effectiveTier(draft.state, contract));
  const ownText = $derived(view.text);
  /** What the box holds, in the words of its buttons: a sound's "prompt", a song's "style". */
  const noun = $derived(task === 'style' ? 'style' : 'prompt');
  const Noun = $derived(noun === 'style' ? 'Style' : 'Prompt');
  /** Phones and tablets open the options as a sheet; only the wide layout has room for a popover beside the box. */
  const sheet = $derived(layout.mode !== 'wide');

  const TIER_OPTIONS = $derived([
    { value: 'fast' as const, label: 'Fast', help: 'The default model.' },
    { value: 'smart' as const, label: 'Smart', help: 'Follows your options and wording more closely.', unavailable: contract.smart?.unavailable ?? null },
  ]);

  let panelOpen = $state(false);
  /** The download plan of the tier about to run is being read, or the dialog about it is open. */
  let asking = $state(false);
  let toggleEl = $state<HTMLButtonElement | null>(null);
  let phase: 'downloading' | 'loading' | 'generating' | 'generating_cpu' | null = null;
  let abort: AbortController | null = null;
  /** Ends this run's hold on Generate (see `textWriting`). */
  let release: (() => void) | null = null;

  const downloadSize = $derived(downloadedTiers.has(tier) ? null : (tier === 'smart' ? contract.smart?.download_size_label : contract.download_size_label) ?? null);
  const fastUndownloaded = $derived(tier === 'fast' && !contract.downloaded && !downloadedTiers.has('fast'));
  /** Why Enhance cannot run now, or null: shown in the popover. */
  const disabledHint = $derived(task === null ? null : enhanceBlock(contract, task, { jobsActive, busy, text: ownText }));
  const enhanceDisabled = $derived(disabledHint !== null || enhancing || asking);

  function updateSettings(next: AudioEnhanceSettings): void {
    if (task === null) return;
    draft.update('enhanceSettingsAudio', { ...draft.state.enhanceSettingsAudio, [task]: next });
  }

  /** Put every option of the task back to its default. */
  function resetSettings(): void {
    if (task === null) return;
    draft.update('enhanceSettingsAudio', { ...draft.state.enhanceSettingsAudio, [task]: null });
  }

  /** Ask before the enhancement downloads its model; true when nothing is missing or the user agrees. */
  async function confirmEnhancerDownload(runTier: EnhancerTier): Promise<boolean> {
    if (!tierMissing(contract, runTier)) return true;
    asking = true;
    try {
      const { confirmed, error } = await confirmDownload(contract, runTier);
      if (error !== null) errorText = error;
      return confirmed;
    } finally {
      asking = false;
    }
  }

  /** Close the panel and put the focus on the Enhance button: the control that had it is closing with the panel. */
  function closePanel(): void {
    panelOpen = false;
    void tick().then(() => toggleEl?.focus());
  }

  /** Run the enhancer on the box's text; also used by the box's Re-enhance link and the Enhance shortcut. */
  export async function runEnhance(): Promise<void> {
    if (enhanceDisabled || task === null || view.kind === null || view.box === null || view.kind === 'speech') return;
    errorText = null;
    if (!(await confirmEnhancerDownload(tier))) return;
    // The box shows the rewrite as it streams in, so the popover or sheet gets out of its way.
    if (panelOpen) closePanel();
    onstart?.();
    const box = view.box;
    const kind = view.kind;
    const input = ownText;
    const model = draft.state.model;
    const runTier = tier;
    const payload = audioSettingsPayload(settings, task, matrix, singing);
    const field = TASK_FIELDS[task];
    enhancing = true;
    release = textWriting.begin('Enhancing');
    let failed = false;
    streamingText = '';
    noteText = null;
    statusText = enhancePhaseMessage('generating', null);
    abort = new AbortController();
    try {
      await enhancePrompt(
        {
          mode: 'audio',
          task,
          model,
          kind,
          prompt: input,
          settings: payload,
          context: audioEnhanceContext(draft.state, task, vocabulary, view),
          ...(runTier === 'smart' ? { enhance_model: 'smart' as const } : {}),
        },
        (frame) => {
          if (frame.type === 'status') {
            phase = frame.phase;
            statusText = enhancePhaseMessage(frame.phase, downloadSize);
          } else if (frame.type === 'text') {
            streamingText = frame.fields[field] ?? frame.text;
          } else if (frame.type === 'done') {
            const text = frame.fields[field] ?? '';
            const record = frame.model ? audioEnhanceRecord(task, frame.model, payload) : undefined;
            const from = { input, model, task, ...(record ? { record } : {}) };
            draft.update('audioRewrites', { ...draft.state.audioRewrites, [box]: pushRewrite(draft.state.audioRewrites[box], { text, from }) });
            noteText = frame.warning ?? null;
            downloadedTiers.add(runTier);
          } else if (frame.type === 'error') {
            failed = true;
            errorText = frame.detail;
          }
        },
        abort.signal,
      );
    } catch (error) {
      if (!abort?.signal.aborted) {
        failed = true;
        errorText = error instanceof Error ? error.message : 'Prompt enhancement failed.';
      }
    } finally {
      enhancing = false;
      release?.();
      release = null;
      phase = null;
      statusText = null;
      streamingText = '';
      abort = null;
    }
    if (failed) onfail?.();
  }

  /** Stop the running enhancement; the header's stop ring and the panel's Stop both call it. */
  export function stop(): void {
    // A download or model load cannot be interrupted; it finishes in the background and then stops.
    if (phase === 'downloading' || phase === 'loading') {
      noteText = phase === 'downloading'
        ? 'Stopped. The enhancer model keeps downloading in the background; generating is available when it finishes.'
        : 'Stopped. The enhancer finishes loading in the background; generating is available in a moment.';
    }
    abort?.abort();
  }

  /** The panel's Stop: stop, and close the panel with the focus back on the Enhance button (Stop is gone with the run). */
  function stopFromPanel(): void {
    stop();
    closePanel();
  }

  // An out-of-date text opens the panel on its fix: the run button, which then reads "Enhance again".
  $effect(() => {
    if (!panelOpen || !stale) return;
    void tick().then(() => document.getElementById('ws-enhance-run')?.focus());
  });

  // The live state is bound into the box, which outlives this component (a kind switch); stop the request and hand back idle state.
  $effect(() => () => {
    abort?.abort();
    release?.();
    release = null;
    enhancing = false;
    streamingText = '';
    statusText = null;
  });
</script>

<button
  type="button"
  id="ws-enhance-toggle"
  bind:this={toggleEl}
  class="ui-btn ui-btn-sm"
  aria-haspopup="dialog"
  aria-expanded={panelOpen}
  aria-controls="ws-enhance-panel"
  aria-label={stale ? 'Enhance (out of date)' : undefined}
  onclick={() => { panelOpen = !panelOpen; }}
>
  {#if stale}<Icon name="alert" size={13} class="ui-icon-warn" />{:else}<Icon name="sparkle" size={13} />{/if}Enhance<Icon name="chevdown" size={12} />
</button>

{#snippet body(titled: boolean)}
  <div class="enhance-head">
    {#if titled}<h2 class="enhance-heading">Enhance</h2>{/if}
    <button type="button" class="enhance-reset ui-btn ui-btn-sm ui-btn-quiet" disabled={!changed || enhancing} onclick={resetSettings}>Reset</button>
  </div>
  <EnhanceOptions {axes} {settings} mode="audio" disabled={enhancing} idPrefix="ws-enhance-audio" onchange={updateSettings} />
  {#if contract.smart === null && contract.model}
    <p class="ui-help enhance-pad" title="Set on the Config page">Enhancer: <span class="font-mono break-all">{contract.model}</span></p>
  {/if}
{/snippet}

<!-- The Enhancer choice and the run button stay in view while the options scroll. -->
{#snippet foot()}
  <div class="enhance-foot">
    {#if contract.smart}
      <EnhanceChoice label="Enhancer" options={TIER_OPTIONS} value={tier} disabled={enhancing} idPrefix="ws-enhance-audio-model" onchange={(value) => draft.update('enhanceModel', value)} />
    {/if}
    <span class="enhance-spacer"></span>
    {#if enhancing}
      <button type="button" class="ui-btn ui-btn-danger" onclick={stopFromPanel}>Stop</button>
    {:else}
      <button type="button" id="ws-enhance-run" class="ui-btn ui-btn-primary" disabled={enhanceDisabled} onclick={runEnhance}>
        {view.rewrites.text.trim() ? 'Enhance again' : `Enhance ${noun}`}
      </button>
    {/if}
    {#if !enhancing && staleReason}
      <p class="ui-help enhance-hint" data-testid="enhance-stale">{staleReason}</p>
    {/if}
    {#if !enhancing}
      {#if disabledHint}
        <p class="ui-help enhance-hint">{disabledHint}</p>
      {:else if downloadSize}
        <p class="ui-help enhance-hint">First use downloads ≈{downloadSize}.</p>
      {:else if fastUndownloaded}
        <p class="ui-help enhance-hint">First use downloads the model.</p>
      {/if}
    {/if}
  </div>
{/snippet}

{#if sheet}
  <!-- A phone or a tablet: the options as a sheet, like every menu there; the box's text on top shows what is rewritten, and the run row is the footer. -->
  <Sheet bind:open={panelOpen} title={`Enhance ${noun}`} footer={foot}>
    <div id="ws-enhance-panel" class="enhance-sheet">
      <div class="enhance-quote">
        <p class="ui-area-label">{Noun}</p>
        <p class="enhance-quote-text" data-testid="enhance-quote">{ownText.trim() || 'Nothing written yet.'}</p>
      </div>
      {@render body(false)}
    </div>
  </Sheet>
{:else if panelOpen}
  <div
    use:popover={{ anchor: toggleEl, align: 'start', onclose: () => { panelOpen = false; } }}
    id="ws-enhance-panel"
    class="enhance-popover ui-overlay"
    role="dialog"
    aria-label="{Noun} enhancer"
  >
    <div class="enhance-body">{@render body(true)}</div>
    <div class="enhance-popover-foot">{@render foot()}</div>
  </div>
{/if}

<style>
  /* Never taller or wider than the window: on small screens the options scroll inside the popover while the run row stays. */
  .enhance-popover { z-index: 70; display: flex; flex-direction: column; width: min(368px, calc(100vw - 16px)); max-height: calc(100dvh - 16px); overflow: hidden; padding: 0; border-radius: var(--radius-md); }
  .enhance-body { flex: 1 1 auto; min-height: 0; overflow-y: auto; overscroll-behavior: contain; padding-top: 8px; }
  .enhance-popover-foot { flex: none; border-top: 1px solid var(--color-border-subtle); }
  .enhance-sheet { padding: 4px max(4px, env(safe-area-inset-right, 0px)) 12px max(4px, env(safe-area-inset-left, 0px)); }
  .enhance-quote { display: grid; gap: 2px; margin: 0 8px 8px; padding: 8px 10px; border-radius: var(--radius-md); background: var(--color-bg-surface); }
  .enhance-quote p { margin: 0; }
  .enhance-quote-text { display: -webkit-box; -webkit-line-clamp: 3; line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; font-size: var(--text-content); line-height: 1.45; color: var(--color-text-secondary); overflow-wrap: anywhere; }
  .enhance-head { display: flex; align-items: center; gap: 8px; padding: 0 8px 0 12px; }
  .enhance-heading { margin: 0; font-family: var(--font-heading); font-size: var(--text-title); font-weight: 800; }
  .enhance-reset { margin-left: auto; }
  .enhance-pad { padding: 0 12px; }
  .enhance-foot { display: flex; flex: 1 1 100%; flex-wrap: wrap; align-items: center; gap: 8px; min-width: 0; }
  .enhance-popover-foot .enhance-foot { padding: 10px 12px 12px; }
  .enhance-hint { flex-basis: 100%; margin: 0; }
  .enhance-spacer { flex: 1; }
</style>
