<script lang="ts">
  import { tick } from 'svelte';
  import { draft } from '$lib/state/draft.svelte';
  import { enhancePrompt } from '$lib/api/promptEnhance';
  import { confirmEnhancerDownload as confirmDownload, downloadedTiers, tierMissing } from '$lib/state/enhancerDownload.svelte';
  import { extendEnhanceHint, extendGuard, type ExtendModelSupport } from '$lib/state/extend';
  import { extendInspection } from '$lib/state/extendInspection.svelte';
  import { rewriteResultPatch } from '$lib/state/rewriteHistory';
  import { EnhanceChoice, EnhanceOptions, EnhanceRewrite, InspectorRow } from '$lib/components/molecules';
  import { Icon, Toggle } from '$lib/components/atoms';
  import { popover } from '$lib/actions/popover';
  import { layout } from '$lib/state/layoutMode.svelte';
  import { textWriting } from '$lib/state/textWriting.svelte';
  import Sheet from '$lib/components/molecules/Sheet.svelte';
  import {
    clampedNote,
    effectiveEnhanceSettings,
    effectiveFormat,
    effectiveTier,
    enhanceContext,
    enhanceSettingsKey,
    isDefaultEnhanceSettings,
    DEFAULT_ENHANCE_SETTINGS,
    enhancedSourceFor,
    enhancePhaseMessage,
    canSubmitText,
    hasAudio,
    actionOnlyBlocked,
    isJsonCaption,
    prettyCaption,
    runParts,
    settingsPayload,
    withRememberedParts,
    workflowMode,
  } from '$lib/state/promptEnhance';
  import type { EnhanceFormat, EnhanceParts, EnhanceSettings, EnhancerTier, PromptEnhancerContract } from '$lib/types';

  interface Props {
    contract: PromptEnhancerContract;
    /** `inline`: options popover with an Enhance prompt button. `file`: options and the each-image toggle only. */
    variant: 'inline' | 'file';
    /** A job is being submitted. */
    busy: boolean;
    /** A job is running or queued; the enhancer would need its memory. */
    jobsActive?: boolean;
    /** Word ceiling for the current model (FLUX.1 reads fewer tokens). */
    maxWords: number;
    /** Live state the prompt box shows in its Enhanced tab. */
    enhancing?: boolean;
    /** The prompt part of the stream (the whole text for images). */
    streamingText?: string;
    /** The scene part of a video stream, as split by the server. */
    streamingScene?: string;
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
    /** Whether the chosen video model can extend a clip; an extension waits for a model that can. */
    extendSupport?: ExtendModelSupport | null;
    /** The clip also ends on a last-frame image, so the rewrite must not invent a different ending. */
    endImage?: boolean;
    /** How many reference images are set (uploads included); the rewrite keeps the prompt's "image 1", "image 2". */
    references?: number;
    /** An audio track is in effect: it is the clip's sound, so the rewrite has no Sound options. */
    track?: boolean;
    /** The image model takes JSON captions, so the popover offers Write as: Text or JSON. */
    jsonCapable?: boolean;
  }

  let {
    contract,
    variant,
    busy,
    jobsActive = false,
    jsonCapable = false,
    endImage = false,
    references = 0,
    track = false,
    extendSupport = null,
    maxWords,
    enhancing = $bindable(false),
    streamingText = $bindable(''),
    streamingScene = $bindable(''),
    errorText = $bindable(null),
    noteText = $bindable(null),
    statusText = $bindable(null),
    stale = false,
    staleReason = null,
    onstart,
    onfail,
  }: Props = $props();

  const mode = $derived(workflowMode(draft.state.workflow));
  /** What the server is asked to write: this enhancer serves the Image and Video tabs; audio text has its own request. */
  const requestMode = $derived<'image' | 'video'>(mode === 'video' ? 'video' : 'image');
  const settings = $derived(effectiveEnhanceSettings(draft.state, contract));
  // A blank Scene means both parts, whatever is remembered; the file variant has no single scene to wait for.
  const rewriteBlocked = $derived(variant === 'inline' && actionOnlyBlocked(draft.state));
  const rewriteParts = $derived<EnhanceParts>(variant === 'inline' ? runParts(draft.state, settings.parts) : (settings.parts ?? 'both'));
  const runSettings = $derived<EnhanceSettings>(mode === 'video' ? { ...settings, parts: rewriteParts } : settings);
  const audioSupported = $derived(contract.video_audio ?? true);
  // An extension of a silent clip is silent whatever the Audio toggle says, so the Sound axis does not apply.
  const sourceHasAudio = $derived(draft.state.workflow === 'extend' ? (draft.extendSource?.has_audio ?? null) : null);
  const audio = $derived(hasAudio(draft.state, audioSupported, sourceHasAudio, track));
  // An extension needs a source it can read before any text can be written for it.
  const sourceGuard = $derived(extendGuard(draft.state, draft.extendSource, extendInspection.statusFor(draft.state.extendSourcePath), extendSupport));
  const tier = $derived(effectiveTier(draft.state, contract));
  const format = $derived(effectiveFormat(draft.state, jsonCapable));
  const caption = $derived(format === 'json');
  const available = $derived(contract.model !== null);
  const TIER_OPTIONS = $derived([
    { value: 'fast' as const, label: 'Fast', help: 'The default model.' },
    { value: 'smart' as const, label: 'Smart', help: 'Follows your options and wording more closely.', unavailable: contract.smart?.unavailable ?? null },
  ]);
  const FORMAT_OPTIONS: { value: EnhanceFormat; label: string; help: string }[] = [
    { value: 'text', label: 'Text', help: 'A rewritten prompt.' },
    { value: 'json', label: 'JSON', help: 'A full Ideogram 4 JSON caption (subjects, lettering, colors).' },
  ];

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
  const enhanceDisabled = $derived(busy || jobsActive || enhancing || asking || !available || !canSubmitText(draft.state) || sourceGuard.blocked);
  const disabledHint = $derived(
    !available ? (contract.error ?? 'No prompt enhancer model is configured.')
      : jobsActive ? 'Available when all jobs have finished.'
      : busy ? 'Available once the job is submitted.'
      : sourceGuard.blocked ? extendEnhanceHint(sourceGuard)
      : !canSubmitText(draft.state) ? (mode === 'video' ? 'Write a scene or an action first.' : 'Write a prompt first.')
      : null
  );

  /** Remember *next* for the current mode; the Rewrite choice of the current workflow lives in `partsByWorkflow` alone. */
  function updateSettings(next: EnhanceSettings): void {
    const { parts: _derived, ...stored } = next;
    draft.update(enhanceSettingsKey(draft.state.workflow), stored);
  }

  /** Put every option of the current mode back to its default; the remembered Rewrite choices stay. */
  function resetSettings(): void {
    const remembered = settings.partsByWorkflow;
    const { parts: _serverDefault, ...defaults } = { ...DEFAULT_ENHANCE_SETTINGS, ...contract.matrix.defaults };
    draft.update(enhanceSettingsKey(draft.state.workflow), remembered ? { ...defaults, partsByWorkflow: remembered } : null);
  }

  // The run row is empty (auto-enhance on, no Smart choice) for the inline variant; no footer is drawn then.
  const hasFoot = $derived(Boolean(contract.smart) || variant === 'file' || (variant === 'inline' && !draft.state.enhanceAuto));

  const changed = $derived(!isDefaultEnhanceSettings(settings, contract.matrix.defaults));

  function updateParts(parts: EnhanceParts): void {
    updateSettings(withRememberedParts(settings, draft.state.workflow, parts));
  }

  /** Ask before the enhancement downloads its model; true when nothing is missing or the user agrees. Says why in the note when the plan can't be read. */
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

  /** Run the enhancer on the current prompt; also used by the Enhance shortcut. */
  export async function runEnhance(): Promise<void> {
    if (enhanceDisabled) return;
    errorText = null;
    if (!(await confirmEnhancerDownload(tier))) return;
    // The box shows the rewrite as it streams in, so the popover or sheet gets out of its way.
    if (panelOpen) closePanel();
    onstart?.();
    const source = enhancedSourceFor(draft.state);
    const requested = runSettings;
    const runTier = tier;
    const runCaption = caption;
    enhancing = true;
    release = textWriting.begin('Enhancing');
    let failed = false;
    streamingText = '';
    streamingScene = '';
    errorText = null;
    noteText = null;
    statusText = enhancePhaseMessage('generating', null, runCaption);
    abort = new AbortController();
    try {
      await enhancePrompt(
        {
          prompt: source.prompt,
          ...(mode === 'video' ? { scene: source.scene } : {}),
          mode: requestMode,
          settings: settingsPayload(requested, mode, audio),
          max_words: maxWords,
          context: enhanceContext(draft.state, audioSupported, sourceHasAudio, endImage, references, track),
          ...(runTier === 'smart' ? { enhance_model: 'smart' as const } : {}),
          ...(runCaption ? { format: 'json' as const } : {}),
        },
        (frame) => {
          if (frame.type === 'status') {
            phase = frame.phase;
            statusText = enhancePhaseMessage(frame.phase, downloadSize, runCaption);
          } else if (frame.type === 'text') {
            // Video frames carry the server's split; images stream their whole text into the prompt.
            streamingText = frame.prompt ?? frame.text;
            streamingScene = frame.scene ?? '';
          } else if (frame.type === 'done') {
            // A caption arrives on one line; it is laid out for reading, and Generate sends it as the JSON prompt.
            const prompt = runCaption ? prettyCaption(frame.prompt) : frame.prompt;
            const record = frame.model ? { settings: settingsPayload(requested, mode, audio), model: frame.model, format: runCaption ? ('json' as const) : ('text' as const) } : undefined;
            const scene = source.mode === 'video' ? frame.scene ?? '' : '';
            const from = record ? { ...source, record } : source;
            draft.patch(rewriteResultPatch(draft.state, { prompt, scene, from }));
            const notes = [frame.clamped ? clampedNote(requested.length) : null, frame.warning ?? null].filter((note) => note !== null);
            noteText = notes.length > 0 ? notes.join(' ') : null;
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
      streamingScene = '';
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

  // The live state is bound into the prompt box, which outlives this component (JSON toggle,
  // prompt source switch); stop the request and hand back idle state so nothing stays stuck.
  $effect(() => () => {
    abort?.abort();
    release?.();
    release = null;
    enhancing = false;
    streamingText = '';
    streamingScene = '';
    statusText = null;
  });
</script>

{#snippet outputRows()}
  {#if mode === 'video'}
    <InspectorRow label="Rewrite">
      <EnhanceRewrite value={rewriteParts} actionOnlyBlocked={rewriteBlocked} disabled={enhancing} showLabel={false} full onchange={updateParts} />
    </InspectorRow>
  {:else if jsonCapable}
    <InspectorRow label="Write as">
      <EnhanceChoice label="Write as" options={FORMAT_OPTIONS} value={format} disabled={enhancing} idPrefix={`ws-enhance-${variant}-format`} showLabel={false} full onchange={(value) => draft.update('enhanceFormat', value)} />
    </InspectorRow>
  {/if}
{/snippet}

<button
  type="button"
  id={variant === 'inline' ? 'ws-enhance-toggle' : 'ws-enhance-toggle-file'}
  bind:this={toggleEl}
  class="ui-btn ui-btn-sm"
  aria-haspopup="dialog"
  aria-expanded={panelOpen}
  aria-controls="ws-enhance-panel"
  aria-label={stale ? 'Enhance (out of date)' : undefined}
  onclick={() => { panelOpen = !panelOpen; }}
>
  {#if stale}<Icon name="alert" size={13} class="ui-icon-warn" />{:else}<Icon name="sparkle" size={13} />{/if}Enhance{draft.state.enhanceAuto ? ' · each image' : ''}<Icon name="chevdown" size={12} />
</button>

{#snippet body(titled: boolean)}
  <div class="enhance-head">
    {#if titled}<h2 class="enhance-heading">Enhance</h2>{/if}
    <button type="button" class="enhance-reset ui-btn ui-btn-sm ui-btn-quiet" disabled={!changed || enhancing} onclick={resetSettings}>Reset</button>
  </div>
  <EnhanceOptions axes={contract.matrix.axes} {settings} {mode} {audio} disabled={enhancing} idPrefix={`ws-enhance-${variant}`} output={outputRows} onchange={updateSettings} />
  {#if contract.smart === null && contract.model}
    <p class="ui-help enhance-pad" title="Set on the Config page">Enhancer: <span class="font-mono break-all">{contract.model}</span></p>
  {/if}
  <div class="enhance-pad">
    <Toggle
      id={`ws-enhance-auto-${variant}`}
      checked={draft.state.enhanceAuto}
      disabled={busy || !available}
      label="Enhance each image when generating"
      onchange={(e) => draft.update('enhanceAuto', (e.currentTarget as HTMLInputElement).checked)}
    />
  </div>
{/snippet}

<!-- The Enhancer choice and the run button stay in view while the options scroll. -->
{#snippet foot()}
  <div class="enhance-foot">
    {#if contract.smart}
      <EnhanceChoice label="Enhancer" options={TIER_OPTIONS} value={tier} disabled={enhancing} idPrefix={`ws-enhance-${variant}-model`} onchange={(value) => draft.update('enhanceModel', value)} />
    {/if}
    {#if variant === 'inline' && !draft.state.enhanceAuto}
      <span class="enhance-spacer"></span>
      {#if enhancing}
        <button type="button" class="ui-btn ui-btn-danger" onclick={stopFromPanel}>Stop</button>
      {:else}
        <button type="button" id="ws-enhance-run" class="ui-btn ui-btn-primary" disabled={enhanceDisabled} onclick={runEnhance}>
          {#if caption}
            {isJsonCaption(draft.state.enhancedPrompt) ? 'Write caption again' : 'Write caption'}
          {:else}
            {draft.state.enhancedPrompt.trim() || draft.state.enhancedScene.trim() ? 'Enhance again' : 'Enhance prompt'}
          {/if}
        </button>
      {/if}
    {/if}
    {#if variant === 'inline' && !draft.state.enhanceAuto && !enhancing && staleReason}
      <p class="ui-help enhance-hint" data-testid="enhance-stale">{staleReason}</p>
    {/if}
    {#if variant === 'inline' && !draft.state.enhanceAuto && !enhancing}
      {#if disabledHint}
        <p class="ui-help enhance-hint">{disabledHint}</p>
      {:else if downloadSize}
        <p class="ui-help enhance-hint">First use downloads ≈{downloadSize}.</p>
      {:else if fastUndownloaded}
        <p class="ui-help enhance-hint">First use downloads the model.</p>
      {/if}
    {/if}
    {#if variant === 'file'}
      <p class="ui-help enhance-hint">Off: entries with their own <span class="font-mono">enhance:</span> setting (✨) are still enhanced.</p>
    {/if}
  </div>
{/snippet}

{#if layout.compact}
  <!-- A phone: the options as a sheet, like every menu there; the run row is the sheet's footer. -->
  <Sheet bind:open={panelOpen} title="Prompt enhancer" footer={hasFoot ? foot : undefined}>
    <div id="ws-enhance-panel" class="enhance-sheet">{@render body(false)}</div>
  </Sheet>
{:else if panelOpen}
  <div
    use:popover={{ anchor: toggleEl, align: 'start', onclose: () => { panelOpen = false; } }}
    id="ws-enhance-panel"
    class="enhance-popover ui-overlay"
    role="dialog"
    aria-label="Prompt enhancer"
  >
    <div class="enhance-body">{@render body(true)}</div>
    {#if hasFoot}
      <div class="enhance-popover-foot">{@render foot()}</div>
    {/if}
  </div>
{/if}

{#if draft.state.enhanceAuto && ((variant === 'inline' && draft.state.promptSource === 'inline') || (variant === 'file' && draft.state.promptSource === 'file'))}
  <input type="hidden" name="enhance_auto" value="true" />
  <input type="hidden" name="enhance_settings" value={JSON.stringify(settingsPayload(runSettings, mode, audio))} />
{/if}
<!-- The enhancer and format apply to the whole job: auto-enhance, or prompt-file entries with their own `enhance:`. -->
{#if (variant === 'inline' && draft.state.promptSource === 'inline' && draft.state.enhanceAuto) || (variant === 'file' && draft.state.promptSource === 'file')}
  {#if tier === 'smart'}<input type="hidden" name="enhance_model" value="smart" />{/if}
  {#if caption}<input type="hidden" name="enhance_format" value="json" />{/if}
{/if}

<style>
  /* Never taller or wider than the window: on small screens the options scroll inside the popover while the run row stays. */
  .enhance-popover { z-index: 70; display: flex; flex-direction: column; width: min(368px, calc(100vw - 16px)); max-height: calc(100dvh - 16px); overflow: hidden; padding: 0; border-radius: var(--radius-md); }
  .enhance-body { flex: 1 1 auto; min-height: 0; overflow-y: auto; overscroll-behavior: contain; padding-top: 8px; }
  .enhance-popover-foot { flex: none; border-top: 1px solid var(--color-border-subtle); }
  .enhance-sheet { padding: 4px max(4px, env(safe-area-inset-right, 0px)) 12px max(4px, env(safe-area-inset-left, 0px)); }
  .enhance-head { display: flex; align-items: center; gap: 8px; padding: 0 8px 0 12px; }
  .enhance-heading { margin: 0; font-family: var(--font-heading); font-size: var(--text-title); font-weight: 800; }
  .enhance-reset { margin-left: auto; }
  .enhance-pad { padding: 0 12px; }
  .enhance-pad:first-of-type { margin-top: 8px; }
  /* In the sheet the footer pads itself; in the popover the row carries its own padding. */
  .enhance-foot { display: flex; flex: 1 1 100%; flex-wrap: wrap; align-items: center; gap: 8px; min-width: 0; }
  .enhance-popover-foot .enhance-foot { padding: 10px 12px 12px; }
  .enhance-hint { flex-basis: 100%; margin: 0; }
  .enhance-spacer { flex: 1; }
</style>
