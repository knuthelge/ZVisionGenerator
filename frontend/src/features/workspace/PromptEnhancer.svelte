<script lang="ts" module>
  import { SvelteSet } from 'svelte/reactivity';
  import type { EnhancerTier } from '$lib/types';

  /** Enhancer models downloaded during this session, so their first-use size stops showing. */
  const downloadedTiers = new SvelteSet<EnhancerTier>();
</script>

<script lang="ts">
  import { draft } from '$lib/state/draft.svelte';
  import { enhancePrompt } from '$lib/api/promptEnhance';
  import { extendEnhanceHint, extendGuard, type ExtendModelSupport } from '$lib/state/extend';
  import { extendInspection } from '$lib/state/extendInspection.svelte';
  import { EnhanceChoice, EnhanceOptions, EnhanceRewrite, InspectorRow } from '$lib/components/molecules';
  import { Icon, Toggle } from '$lib/components/atoms';
  import { popover } from '$lib/actions/popover';
  import { layout } from '$lib/state/layoutMode.svelte';
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
  import type { EnhanceFormat, EnhanceParts, EnhanceSettings, PromptEnhancerContract } from '$lib/types';

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
    /** Called when an enhancement starts, so the Enhanced tab can come forward. */
    onstart?: () => void;
    /** Whether the chosen video model can extend a clip; an extension waits for a model that can. */
    extendSupport?: ExtendModelSupport | null;
    /** The clip also ends on a last-frame image, so the rewrite must not invent a different ending. */
    endImage?: boolean;
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
    extendSupport = null,
    maxWords,
    enhancing = $bindable(false),
    streamingText = $bindable(''),
    streamingScene = $bindable(''),
    errorText = $bindable(null),
    noteText = $bindable(null),
    statusText = $bindable(null),
    onstart,
  }: Props = $props();

  const mode = $derived(workflowMode(draft.state.workflow));
  const settings = $derived(effectiveEnhanceSettings(draft.state, contract));
  // A blank Scene means both parts, whatever is remembered; the file variant has no single scene to wait for.
  const rewriteBlocked = $derived(variant === 'inline' && actionOnlyBlocked(draft.state));
  const rewriteParts = $derived<EnhanceParts>(variant === 'inline' ? runParts(draft.state, settings.parts) : (settings.parts ?? 'both'));
  const runSettings = $derived<EnhanceSettings>(mode === 'video' ? { ...settings, parts: rewriteParts } : settings);
  const audioSupported = $derived(contract.video_audio ?? true);
  // An extension of a silent clip is silent whatever the Audio toggle says, so the Sound axis does not apply.
  const sourceHasAudio = $derived(draft.state.workflow === 'extend' ? (draft.extendSource?.has_audio ?? null) : null);
  const audio = $derived(hasAudio(draft.state, audioSupported, sourceHasAudio));
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
  let toggleEl = $state<HTMLButtonElement | null>(null);
  let phase: 'downloading' | 'loading' | 'generating' | 'generating_cpu' | null = null;
  let abort: AbortController | null = null;

  const downloadSize = $derived(downloadedTiers.has(tier) ? null : (tier === 'smart' ? contract.smart?.download_size_label : contract.download_size_label) ?? null);
  const fastUndownloaded = $derived(tier === 'fast' && !contract.downloaded && !downloadedTiers.has('fast'));
  const enhanceDisabled = $derived(busy || jobsActive || enhancing || !available || !canSubmitText(draft.state) || sourceGuard.blocked);
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

  const changed = $derived(!isDefaultEnhanceSettings(settings, contract.matrix.defaults));

  function updateParts(parts: EnhanceParts): void {
    updateSettings(withRememberedParts(settings, draft.state.workflow, parts));
  }

  /** Run the enhancer on the current prompt; also used by the prompt box's Re-enhance link. */
  export async function runEnhance(): Promise<void> {
    if (enhanceDisabled) return;
    onstart?.();
    const source = enhancedSourceFor(draft.state);
    const requested = runSettings;
    const parts = rewriteParts;
    const runTier = tier;
    const runCaption = caption;
    enhancing = true;
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
          mode,
          settings: settingsPayload(requested, mode, audio),
          max_words: maxWords,
          context: enhanceContext(draft.state, audioSupported, sourceHasAudio, endImage),
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
            draft.patch({ enhancedPrompt: prompt, enhancedScene: source.mode === 'video' && parts === 'both' ? frame.scene ?? '' : '', enhancedFrom: record ? { ...source, record } : source });
            const notes = [frame.clamped ? clampedNote(requested.length) : null, frame.warning ?? null].filter((note) => note !== null);
            noteText = notes.length > 0 ? notes.join(' ') : null;
            downloadedTiers.add(runTier);
          } else if (frame.type === 'error') {
            errorText = frame.detail;
          }
        },
        abort.signal,
      );
    } catch (error) {
      if (!abort?.signal.aborted) errorText = error instanceof Error ? error.message : 'Prompt enhancement failed.';
    } finally {
      enhancing = false;
      phase = null;
      statusText = null;
      streamingText = '';
      streamingScene = '';
      abort = null;
    }
  }

  function stopEnhance(): void {
    // A download or model load cannot be interrupted; it finishes in the background and then stops.
    if (phase === 'downloading' || phase === 'loading') {
      noteText = phase === 'downloading'
        ? 'Stopped. The enhancer model keeps downloading in the background; generating is available when it finishes.'
        : 'Stopped. The enhancer finishes loading in the background; generating is available in a moment.';
    }
    abort?.abort();
  }

  // The live state is bound into the prompt box, which outlives this component (JSON toggle,
  // prompt source switch); stop the request and hand back idle state so nothing stays stuck.
  $effect(() => () => {
    abort?.abort();
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
  onclick={() => { panelOpen = !panelOpen; }}
>
  <Icon name="sparkle" size={13} />Enhance{draft.state.enhanceAuto ? ' · each image' : ''}<Icon name="chevdown" size={12} />
</button>

{#snippet panel(titled: boolean)}
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
  <div class="enhance-foot">
    {#if contract.smart}
      <EnhanceChoice label="Enhancer" options={TIER_OPTIONS} value={tier} disabled={enhancing} idPrefix={`ws-enhance-${variant}-model`} onchange={(value) => draft.update('enhanceModel', value)} />
    {/if}
    {#if variant === 'inline' && !draft.state.enhanceAuto}
      <span class="enhance-spacer"></span>
      {#if enhancing}
        <button type="button" class="ui-btn ui-btn-danger" onclick={stopEnhance}>Stop</button>
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
  </div>
  {#if variant === 'inline' && !draft.state.enhanceAuto && !enhancing}
    {#if disabledHint}
      <p class="ui-help enhance-pad">{disabledHint}</p>
    {:else if downloadSize}
      <p class="ui-help enhance-pad">First use downloads ≈{downloadSize}.</p>
    {:else if fastUndownloaded}
      <p class="ui-help enhance-pad">First use downloads the model.</p>
    {/if}
  {/if}
  {#if variant === 'file'}
    <p class="ui-help enhance-pad">Off: entries with their own <span class="font-mono">enhance:</span> setting (✨) are still enhanced.</p>
  {/if}
{/snippet}

{#if layout.compact}
  <!-- A phone: the options as a sheet, like every menu there. -->
  <Sheet bind:open={panelOpen} title="Prompt enhancer">
    <div id="ws-enhance-panel" class="enhance-sheet">{@render panel(false)}</div>
  </Sheet>
{:else if panelOpen}
  <div
    use:popover={{ anchor: toggleEl, align: 'start', onclose: () => { panelOpen = false; } }}
    id="ws-enhance-panel"
    class="enhance-popover ui-overlay"
    role="dialog"
    aria-label="Prompt enhancer"
  >
    {@render panel(true)}
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
  /* Never taller or wider than the window: on small screens the options scroll inside the popover. */
  .enhance-popover { z-index: 70; width: min(368px, calc(100vw - 16px)); max-height: calc(100dvh - 16px); overflow-y: auto; padding: 8px 0 0; border-radius: var(--radius-md); }
  .enhance-sheet { padding: 4px max(4px, env(safe-area-inset-right, 0px)) 12px max(4px, env(safe-area-inset-left, 0px)); }
  .enhance-head { display: flex; align-items: center; gap: 8px; padding: 0 8px 0 12px; }
  .enhance-heading { margin: 0; font-family: var(--font-heading); font-size: var(--text-title); font-weight: 800; }
  .enhance-reset { margin-left: auto; }
  .enhance-pad { padding: 0 12px; }
  .enhance-pad:first-of-type { margin-top: 8px; }
  .enhance-foot { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; padding: 10px 12px 12px; }
  .enhance-spacer { flex: 1; }
</style>
