<script lang="ts">
  import { draft } from '$lib/state/draft.svelte';
  import { enhancePrompt } from '$lib/api/promptEnhance';
  import { EnhanceOptions } from '$lib/components/molecules';
  import { Toggle } from '$lib/components/atoms';
  import {
    clampedNote,
    effectiveEnhanceSettings,
    enhancePhaseMessage,
    isEnhancedStale,
    isNoOpSettings,
    settingsPayload,
    workflowMode,
  } from '$lib/state/promptEnhance';
  import type { EnhanceSettings, PromptEnhancerContract } from '$lib/types';

  interface Props {
    contract: PromptEnhancerContract;
    /** `inline`: Enhance button, options, Enhanced box, and auto toggle. `file`: auto toggle and options only. */
    variant: 'inline' | 'file';
    /** A job is running or being submitted. */
    busy: boolean;
    /** Word ceiling for the current model (FLUX.1 reads fewer tokens). */
    maxWords: number;
  }

  let { contract, variant, busy, maxWords }: Props = $props();

  const mode = $derived(workflowMode(draft.state.workflow));
  const settings = $derived(effectiveEnhanceSettings(draft.state, contract));
  const noOp = $derived(isNoOpSettings(settings, mode));
  const stale = $derived(isEnhancedStale(draft.state));
  const available = $derived(contract.model !== null);

  let panelOpen = $state(false);
  let enhancing = $state(false);
  let streamingText = $state('');
  let statusText = $state<string | null>(null);
  let errorText = $state<string | null>(null);
  let noteText = $state<string | null>(null);
  let downloadedThisSession = $state(false);
  let phase: 'downloading' | 'loading' | 'generating' | 'generating_cpu' | null = null;
  let abort: AbortController | null = null;

  const downloadSize = $derived(downloadedThisSession ? null : contract.download_size_label);
  const enhanceDisabled = $derived(busy || enhancing || noOp || !available || draft.state.prompt.trim() === '');
  const disabledHint = $derived(
    !available ? (contract.error ?? 'No prompt enhancer model is configured.')
      : busy ? 'Available when the current job finishes.'
      : noOp ? 'Pick a style, a detail, or a length.'
      : draft.state.prompt.trim() === '' ? 'Write a prompt first.'
      : null
  );

  function updateSettings(next: EnhanceSettings): void {
    draft.update('enhanceSettings', next);
  }

  async function runEnhance(): Promise<void> {
    if (enhanceDisabled) return;
    const source = { prompt: draft.state.prompt, mode };
    const requested = settings;
    enhancing = true;
    streamingText = '';
    errorText = null;
    noteText = null;
    statusText = enhancePhaseMessage('generating', null);
    abort = new AbortController();
    try {
      await enhancePrompt(
        { prompt: source.prompt, mode, settings: settingsPayload(requested, mode), max_words: maxWords },
        (frame) => {
          if (frame.type === 'status') {
            phase = frame.phase;
            statusText = enhancePhaseMessage(frame.phase, downloadSize);
          } else if (frame.type === 'text') {
            streamingText = frame.text;
          } else if (frame.type === 'done') {
            draft.update('enhancedPrompt', frame.prompt);
            draft.update('enhancedFrom', source);
            noteText = frame.clamped ? clampedNote(requested.length) : null;
            downloadedThisSession = true;
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

  function clearEnhanced(): void {
    draft.update('enhancedPrompt', '');
    draft.update('enhancedFrom', null);
    noteText = null;
  }

  function onEnhancedInput(value: string): void {
    draft.update('enhancedPrompt', value);
    // Hand-typed text (no source) is never "out of date"; edits to a generated rewrite keep its source.
    if (value.trim() === '') draft.update('enhancedFrom', null);
  }
</script>

{#if variant === 'inline'}
  <div class="mb-3 flex items-center justify-between gap-2">
    <button
      type="button"
      id="ws-enhance-toggle"
      class="surface-link-muted inline-flex items-center gap-1 text-xs font-medium transition"
      aria-expanded={panelOpen}
      aria-controls="ws-enhance-panel"
      onclick={() => { panelOpen = !panelOpen; }}
    >
      <span aria-hidden="true">✨</span> Enhance{draft.state.enhanceAuto ? ' · each image' : ''}
      <svg class="h-3 w-3 transition-transform" class:rotate-180={panelOpen} fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path>
      </svg>
    </button>
    {#if enhancing}
      <span class="text-xs text-text-muted" role="status">{statusText}</span>
    {/if}
  </div>
{/if}

{#if variant === 'file' || panelOpen}
  <div id="ws-enhance-panel" class="surface-card-muted mb-3 space-y-3 rounded-md p-3">
    {#if variant === 'file'}
      <p class="field-label">Prompt enhancer</p>
    {/if}
    <EnhanceOptions axes={contract.matrix.axes} {settings} {mode} disabled={enhancing} idPrefix={`ws-enhance-${variant}`} onchange={updateSettings} />

    <Toggle
      id={`ws-enhance-auto-${variant}`}
      checked={draft.state.enhanceAuto}
      disabled={busy || !available}
      label="Enhance each image when generating"
      onchange={(e) => draft.update('enhanceAuto', (e.currentTarget as HTMLInputElement).checked)}
    />
    {#if variant === 'file'}
      <p class="field-hint-label">Off: entries with their own <span class="font-mono">enhance:</span> setting (✨) are still enhanced.</p>
    {/if}

    {#if variant === 'inline' && !draft.state.enhanceAuto}
      <div class="flex flex-wrap items-center gap-2">
        {#if enhancing}
          <button type="button" class="surface-button rounded-full px-3 py-1.5 text-xs" onclick={stopEnhance}>Stop</button>
        {:else}
          <button
            type="button"
            id="ws-enhance-run"
            class="surface-button surface-button-primary rounded-full px-3 py-1.5 text-xs"
            disabled={enhanceDisabled}
            onclick={runEnhance}
          >
            {draft.state.enhancedPrompt.trim() ? 'Enhance again' : 'Enhance prompt'}
          </button>
        {/if}
        {#if disabledHint && !enhancing}
          <span class="text-xs text-text-muted">{disabledHint}</span>
        {:else if downloadSize && !enhancing}
          <span class="text-xs text-text-muted">First use downloads ≈{downloadSize}.</span>
        {:else if !contract.downloaded && !downloadedThisSession && !enhancing}
          <span class="text-xs text-text-muted">First use downloads the model.</span>
        {/if}
      </div>
    {:else if draft.state.enhanceAuto && noOp}
      <p class="text-xs text-warning" role="alert">Pick a style, a detail, or a length.</p>
    {/if}
  </div>
{/if}

{#if variant === 'inline'}
  {#if errorText}
    <p class="mb-2 text-xs text-error" role="alert">{errorText}</p>
  {/if}
  <div class="mb-3">
    <div class="mb-2 flex flex-wrap items-center justify-between gap-x-2 gap-y-1">
      <label class="field-label block whitespace-nowrap" for="ws-enhanced-prompt">Enhanced prompt</label>
      <div class="flex flex-wrap items-center gap-x-2 gap-y-1 whitespace-nowrap">
        {#if stale && !draft.state.enhanceAuto}
          <span class="enhance-stale-badge" title="The prompt or workflow changed after this was enhanced. It is still used until you clear it.">Out of date</span>
          <button type="button" class="surface-link-muted text-[11px] font-medium transition" disabled={enhanceDisabled} onclick={runEnhance}>Re-enhance</button>
        {/if}
        {#if draft.state.enhancedPrompt && !draft.state.enhanceAuto}
          <button type="button" class="surface-link-muted text-[11px] font-medium transition" onclick={clearEnhanced}>Clear</button>
        {/if}
      </div>
    </div>
    <textarea
      id="ws-enhanced-prompt"
      rows="4"
      class="surface-textarea w-full rounded-md shadow-sm transition placeholder-zinc-600 focus:border-primary-main focus:ring-4 focus:ring-primary-main"
      placeholder={draft.state.enhanceAuto ? 'Generated per image when the job runs.' : draft.state.prompt || 'Enhance the prompt, or type a variant here. When this has text, it is what gets generated.'}
      disabled={draft.state.enhanceAuto}
      readonly={enhancing}
      aria-busy={enhancing}
      value={enhancing ? streamingText : (draft.state.enhanceAuto ? '' : draft.state.enhancedPrompt)}
      oninput={(e) => onEnhancedInput((e.currentTarget as HTMLTextAreaElement).value)}
    ></textarea>
    {#if noteText}
      <p class="field-hint-label mt-1">{noteText}</p>
    {:else if draft.state.enhancedPrompt.trim() && !draft.state.enhanceAuto}
      <p class="field-hint-label mt-1">This text is generated instead of the prompt above.</p>
    {/if}
  </div>
{/if}

{#if draft.state.enhanceAuto && ((variant === 'inline' && draft.state.promptSource === 'inline') || (variant === 'file' && draft.state.promptSource === 'file'))}
  <input type="hidden" name="enhance_auto" value="true" />
  <input type="hidden" name="enhance_settings" value={JSON.stringify(settingsPayload(settings, mode))} />
{/if}

<style>
  .enhance-stale-badge {
    border: 1px solid var(--color-warning-border);
    background: var(--color-warning-surface);
    color: var(--color-warning);
    border-radius: 9999px;
    padding: 0 8px;
    font-size: 11px;
    line-height: 18px;
  }
</style>
