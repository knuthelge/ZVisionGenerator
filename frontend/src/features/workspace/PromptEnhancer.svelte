<script lang="ts">
  import { draft } from '$lib/state/draft.svelte';
  import { enhancePrompt } from '$lib/api/promptEnhance';
  import { EnhanceOptions } from '$lib/components/molecules';
  import { Icon, Toggle } from '$lib/components/atoms';
  import { popover } from '$lib/actions/popover';
  import {
    clampedNote,
    effectiveEnhanceSettings,
    enhancePhaseMessage,
    isNoOpSettings,
    settingsPayload,
    workflowMode,
  } from '$lib/state/promptEnhance';
  import type { EnhanceSettings, PromptEnhancerContract } from '$lib/types';

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
    streamingText?: string;
    errorText?: string | null;
    noteText?: string | null;
    statusText?: string | null;
    /** Called when an enhancement starts, so the Enhanced tab can come forward. */
    onstart?: () => void;
  }

  let {
    contract,
    variant,
    busy,
    jobsActive = false,
    maxWords,
    enhancing = $bindable(false),
    streamingText = $bindable(''),
    errorText = $bindable(null),
    noteText = $bindable(null),
    statusText = $bindable(null),
    onstart,
  }: Props = $props();

  const mode = $derived(workflowMode(draft.state.workflow));
  const settings = $derived(effectiveEnhanceSettings(draft.state, contract));
  const noOp = $derived(isNoOpSettings(settings, mode));
  const available = $derived(contract.model !== null);

  let panelOpen = $state(false);
  let toggleEl = $state<HTMLButtonElement | null>(null);
  let downloadedThisSession = $state(false);
  let phase: 'downloading' | 'loading' | 'generating' | 'generating_cpu' | null = null;
  let abort: AbortController | null = null;

  const downloadSize = $derived(downloadedThisSession ? null : contract.download_size_label);
  const enhanceDisabled = $derived(busy || jobsActive || enhancing || noOp || !available || draft.state.prompt.trim() === '');
  const disabledHint = $derived(
    !available ? (contract.error ?? 'No prompt enhancer model is configured.')
      : jobsActive ? 'Available when all jobs have finished.'
      : busy ? 'Available once the job is submitted.'
      : noOp ? 'Pick a style, a mood, a detail, or a length.'
      : draft.state.prompt.trim() === '' ? 'Write a prompt first.'
      : null
  );

  function updateSettings(next: EnhanceSettings): void {
    draft.update('enhanceSettings', next);
  }

  /** Run the enhancer on the current prompt; also used by the prompt box's Re-enhance link. */
  export async function runEnhance(): Promise<void> {
    if (enhanceDisabled) return;
    onstart?.();
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

  // The live state is bound into the prompt box, which outlives this component (JSON toggle,
  // prompt source switch); stop the request and hand back idle state so nothing stays stuck.
  $effect(() => () => {
    abort?.abort();
    enhancing = false;
    streamingText = '';
    statusText = null;
  });
</script>

{#snippet options()}
  <EnhanceOptions axes={contract.matrix.axes} {settings} {mode} disabled={enhancing} idPrefix={`ws-enhance-${variant}`} onchange={updateSettings} />

  <Toggle
    id={`ws-enhance-auto-${variant}`}
    checked={draft.state.enhanceAuto}
    disabled={busy || !available}
    label="Enhance each image when generating"
    onchange={(e) => draft.update('enhanceAuto', (e.currentTarget as HTMLInputElement).checked)}
  />
{/snippet}

<button
  type="button"
  id={variant === 'inline' ? 'ws-enhance-toggle' : 'ws-enhance-toggle-file'}
  bind:this={toggleEl}
  class="prompt-tool"
  aria-haspopup="dialog"
  aria-expanded={panelOpen}
  aria-controls="ws-enhance-panel"
  onclick={() => { panelOpen = !panelOpen; }}
>
  <Icon name="sparkle" size={13} />Enhance{draft.state.enhanceAuto ? ' · each image' : ''}<Icon name="chevdown" size={12} />
</button>

{#if panelOpen}
  <div
    use:popover={{ anchor: toggleEl, align: 'start', onclose: () => { panelOpen = false; } }}
    id="ws-enhance-panel"
    class="enhance-popover surface-popover space-y-3"
    role="dialog"
    aria-label="Prompt enhancer"
  >
    {@render options()}
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
      <p class="text-xs text-warning" role="alert">Pick a style, a mood, a detail, or a length.</p>
    {/if}
    {#if variant === 'file'}
      <p class="field-hint-label">Off: entries with their own <span class="font-mono">enhance:</span> setting (✨) are still enhanced.</p>
    {/if}
  </div>
{/if}

{#if draft.state.enhanceAuto && ((variant === 'inline' && draft.state.promptSource === 'inline') || (variant === 'file' && draft.state.promptSource === 'file'))}
  <input type="hidden" name="enhance_auto" value="true" />
  <input type="hidden" name="enhance_settings" value={JSON.stringify(settingsPayload(settings, mode))} />
{/if}

<style>
  .enhance-popover { z-index: 70; width: 320px; padding: 12px; border-radius: var(--radius-md); }
</style>
