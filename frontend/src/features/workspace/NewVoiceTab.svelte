<script lang="ts">
  import { ApiError, errorText } from '$lib/api/client';
  import { candidateAudioUrl, planVoiceDesign, submitVoiceDesign, type VoiceDesignBody } from '$lib/api/voices';
  import { sendJobControl } from '$lib/api/workspace';
  import { autogrow } from '$lib/actions/autogrow';
  import { Icon } from '$lib/components/atoms';
  import { ActionMenu, requestConfirm, VoicePlay, type ActionMenuEntry } from '$lib/components/molecules';
  import { formatTimestamp } from '$lib/state/audioTrack';
  import { confirmDownloads } from '$lib/state/confirmDownloads';
  import { draft } from '$lib/state/draft.svelte';
  import { jobStore } from '$lib/state/job.svelte';
  import { isVoiceDesignJob, jobBar, jobPhase, voiceDesignProgress } from '$lib/state/jobView';
  import { layout } from '$lib/state/layoutMode.svelte';
  import { newVoiceForm as form } from '$lib/state/newVoiceForm.svelte';
  import { addToast } from '$lib/state/toasts.svelte';
  import { voicesStore } from '$lib/state/voices.svelte';
  import type { AudioModelOption, AudioSettingsVocabulary, SavedVoice, VoiceCandidate } from '$lib/types';
  import { DESCRIPTION_MAX, defaultPreviewLine, EXAMPLE_DESCRIPTIONS, lineHint, PREVIEW_MAX, runLine } from './newVoice';

  interface Props {
    /** The open speech model: it designs the voices, and its voice lengths bound the preview line. */
    model: AudioModelOption | null;
    /** How fast speech is read, for the preview line's read time; without it the line says nothing about its length. */
    vocabulary: AudioSettingsVocabulary | undefined;
    /** A candidate was saved as a voice and chosen: the window closes. */
    onsaved: (voice: SavedVoice) => void;
  }

  let { model, vocabulary, onsaved }: Props = $props();

  let starting = $state(false);
  let startError = $state<string | null>(null);
  // The job a Stop was pressed for: its button reads Stopping… until the job is gone.
  let stopping = $state<string | null>(null);
  let chosen = $state<string | null>(null);
  let name = $state('');
  let saving = $state(false);
  let saveError = $state<string | null>(null);
  let announced = $state('');
  let unfolded = $state(false);
  let examplesOpen = $state(false);
  let examplesButton = $state<HTMLButtonElement | null>(null);
  let panelEl = $state<HTMLElement | null>(null);

  const range = $derived(model?.kind === 'speech' ? { min: model.limits.min_voice_seconds, max: model.limits.max_voice_seconds } : null);
  const line = $derived(form.lineOr(defaultPreviewLine(draft.state.speechText)));
  const hint = $derived(range === null ? null : lineHint(line, range, vocabulary?.reading));

  // The design job is found by what it is, not by a remembered id, so a reloaded page finds it too.
  const running = $derived(jobStore.current !== null && isVoiceDesignJob(jobStore.current) && jobStore.isRunning ? jobStore.current : null);
  const queued = $derived(running === null ? (jobStore.queuedJobs.find(isVoiceDesignJob) ?? null) : null);
  const ended = $derived(
    running === null && jobStore.current !== null && isVoiceDesignJob(jobStore.current) && (jobStore.current.status === 'failed' || jobStore.current.status === 'cancelled') ? jobStore.current : null,
  );
  const busy = $derived(running !== null || queued !== null);
  const activeId = $derived(running?.job_id ?? queued?.job_id ?? queued?.id ?? null);

  const batches = $derived(voicesStore.design.batches);
  const total = $derived(batches.reduce((sum, batch) => sum + batch.candidates.length, 0));
  const unsaved = $derived(batches.reduce((sum, batch) => sum + batch.candidates.filter((candidate) => candidate.saved_as === null).length, 0));
  const folded = $derived(layout.compact && total > 0 && !unfolded && form.description.trim() !== '');

  const waiting = $derived(queued !== null || (running !== null && (running.status === 'queued' || jobPhase(running) === 'queue')));
  const statusLine = $derived(queued !== null ? 'Waiting in the queue' : running !== null ? runLine(running, total > 0) : '');
  const statusFraction = $derived.by(() => {
    if (running === null || waiting) return 0;
    const bar = jobBar(running);
    return bar.kind === 'segments' ? 0 : (bar.fraction ?? 0);
  });
  const endedDone = $derived(ended === null ? 0 : voiceDesignProgress(ended).done);
  const endedTotal = $derived(ended === null ? 0 : voiceDesignProgress(ended).total);
  const runKey = $derived(busy ? (waiting ? 'waiting' : 'making') : (ended?.status ?? 'idle'));

  const chosenCandidate = $derived(batches.flatMap((batch) => batch.candidates).find((candidate) => candidate.id === chosen && candidate.saveable && candidate.saved_as === null) ?? null);
  const canMake = $derived(model !== null && !busy && !starting && form.description.trim() !== '' && line.trim() !== '');
  const canSave = $derived(chosenCandidate !== null && name.trim() !== '' && !saving);

  const examples = $derived<ActionMenuEntry[]>(
    EXAMPLE_DESCRIPTIONS.map((example, index) => ({ kind: 'item', id: `example-${index}`, label: example.label, description: example.text, onselect: () => { form.description = example.text; } })),
  );

  // The candidates are read when the tab opens; the app reads them again whenever a design job ends.
  $effect(() => {
    void voicesStore.loadDesign();
  });

  // A selection that is gone, or saved from another tab, is not kept.
  $effect(() => {
    if (chosen !== null && chosenCandidate === null) chosen = null;
  });

  // New candidates are announced without moving focus.
  let knownTotal = -1;
  $effect(() => {
    const now = total;
    if (knownTotal >= 0 && now > knownTotal) announced = `${now - knownTotal} voices ready. Choose one.`;
    knownTotal = now;
  });

  // A phone shows the list below the form: a run's state scrolls into view when it starts or ends.
  let lastRun = 'idle';
  $effect(() => {
    const key = runKey;
    if (key !== lastRun && key !== 'idle' && layout.compact) {
      const calm = typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches;
      panelEl?.scrollIntoView?.({ block: 'start', behavior: calm ? 'auto' : 'smooth' });
    }
    lastRun = key;
  });

  /** Ask for 3 voices: approve what has to be downloaded, then queue the job. A declined download queues nothing. */
  async function make(): Promise<void> {
    if (!canMake || model === null) return;
    starting = true;
    startError = null;
    const body: VoiceDesignBody = { model: model.id, description: form.description.trim(), text: line.trim() };
    try {
      if (!(await confirmDownloads(await planVoiceDesign(body)))) return;
      jobStore.jobSubmitted(await submitVoiceDesign(body));
    } catch (err) {
      startError = errorText(err, 'Could not start making voices.');
    } finally {
      starting = false;
    }
  }

  /** Stop the running job, or take the queued one off the queue. */
  async function stop(): Promise<void> {
    const id = activeId;
    if (id === null) return;
    stopping = id;
    try {
      if (running !== null) await sendJobControl(id, 'quit');
      else await jobStore.removeQueued(id);
    } catch (err) {
      stopping = null;
      addToast(errorText(err, 'Could not stop making voices.'), 'error');
    }
  }

  /** Save the chosen candidate as a voice, choose it for the next takes and close the window. */
  async function save(): Promise<void> {
    if (chosenCandidate === null || !canSave) return;
    saving = true;
    saveError = null;
    try {
      const made = await voicesStore.saveCandidate(chosenCandidate.id, name.trim());
      draft.setSavedVoice(made.slug);
      name = '';
      chosen = null;
      onsaved(made);
    } catch (err) {
      if (err instanceof ApiError && err.status === 404) {
        // The kept voices changed under this page (another tab cleared them): read them again.
        chosen = null;
        await voicesStore.loadDesign();
        saveError = 'That voice is no longer kept. The list is up to date; choose again.';
      } else {
        saveError = errorText(err, 'Could not save the voice.');
      }
    } finally {
      saving = false;
    }
  }

  /** Empty the list after asking; the server refuses while a design job is queued or running. */
  async function clear(): Promise<void> {
    const info = unsaved === 0 ? 'The list is emptied; the voices you saved stay.' : unsaved === 1 ? 'The candidate you have not saved will be gone.' : `The ${unsaved} candidates you have not saved will be gone.`;
    if (!(await requestConfirm({ question: 'Clear voices?', info, confirmLabel: 'Clear voices', danger: true }))) return;
    try {
      await voicesStore.clearCandidates();
      saveError = null;
    } catch (err) {
      addToast(errorText(err, 'Could not clear the voices.'), 'error');
    }
  }

  function onNameKeydown(event: KeyboardEvent): void {
    if (event.key !== 'Enter' || event.isComposing) return;
    event.preventDefault();
    void save();
  }

  function onChoose(candidate: VoiceCandidate): void {
    chosen = candidate.id;
    saveError = null;
  }
</script>

{#snippet makeButton()}
  <div class="ui-action-bar nv-make">
    <button type="button" class="ui-btn ui-btn-main" class:ui-btn-primary={total === 0} data-action="make" disabled={!canMake} aria-busy={starting} onclick={() => void make()}>
      <Icon name="sparkle" size={14} />{total > 0 ? 'Make 3 more' : 'Make 3'}
    </button>
  </div>
  {#if startError}<p class="ui-help ui-help-error nv-start-error" role="alert" data-testid="start-error">{startError}</p>{/if}
{/snippet}

<div class="nv-tab" data-mode={layout.mode} data-compact={layout.compact} data-testid="new-voice-tab">
  <div class="nv-body">
    <div class="nv-grid">
      <div class="nv-col nv-form">
        {#if folded}
          <button type="button" class="nv-fold" aria-expanded="false" aria-label="Edit the description" data-testid="description-fold" onclick={() => { unfolded = true; }}>
            <span>{form.description}</span><Icon name="chevdown" size={14} />
          </button>
        {:else}
          <div class="nv-field">
            <label class="ui-label" for="nv-desc">Describe the voice <span class="nv-required">required</span></label>
            <div class="ui-prompt-box nv-box">
              <textarea
                id="nv-desc"
                rows="2"
                maxlength={DESCRIPTION_MAX}
                aria-required="true"
                aria-describedby="nv-desc-help"
                placeholder="A warm woman in her thirties, calm and clear, speaking slowly"
                bind:value={form.description}
                use:autogrow={form.description}
              ></textarea>
              <div class="ui-prompt-tools">
                <button
                  bind:this={examplesButton}
                  type="button"
                  id="nv-examples"
                  class="ui-btn ui-btn-sm"
                  aria-haspopup="menu"
                  aria-expanded={examplesOpen}
                  onclick={() => { examplesOpen = !examplesOpen; }}
                >Examples<Icon name="chevdown" size={12} /></button>
                <span class="nv-count" data-testid="description-count">{form.description.length} / {DESCRIPTION_MAX}</span>
              </div>
            </div>
            <p id="nv-desc-help" class="ui-help">Say how it should sound as well: warm, slow, whispered, excited.</p>
          </div>
          <div class="nv-field">
            <label class="ui-label" for="nv-line">Preview line <span class="nv-required">required</span></label>
            <div class="ui-prompt-box nv-box">
              <textarea
                id="nv-line"
                rows="2"
                maxlength={PREVIEW_MAX}
                aria-required="true"
                aria-describedby="nv-line-meta nv-line-help"
                value={line}
                oninput={(event) => form.setLine(event.currentTarget.value)}
                use:autogrow={line}
              ></textarea>
              <div class="ui-prompt-tools" id="nv-line-meta">
                <span></span>
                {#if hint}<span class="nv-meta" class:warn={hint.warning !== null} data-testid="line-read">{#if hint.warning}<Icon name="alert" size={12} />{/if}{hint.read}</span>{/if}
              </div>
            </div>
            {#if hint?.warning}
              <p id="nv-line-help" class="ui-help ui-help-warning" data-testid="line-warning">{hint.warning}</p>
            {:else}
              <p id="nv-line-help" class="ui-help">Each voice reads this line, and it’s saved with the voice.</p>
            {/if}
          </div>
        {/if}
        {@render makeButton()}
      </div>

      <div class="nv-col">
        <section class="nv-result" aria-label="Candidates" bind:this={panelEl} data-testid="candidates-panel">
          <div class="nv-head">
            <h3 class="ui-area-label">Candidates</h3>
            {#if total > 0}
              <span class="nv-count" data-testid="candidate-count">{total}</span>
              <button type="button" class="ui-btn ui-btn-quiet ui-btn-sm" data-action="clear-voices" disabled={busy} onclick={() => void clear()}>Clear voices</button>
            {/if}
          </div>

          {#if busy}
            <div class="nv-status" data-solo={total === 0} role="status" aria-live="polite" data-testid="design-status">
              <div class="nv-status-top">
                <p class="nv-status-line">{#if waiting}<Icon name="clock" size={14} />{/if}{statusLine}</p>
                <button type="button" class="ui-btn ui-btn-sm" data-action="stop" disabled={stopping === activeId} onclick={() => void stop()}>
                  <Icon name="stop" size={14} />{stopping === activeId ? 'Stopping…' : 'Stop'}
                </button>
              </div>
              <div class="ui-progress"><span class="ui-progress-fill" style:width="{statusFraction * 100}%"></span></div>
              {#if total === 0}
                <p class="ui-help nv-status-help">{waiting ? 'It starts when the job ahead ends. You can close this; it keeps its place.' : 'You can close this; the job keeps running.'}</p>
              {/if}
            </div>
          {:else if ended !== null && ended.status === 'failed'}
            <div class="ui-alert ui-alert-error" role="alert" data-testid="design-failed"><Icon name="alert" size={16} /><span>{ended.message || 'Making the voices failed.'}</span></div>
          {:else if ended !== null}
            <p class="ui-help nv-note" data-testid="design-stopped">{endedDone > 0 ? `Stopped after ${endedDone} of ${endedTotal}. These are the voices it finished.` : 'Stopped before the first voice was ready.'}</p>
          {/if}

          {#if total > 0}
            <div class="cand-scroll" role="radiogroup" aria-label="Choose a voice" data-testid="candidate-list">
              {#each batches as batch (batch.batch_id)}
                {#if batches.length > 1}<div class="cand-batch" title={batch.description} data-testid="candidate-batch">{batch.description}</div>{/if}
                {#each batch.candidates as candidate (candidate.id)}
                  {@const label = `Voice ${candidate.number}`}
                  {@const taken = candidate.saved_as !== null}
                  {@const off = !candidate.saveable && !taken}
                  <div class="cand" data-testid="candidate" data-id={candidate.id} data-selected={chosen === candidate.id} data-saved={taken} aria-disabled={off ? 'true' : undefined}>
                    <VoicePlay src={candidateAudioUrl(candidate.id)} {label} class="cand-play" />
                    <label class="cand-pick">
                      <span class="cand-name">{label}</span>
                      <span class="cand-len">{formatTimestamp(Math.round(candidate.seconds))}</span>
                      {#if taken}
                        <span class="cand-saved"><Icon name="check" size={12} />Saved as {candidate.saved_as}</span>
                      {:else}
                        <input
                          type="radio"
                          class="ui-checkbox voice-radio"
                          name="candidate"
                          value={candidate.id}
                          checked={chosen === candidate.id}
                          disabled={off}
                          aria-label="Choose {label}"
                          aria-describedby={off ? `cand-reason-${candidate.id}` : undefined}
                          onchange={() => onChoose(candidate)}
                        >
                      {/if}
                    </label>
                    {#if off && candidate.reason}<p id="cand-reason-{candidate.id}" class="cand-reason" data-testid="candidate-reason">{candidate.reason}</p>{/if}
                  </div>
                {/each}
              {/each}
            </div>
          {:else if !busy && ended === null}
            <div class="nv-empty"><Icon name="waveform" size={20} /><span>Voices to compare appear here, 3 at a time.</span></div>
          {/if}
        </section>
      </div>
    </div>
  </div>

  {#if total > 0}
    <div class="nv-foot" data-testid="save-bar">
      <input
        id="nv-name"
        class="ui-field nv-name"
        placeholder="Name the chosen voice"
        maxlength="60"
        autocomplete="off"
        spellcheck="false"
        aria-label="Name"
        aria-invalid={saveError ? 'true' : undefined}
        aria-describedby={saveError ? 'nv-name-error' : undefined}
        bind:value={name}
        oninput={() => { saveError = null; }}
        onkeydown={onNameKeydown}
      >
      <button type="button" class="ui-btn ui-btn-primary" data-action="save-voice" disabled={!canSave} aria-busy={saving} onclick={() => void save()}>{saving ? 'Saving…' : 'Save voice'}</button>
      {#if saveError}<p id="nv-name-error" class="ui-help ui-help-error nv-save-error" role="alert">{saveError}</p>{/if}
    </div>
  {/if}
  <p class="sr-only" role="status" aria-live="polite">{announced}</p>
</div>
<ActionMenu open={examplesOpen} anchor={examplesButton} items={examples} label="Example descriptions" align="start" onclose={() => { examplesOpen = false; }} />

<style>
  .nv-tab { display: flex; flex: 1; min-height: 0; flex-direction: column; }
  .nv-body { display: flex; flex: 1; min-height: 0; flex-direction: column; padding: 12px 16px; overflow: hidden; }
  .nv-grid { display: grid; flex: 1; min-height: 0; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); grid-template-rows: minmax(0, 1fr); gap: 16px; align-items: stretch; }
  .nv-col { display: flex; min-width: 0; min-height: 0; flex-direction: column; gap: 8px; }
  .nv-form { overflow-y: auto; overscroll-behavior: contain; }
  .nv-field { display: flex; flex-direction: column; gap: 4px; }
  .nv-required { margin-left: 6px; font-size: var(--text-meta); font-weight: 400; color: var(--color-text-muted); }
  .nv-box { display: flex; flex-direction: column; }
  .nv-box textarea { display: block; width: 100%; padding: 8px 10px 6px; border: 0; background: transparent; color: var(--color-text-primary); font-family: var(--font-sans); font-size: var(--text-content); line-height: 1.5; resize: none; outline: none; min-height: 3.6em; max-height: 9em; }
  .nv-box textarea::placeholder { color: var(--color-zinc-600); }
  .nv-box .ui-prompt-tools { justify-content: space-between; }
  .nv-count { font-family: var(--font-mono); font-size: var(--text-meta); color: var(--color-text-muted); }
  .nv-meta { padding: 0 2px; font-size: var(--text-meta); color: var(--color-text-muted); }
  .nv-meta.warn { display: inline-flex; align-items: center; gap: 4px; color: var(--color-warning); }
  .nv-make :global(.ui-btn) { width: 100%; }
  .nv-start-error { margin: 0; }
  .nv-fold { display: flex; align-items: center; gap: 8px; min-height: 44px; padding: 0 10px; border-radius: var(--radius-sm); background: var(--color-bg-surface-hover); text-align: left; font-size: 14px; color: var(--color-text-secondary); }
  .nv-fold span { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

  .nv-result { display: flex; flex: 1; min-height: 0; flex-direction: column; gap: 8px; padding: 10px 12px 12px; border-radius: var(--radius-md); background: var(--color-bg-surface); }
  .nv-head { display: flex; align-items: center; gap: 8px; min-height: var(--spacing-control-sm); }
  .nv-head :global(.ui-area-label) { margin: 0; }
  .nv-head .ui-btn { margin-left: auto; }
  .nv-head .nv-count { margin-left: -2px; }
  .nv-empty { display: flex; flex: 1; flex-direction: column; align-items: center; justify-content: center; gap: 6px; padding: 24px 12px; text-align: center; font-size: var(--text-ui); color: var(--color-text-muted); }
  .nv-status { display: flex; flex-direction: column; gap: 8px; }
  .nv-status[data-solo='true'] { flex: 1; justify-content: center; }
  .nv-status-line { display: flex; align-items: center; gap: 8px; margin: 0; font-size: var(--text-content); font-weight: 600; color: var(--color-text-primary); }
  .nv-status-top { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
  .nv-status .ui-progress { height: 6px; }
  .nv-status-help, .nv-note { margin: 0; }

  .cand-scroll { display: flex; flex: 1; min-height: 0; flex-direction: column; gap: 3px; overflow-y: auto; overscroll-behavior: contain; }
  .cand-scroll > :global(*) { flex: none; }
  .cand-batch { overflow: hidden; margin: 4px 0 1px; padding: 0 2px; font-size: var(--text-meta); color: var(--color-text-muted); white-space: nowrap; text-overflow: ellipsis; }
  .cand-batch:first-child { margin-top: 0; }
  .cand { display: grid; grid-template-columns: auto minmax(0, 1fr); column-gap: 6px; align-items: center; padding: 3px 8px 3px 3px; border-radius: var(--radius-sm); background: var(--color-bg-surface-hover); }
  .cand[data-selected='true'] { background: color-mix(in srgb, var(--color-primary-main) 14%, var(--color-bg-surface-hover)); box-shadow: inset 0 0 0 1px var(--color-primary-border); }
  .cand[data-saved='true'] { background: transparent; }
  .cand :global(.cand-play) { background: var(--color-bg-control-hover); }
  .cand-pick { display: flex; min-height: 24px; align-items: center; gap: 8px; cursor: pointer; }
  .cand-name { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: var(--text-ui); font-weight: 600; color: var(--color-text-primary); }
  .cand-len { flex: none; font-family: var(--font-mono); font-size: var(--text-meta); color: var(--color-text-secondary); }
  .cand-reason { grid-column: 2; margin: 0 0 3px; font-size: var(--text-meta); line-height: 1.4; color: var(--color-warning); }
  .cand[aria-disabled='true'] .cand-pick { cursor: not-allowed; }
  .cand[aria-disabled='true'] .cand-name { color: var(--color-text-muted); }
  .cand-saved { display: inline-flex; flex: none; align-items: center; gap: 4px; font-size: var(--text-meta); color: var(--color-success); }
  .voice-radio { border-radius: 50%; }
  .voice-radio:checked { background-image: radial-gradient(circle at center, var(--color-primary-ink) 0 34%, transparent 36%); }

  /* Save bar: the name and Save voice are always in the foot, so a long list never pushes them away. */
  .nv-foot { display: flex; flex: none; flex-wrap: wrap; align-items: center; gap: 8px; padding: 10px 16px; border-top: 1px solid var(--color-border-subtle); }
  .nv-name { flex: 1 1 200px; min-width: 0; }
  .nv-save-error { flex: 1 1 100%; margin: 0; }

  /* A tablet: 40 px rows. */
  .nv-tab[data-mode='tall'] .cand :global(.cand-play) { width: 40px; height: 40px; }
  .nv-tab[data-mode='tall'] .cand-pick { min-height: 40px; }
  .nv-tab[data-mode='tall'] .cand-name { font-size: 14px; }
  .nv-tab[data-mode='tall'] .voice-radio { width: 22px; height: 22px; }

  /* A phone: one column that scrolls as a whole, and 44 px targets. */
  .nv-tab[data-compact='true'] .nv-body { padding: 12px; overflow-y: auto; overscroll-behavior: contain; }
  .nv-tab[data-compact='true'] .nv-grid { flex: none; grid-template-columns: minmax(0, 1fr); grid-template-rows: none; }
  .nv-tab[data-compact='true'] .nv-form { overflow: visible; }
  .nv-tab[data-compact='true'] .nv-result { flex: none; }
  .nv-tab[data-compact='true'] .cand-scroll { flex: none; overflow: visible; }
  .nv-tab[data-compact='true'] .nv-make :global(.ui-btn) { height: 44px; font-size: 15px; }
  .nv-tab[data-compact='true'] .nv-box textarea { font-size: 16px; }
  .nv-tab[data-compact='true'] .nv-box .ui-prompt-tools .ui-btn { height: 44px; padding: 0 10px; font-size: 14px; }
  .nv-tab[data-compact='true'] .nv-head .ui-btn, .nv-tab[data-compact='true'] .nv-status .ui-btn { height: 44px; }
  .nv-tab[data-compact='true'] .cand { padding: 6px 12px 6px 6px; }
  .nv-tab[data-compact='true'] .cand :global(.cand-play) { width: 44px; height: 44px; }
  .nv-tab[data-compact='true'] .cand-pick { min-height: 44px; }
  .nv-tab[data-compact='true'] .cand-name { font-size: 15px; }
  .nv-tab[data-compact='true'] .cand-len { font-size: 13px; }
  .nv-tab[data-compact='true'] .cand-reason { font-size: 12px; }
  .nv-tab[data-compact='true'] .voice-radio { width: 22px; height: 22px; }
  .nv-tab[data-compact='true'] .nv-foot { padding: 10px 12px; }
  .nv-tab[data-compact='true'] .nv-name { height: 44px; font-size: 16px; }
  .nv-tab[data-compact='true'] .nv-foot .ui-btn { height: 44px; font-size: 15px; }
</style>
