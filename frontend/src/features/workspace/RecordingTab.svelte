<script lang="ts">
  import { errorText } from '$lib/api/client';
  import { browseMediaUrl } from '$lib/api/browse';
  import { Icon } from '$lib/components/atoms';
  import { ConsentTick, PathBrowser, VoicePlay } from '$lib/components/molecules';
  import { fileName } from '$lib/components/molecules/assetDetails';
  import { autogrow } from '$lib/actions/autogrow';
  import { formatTimestamp, takeAudioFiles, AUDIO_ACCEPT } from '$lib/state/audioTrack';
  import { trackKey, trackLengths } from '$lib/state/audioTrackLength.svelte';
  import { draft } from '$lib/state/draft.svelte';
  import { hasPath } from '$lib/state/imageSlots';
  import { layout } from '$lib/state/layoutMode.svelte';
  import { recordingForm as form } from '$lib/state/recordingForm.svelte';
  import { voiceConsent } from '$lib/state/voiceConsent.svelte';
  import { voicesStore } from '$lib/state/voices.svelte';
  import type { AudioModelOption, SavedVoice } from '$lib/types';
  import { recordingLengthError, VOICE_WORDS } from './audioView';
  import ImageSources from './ImageSources.svelte';

  interface Props {
    /** The open speech model: its voice lengths bound the recording, and it is the model a saved voice is made for. */
    model: AudioModelOption | null;
    /** The recording is the voice for the next takes (Use once): the window closes. */
    onuse: () => void;
    /** The recording is saved as a voice and chosen: the window closes. */
    onsaved: (voice: SavedVoice) => void;
  }

  let { model, onuse, onsaved }: Props = $props();

  let saving = $state(false);
  let serverError = $state<string | null>(null);
  let dragOver = $state(false);
  let fileInput = $state<HTMLInputElement | null>(null);
  let browsing = $state(false);

  const range = $derived(model?.kind === 'speech' ? { min: model.limits.min_voice_seconds, max: model.limits.max_voice_seconds } : null);
  const length = $derived(form.recorded ? trackLengths.lengthOf(trackKey(form.file === null ? form.path : null, form.file)) : null);
  const outOfRange = $derived(range !== null && length !== null && (length < range.min || length > range.max));
  const recordingError = $derived(outOfRange && range !== null && length !== null ? recordingLengthError(length, range) : null);
  const recordingName = $derived(form.file?.name ?? (form.path !== null && hasPath(form.path) ? fileName(form.path) : ''));
  const named = $derived(form.name.trim() !== '');
  const ready = $derived(form.recorded && !outOfRange && form.text.trim() !== '' && (!form.needsTick || form.ticked));
  const canSave = $derived(ready && named && !saving);

  // The recording's length is read once, for the line next to its name and for the length check.
  $effect(() => {
    if (form.recorded) trackLengths.ensure(form.file === null ? form.path : null, form.file);
  });

  function takeFiles(files: File[]): void {
    takeAudioFiles(files, (picked) => { form.setRecording(null, picked); serverError = null; });
  }

  function takePath(path: string): void {
    form.setRecording(path, null);
    serverError = null;
  }

  function onDrop(event: DragEvent): void {
    event.preventDefault();
    dragOver = false;
    takeFiles([...(event.dataTransfer?.files ?? [])]);
  }

  /** Make the recording the voice of the next takes; it stays under the Voice select until it is removed. */
  function useOnce(): void {
    if (!ready) return;
    const tick = form.needsTick;
    draft.setVoice(form.file !== null ? { file: form.file } : { path: form.path! });
    draft.update('voiceText', form.text);
    // The tick given here is the speaker's for this recording; a take that is in a built-in voice needs none.
    if (tick) voiceConsent.ticked = true;
    else voiceConsent.reset(true);
    form.clear();
    onuse();
  }

  async function save(): Promise<void> {
    if (!canSave) return;
    saving = true;
    serverError = null;
    try {
      const saved = await voicesStore.saveRecording(form.formData(model?.id ?? ''));
      draft.setSavedVoice(saved.slug);
      form.clear();
      onsaved(saved);
    } catch (err) {
      serverError = errorText(err, 'Could not save the voice.');
    } finally {
      saving = false;
    }
  }

  /** Enter in Name does what the primary button does. */
  function onNameKeydown(event: KeyboardEvent): void {
    if (event.key !== 'Enter' || event.isComposing) return;
    event.preventDefault();
    if (named) void save();
    else useOnce();
  }
</script>

<div class="rec-tab" data-compact={layout.compact} data-testid="recording-tab">
  <div class="rec-body">
    <div class="rec-field" role="group" aria-labelledby="rec-label">
      <span id="rec-label" class="ui-label">Recording</span>
      {#if form.recorded}
        <!-- svelte-ignore a11y_no_static_element_interactions because dropping a file is a pointer shortcut; the buttons take the same files -->
        <div class="rec-clip" data-dragover={dragOver} ondragover={(e) => { e.preventDefault(); dragOver = true; }} ondragleave={() => { dragOver = false; }} ondrop={onDrop}>
          <VoicePlay src={form.file === null && form.path !== null ? browseMediaUrl('audio_track', form.path) : null} file={form.file} label={recordingName || 'the recording'} class="rec-play" />
          <span class="rec-name" title={recordingName}>{recordingName}</span>
          {#if length !== null}<span class="rec-len" data-testid="recording-length">{formatTimestamp(length)}</span>{/if}
          <button type="button" class="ui-btn ui-btn-row rec-clear" aria-label="Clear the voice recording" title="Clear" onclick={() => form.setRecording(null, null)}><Icon name="close" size={14} /></button>
        </div>
        {#if recordingError}<p class="ui-help ui-help-error" role="alert">{recordingError}</p>{/if}
      {:else if layout.compact}
        <ImageSources kind="audio" browseFrom={form.path} onfiles={takeFiles} onpath={takePath} />
      {:else}
        <!-- svelte-ignore a11y_no_static_element_interactions because dropping a file is a pointer shortcut; the buttons take the same files -->
        <div class="rec-drop" data-dragover={dragOver} ondragover={(e) => { e.preventDefault(); dragOver = true; }} ondragleave={() => { dragOver = false; }} ondrop={onDrop}>
          <span>{dragOver ? 'Drop to use it' : `Drop an audio file${range ? `, ${range.min} to ${range.max} seconds` : ''}`}</span>
          <button type="button" class="ui-btn ui-btn-sm" onclick={() => fileInput?.click()}><Icon name="upload" size={14} />Choose from this device</button>
          <input bind:this={fileInput} type="file" accept={AUDIO_ACCEPT} class="hidden" tabindex="-1" aria-hidden="true" onchange={(e) => { const chosen = [...(e.currentTarget.files ?? [])]; e.currentTarget.value = ''; takeFiles(chosen); }}>
          <button type="button" class="ui-btn ui-btn-sm" onclick={() => { browsing = true; }}><Icon name="folder" size={14} />Browse the server</button>
        </div>
      {/if}
    </div>
    <div class="rec-field rec-words">
      <label class="ui-label" for="rec-text">What it says <span class="voice-required">required</span></label>
      <textarea
        id="rec-text"
        class="ui-field rec-text"
        rows="3"
        aria-required="true"
        aria-invalid={form.recorded && form.text.trim() === '' ? 'true' : undefined}
        placeholder={VOICE_WORDS.placeholder}
        bind:value={form.text}
        use:autogrow={form.text}
      ></textarea>
    </div>
    {#if form.needsTick}
      <ConsentTick id="rec-consent" checked={form.ticked} onchange={(checked) => { form.ticked = checked; }} />
    {/if}
  </div>
  <div class="rec-foot" data-testid="recording-foot">
    <input
      id="rec-name"
      class="ui-field rec-name-field"
      placeholder="Name it to save it"
      maxlength="60"
      autocomplete="off"
      spellcheck="false"
      aria-label="Name, to save it as a voice"
      aria-invalid={serverError ? 'true' : undefined}
      aria-describedby={serverError ? 'rec-name-error' : undefined}
      bind:value={form.name}
      oninput={() => { serverError = null; }}
      onkeydown={onNameKeydown}
    >
    <button type="button" class="ui-btn" class:ui-btn-primary={!named} data-action="use-once" disabled={!ready} onclick={useOnce}>Use once</button>
    <button type="button" class="ui-btn" class:ui-btn-primary={named} data-action="save-voice" disabled={!canSave} aria-busy={saving} onclick={() => void save()}>{saving ? 'Saving…' : 'Save voice'}</button>
    {#if serverError}<p id="rec-name-error" class="ui-help ui-help-error rec-error" role="alert">{serverError}</p>{/if}
  </div>
</div>
<PathBrowser bind:open={browsing} purpose="audio_track" initialPath={form.path} onchoose={takePath} />

<style>
  .rec-tab { display: flex; flex: 1; min-height: 0; flex-direction: column; }
  .rec-body { display: flex; flex: 1; min-height: 0; flex-direction: column; gap: 14px; padding: 12px 16px; overflow-y: auto; overscroll-behavior: contain; }
  .rec-field { display: flex; flex-direction: column; gap: 4px; }
  .rec-words { flex: 1; min-height: 0; }
  .rec-text { flex: 1; min-height: 72px; resize: none; padding: 8px; font-size: var(--text-content); line-height: 1.45; }
  .rec-text[aria-invalid='true'] { box-shadow: inset 0 0 0 1px var(--color-error); }
  .voice-required { margin-left: 6px; font-size: var(--text-meta); font-weight: 400; text-transform: none; letter-spacing: 0; color: var(--color-text-muted); }
  .rec-clip { display: flex; align-items: center; gap: 10px; min-height: 40px; padding: 4px 10px 4px 6px; border-radius: var(--radius-sm); background: var(--color-bg-surface); }
  .rec-clip[data-dragover='true'], .rec-drop[data-dragover='true'] { outline: 2px dashed var(--color-primary-main); outline-offset: -2px; }
  .rec-clip :global(.rec-play) { width: 28px; height: 28px; background: var(--color-bg-surface-hover); }
  .rec-name { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: var(--text-ui); color: var(--color-text-primary); }
  .rec-len { flex: none; font-family: var(--font-mono); font-size: var(--text-meta); color: var(--color-text-muted); }
  .rec-drop { display: flex; align-items: center; gap: 8px; min-height: 44px; padding: 6px 8px 6px 12px; border: 1px dashed var(--color-border-strong); border-radius: var(--radius-sm); font-size: var(--text-ui); color: var(--color-text-muted); }
  .rec-drop span { flex: 1; min-width: 0; }
  .rec-foot { display: flex; flex: none; flex-wrap: wrap; align-items: center; gap: 8px; padding: 10px 16px; border-top: 1px solid var(--color-border-subtle); }
  .rec-name-field { flex: 1 1 180px; min-width: 0; }
  .rec-error { flex: 1 1 100%; margin: 0; }
  @media (pointer: coarse) {
    .rec-clip :global(.rec-play), .rec-clip .rec-clear { width: 40px; height: 40px; }
    .rec-drop { min-height: 56px; }
    .rec-text { font-size: 16px; }
  }
  /* A phone: 44 px targets, the name on its own line and the two buttons under it. */
  .rec-tab[data-compact='true'] .rec-body { padding: 12px; }
  .rec-tab[data-compact='true'] .rec-clip { min-height: 56px; }
  .rec-tab[data-compact='true'] .rec-clip :global(.rec-play), .rec-tab[data-compact='true'] .rec-clip .rec-clear { width: 44px; height: 44px; }
  .rec-tab[data-compact='true'] .rec-name { font-size: 15px; }
  .rec-tab[data-compact='true'] .rec-len { font-size: 13px; }
  .rec-tab[data-compact='true'] .rec-foot { padding: 10px 12px; }
  .rec-tab[data-compact='true'] .rec-name-field { flex-basis: 100%; height: 44px; font-size: 16px; }
  .rec-tab[data-compact='true'] .rec-foot .ui-btn { flex: 1 1 0; height: 44px; font-size: 15px; }
</style>
