<script lang="ts">
  import { untrack } from 'svelte';
  import { ApiError, errorText } from '$lib/api/client';
  import { saveVoiceStart } from '$lib/state/assetActions';
  import { formatTimestamp } from '$lib/state/audioTrack';
  import { layout } from '$lib/state/layoutMode.svelte';
  import { addToast } from '$lib/state/toasts.svelte';
  import { voicesStore } from '$lib/state/voices.svelte';
  import type { AudioSettingsVocabulary, GalleryAsset, SavedVoice } from '$lib/types';
  import { autogrow } from '$lib/actions/autogrow';
  import ConsentTick from './ConsentTick.svelte';
  import VoicePlay from './VoicePlay.svelte';
  import Modal from './Modal.svelte';

  interface Props {
    /** The speech take to keep as a voice; the dialog is open while there is one. */
    asset: GalleryAsset | null;
    /** The id of the speech model the voice is made for; null when no speech model here can clone, which turns Save voice off. */
    model: string | null;
    /** The audio settings vocabulary: its Delivery marks steer the delivery and are not said, so they are left out of the words. */
    vocabulary?: AudioSettingsVocabulary;
    /** The dialog was closed: cancelled, dismissed or saved. */
    onclose: () => void;
  }

  let { asset, model, vocabulary, onclose }: Props = $props();

  let name = $state('');
  let text = $state('');
  let ticked = $state(false);
  let consentNeeded = $state(false);
  let saving = $state(false);
  let nameError = $state<string | null>(null);
  let failure = $state<string | null>(null);

  const trimmedName = $derived(name.trim());
  const canSave = $derived(asset !== null && model !== null && trimmedName !== '' && text.trim() !== '' && (!consentNeeded || ticked) && !saving);

  // Each take it opens on starts from its own words, with no name and no tick. Only the take re-runs this: the vocabulary is read untracked,
  // so a refreshed context while the dialog is open leaves what was typed alone.
  $effect(() => {
    if (asset === null) return;
    const take = asset;
    const start = untrack(() => saveVoiceStart(take, vocabulary?.marks.map((mark) => mark.id) ?? []));
    name = '';
    text = start.text;
    consentNeeded = start.consentNeeded;
    ticked = false;
    saving = false;
    nameError = null;
    failure = null;
  });

  function formData(take: GalleryAsset, modelId: string): FormData {
    const form = new FormData();
    form.set('model', modelId);
    form.set('name', trimmedName);
    form.set('voice_path', take.file_path ?? '');
    form.set('voice_text', text);
    if (consentNeeded && ticked) form.set('voice_consent', 'on');
    return form;
  }

  async function save(): Promise<void> {
    if (!canSave || asset === null || model === null) return;
    saving = true;
    nameError = null;
    failure = null;
    try {
      const saved = await voicesStore.saveRecording(formData(asset, model));
      addToast(`Saved voice "${saved.name}"`, 'success');
      onclose();
    } catch (err) {
      const reason = errorText(err, 'Could not save the voice.');
      // A name that is taken belongs under Name; any other refusal is about the take.
      if (err instanceof ApiError && err.status === 409) nameError = reason;
      else failure = reason;
    } finally {
      saving = false;
    }
  }
</script>

<Modal open={asset !== null} title="Save as voice" size="md" elevated {onclose}>
  {#snippet children()}
    {#if asset}
      <form class="sv-body" data-compact={layout.compact} onsubmit={(event) => { event.preventDefault(); void save(); }}>
        <div class="sv-clip">
          <VoicePlay src={asset.url} label={asset.filename} class="sv-play" />
          <span class="sv-name" title={asset.filename}>{asset.filename}</span>
          {#if asset.duration != null}<span class="sv-len" data-testid="save-voice-length">{formatTimestamp(asset.duration)}</span>{/if}
        </div>
        <div class="sv-field">
          <label class="ui-label" for="sv-name">Name</label>
          <input
            id="sv-name"
            class="ui-field"
            placeholder="Name this voice"
            maxlength="60"
            spellcheck="false"
            autocomplete="off"
            aria-invalid={nameError ? 'true' : undefined}
            aria-describedby={nameError ? 'sv-name-error' : undefined}
            bind:value={name}
            oninput={() => { nameError = null; }}
          >
          {#if nameError}<p id="sv-name-error" class="ui-help ui-help-error" role="alert">{nameError}</p>{/if}
        </div>
        <div class="sv-field">
          <label class="ui-label" for="sv-text">What it says</label>
          <textarea id="sv-text" class="ui-field sv-text" rows="3" aria-required="true" bind:value={text} use:autogrow={text}></textarea>
        </div>
        {#if consentNeeded}
          <ConsentTick checked={ticked} onchange={(checked) => { ticked = checked; }} />
        {/if}
        {#if failure}<p class="ui-help ui-help-error" role="alert">{failure}</p>{/if}
      </form>
    {/if}
  {/snippet}
  {#snippet footer()}
    {#if !layout.compact}<button type="button" class="ui-btn" onclick={onclose}>Cancel</button>{/if}
    <button type="button" class="ui-btn ui-btn-primary sv-save" class:wide={layout.compact} data-action="save-voice" disabled={!canSave} aria-busy={saving} onclick={() => void save()}>{saving ? 'Saving…' : 'Save voice'}</button>
  {/snippet}
</Modal>

<style>
  .sv-body { display: flex; flex-direction: column; gap: 12px; }
  .sv-field { display: flex; flex-direction: column; gap: 4px; }
  .sv-clip { display: flex; align-items: center; gap: 10px; min-height: 40px; padding: 4px 10px 4px 6px; border-radius: var(--radius-sm); background: var(--color-bg-surface); }
  .sv-clip :global(.sv-play) { width: 28px; height: 28px; background: var(--color-bg-surface-hover); }
  .sv-name { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: var(--text-ui); color: var(--color-text-primary); }
  .sv-len { flex: none; font-family: var(--font-mono); font-size: var(--text-meta); color: var(--color-text-muted); }
  .sv-text { min-height: 72px; resize: none; padding: 8px; font-size: var(--text-content); line-height: 1.45; }
  @media (pointer: coarse) {
    .sv-clip :global(.sv-play) { width: 40px; height: 40px; }
    .sv-text { font-size: 16px; }
  }
  /* A phone: 44 px targets and one full-width button in the foot. */
  .sv-body[data-compact='true'] .sv-clip { min-height: 56px; }
  .sv-body[data-compact='true'] .sv-clip :global(.sv-play) { width: 44px; height: 44px; }
  .sv-body[data-compact='true'] .sv-name { font-size: 15px; }
  .sv-body[data-compact='true'] .sv-len { font-size: 13px; }
  .sv-body[data-compact='true'] :global(input.ui-field) { height: 44px; font-size: 16px; }
  .sv-save.wide { flex: 1 1 100%; height: 44px; font-size: 15px; }
</style>
