<script lang="ts">
  import { Icon } from '$lib/components/atoms';
  import { onMount } from 'svelte';
  import { ConsentTick, Select, VoicePlay } from '$lib/components/molecules';
  import { autogrow } from '$lib/actions/autogrow';
  import { formatTimestamp, takeAudioFiles } from '$lib/state/audioTrack';
  import { draft } from '$lib/state/draft.svelte';
  import { voiceConsent } from '$lib/state/voiceConsent.svelte';
  import { voicesStore } from '$lib/state/voices.svelte';
  import type { AudioModelOption } from '$lib/types';
  import { chooseVoice, VOICE_WORDS, type VoiceView } from './audioView';
  import ImageSources from './ImageSources.svelte';
  import VoiceActions from './VoiceActions.svelte';

  interface Props {
    /** The voice of the open Speech tab, computed once by the Workspace. */
    voice: VoiceView;
    /** The open speech model, which says whether the New voice tab can design a voice from a description. */
    model: AudioModelOption | null;
  }

  let { voice, model }: Props = $props();

  // The saved voices are read again each time the sheet opens, so voices saved or removed elsewhere (another tab) show up.
  onMount(() => { void voicesStore.load(); });

  function takeFiles(files: File[]): void {
    takeAudioFiles(files, (file) => draft.setVoice({ file }));
  }
</script>

<div class="voice-pop">
  <div class="voice-pop-row">
    <Select
      class="pop-select"
      triggerClass={voice.deleted ? 'pop-select-trigger voice-deleted' : voice.cloned ? 'pop-select-trigger voice-named' : 'pop-select-trigger'}
      label="Voice"
      testId="pop-voice-select"
      value={voice.choice}
      options={voice.options}
      placeholder={voice.deleted ? VOICE_WORDS.deletedPlaceholder : voice.recording ? VOICE_WORDS.recording : voice.saved ? voice.chip : undefined}
      onchange={(choice) => chooseVoice(voice, choice)}
    />
    <VoicePlay src={voice.play.src} file={voice.play.file} label={voice.play.label} />
  </div>
  {#if voice.cloneDownload}<p class="ui-help" data-testid="pop-voice-download-hint">{VOICE_WORDS.downloadHint}</p>{/if}
  {#if voice.recording}
    <div class="pop-block" data-testid="voice-one-off">
      <div class="pop-group" role="group" aria-label="Recording">
        {#if voice.set}
          <div class="rec-file">
            <span class="rec-name" class:missing={voice.gone} title={voice.name}>{voice.name}</span>
            {#if voice.length !== null}<span class="rec-length" data-testid="pop-voice-length">{formatTimestamp(voice.length)}</span>{/if}
            {#if !voice.gone}<button type="button" class="ui-btn ui-btn-icon rec-clear" aria-label="Remove the voice recording" onclick={() => draft.leaveRecording()}><Icon name="close" size={16} /></button>{/if}
          </div>
        {/if}
        {#if voice.error}<p class="ui-help ui-help-error">{voice.error}</p>{/if}
        {#if voice.gone || !voice.set}<ImageSources kind="audio" browseFrom={voice.path} onfiles={takeFiles} onpath={(path) => draft.setVoice({ path })} />{/if}
      </div>
      <div class="pop-group">
        <label class="ui-label" for="ws-pop-voice-text">What it says <span class="voice-required">required</span></label>
        <textarea
          id="ws-pop-voice-text"
          class="ui-field voice-field"
          rows="2"
          aria-required="true"
          aria-invalid={voice.set && voice.text.trim() === '' ? 'true' : undefined}
          placeholder={VOICE_WORDS.placeholder}
          value={voice.text}
          use:autogrow={voice.text}
          oninput={(e) => draft.update('voiceText', e.currentTarget.value)}
        ></textarea>
      </div>
      {#if voice.needsConsent}
        <ConsentTick large checked={voice.ticked} onchange={(checked) => { voiceConsent.ticked = checked; }} />
      {/if}
    </div>
  {/if}
  <VoiceActions {voice} {model} />
</div>

<style>
  /* The chosen voice is gone: the select has no name to show, so its placeholder says so in the error colour. */
  :global(.voice-deleted .ui-select-value.placeholder) { color: var(--color-error); }
  /* A saved or one-off voice reads as a choice made, in the normal text colour, even where the select shows it as its placeholder. */
  :global(.voice-named .ui-select-value.placeholder) { color: var(--color-text-primary); }
  .voice-pop { display: flex; flex-direction: column; gap: 16px; }
  .voice-pop-row { display: flex; align-items: center; gap: 8px; }
  .voice-pop :global(.pop-select) { display: flex; flex: 1; min-width: 0; }
  .voice-pop :global(.pop-select-trigger) { height: 44px; font-size: 15px; }
  .pop-block { display: flex; flex-direction: column; gap: 16px; }
  .pop-group { display: flex; flex-direction: column; gap: 8px; padding: 0 2px; }
  /* The popover has its own padding. */
  .voice-pop :global(.voice-actions) { padding: 0; }
  .rec-file { display: flex; align-items: center; gap: 12px; min-height: 48px; font-size: 15px; }
  .rec-name { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--color-text-primary); }
  .rec-name.missing { color: var(--color-error); }
  .rec-length { flex: none; font-family: var(--font-mono); font-size: 13px; color: var(--color-text-muted); white-space: nowrap; }
  .rec-clear { flex: none; width: 44px; height: 44px; }
  .voice-required { margin-left: 6px; font-size: var(--text-meta); font-weight: 400; text-transform: none; letter-spacing: 0; color: var(--color-text-muted); }
  .voice-field { min-height: 64px; max-height: 120px; resize: none; padding: 8px 10px; font-size: 16px; line-height: 1.45; }
  .voice-field[aria-invalid='true'] { box-shadow: inset 0 0 0 1px var(--color-error); }
</style>
