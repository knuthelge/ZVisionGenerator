<script lang="ts">
  import { Icon } from '$lib/components/atoms';
  import { Segmented } from '$lib/components/molecules';
  import { autogrow } from '$lib/actions/autogrow';
  import { formatTimestamp, takeAudioFiles } from '$lib/state/audioTrack';
  import { draft } from '$lib/state/draft.svelte';
  import { voiceConsent } from '$lib/state/voiceConsent.svelte';
  import { VOICE_WORDS, type VoiceView } from './audioView';
  import ImageSources from './ImageSources.svelte';

  interface Props {
    /** The voice of the open Speech tab, computed once by the Workspace. */
    voice: VoiceView;
  }

  let { voice }: Props = $props();

  function takeFiles(files: File[]): void {
    takeAudioFiles(files, (file) => draft.setVoice({ file }));
  }
</script>

<div class="voice-pop">
  <Segmented
    label="Voice"
    class="pop-seg"
    value={voice.recording ? 'recording' : 'any'}
    options={VOICE_WORDS.modes}
    onchange={(mode) => draft.update('voiceMode', mode)}
  />
  {#if !voice.recording}
    <p class="ui-help">{VOICE_WORDS.anyHint}</p>
  {:else}
    <div class="pop-group" role="group" aria-labelledby="pop-voice-group">
      <h4 id="pop-voice-group" class="ui-area-label pop-group-label">Recording</h4>
      <div class="picker-row">
        <span class="picker-thumb" class:missing={voice.gone}><Icon name={voice.gone ? 'alert' : 'waveform'} size={20} /></span>
        <span class="picker-name" class:muted={!voice.name} class:missing={voice.gone} title={voice.name}>{voice.name || 'No recording yet'}</span>
        {#if voice.length !== null}<span class="picker-length" data-testid="pop-voice-length">{formatTimestamp(voice.length)}</span>{/if}
        {#if voice.set}<button type="button" class="ui-btn ui-btn-icon" aria-label="Clear the voice recording" onclick={() => draft.setVoice(null)}><Icon name="close" size={16} /></button>{/if}
      </div>
      {#if voice.error}<p class="ui-help ui-help-error">{voice.error}</p>{/if}
      <ImageSources kind="audio" browseFrom={voice.path} onfiles={takeFiles} onpath={(path) => draft.setVoice({ path })} />
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
      <label class="voice-consent" class:ticked={voice.ticked}>
        <input type="checkbox" class="ui-checkbox" checked={voice.ticked} onchange={(e) => (voiceConsent.ticked = e.currentTarget.checked)}>
        <span>{VOICE_WORDS.consent}</span>
      </label>
    {/if}
  {/if}
</div>

<style>
  .voice-pop { display: flex; flex-direction: column; gap: 16px; }
  .voice-pop :global(.pop-seg) { display: flex; width: 100%; }
  .voice-pop :global(.pop-seg > button) { flex: 1; height: 44px; font-size: 14px; }
  .pop-group { display: flex; flex-direction: column; gap: 8px; padding: 0 2px; }
  .pop-group-label { padding: 0; }
  .picker-row { display: flex; align-items: center; gap: 12px; min-height: 56px; font-size: 15px; }
  .picker-thumb { display: grid; flex: none; place-items: center; width: 56px; height: 56px; border-radius: var(--radius-sm); background: var(--color-bg-surface-hover); color: var(--color-text-muted); }
  .picker-thumb.missing { background: var(--color-error-surface); color: var(--color-error); }
  .picker-name { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .picker-length { flex: none; font-family: var(--font-mono); font-size: 13px; color: var(--color-text-muted); white-space: nowrap; }
  .picker-name.muted { color: var(--color-text-muted); }
  .picker-name.missing { color: var(--color-error); }
  .voice-required { margin-left: 6px; font-size: var(--text-meta); font-weight: 400; text-transform: none; letter-spacing: 0; color: var(--color-text-muted); }
  .voice-field { min-height: 64px; max-height: 120px; resize: none; padding: 8px 10px; font-size: 16px; line-height: 1.45; }
  .voice-consent { display: flex; align-items: flex-start; gap: 10px; min-height: 44px; font-size: 15px; line-height: 1.4; color: var(--color-text-secondary); cursor: pointer; }
  .voice-consent.ticked { color: var(--color-text-primary); }
  .voice-consent input { flex: none; width: 20px; height: 20px; margin-top: 1px; }
</style>
