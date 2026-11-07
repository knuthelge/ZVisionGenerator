<script lang="ts">
  import { InspectorRow, InspectorSection, Select } from '$lib/components/molecules';
  import { autogrow } from '$lib/actions/autogrow';
  import { formatTimestamp } from '$lib/state/audioTrack';
  import { draft } from '$lib/state/draft.svelte';
  import { voiceConsent } from '$lib/state/voiceConsent.svelte';
  import { chooseVoice, VOICE_WORDS, type VoiceView } from './audioView';
  import ReferenceSlot from './ReferenceSlot.svelte';

  interface Props {
    /** The voice of the open Speech tab, computed once by the Workspace. */
    voice: VoiceView;
    /** The voice differs from its default: the row shows its changed mark and its reset. */
    changed: boolean;
    /** Put the voice back: a built-in voice of the model's default, with no recording. */
    onreset: () => void;
  }

  let { voice, changed, onreset }: Props = $props();
</script>

<InspectorSection id="voice" title="Voice" summary={voice.summary.text} active={voice.summary.tone === 'active'} error={voice.summary.tone === 'error'}>
  <InspectorRow label="Voice" forId="ws-voice-select" changed={changed} onreset={onreset}>
    <Select
      id="ws-voice-select"
      variant="bare"
      class="flex-1"
      triggerClass="inspector-select"
      label="Voice"
      testId="voice-select"
      value={voice.choice}
      options={voice.options}
      onchange={(choice) => chooseVoice(voice, choice)}
    />
  </InspectorRow>
  {#if voice.recording}
    {#if voice.cloneDownload}
      <p class="row-hint" data-testid="voice-download-hint">{VOICE_WORDS.downloadHint}</p>
    {/if}
    <ReferenceSlot
      idBase="ws-voice"
      kind="audio"
      label="Recording"
      names={{ browse: 'Browse for a voice recording', upload: 'Upload a voice recording from this device', clear: 'Clear the voice recording', preview: 'Voice recording' }}
      emptyText="Browse or drop"
      clearable={voice.set}
      path={voice.path}
      file={voice.file}
      name={voice.name}
      detail={voice.length === null ? '' : formatTimestamp(voice.length)}
      dropText={voice.set ? 'Drop to replace' : 'Drop an audio file'}
      missing={voice.gone}
      error={voice.error ?? undefined}
      onfilechange={(file) => { if (file !== null) draft.setVoice({ file }); }}
      onpathchange={(path) => draft.setVoice(path === null ? null : { path })}
    />
    <div class="voice-text">
      <label class="ui-label" for="ws-voice-text">What it says <span class="voice-required">required</span></label>
      <textarea
        id="ws-voice-text"
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
        <input type="checkbox" class="ui-checkbox" id="ws-voice-consent" checked={voice.ticked} onchange={(e) => (voiceConsent.ticked = e.currentTarget.checked)}>
        <span>{VOICE_WORDS.consent}</span>
      </label>
    {/if}
  {/if}
</InspectorSection>


<style>
  .voice-text { display: flex; flex-direction: column; gap: 4px; padding: 6px 12px 4px; }
  .voice-required { margin-left: 6px; font-size: var(--text-meta); font-weight: 400; text-transform: none; letter-spacing: 0; color: var(--color-text-muted); }
  .voice-field { min-height: 52px; max-height: 104px; resize: none; padding: 6px 8px; font-size: var(--text-ui); line-height: 1.45; }
  .voice-field[aria-invalid='true'] { box-shadow: inset 0 0 0 1px var(--color-error); }
  .voice-consent { display: flex; align-items: flex-start; gap: 8px; min-height: 30px; padding: 6px 12px 8px; font-size: var(--text-ui); line-height: 1.4; color: var(--color-text-secondary); cursor: pointer; }
  .voice-consent.ticked { color: var(--color-text-primary); }
  .voice-consent input { margin-top: 2px; }
  @media (pointer: coarse) {
    .voice-field { min-height: 64px; font-size: 16px; }
    .voice-consent { min-height: 44px; font-size: 15px; }
    .voice-consent input { width: 20px; height: 20px; }
  }
</style>
