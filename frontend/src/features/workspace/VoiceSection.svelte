<script lang="ts">
  import { onMount } from 'svelte';
  import { ConsentTick, InspectorRow, InspectorSection, Select, VoicePlay } from '$lib/components/molecules';
  import { autogrow } from '$lib/actions/autogrow';
  import { formatTimestamp } from '$lib/state/audioTrack';
  import { draft } from '$lib/state/draft.svelte';
  import { voiceConsent } from '$lib/state/voiceConsent.svelte';
  import { voicesStore } from '$lib/state/voices.svelte';
  import type { AudioModelOption } from '$lib/types';
  import { chooseVoice, VOICE_WORDS, type VoiceView } from './audioView';
  import ReferenceSlot from './ReferenceSlot.svelte';
  import VoiceActions from './VoiceActions.svelte';

  interface Props {
    /** The voice of the open Speech tab, computed once by the Workspace. */
    voice: VoiceView;
    /** The open speech model, which says whether the New voice tab can design a voice from a description. */
    model: AudioModelOption | null;
    /** The voice differs from its default: the row shows its changed mark and its reset. */
    changed: boolean;
    /** Put the voice back: a default voice of the model's default, with no recording. */
    onreset: () => void;
  }

  let { voice, model, changed, onreset }: Props = $props();

  // The saved voices are read again whenever the section appears, so voices saved or removed elsewhere (another tab) show up.
  onMount(() => { void voicesStore.load(); });
</script>

<InspectorSection id="voice" title="Voice" summary={voice.summary.text} active={voice.summary.tone === 'active'} error={voice.summary.tone === 'error'}>
  <InspectorRow label="Voice" forId="ws-voice-select" changed={changed} onreset={onreset}>
    <Select
      id="ws-voice-select"
      variant="bare"
      class="flex-1"
      triggerClass={voice.deleted ? 'inspector-select voice-deleted' : voice.cloned ? 'inspector-select voice-named' : 'inspector-select'}
      label="Voice"
      testId="voice-select"
      value={voice.choice}
      options={voice.options}
      placeholder={voice.deleted ? VOICE_WORDS.deletedPlaceholder : voice.recording ? VOICE_WORDS.recording : voice.saved ? voice.chip : undefined}
      onchange={(choice) => chooseVoice(voice, choice)}
    />
    <VoicePlay src={voice.play.src} file={voice.play.file} label={voice.play.label} />
  </InspectorRow>
  {#if voice.cloneDownload}
    <p class="row-hint" data-testid="voice-download-hint">{VOICE_WORDS.downloadHint}</p>
  {/if}
  {#if voice.recording}
    <div class="rec-block" data-testid="voice-one-off">
      <ReferenceSlot
        idBase="ws-voice"
        kind="audio"
        label="Recording"
        thumb={false}
        swappable={false}
        names={{ browse: 'Browse for a voice recording', upload: 'Upload a voice recording from this device', clear: 'Remove the voice recording', preview: 'Voice recording' }}
        emptyText="Browse or drop"
        clearable={!voice.gone}
        path={voice.path}
        file={voice.file}
        name={voice.name}
        detail={voice.length === null ? '' : formatTimestamp(voice.length)}
        dropText={voice.set ? 'Drop to replace' : 'Drop an audio file'}
        missing={voice.gone}
        error={voice.error ?? undefined}
        onfilechange={(file) => { if (file !== null) draft.setVoice({ file }); }}
        onpathchange={(path) => { if (path === null) draft.leaveRecording(); else draft.setVoice({ path }); }}
      />
      <InspectorRow label="What it says" forId="ws-voice-text">
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
      </InspectorRow>
      {#if voice.needsConsent}
        <div class="rec-indent">
          <ConsentTick id="ws-voice-consent" checked={voice.ticked} onchange={(checked) => { voiceConsent.ticked = checked; }} />
        </div>
      {/if}
    </div>
  {/if}
  <VoiceActions {voice} {model} />
</InspectorSection>

<style>
  /* The chosen voice is gone: the select has no name to show, so its placeholder says so in the error colour. */
  :global(.voice-deleted .ui-select-value.placeholder) { color: var(--color-error); }
  /* A saved or one-off voice reads as a choice made, in the normal text colour, even where the select shows it as its placeholder. */
  :global(.voice-named .ui-select-value.placeholder) { color: var(--color-text-primary); }
  .rec-block { display: flex; flex-direction: column; padding-bottom: 6px; }
  .rec-indent { padding: 2px 8px 2px calc(var(--inspector-label-width, 96px) + 20px); }
  .voice-field { width: 100%; min-height: 52px; max-height: 104px; resize: none; padding: 6px 8px; font-size: var(--text-ui); line-height: 1.45; }
  .voice-field[aria-invalid='true'] { box-shadow: inset 0 0 0 1px var(--color-error); }
  @media (pointer: coarse) {
    .voice-field { min-height: 64px; font-size: 16px; }
  }
</style>
