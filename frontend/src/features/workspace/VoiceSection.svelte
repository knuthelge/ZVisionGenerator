<script lang="ts">
  import { InspectorRow, InspectorSection, Segmented } from '$lib/components/molecules';
  import { autogrow } from '$lib/actions/autogrow';
  import { formatTimestamp } from '$lib/state/audioTrack';
  import { draft } from '$lib/state/draft.svelte';
  import { voiceConsent } from '$lib/state/voiceConsent.svelte';
  import type { VoiceView } from './audioView';
  import ReferenceSlot from './ReferenceSlot.svelte';

  interface Props {
    /** The voice of the open Speech tab, computed once by the Workspace. */
    voice: VoiceView;
  }

  let { voice }: Props = $props();

  const consentLabel = "This is my voice, or I have the speaker's permission to clone it.";
</script>

{#if voice.visible}
  <InspectorSection id="voice" title="Voice" summary={voice.summary.text} active={voice.summary.tone === 'active'} error={voice.summary.tone === 'error'}>
    <InspectorRow label="Voice">
      <Segmented
        size="sm"
        label="Voice"
        class="voice-mode"
        value={voice.mode}
        options={[{ value: 'any', label: 'Any' }, { value: 'recording', label: 'From a recording' }]}
        onchange={(mode) => draft.update('voiceMode', mode)}
      />
    </InspectorRow>
    {#if !voice.recording}
      <p class="row-hint" data-testid="voice-any-hint">The model picks a voice, and it changes from take to take. To keep one, use a take as the voice: Gallery, Use as → Voice.</p>
    {:else}
      <ReferenceSlot
        idBase="ws-voice"
        kind="audio"
        label="Recording"
        names={{ browse: 'Browse for a voice recording', upload: 'Upload a voice recording from this device', clear: 'Clear the voice recording', preview: 'Voice recording' }}
        emptyText="Browse or drop"
        clearable={voice.set}
        path={voice.path}
        file={voice.file}
        previewUrl={null}
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
          placeholder="The words spoken in the recording"
          value={voice.text}
          use:autogrow={voice.text}
          oninput={(e) => draft.update('voiceText', e.currentTarget.value)}
        ></textarea>
      </div>
      {#if voice.needsConsent}
        <label class="voice-consent" class:ticked={voice.ticked}>
          <input type="checkbox" class="ui-checkbox" id="ws-voice-consent" checked={voice.ticked} onchange={(e) => voiceConsent.set(e.currentTarget.checked)}>
          <span>{consentLabel}</span>
        </label>
      {/if}
    {/if}
  </InspectorSection>
{/if}

<style>
  :global(.voice-mode) { flex: 1; min-width: 0; }
  :global(.voice-mode > button) { flex: 1; min-width: 0; padding: 0 2px; }
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
