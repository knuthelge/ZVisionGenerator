<script lang="ts">
  import { InspectorNumber } from '$lib/components/atoms';
  import { InspectorRow, InspectorSection, Segmented } from '$lib/components/molecules';
  import { clampSeconds, secondsRange } from '$lib/state/audioDraft';
  import { draft } from '$lib/state/draft.svelte';
  import { autoSecondsText, autoSoundSeconds, soundSummary } from '$lib/state/soundSettings';
  import type { AudioModelOption, AudioSettingsVocabulary } from '$lib/types';
  import type { AudioShown } from './audioView';
  import type { RowId } from './settingRows';

  interface Props {
    /** The Sound Types and the length of each type's Auto, from the Workspace payload. */
    vocabulary: AudioSettingsVocabulary;
    /** The open Sound row: it sets the range a typed length can take. */
    model: AudioModelOption | null;
    /** The controls the open row shows. */
    shown: Pick<AudioShown, 'showSoundType' | 'showLength'>;
    /** A submit is in flight: the scrubber waits. */
    busy: boolean;
    /** Whether a row differs from its default: its changed mark and its reset. */
    changed: (id: RowId) => boolean;
    /** Put a row back to its default. */
    onreset: (id: RowId) => void;
  }

  let { vocabulary, model, shown, busy, changed, onreset }: Props = $props();

  const s = $derived(draft.state);
  const lengthRange = $derived(secondsRange(model));
  const lengthSpec = $derived(lengthRange && { step: 1, ...lengthRange });
  const auto = $derived(autoSoundSeconds(s.soundType, vocabulary));
  // Blank is Auto: the field shows the number the type will use, and typing locks one.
  const locked = $derived(s.audioSeconds > 0 ? s.audioSeconds : null);
  const typeOptions = $derived(vocabulary.sound_types.map((type) => ({ value: type.id, label: type.label })));
  const summary = $derived(soundSummary(s, vocabulary));

  const setLength = (seconds: number | null): void => draft.update('audioSeconds', seconds === null ? 0 : clampSeconds(seconds, model));
</script>

<InspectorSection id="sound" title="Sound" {summary} active={s.soundType !== 'auto' || s.audioSeconds > 0}>
  {#if shown.showSoundType}
    <InspectorRow label="Type" changed={changed('soundType')} onreset={() => onreset('soundType')}>
      <Segmented label="Type" size="sm" class="sound-seg" testId="sound-type" options={typeOptions} value={s.soundType} onchange={(type) => draft.update('soundType', type)} />
    </InspectorRow>
  {/if}
  {#if shown.showLength && lengthSpec}
    <InspectorRow label="Length (s)" forId="ws-length" scrubber={{ ...lengthSpec, value: locked ?? auto, onchange: setLength, disabled: busy }} changed={changed('length')} onreset={() => onreset('length')}>
      <InspectorNumber id="ws-length" value={locked} {...lengthSpec} nullable nudgeFrom={auto} commitOnly placeholder="auto · {autoSecondsText(s.soundType, vocabulary)}" ariaLabel="Length in seconds, empty for Auto" onchange={setLength} />
      <span class="unit" data-testid="length-max">max {lengthSpec.max}</span>
    </InspectorRow>
  {/if}
</InspectorSection>

<style>
  :global(.sound-seg) { display: flex; flex: 1; min-width: 0; }
  :global(.sound-seg > button) { flex: 1 1 auto; min-width: 0; padding: 0 4px; }
  .unit { flex: none; font-family: var(--font-mono); font-size: var(--text-meta); color: var(--color-text-muted); white-space: nowrap; }
</style>
