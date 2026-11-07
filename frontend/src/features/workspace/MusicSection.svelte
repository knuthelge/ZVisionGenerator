<script lang="ts">
  import { InspectorNumber } from '$lib/components/atoms';
  import { InspectorRow, InspectorSection, Select } from '$lib/components/molecules';
  import { musicSecondsShown, secondsRange, settleMusicSeconds } from '$lib/state/audioDraft';
  import { markedFields } from '$lib/state/audioFill';
  import { draft } from '$lib/state/draft.svelte';
  import { keyOptions, musicSummary } from '$lib/state/musicSettings';
  import type { AudioModelOption, AudioSettingsVocabulary } from '$lib/types';
  import type { AudioShown } from './audioView';
  import FilledMark from './FilledMark.svelte';
  import type { RowId } from './settingRows';

  interface Props {
    /** The tempo range and the keys, from the Workspace payload. */
    vocabulary: AudioSettingsVocabulary;
    /** The open Music row; a model that sets its length also gets the Length row. */
    model: AudioModelOption | null;
    /** The controls the open row shows. */
    shown: Pick<AudioShown, 'showTempo' | 'showKey' | 'showLength'>;
    /** A submit is in flight: the scrubbers wait. */
    busy: boolean;
    /** Whether a row differs from its default: its changed mark and its reset. */
    changed: (id: RowId) => boolean;
    /** Put a row back to its default. */
    onreset: (id: RowId) => void;
  }

  let { vocabulary, model, shown, busy, changed, onreset }: Props = $props();

  const s = $derived(draft.state);
  const tempoSpec = $derived({ step: 1, min: vocabulary.tempo.min, max: vocabulary.tempo.max });
  const lengthRange = $derived(secondsRange(model));
  const lengthSpec = $derived(lengthRange && { step: 1, ...lengthRange });
  const shownLength = $derived(musicSecondsShown(s.musicSeconds, model));
  // Auto first, then the twelve major and the twelve minor keys in two columns.
  const keySelect = $derived(keyOptions(vocabulary));
  const summary = $derived(musicSummary(s, model, vocabulary));
  /** The fields Describe it wrote that are still as written: their rows carry a ✦. */
  const marked = $derived(markedFields(s, 'music'));

  const setTempo = (tempo: number | null): void => draft.update('audioTempo', tempo);
  const setLength = (seconds: number): void => draft.update('musicSeconds', settleMusicSeconds(seconds, model));
</script>

<InspectorSection id="music" title="Music" {summary} active={s.audioTempo !== null || s.audioKey !== null}>
  {#if shown.showTempo}
    <InspectorRow label="Tempo" forId="ws-tempo" scrubber={{ ...tempoSpec, value: s.audioTempo, onchange: setTempo, disabled: busy }} changed={changed('tempo')} onreset={() => onreset('tempo')}>
      {#snippet mark()}{#if marked.includes('tempo')}<FilledMark />{/if}{/snippet}
      <span class="tempo-field">
        <InspectorNumber id="ws-tempo" value={s.audioTempo} {...tempoSpec} nullable commitOnly placeholder="Auto" ariaLabel="Tempo in BPM, empty for Auto" onchange={setTempo} />
        <span class="unit">BPM</span>
      </span>
    </InspectorRow>
  {/if}
  {#if shown.showKey}
    <InspectorRow label="Key" forId="ws-key" changed={changed('key')} onreset={() => onreset('key')}>
      {#snippet mark()}{#if marked.includes('key')}<FilledMark />{/if}{/snippet}
      <Select
        id="ws-key"
        variant="bare"
        class="flex-1"
        triggerClass="inspector-select"
        label="Key"
        testId="key-select"
        value={s.audioKey ?? ''}
        options={keySelect}
        groupColumns
        onchange={(key) => draft.update('audioKey', key === '' ? null : key)}
      />
    </InspectorRow>
  {/if}
  {#if shown.showLength && lengthSpec && shownLength !== null}
    <InspectorRow label="Length (s)" forId="ws-music-length" scrubber={{ ...lengthSpec, value: shownLength, onchange: setLength, disabled: busy }} changed={changed('musicLength')} onreset={() => onreset('musicLength')}>
      <InspectorNumber id="ws-music-length" value={shownLength} {...lengthSpec} commitOnly onchange={(v) => v !== null && setLength(v)} />
      <span class="unit" data-testid="music-length-max">max {lengthSpec.max}</span>
    </InspectorRow>
  {/if}
</InspectorSection>

<style>
  .tempo-field { display: flex; flex: 1; min-width: 0; align-items: center; gap: 4px; }
  /* The unit sits against the number; on a touch screen the stepper keeps its own width. */
  .tempo-field :global(.inspector-number-wrap:not(.with-steps)) { flex: 0 1 72px; }
  .unit { flex: none; font-family: var(--font-mono); font-size: var(--text-meta); color: var(--color-text-muted); white-space: nowrap; }
</style>
