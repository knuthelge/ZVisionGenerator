import type { AudioSettingsVocabulary, DraftState, SoundType } from '$lib/types';

const fmt = (value: number): string => (Number.isInteger(value) ? String(value) : String(Number(value.toFixed(2))));

/** The label of a Sound Type as the control writes it (`one-shot` is `One-shot`); the id itself when the vocabulary does not list it. */
export function soundTypeLabel(type: SoundType, vocabulary: AudioSettingsVocabulary): string {
  return vocabulary.sound_types.find((item) => item.id === type)?.label ?? type;
}

/** The length in seconds a sound of this type has while its Length is on Auto. */
export function autoSoundSeconds(type: SoundType, vocabulary: AudioSettingsVocabulary): number {
  return vocabulary.auto_seconds[type];
}

/** What Auto stands for, as the Length field and the phone chip write it: `≈ 12 s`. */
export function autoSecondsText(type: SoundType, vocabulary: AudioSettingsVocabulary): string {
  return `≈ ${fmt(autoSoundSeconds(type, vocabulary))} s`;
}

/** The Sound section's header summary: the type, then the length it will use (`Ambience · ≈ 12 s`) or the one that is set (`Effect · 8 s`). */
export function soundSummary(state: Pick<DraftState, 'soundType' | 'audioSeconds'>, vocabulary: AudioSettingsVocabulary): string {
  const length = state.audioSeconds > 0 ? `${fmt(state.audioSeconds)} s` : autoSecondsText(state.soundType, vocabulary);
  return `${soundTypeLabel(state.soundType, vocabulary)} · ${length}`;
}
