import { musicSecondsShown } from './audioDraft';
import type { AudioModelOption, AudioSettingsVocabulary, DraftState } from '$lib/types';

/** A key's words as the Web UI writes them: the sharp and flat of the root as ♯ and ♭ (`F# minor` is `F♯ minor`). */
export function keyName(words: string): string {
  return words.replace(/^([A-G])#/, '$1♯').replace(/^([A-G])b/, '$1♭');
}

/** The name of the key with this id (`Am` is `A minor`); the id itself when the vocabulary does not list it. */
export function keyNameOf(vocabulary: AudioSettingsVocabulary | undefined, id: string): string {
  const key = vocabulary?.keys.find((item) => item.id === id);
  return key === undefined ? id : keyName(key.words);
}

/** The Key select's options: Auto first, then the twelve major and the twelve minor keys in two groups. Auto alone while the vocabulary is not at hand. */
export function keyOptions(vocabulary: Pick<AudioSettingsVocabulary, 'keys'> | undefined): { value: string; label: string; hint?: string; group?: string }[] {
  return [
    { value: '', label: 'Auto', hint: 'The model picks' },
    ...(vocabulary?.keys ?? []).map((key) => ({ value: key.id, label: keyName(key.words), group: key.mode === 'major' ? 'Major' : 'Minor' })),
  ];
}

/** A key's name in the width of a phone chip: `A minor` is `A min`. */
export function keyShortName(name: string): string {
  return name.replace(' major', ' maj').replace(' minor', ' min');
}

/** The Music section's header summary: `Auto`, or the tempo and key that are set (`92 BPM · A minor`), then the length of a model that sets one (`Auto · 30 s`). */
export function musicSummary(state: Pick<DraftState, 'audioTempo' | 'audioKey' | 'musicSeconds'>, model: AudioModelOption | null, vocabulary: AudioSettingsVocabulary | undefined): string {
  const set = [state.audioTempo === null ? null : `${state.audioTempo} BPM`, state.audioKey === null ? null : keyNameOf(vocabulary, state.audioKey)].filter((part) => part !== null);
  const seconds = musicSecondsShown(state.musicSeconds, model);
  return [...(set.length > 0 ? set : ['Auto']), ...(seconds === null ? [] : [`${seconds} s`])].join(' · ');
}
