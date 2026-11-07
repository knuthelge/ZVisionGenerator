import { DELIVERY_FIELDS as FIELDS } from './audioDraft';
import type { AudioSettingsVocabulary, SpeechDeliveryValues, VibeOption } from '$lib/types';

/** The Vibe whose five values are exactly *values*, or null when they match none (the voice is steered by hand, or not at all). */
export function vibeOf(values: SpeechDeliveryValues, vibes: readonly VibeOption[]): VibeOption | null {
  return vibes.find((vibe) => FIELDS.every((field) => vibe.delivery[field] === values[field])) ?? null;
}

/** Whether *a* and *b* are the same five values. */
export function sameDelivery(a: SpeechDeliveryValues, b: SpeechDeliveryValues): boolean {
  return FIELDS.every((field) => a[field] === b[field]);
}

/** The Delivery section's header summary and the phone chip's value: the matching Vibe, `Natural` at the defaults, else `Custom`. */
export function deliverySummary(values: SpeechDeliveryValues, vocabulary: AudioSettingsVocabulary): string {
  if (sameDelivery(values, vocabulary.delivery_defaults)) return 'Natural';
  return vibeOf(values, vocabulary.vibes)?.label ?? 'Custom';
}

/** A choice's label in the vocabulary of its setting, or its id when the vocabulary does not list it. */
export function deliveryLabel(vocabulary: AudioSettingsVocabulary, field: keyof SpeechDeliveryValues, id: string): string {
  const choices = { emotion: vocabulary.emotions, manner: vocabulary.manners, pace: vocabulary.paces, pitch: vocabulary.levels, expressiveness: vocabulary.levels }[field];
  return choices.find((choice) => choice.id === id)?.label ?? id;
}

/**
 * What a Vibe tile says under its name: the emotion, the manner and the pace, each only when it is not the default
 * (`Thoughtful · slow`, `Whisper · slow`). Pitch and Expressiveness are in the rows under the tiles.
 */
export function vibeDetail(vibe: VibeOption, vocabulary: AudioSettingsVocabulary): string {
  const defaults = vocabulary.delivery_defaults;
  const parts = (['emotion', 'manner', 'pace'] as const)
    .filter((field) => vibe.delivery[field] !== defaults[field])
    .map((field) => deliveryLabel(vocabulary, field, vibe.delivery[field]));
  return parts.map((part, index) => (index === 0 ? part : part.toLowerCase())).join(' · ');
}

/** A bracketed run of 2 to 20 letters and spaces: a mark when it names one, a mistake when it does not (the server's rule, `utils/speech_text.py`). */
export const BRACKETED = /\[([A-Za-z ]{2,20})\]/g;

/** The name a bracketed run stands for: lower case, one space between words. */
export function markName(name: string): string {
  return name.toLowerCase().split(/\s+/).filter(Boolean).join(' ');
}

/** The text without its `[marks]`: the characters that are read. A bracketed run counts as a mark when it names one in *marks*, as the server counts. */
export function stripMarks(text: string, marks: readonly string[]): string {
  if (marks.length === 0) return text;
  return text.replace(BRACKETED, (found, name: string) => (marks.includes(markName(name)) ? '' : found));
}

/**
 * The bracketed runs of *text* that look like a mark but are not one in *marks*, as written (`[sadd]`), in order: the ones the server
 * refuses. A run of spaces alone is plain text. With no *marks* (the vocabulary has not loaded) nothing is reported.
 */
export function unknownMarks(text: string, marks: readonly string[]): string[] {
  if (marks.length === 0) return [];
  return [...text.matchAll(BRACKETED)].filter(([, name]) => name.trim() !== '' && !marks.includes(markName(name))).map(([found]) => found);
}

/**
 * What the user reads when the bracketed run *found* (as `unknownMarks` gives it) is refused: it names the way to get a pause or a laugh when
 * that is what was meant. The server words it the same way (`unknown_mark_message` in `utils/speech_text.py`).
 */
export function unknownMarkMessage(found: string): string {
  const refused = `${found} isn't a Delivery mark.`;
  const name = markName(found.slice(1, -1));
  if (name === 'pause' || name === 'pauses') return `${refused} For a pause, type … (the Insert menu adds one).`;
  if (name === 'laugh' || name === 'laughs' || name === 'laughing') return `${refused} For a laugh, write haha (the Insert menu adds one).`;
  return `${refused} Use one of the Delivery marks, or remove the brackets.`;
}

/** How many seconds *characters* characters take to read at *pace*; 0 for no text. A pace the vocabulary does not list reads at Normal. */
export function readingSeconds(characters: number, pace: string, reading: AudioSettingsVocabulary['reading']): number {
  if (characters <= 0) return 0;
  return characters / (reading.chars_per_second * (reading.pace_factor[pace] ?? 1));
}

/** A time as the status line writes it: `1:24`, rounded to the second. */
export function formatReadingTime(seconds: number): string {
  const whole = Math.round(seconds);
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, '0')}`;
}
