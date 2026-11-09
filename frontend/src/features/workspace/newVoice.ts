// What the New voice tab knows without a screen: the line every designed voice reads, in the language of the take's text, how long
// that line takes to read, and the example descriptions. Pure functions, so each is tested with plain strings.

import { jobPhase, voiceDesignLine, voiceDesignProgress } from '$lib/state/jobView';
import { formatReadingTime, readingSeconds } from '$lib/state/speechDelivery';
import type { ActiveJobState, AudioSettingsVocabulary } from '$lib/types';

/** The longest description the server takes. */
export const DESCRIPTION_MAX = 500;

/** The longest preview line the server takes: one part of the speech model (`part_chars`). */
export const PREVIEW_MAX = 500;

/** The ten languages the speech model reads. */
export type PreviewLanguage = 'Chinese' | 'English' | 'Japanese' | 'Korean' | 'German' | 'French' | 'Russian' | 'Portuguese' | 'Spanish' | 'Italian';

/** One natural sentence in each language, about 6 to 10 seconds at a normal pace: the line a designed voice reads when the user writes none. */
export const PREVIEW_LINES: Readonly<Record<PreviewLanguage, string>> = {
  Chinese: '每一种声音都是一个小小的世界，让我慢慢地、清楚地为你读一读这一句话。',
  English: 'Every voice is a small world. Let me show you this one, slowly and clearly.',
  Japanese: 'どんな声にも、小さな世界があります。この声を、ゆっくりはっきりとお聞かせします。',
  Korean: '모든 목소리는 작은 세상입니다. 이 목소리를 천천히 또렷하게 들려 드릴게요.',
  German: 'Jede Stimme ist eine kleine Welt. Lass mich dir diese hier langsam und deutlich zeigen.',
  French: 'Chaque voix est un petit monde. Laissez-moi vous présenter celle-ci, lentement et clairement.',
  Russian: 'Каждый голос — это маленький мир. Позвольте мне показать вам этот голос, медленно и чётко.',
  Portuguese: 'Cada voz é um pequeno mundo. Deixe-me mostrar-lhe esta, devagar e com clareza.',
  Spanish: 'Cada voz es un pequeño mundo. Déjame mostrarte esta, despacio y con claridad.',
  Italian: 'Ogni voce è un piccolo mondo. Lascia che ti mostri questa, lentamente e con chiarezza.',
};

/** Words that are common in one Latin-script language and rare in the others; a text votes for the language that owns most of them. */
const STOP_WORDS: Readonly<Record<'English' | 'German' | 'French' | 'Spanish' | 'Italian' | 'Portuguese', ReadonlySet<string>>> = {
  English: new Set(['the', 'and', 'is', 'are', 'you', 'of', 'to', 'in', 'it', 'that', 'this', 'with', 'for', 'have', 'was', 'what', 'hello', 'there', 'my', 'we']),
  German: new Set(['der', 'die', 'das', 'und', 'ist', 'nicht', 'ich', 'du', 'sie', 'ein', 'eine', 'mit', 'auf', 'zu', 'den', 'dem', 'wir', 'haben', 'hallo', 'auch', 'mich', 'dir']),
  French: new Set(['le', 'les', 'des', 'et', 'est', 'une', 'un', 'que', 'pour', 'pas', 'vous', 'nous', 'je', 'il', 'dans', 'avec', 'bonjour', 'ce', 'sur', 'qui']),
  Spanish: new Set(['el', 'los', 'las', 'y', 'es', 'una', 'un', 'que', 'para', 'por', 'no', 'con', 'hola', 'muy', 'está', 'como', 'pero', 'del', 'se']),
  Italian: new Set(['il', 'lo', 'gli', 'che', 'di', 'è', 'e', 'una', 'un', 'per', 'non', 'con', 'ciao', 'sono', 'molto', 'come', 'questo', 'della', 'nel', 'ti']),
  Portuguese: new Set(['os', 'as', 'é', 'e', 'uma', 'um', 'que', 'para', 'não', 'com', 'olá', 'muito', 'você', 'como', 'mas', 'em', 'está', 'dos', 'da']),
};

const HANGUL = /\p{Script=Hangul}/gu;
const KANA = /[\p{Script=Hiragana}\p{Script=Katakana}]/gu;
const HAN = /\p{Script=Han}/gu;
const CYRILLIC = /\p{Script=Cyrillic}/gu;
const LATIN = /\p{Script=Latin}/gu;

/** How many characters of *text* match *pattern*. */
function count(text: string, pattern: RegExp): number {
  return text.match(pattern)?.length ?? 0;
}

/** The text without its `[marks]`, which steer a take and are not read. */
function withoutMarks(text: string): string {
  return text.replace(/\[[^\]]*\]/g, ' ');
}

/** The language of the Latin-script *text*, by a stop-word vote; English when no language owns more words than the others. */
function latinLanguage(text: string): PreviewLanguage {
  const words = text.toLowerCase().match(/\p{L}+/gu) ?? [];
  let best: PreviewLanguage = 'English';
  let bestVotes = 0;
  let tied = false;
  for (const [language, stops] of Object.entries(STOP_WORDS) as [keyof typeof STOP_WORDS, ReadonlySet<string>][]) {
    const votes = words.filter((word) => stops.has(word)).length;
    if (votes > bestVotes) { best = language; bestVotes = votes; tied = false; }
    else if (votes === bestVotes && votes > 0) tied = true;
  }
  return tied ? 'English' : best;
}

/**
 * The language of *text*, to choose the line its voices read: the script with the most letters decides (Hangul is Korean, kana with or
 * without kanji is Japanese, Han alone is Chinese, Cyrillic is Russian), and Latin script goes to a stop-word vote among English,
 * German, French, Spanish, Italian and Portuguese. English for empty text, a text with no letters and a tie.
 */
export function previewLanguage(text: string): PreviewLanguage {
  const plain = withoutMarks(text);
  const hangul = count(plain, HANGUL);
  const kana = count(plain, KANA);
  const han = count(plain, HAN);
  const cyrillic = count(plain, CYRILLIC);
  const latin = count(plain, LATIN);
  const scripts: [PreviewLanguage | null, number][] = [
    ['Korean', hangul],
    [kana > 0 ? 'Japanese' : 'Chinese', kana + han],
    ['Russian', cyrillic],
    [null, latin],
  ];
  const [language, letters] = scripts.reduce((best, entry) => (entry[1] > best[1] ? entry : best), scripts[3]);
  if (letters === 0) return 'English';
  return language ?? latinLanguage(plain);
}

/** The line the voices read when the user writes none: the sentence in the language of the take's text. */
export function defaultPreviewLine(speechText: string): string {
  return PREVIEW_LINES[previewLanguage(speechText)];
}

/** How many Latin characters a Chinese, Japanese or Korean character takes to read: they carry about twice the sound of a letter. */
const CJK_WEIGHT = 2;

/** The number of characters that count toward the reading time of *line*: a Chinese, Japanese or Korean character counts twice. */
function spokenCharacters(line: string): number {
  const text = withoutMarks(line).trim();
  const wide = count(text, HANGUL) + count(text, KANA) + count(text, HAN);
  return text.length + wide * (CJK_WEIGHT - 1);
}

/** What sits under the preview line: the read time, and a warning when the time is outside the range of a voice. */
export interface LineHint {
  /** "About 6 seconds to read". */
  read: string;
  /** The warning that replaces the help line, or null when the line is long enough and not too long. */
  warning: string | null;
}

/**
 * Say how long *line* takes to read at a normal pace, and warn when that is outside *range* (the seconds a saved voice may last). It is a
 * warning, not a block: the estimate is rough, and the server has the last word. Null when the vocabulary has not loaded or the line is empty.
 */
export function lineHint(line: string, range: { min: number; max: number }, reading: AudioSettingsVocabulary['reading'] | undefined): LineHint | null {
  if (reading === undefined) return null;
  const characters = spokenCharacters(line);
  if (characters === 0) return null;
  const seconds = readingSeconds(characters, 'normal', reading);
  const whole = Math.max(1, Math.round(seconds));
  const read = seconds >= 60 ? `About ${formatReadingTime(seconds)} to read` : `About ${whole} ${whole === 1 ? 'second' : 'seconds'} to read`;
  const outside = seconds < range.min || seconds > range.max;
  return { read, warning: outside ? `A voice needs ${range.min} to ${range.max} seconds, so this one may not be saveable.` : null };
}

/** One example description: the short name in the menu, and the sentence it fills into the box. */
export interface ExampleDescription {
  label: string;
  text: string;
}

/** Examples across both genders, several ages and accents; a choice fills the description box, which stays editable. */
export const EXAMPLE_DESCRIPTIONS: readonly ExampleDescription[] = [
  { label: 'Warm British woman', text: 'A warm British woman in her thirties, calm and clear' },
  { label: 'Gravelly older man', text: 'A deep, gravelly man in his sixties, slow and weathered' },
  { label: 'Bright young woman', text: 'A bright, cheerful young woman from California, quick and friendly' },
  { label: 'Documentary narrator', text: 'A calm man in his fifties with a low voice, like a nature documentary narrator' },
  { label: 'Soft Irish grandmother', text: 'A soft-spoken grandmother with a faint Irish lilt, gentle and slow' },
  { label: 'Curious teen', text: 'A curious teenage boy, quick and full of wonder' },
];

/**
 * The one line that says where a design job is. It is `voiceDesignLine` for a first run; once earlier candidates are listed, the voices
 * being made are *more* of them, so the making phase reads "Making 3 more: 2 of 3".
 */
export function runLine(job: ActiveJobState, more: boolean): string {
  if (!more || jobPhase(job) !== 'generate') return voiceDesignLine(job);
  const { done, total } = voiceDesignProgress(job);
  return done > 0 ? `Making ${total} more: ${done} of ${total}` : `Making ${total} more`;
}
