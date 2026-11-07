/** What a music take's Lyrics box makes: a song with words, or an instrumental made from `[section]` tags. */
type MusicKind = 'song' | 'instrumental';

/** The result of `musicMode`: what the model is given, and how many sections that is. */
export interface MusicMode {
  kind: MusicKind;
  /** The lyrics the model is given: as written for a song, one tag per paragraph for an instrumental. */
  lyrics: string;
  /** The number of `[section]` tags in those lyrics. */
  sectionCount: number;
}

// One `[tag]`: no brackets or line breaks inside, so an unclosed bracket is a word.
const TAG_SOURCE = String.raw`\[[^[\]\n]+\]`;
const TAG = new RegExp(TAG_SOURCE, 'g');
const SECTION_LINE = new RegExp(`^${TAG_SOURCE}$`);

/** Whether a lyric line only names a section: `[verse]`, `[chorus 2]`, with spaces around it. Shares its tag rule with `sectionTags`. */
export function isSectionLine(line: string): boolean {
  return SECTION_LINE.test(line.trim());
}

/** Return the `[section]` tags of *text*, in order and as written. */
export function sectionTags(text: string): string[] {
  return text.match(TAG) ?? [];
}

/**
 * Decide whether *lyrics* make a song or an instrumental. This mirrors `music_mode` in `zvisiongenerator/utils/music_mode.py`,
 * and both are tested against `tests/audio/music_mode_cases.json`.
 *
 * Words outside `[tags]` make a song. Empty lyrics make an instrumental from *defaultSections*, and lyrics of only tags make
 * one from those tags.
 *
 * @param lyrics The Lyrics box.
 * @param defaultSections The section list an empty Lyrics box stands for, such as `[intro] [verse] [chorus]`.
 * @returns The mode, or null when the lyrics are empty and *defaultSections* holds no tag (the server refuses that job).
 */
export function musicMode(lyrics: string, defaultSections: string): MusicMode | null {
  const tags = sectionTags(lyrics);
  if (lyrics.replace(TAG, '').trim() !== '') return { kind: 'song', lyrics, sectionCount: tags.length };
  const sections = tags.length > 0 ? tags : sectionTags(defaultSections);
  if (sections.length === 0) return null;
  return { kind: 'instrumental', lyrics: sections.join('\n\n'), sectionCount: sections.length };
}

/** The status line under the Lyrics box: what the take will be, and how its length is set. */
export function musicStatus(mode: MusicMode | null): string {
  if (mode === null) return '';
  if (mode.kind === 'song') return 'Song · length follows the lyrics';
  return `Instrumental · ${mode.sectionCount} ${mode.sectionCount === 1 ? 'section' : 'sections'} · add or remove section tags to change its length`;
}
