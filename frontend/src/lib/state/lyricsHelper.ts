import type { LyricsEnhanceRequest } from '$lib/api/promptEnhance';
import type { LyricsImprovement } from '$lib/types';
import { hasWords, parseSong, tagName, type Song } from './songBlocks';

/** What the lyrics helper is open on: the whole song, or one section of it to improve or (an empty one) to write. */
export interface LyricsTarget {
  /** The section's index among the `[section]` tags from 0; null is the whole song. */
  section: number | null;
  /** Fill the empty section instead of improving the words. */
  write: boolean;
}

/** The ways to improve lyrics, in the order the popover lists them. `fit` only makes sense for the whole song. */
export const IMPROVEMENTS: readonly { slug: LyricsImprovement; label: string; help: string; wholeOnly: boolean }[] = [
  { slug: 'tighten', label: 'Tighten', help: 'Cut filler and repeated ideas; keep the meaning', wholeOnly: false },
  { slug: 'fit', label: 'Fit the sections', help: 'Make each section\'s length suit its role', wholeOnly: true },
  { slug: 'rhymes', label: 'Smooth rhymes', help: 'A clearer, steadier rhyme at the line ends', wholeOnly: false },
  { slug: 'simpler', label: 'Simpler words', help: 'Plain words that are easy to sing', wholeOnly: false },
];

/** The improvements ticked when the helper opens; unticking them all runs a general polish. */
export const DEFAULT_PICKS: readonly LyricsImprovement[] = ['tighten'];

/** The longest direction the server takes. */
export const NOTE_MAX_CHARS = 300;
/** How much of the style the request carries; the server keeps the same. */
export const STYLE_CHARS = 400;

/** The lyrics helper's panel for a target: what it says, offers and asks the server for. */
export interface HelperView {
  /** The panel's title. */
  title: string;
  /** The run button's words. */
  runLabel: string;
  /** What is being worked on: shown on a sheet so the text stays in sight. */
  quote: { label: string; text: string };
  /** Why there is nothing to run, in the words the panel shows; null when it can run. */
  problem: string | null;
  /** The song has no words for the target to work on (a whole-song Improve, or a Write with nothing to fit): the panel explains where new lyrics come from and offers Fill in. */
  emptySong: boolean;
  /** The improvements the panel offers (none for a Write). */
  options: typeof IMPROVEMENTS;
  /** The line about what the words are written to match. */
  context: string;
}

/** Whether any part of the song has words (a song of only tags is an instrumental). */
export function songHasWords(song: Song): boolean {
  return hasWords(song.lead) || song.blocks.some(hasWords);
}

const NOTHING_TO_IMPROVE = 'Nothing to improve yet. New lyrics come from Describe it; this helper polishes the words that are there.';
const NOTHING_TO_IMPROVE_IN_TAGS = 'Nothing to improve yet: your sections have no words. Fill in writes lyrics into them from Describe it; this helper polishes the words that are there.';
const NOTHING_TO_FIT = 'There are no words yet. New lyrics come from Describe it; Write fills a section to match the rest of the song.';
const NOTHING_TO_FIT_IN_TAGS = 'There are no words yet. Fill in writes lyrics into your sections from Describe it; Write fills a section to match the rest of the song.';

function contextLine(style: string, others: number | null): string {
  const cleaned = style.replace(/\s+/g, ' ').trim();
  if (others !== null) {
    const known = [cleaned === '' ? null : 'the style', others === 0 ? null : `the other ${others} ${others === 1 ? 'section' : 'sections'}`].filter((part) => part !== null);
    return known.length === 0 ? 'Works on this section alone.' : `Knows ${known.join(' and ')}.`;
  }
  return cleaned === '' ? 'No style written yet: the words follow the lyrics.' : `Written to match your style: ${cleaned.length > 64 ? `${cleaned.slice(0, 63)}…` : cleaned}`;
}

/** What the panel shows for *target* on *lyrics* with the open *style*. */
export function helperView(lyrics: string, target: LyricsTarget, style: string): HelperView {
  const song = parseSong(lyrics);
  const { section } = target;
  if (section === null) {
    const words = songHasWords(song);
    return {
      title: 'Improve lyrics',
      runLabel: 'Improve lyrics',
      quote: { label: 'Lyrics', text: lyrics.trim().replace(/\n\s*\n/g, '\n') },
      problem: words ? null : song.blocks.length > 0 ? NOTHING_TO_IMPROVE_IN_TAGS : NOTHING_TO_IMPROVE,
      emptySong: !words,
      options: IMPROVEMENTS,
      context: contextLine(style, null),
    };
  }
  const block = song.blocks[section];
  const base = { options: target.write ? [] : IMPROVEMENTS.filter((item) => !item.wholeOnly), emptySong: false };
  if (block === undefined) {
    return { ...base, title: 'Section', runLabel: 'Improve', quote: { label: 'Section', text: '' }, problem: 'That section is not in the song any more.', context: '' };
  }
  const name = tagName(block.tag) || 'Section';
  const others = song.blocks.length - 1;
  const rest = songHasWords({ lead: song.lead, blocks: song.blocks.filter((_, at) => at !== section) });
  if (target.write) {
    const sections = { ...base, title: `Write ${name}`, runLabel: `Write ${name.toLowerCase()}`, quote: { label: name, text: 'No words yet' }, context: contextLine(style, others) };
    if (hasWords(block)) return { ...sections, problem: 'This section has words now. Use Improve to change them.' };
    // Nothing in the song to fit: the panel offers Fill in, which writes lyrics into the sections the song already has.
    return rest ? { ...sections, problem: null } : { ...sections, problem: song.blocks.length > 0 ? NOTHING_TO_FIT_IN_TAGS : NOTHING_TO_FIT, emptySong: true };
  }
  return {
    ...base,
    title: `Improve ${name}`,
    runLabel: `Improve ${name.toLowerCase()}`,
    quote: { label: name, text: block.text },
    problem: hasWords(block) ? null : 'This section has no words yet. Use Write to fill it.',
    context: contextLine(style, others),
  };
}

/** The request for *target*: the picks and the direction for an improvement, the direction alone for a write. */
export function lyricsRequest(model: string, lyrics: string, style: string, target: LyricsTarget, picks: readonly LyricsImprovement[], note: string): LyricsEnhanceRequest {
  const direction = note.trim() === '' ? {} : { note: note.trim() };
  const settings = target.write
    ? { action: 'write' as const, section: target.section, ...direction }
    : {
        improve: IMPROVEMENTS.filter((item) => picks.includes(item.slug) && !(item.wholeOnly && target.section !== null)).map((item) => item.slug),
        ...direction,
        ...(target.section === null ? {} : { section: target.section }),
      };
  const styled = style.trim().slice(0, STYLE_CHARS);
  return { mode: 'audio', task: 'lyrics', model, kind: 'music', prompt: lyrics, settings, ...(styled === '' ? {} : { context: { style: styled } }) };
}
