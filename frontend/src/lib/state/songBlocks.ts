import { isSectionLine } from './musicMode';

/**
 * Lyrics as section blocks: a view over the tagged lyrics text, never a second copy of it. `parseSong` cuts the text at its
 * `[section]` lines and `serializeSong` joins the pieces again, so `serializeSong(parseSong(text))` is the text for every input
 * (apart from CRLF line ends, which read as LF). Words before the first tag, tags the app does not know, words on a tag's own
 * line, blank lines and trailing newlines all stay where they are.
 */

/** The section names the Add menu offers: the ones MiniMax and YuE2 know. New blocks write them in title case. */
export const SECTION_NAMES = ['Intro', 'Verse', 'Pre-Chorus', 'Chorus', 'Post-Chorus', 'Bridge', 'Solo', 'Instrumental', 'Outro'] as const;

/** The words under one `[section]` line, or before the first. */
export interface SongBody {
  /** The lines as written, without the blank lines that close them. Empty for an instrumental section. */
  text: string;
  /** The blank lines that close the body: the gap before the next tag, or the end of the text. */
  blank: number;
}

/** One section: its tag line and the words under it. */
export interface SongBlock extends SongBody {
  /** The tag line as written, such as `[Verse]` (spaces around it kept). `tagName` reads its name. */
  tag: string;
}

/** The lyrics as sections. */
export interface Song {
  /** The words before the first tag; empty when the text starts at a tag. */
  lead: SongBody;
  blocks: SongBlock[];
}

/** The blank lines a block gets after it when the song has no gap of its own to copy. */
const DEFAULT_BLANK = 1;

function splitBody(lines: string[]): SongBody {
  let end = lines.length;
  while (end > 0 && lines[end - 1] === '') end -= 1;
  return { text: lines.slice(0, end).join('\n'), blank: lines.length - end };
}

function bodyLines({ text, blank }: SongBody): string[] {
  return [...(text === '' ? [] : text.split('\n')), ...Array<string>(blank).fill('')];
}

/** Cut *text* into the words before its first `[section]` line and one block per section. Total: any string parses. */
export function parseSong(text: string): Song {
  const lines = text.replace(/\r\n/g, '\n').split('\n');
  const lead: string[] = [];
  const groups: { tag: string; lines: string[] }[] = [];
  for (const line of lines) {
    if (isSectionLine(line)) groups.push({ tag: line, lines: [] });
    else (groups.at(-1)?.lines ?? lead).push(line);
  }
  // Empty text has one empty line, which is nothing, not a blank line to keep.
  const empty = groups.length === 0 && lead.length === 1 && lead[0] === '';
  return { lead: empty ? { text: '', blank: 0 } : splitBody(lead), blocks: groups.map((group) => ({ tag: group.tag, ...splitBody(group.lines) })) };
}

/** Join a song into its lyrics text: the inverse of `parseSong`. */
export function serializeSong(song: Song): string {
  return [...bodyLines(song.lead), ...song.blocks.flatMap((block) => [block.tag, ...bodyLines(block)])].join('\n');
}

/** The name in a tag line: `Verse` for `[Verse]`. */
export function tagName(tag: string): string {
  return tag.trim().slice(1, -1).trim();
}

/** The tag line that names a section: `[Verse]`. Brackets and line breaks cannot be part of a name; null when nothing is left. */
function tagFor(name: string): string | null {
  const clean = name.replace(/[[\]\r\n]/g, ' ').replace(/\s+/g, ' ').trim();
  return clean === '' ? null : `[${clean}]`;
}

/** Whether a block has words (a line that is not blank); a block without any makes an instrumental section. */
export function hasWords(block: SongBody): boolean {
  return block.text.trim() !== '';
}

/** The number of non-blank lines of a block. */
export function lineCount(block: SongBody): number {
  return block.text.split('\n').filter((line) => line.trim() !== '').length;
}

/**
 * Give the blocks of *next* the blank lines they need: the one that ends the song keeps the song's closing blank lines, and a
 * block that was the last of *previous* and is not any more gets the gap the song puts between its other blocks.
 */
function restitch(previous: SongBlock[], next: SongBlock[]): SongBlock[] {
  const wasLast = previous.at(-1);
  const closing = wasLast?.blank ?? 0;
  const gap = previous.length > 1 ? previous[0].blank : DEFAULT_BLANK;
  return next.map((block, index) => ({ ...block, blank: index === next.length - 1 ? closing : block === wasLast ? gap : block.blank }));
}

/** Words directly before a tag are separated from it by a blank line. */
function leadBeforeBlocks(lead: SongBody, blocks: SongBlock[]): SongBody {
  return blocks.length > 0 && lead.text !== '' && lead.blank === 0 ? { ...lead, blank: 1 } : lead;
}

function withBlocks(song: Song, blocks: SongBlock[]): Song {
  const next = restitch(song.blocks, blocks);
  // Words with no tag after them end the text: the blank line that parted them from the last tag goes with it.
  const lead = next.length === 0 && song.blocks.length > 0 && song.lead.text !== '' ? { ...song.lead, blank: 0 } : leadBeforeBlocks(song.lead, next);
  return { lead, blocks: next };
}

function inRange(song: Song, index: number): boolean {
  return Number.isInteger(index) && index >= 0 && index < song.blocks.length;
}

/**
 * Replace the words before the first tag with what was typed. Newlines at the end of *typed* are the textarea's own (Enter at
 * the end of the words): the blank lines that close the body are kept in `blank`, so they are not counted a second time.
 */
export function setLead(song: Song, typed: string): Song {
  const text = typed.replace(/\n+$/, '');
  if (text === song.lead.text) return song;
  return { ...song, lead: leadBeforeBlocks({ ...song.lead, text }, song.blocks) };
}

/**
 * Replace the words under block *index* with what was typed. Newlines at the end of *typed* are the textarea's own and are
 * not kept (see `setLead`); so are the ones at its start in a block that had no words, which the user typed to move below its tag.
 */
export function setText(song: Song, index: number, typed: string): Song {
  if (!inRange(song, index)) return song;
  const current = song.blocks[index].text;
  const text = (current === '' ? typed.replace(/^\n+/, '') : typed).replace(/\n+$/, '');
  if (current === text) return song;
  return { ...song, blocks: song.blocks.map((block, at) => (at === index ? { ...block, text } : block)) };
}

/** Rename block *index* to *name*; an empty name changes nothing. */
export function setTag(song: Song, index: number, name: string): Song {
  const tag = tagFor(name);
  if (tag === null || !inRange(song, index)) return song;
  return { ...song, blocks: song.blocks.map((block, at) => (at === index ? { ...block, tag } : block)) };
}

function swap(song: Song, from: number, to: number): Song {
  if (!inRange(song, from) || !inRange(song, to)) return song;
  const blocks = [...song.blocks];
  [blocks[from], blocks[to]] = [blocks[to], blocks[from]];
  return withBlocks(song, blocks);
}

/** Move block *index* one place up. The first block stays. */
export function moveUp(song: Song, index: number): Song {
  return swap(song, index, index - 1);
}

/** Move block *index* one place down. The last block stays. */
export function moveDown(song: Song, index: number): Song {
  return swap(song, index, index + 1);
}

/** Copy block *index* (its tag and words) below itself. */
export function duplicate(song: Song, index: number): Song {
  if (!inRange(song, index)) return song;
  return withBlocks(song, [...song.blocks.slice(0, index + 1), { ...song.blocks[index] }, ...song.blocks.slice(index + 1)]);
}

/** Add an empty block named *name* after block *after*; -1 adds it before the first. A name with nothing in it changes nothing. */
export function add(song: Song, after: number, name: string): Song {
  const tag = tagFor(name);
  if (tag === null || !Number.isInteger(after) || after < -1 || after >= song.blocks.length) return song;
  return withBlocks(song, [...song.blocks.slice(0, after + 1), { tag, text: '', blank: DEFAULT_BLANK }, ...song.blocks.slice(after + 1)]);
}

/** Remove block *index* with its words. */
export function remove(song: Song, index: number): Song {
  if (!inRange(song, index)) return song;
  return withBlocks(song, song.blocks.filter((_, at) => at !== index));
}
