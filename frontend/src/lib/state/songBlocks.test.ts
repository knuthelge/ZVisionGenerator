import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

import { musicMode } from './musicMode';
import { SECTION_NAMES, add, duplicate, hasWords, lineCount, moveDown, moveUp, parseSong, remove, serializeSong, setLead, setTag, setText, tagName, type Song } from './songBlocks';

interface Case {
  name: string;
  lyrics: string;
  default_sections: string;
}

// The lyrics the Python rule (`music_mode`) is tested against: every one must come back from blocks as it went in.
const MUSIC_CASES = JSON.parse(readFileSync(resolve(process.cwd(), '..', 'tests', 'audio', 'music_mode_cases.json'), 'utf8')) as Case[];

const TRICKY = [
  '',
  '\n',
  '\n\n\n',
  '   ',
  'la la la',
  'la la\n',
  'words before\n\n[Verse]\nfirst\n\n[Chorus]\nsecond',
  'words before\n[Verse]\nfirst',
  '[Pre-Chorus]\nbuilding\n[Chorus]\nbig',
  '[Verse]',
  '[Verse]\n',
  '[Verse]\n\n',
  '[A]\n[B]',
  '[A]\n\n[B]\n\n\n[C]\n',
  '\n[A]\n',
  '  [Intro]  \n\t[Verse]\t\nla',
  '[Chorus] oh oh\nmore',
  'x [Chorus]\n[Bridge] y',
  '[Verse\nstill words\n]',
  '[]\n[ ]\n[[x]]\n[a][b]',
  '[Verse 2]\r\nline one\r\n\r\n[Chorus]\r\nline two\r\n',
  'a\rb\r[Verse]\rc',
  'ÆØÅ [verse]\n日本語\n [chorus] \n🎵',
  '[verse]\n[chorus]\n[verse]\n[chorus]',
  '\n\n[Verse]\n\n\nla\n\n\n',
];

/** What the text reads as once CRLF line ends are made LF: the one change a round trip makes. */
const lf = (text: string): string => text.replace(/\r\n/g, '\n');

function randomTexts(count: number): string[] {
  // A small deterministic generator, so a failure names the same text on every run.
  let seed = 20260611;
  const next = (limit: number): number => {
    seed = (seed * 1664525 + 1013904223) % 2 ** 32;
    return seed % limit;
  };
  const pieces = ['[Verse]', '[chorus]', '[Pre-Chorus]', '[x', 'y]', '[]', 'la la', 'words', ' ', '\t', '\n', '\n', '\n', '\r\n', '\r', '[a] b', '  [Bridge]  ', 'é'];
  return Array.from({ length: count }, () => Array.from({ length: 1 + next(14) }, () => pieces[next(pieces.length)]).join(next(3) === 0 ? '' : '\n'));
}

// tests/fixtures/song_parse_cases.json is also checked by pytest against the Python `parse_song` of the lyrics helper: a section index must mean the same block on both sides.
const PARSE_CASES = JSON.parse(readFileSync(resolve(process.cwd(), '..', 'tests', 'fixtures', 'song_parse_cases.json'), 'utf8')) as { text: string; song: Song }[];

describe('parseSong against the shared cases', () => {
  it.each(PARSE_CASES.map((entry) => [JSON.stringify(entry.text), entry] as const))('cuts %s as the fixture says', (_label, entry) => {
    expect(parseSong(entry.text)).toEqual(entry.song);
  });
});

describe('parseSong and serializeSong', () => {
  const inputs = [...MUSIC_CASES.flatMap((entry) => [entry.lyrics, entry.default_sections]), ...TRICKY, ...randomTexts(400)];

  it.each(inputs.map((text) => [JSON.stringify(text), text] as const))('gives back %s', (_label, text) => {
    expect(serializeSong(parseSong(text))).toBe(lf(text));
  });

  it('cuts words before the first tag, unknown tags and blank lines into blocks', () => {
    const song = parseSong('Intro words\n\n[Pre-Chorus]\nbuilding\n\n[Hook 2]\nla\n');
    expect(song.lead).toEqual({ text: 'Intro words', blank: 1 });
    expect(song.blocks).toEqual([
      { tag: '[Pre-Chorus]', text: 'building', blank: 1 },
      { tag: '[Hook 2]', text: 'la', blank: 1 },
    ]);
  });

  it('reads a tag with words beside it as words, not as a section', () => {
    const song = parseSong('[Verse]\nla\n[Chorus] oh oh');
    expect(song.blocks.map((block) => tagName(block.tag))).toEqual(['Verse']);
    expect(song.blocks[0].text).toBe('la\n[Chorus] oh oh');
  });

  it('makes a list of tags a list of empty blocks, as an instrumental is', () => {
    const song = parseSong('[Intro]\n\n[Verse]\n\n[Outro]');
    expect(song.blocks.map((block) => [tagName(block.tag), hasWords(block)])).toEqual([['Intro', false], ['Verse', false], ['Outro', false]]);
    expect(musicMode(serializeSong(song), '')?.kind).toBe('instrumental');
  });

  it('finds the same tags as the song-or-instrumental rule on every shared case', () => {
    for (const entry of MUSIC_CASES) {
      const song = parseSong(entry.lyrics);
      const sections = musicMode(entry.lyrics, '')?.sectionCount ?? 0;
      // A tag with words beside it is words to the blocks and a tag to the rule, so the blocks never find more.
      expect(song.blocks.length).toBeLessThanOrEqual(sections);
    }
  });

  it('counts a block\'s lines', () => {
    expect(lineCount({ text: 'a\n\n b \n', blank: 0 })).toBe(2);
    expect(lineCount({ text: '', blank: 1 })).toBe(0);
  });

  it('has no blocks for empty text and one empty body', () => {
    expect(parseSong('')).toEqual({ lead: { text: '', blank: 0 }, blocks: [] });
  });
});

describe('editing blocks', () => {
  const text = 'Words first\n\n[Verse]\nla la\nlo lo\n\n[Chorus]\nhey\n\n[Outro]';
  const song = parseSong(text);
  const names = (value: ReturnType<typeof parseSong>): string[] => value.blocks.map((block) => tagName(block.tag));

  it('changes the words of one block and nothing else', () => {
    expect(serializeSong(setText(song, 1, 'hey\nho'))).toBe('Words first\n\n[Verse]\nla la\nlo lo\n\n[Chorus]\nhey\nho\n\n[Outro]');
    expect(serializeSong(setLead(song, 'New first'))).toBe(text.replace('Words first', 'New first'));
  });

  it('leaves the song as it is when the words do not change', () => {
    expect(setLead(song, 'Words first')).toBe(song);
    expect(setText(song, 0, 'la la\nlo lo')).toBe(song);
  });

  it('renames a section to any name and refuses an empty one', () => {
    expect(serializeSong(setTag(song, 0, 'Hook 2'))).toBe(text.replace('[Verse]', '[Hook 2]'));
    expect(serializeSong(setTag(song, 0, ' [Pre\n-Chorus] '))).toBe(text.replace('[Verse]', '[Pre -Chorus]'));
    expect(setTag(song, 0, '  ')).toBe(song);
    expect(setTag(song, 0, '[]')).toBe(song);
  });

  it('moves a block up and down, with the first and last staying put', () => {
    expect(names(moveUp(song, 1))).toEqual(['Chorus', 'Verse', 'Outro']);
    expect(names(moveDown(song, 0))).toEqual(['Chorus', 'Verse', 'Outro']);
    expect(moveUp(song, 0)).toBe(song);
    expect(moveDown(song, 2)).toBe(song);
    expect(serializeSong(moveDown(song, 0))).toBe('Words first\n\n[Chorus]\nhey\n\n[Verse]\nla la\nlo lo\n\n[Outro]');
  });

  it('keeps the way the text ends when the last block moves', () => {
    const open = parseSong('[A]\nx\n\n[B]\ny\n');
    expect(serializeSong(moveUp(open, 1))).toBe('[B]\ny\n\n[A]\nx\n');
    const tight = parseSong('[A]\nx\n[B]\ny');
    expect(serializeSong(moveUp(tight, 1))).toBe('[B]\ny\n[A]\nx');
  });

  it('puts a move back with the opposite move, in a song whose gaps are even', () => {
    for (const input of [text, '[A]\nx\n[B]\ny\n[C]', '[A]\n\n[B]\n\n[C]\n', '[Intro]\n\n[Verse]\nla\n\n[Outro]']) {
      const parsed = parseSong(input);
      for (let index = 1; index < parsed.blocks.length; index += 1) {
        expect(serializeSong(moveDown(moveUp(parsed, index), index - 1))).toBe(lf(input));
      }
    }
  });

  it('copies a block below itself', () => {
    const copy = duplicate(song, 1);
    expect(names(copy)).toEqual(['Verse', 'Chorus', 'Chorus', 'Outro']);
    expect(serializeSong(copy)).toBe('Words first\n\n[Verse]\nla la\nlo lo\n\n[Chorus]\nhey\n\n[Chorus]\nhey\n\n[Outro]');
    expect(serializeSong(duplicate(parseSong('[A]\nx'), 0))).toBe('[A]\nx\n\n[A]\nx');
  });

  it('adds an empty block after a block, at the start or at the end, in title case', () => {
    expect(serializeSong(add(song, 0, 'Pre-Chorus'))).toBe('Words first\n\n[Verse]\nla la\nlo lo\n\n[Pre-Chorus]\n\n[Chorus]\nhey\n\n[Outro]');
    expect(serializeSong(add(song, -1, 'Intro'))).toBe('Words first\n\n[Intro]\n\n[Verse]\nla la\nlo lo\n\n[Chorus]\nhey\n\n[Outro]');
    expect(serializeSong(add(song, 2, 'Bridge'))).toBe(`${text}\n\n[Bridge]`);
    expect(add(song, 9, 'Bridge')).toBe(song);
  });

  it('builds a song from nothing, as a list of sections', () => {
    let built = parseSong('');
    for (const name of ['Intro', 'Verse', 'Outro']) built = add(built, built.blocks.length - 1, name);
    expect(serializeSong(built)).toBe('[Intro]\n\n[Verse]\n\n[Outro]');
    expect(musicMode(serializeSong(built), '')).toMatchObject({ kind: 'instrumental', sectionCount: 3 });
  });

  it('puts every tag on its own line, after words that had no tag', () => {
    expect(serializeSong(add(parseSong('la la'), 0 - 1, 'Verse'))).toBe('la la\n\n[Verse]');
    expect(serializeSong(remove(add(parseSong('la la'), -1, 'Verse'), 0))).toBe('la la');
  });

  it('does not grow the gaps or the end of the text when words are typed after a trailing newline', () => {
    // The component writes each edit, then parses the text again; the textarea keeps the newline it was typed with.
    const again = (value: ReturnType<typeof parseSong>): ReturnType<typeof parseSong> => parseSong(serializeSong(value));
    const middle = parseSong('[A]\nx\n\n[B]\ny');
    expect(serializeSong(setText(again(setText(middle, 0, 'x\n')), 0, 'x\nz'))).toBe('[A]\nx\nz\n\n[B]\ny');
    const last = parseSong('[A]\nx');
    expect(serializeSong(setText(again(setText(again(setText(last, 0, 'x\n')), 0, 'x\n\n')), 0, 'x\n\nz'))).toBe('[A]\nx\n\nz');
    const lead = parseSong('lead\n\n[A]\nx');
    expect(serializeSong(setLead(again(setLead(lead, 'lead\n')), 'lead\nmore'))).toBe('lead\nmore\n\n[A]\nx');
  });

  it('drops the newline typed under a tag with no words, so the words start on its next line', () => {
    expect(serializeSong(setText(parseSong('[A]'), 0, '\nla'))).toBe('[A]\nla');
    expect(serializeSong(setText(parseSong('[A]\n\nla'), 0, '\nla la'))).toBe('[A]\n\nla la');
  });

  it('removes a block with its words', () => {
    expect(serializeSong(remove(song, 1))).toBe('Words first\n\n[Verse]\nla la\nlo lo\n\n[Outro]');
    expect(serializeSong(remove(song, 2))).toBe('Words first\n\n[Verse]\nla la\nlo lo\n\n[Chorus]\nhey');
    expect(serializeSong(remove(parseSong('[A]'), 0))).toBe('');
    expect(remove(song, 7)).toBe(song);
  });

  it('keeps every line of the text through any run of edits', () => {
    // Each edit moves, copies or removes whole blocks; nothing is cut inside one.
    let current = song;
    current = moveUp(duplicate(add(current, 1, 'Solo'), 0), 2);
    current = remove(current, 0);
    const lines = serializeSong(current).split('\n').filter((line) => line !== '');
    expect(lines).toContain('Words first');
    expect(lines).toContain('hey');
    expect(lines.filter((line) => line === '[Verse]').length).toBe(1);
  });

  it('offers the sections MiniMax knows', () => {
    expect(SECTION_NAMES).toEqual(['Intro', 'Verse', 'Pre-Chorus', 'Chorus', 'Post-Chorus', 'Bridge', 'Solo', 'Instrumental', 'Outro']);
  });
});
