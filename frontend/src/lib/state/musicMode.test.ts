import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

import { isSectionLine, musicMode, musicStatus, sectionTags } from './musicMode';

interface Case {
  name: string;
  lyrics: string;
  default_sections: string;
  expected: { cot: string; lyrics: string; kind: 'song' | 'instrumental'; section_count: number };
}

// The cases the Python rule (`music_mode`) is tested against: the two implementations must agree on every one.
const CASES = JSON.parse(readFileSync(resolve(process.cwd(), '..', 'tests', 'audio', 'music_mode_cases.json'), 'utf8')) as Case[];

describe('musicMode', () => {
  it.each(CASES.map((entry) => [entry.name, entry] as const))('%s', (_name, entry) => {
    expect(musicMode(entry.lyrics, entry.default_sections)).toEqual({ kind: entry.expected.kind, lyrics: entry.expected.lyrics, sectionCount: entry.expected.section_count });
  });

  it('is undecided when the lyrics are empty and the default list holds no tag', () => {
    expect(musicMode('', 'no tags here')).toBeNull();
    expect(musicMode('words', 'no tags here')?.kind).toBe('song');
  });

  it('keeps tags in order and as written', () => {
    expect(sectionTags('[Intro] la [Verse 2]\n[x]')).toEqual(['[Intro]', '[Verse 2]', '[x]']);
  });
});

describe('isSectionLine', () => {
  it('is a line of one tag, with spaces allowed around it', () => {
    expect(['[verse]', '  [Chorus 2] ', '[Pre-Chorus]'].map(isSectionLine)).toEqual([true, true, true]);
    expect(['[Chorus] oh', 'oh [Chorus]', '[a][b]', '[unclosed', '[]', '[[x]]', ''].map(isSectionLine)).toEqual([false, false, false, false, false, false, false]);
  });
});

describe('musicStatus', () => {
  it('says what the take will be and how its length is set', () => {
    expect(musicStatus(musicMode('[verse]\nla', '[intro]'))).toBe('Song · length follows the lyrics');
    expect(musicStatus(musicMode('', '[intro] [verse] [chorus]'))).toBe('Instrumental · 3 sections · add or remove section tags to change its length');
    expect(musicStatus(musicMode('[intro]', ''))).toBe('Instrumental · 1 section · add or remove section tags to change its length');
  });

  it('says nothing without a mode', () => {
    expect(musicStatus(null)).toBe('');
  });
});
