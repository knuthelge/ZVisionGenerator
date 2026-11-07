import { describe, expect, it } from 'vitest';
import { DEFAULT_PICKS, IMPROVEMENTS, helperView, lyricsRequest, songHasWords } from './lyricsHelper';
import { parseSong } from './songBlocks';

const SONG = '[Verse]\nThe porch light flickers\nI keep your coat\n\n[Chorus]\nCome back home\n\n[Bridge]\n\n[Outro]\n';

describe('the helper\'s panel', () => {
  it('improves the whole song when it has words', () => {
    const view = helperView(SONG, { section: null, write: false }, 'warm folk');

    expect(view).toMatchObject({ title: 'Improve lyrics', runLabel: 'Improve lyrics', problem: null, emptySong: false });
    expect(view.options.map((item) => item.slug)).toEqual(['tighten', 'fit', 'rhymes', 'simpler']);
    expect(view.options.map((item) => item.label)).toEqual(['Tighten', 'Fit the sections', 'Smooth rhymes', 'Simpler words']);
    expect(view.quote).toEqual({ label: 'Lyrics', text: '[Verse]\nThe porch light flickers\nI keep your coat\n[Chorus]\nCome back home\n[Bridge]\n[Outro]' });
    expect(view.context).toBe('Written to match your style: warm folk');
  });

  it.each(['', '  \n', '[Verse]\n\n[Chorus]\n'])('says there is nothing to improve in an empty song (%j)', (lyrics) => {
    const view = helperView(lyrics, { section: null, write: false }, '');

    expect(view.emptySong).toBe(true);
    expect(view.problem).not.toBeNull();
    expect(view.problem).toMatch(/Describe it/);
  });

  it('opens with Tighten picked', () => {
    expect(DEFAULT_PICKS).toEqual(['tighten']);
    expect(IMPROVEMENTS.map((item) => item.slug)).toContain(DEFAULT_PICKS[0]);
  });

  it('improves one section with the options that make sense for a section', () => {
    const view = helperView(SONG, { section: 1, write: false }, '');

    expect(view).toMatchObject({ title: 'Improve Chorus', runLabel: 'Improve chorus', problem: null });
    expect(view.options.map((item) => item.slug)).toEqual(['tighten', 'rhymes', 'simpler']);
    expect(view.quote).toEqual({ label: 'Chorus', text: 'Come back home' });
    expect(view.context).toBe('Knows the other 3 sections.');
  });

  it('writes an empty section to fit the rest, with no improvements to pick', () => {
    const view = helperView(SONG, { section: 2, write: true }, 'warm folk');

    expect(view).toMatchObject({ title: 'Write Bridge', runLabel: 'Write bridge', problem: null, options: [] });
    expect(view.context).toBe('Knows the style and the other 3 sections.');
  });

  it.each(['[Verse]\n\n[Chorus]\n', '[Verse]\n\n[Chorus]\n\n[Bridge]'])('opens Write on a song with no words as an empty song, so the panel offers Fill in (%j)', (lyrics) => {
    const view = helperView(lyrics, { section: 1, write: true }, '');

    expect(view.emptySong).toBe(true);
    expect(view.problem).toMatch(/^There are no words yet\./);
    expect(view.problem).toMatch(/Describe it/);
  });

  it('cannot improve an empty section or write over one that has words', () => {
    expect(helperView(SONG, { section: 2, write: false }, '').problem).toMatch(/no words yet.*Write/);
    expect(helperView(SONG, { section: 0, write: true }, '').problem).toMatch(/has words now.*Improve/);
  });

  it('says so when the section has gone', () => {
    expect(helperView(SONG, { section: 7, write: false }, '').problem).toMatch(/not in the song any more/);
  });

  it('counts words before the first tag as words', () => {
    expect(songHasWords(parseSong('just some words'))).toBe(true);
    expect(songHasWords(parseSong('[Verse]\n'))).toBe(false);
  });

  it('offers Fit the sections on the whole song only', () => {
    expect(IMPROVEMENTS.filter((item) => item.wholeOnly).map((item) => item.slug)).toEqual(['fit']);
  });
});

describe('the request', () => {
  it('improves the whole song with the picks in the panel\'s order, the direction and the style', () => {
    const request = lyricsRequest('yue2', SONG, '  warm folk ', { section: null, write: false }, ['simpler', 'tighten'], '  more hopeful ');

    expect(request).toEqual({
      mode: 'audio',
      task: 'lyrics',
      model: 'yue2',
      kind: 'music',
      prompt: SONG,
      settings: { improve: ['tighten', 'simpler'], note: 'more hopeful' },
      context: { style: 'warm folk' },
    });
  });

  it('sends no direction, style or picks it was not given', () => {
    const request = lyricsRequest('yue2', SONG, ' ', { section: null, write: false }, [], '  ');

    expect(request.settings).toEqual({ improve: [] });
    expect(request).not.toHaveProperty('context');
  });

  it('improves one section by its index and leaves out the pick that is for the whole song', () => {
    const request = lyricsRequest('yue2', SONG, '', { section: 1, write: false }, ['fit', 'rhymes'], '');

    expect(request.settings).toEqual({ improve: ['rhymes'], section: 1 });
  });

  it('writes one section with the direction alone', () => {
    const request = lyricsRequest('yue2', SONG, 'x'.repeat(900), { section: 2, write: true }, ['tighten'], 'a hopeful turn');

    expect(request.settings).toEqual({ action: 'write', section: 2, note: 'a hopeful turn' });
    expect(request.context?.style).toHaveLength(400);
  });
});
