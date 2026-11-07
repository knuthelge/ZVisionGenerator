import { describe, expect, it } from 'vitest';

import { AUDIO_SETTINGS } from '../../test-utils/audio';
import { TEXT_SNIPPETS, insertSnippet, markGroups, setDeliveryMark } from './speechMarks';

const marks = AUDIO_SETTINGS.marks.map((mark) => mark.id);
const snippet = (id: string) => TEXT_SNIPPETS.find((item) => item.id === id)!;
const mark = (text: string, caret: number, id: string) => setDeliveryMark(text, caret, id, marks);

describe('markGroups', () => {
  const groups = markGroups(AUDIO_SETTINGS);

  it('lists every served mark once: the emotions by group, the manners, then as set', () => {
    expect(groups.map((group) => group.title)).toEqual(['Bright', 'Warm', 'Thoughtful', 'Heavy', 'Manner', 'Reset']);
    expect(groups.flatMap((group) => group.marks.map((item) => item.id)).sort()).toEqual([...marks].sort());
    expect(groups.find((group) => group.title === 'Manner')?.marks.map((item) => item.label)).toEqual(['Whisper', 'Shout']);
    expect(groups.at(-1)?.marks).toEqual([{ id: 'as set', label: 'As set', hint: 'your settings' }]);
  });

  it('writes an emotion by its label', () => {
    expect(groups[0].marks[0]).toEqual({ id: 'elated', label: 'Elated' });
  });
});

describe('insertSnippet', () => {
  it('puts a pause next to the word before it and spaces it off the word after', () => {
    expect(insertSnippet('Well I think so', 4, 4, snippet('pause'))).toEqual({ text: 'Well… I think so', caret: 5 });
    expect(insertSnippet('Well I think so', 5, 5, snippet('pause')).text).toBe('Well … I think so');
    expect(insertSnippet('Well, I think so', 4, 4, snippet('pause')).text).toBe('Well…, I think so');
  });

  it('spaces a laugh off the words on both sides', () => {
    expect(insertSnippet('That was funny', 4, 4, snippet('laugh'))).toEqual({ text: 'That haha was funny', caret: 9 });
    expect(insertSnippet('That was funny.', 15, 15, snippet('laugh')).text).toBe('That was funny. haha');
  });

  it('replaces the selection', () => {
    expect(insertSnippet('That was funny', 5, 8, snippet('laugh')).text).toBe('That haha funny');
  });

  it('offers a pause and a laugh and no long pause', () => {
    expect(TEXT_SNIPPETS.map((item) => item.text)).toEqual(['…', 'haha']);
  });
});

describe('setDeliveryMark', () => {
  it('puts the mark at the start of the sentence holding the caret, whichever word the caret is in', () => {
    const text = 'I was fine. You never called me. We spoke later.';
    expect(mark(text, 20, 'sad').text).toBe('I was fine. [sad] You never called me. We spoke later.');
    expect(mark(text, 0, 'sad').text).toBe('[sad] I was fine. You never called me. We spoke later.');
    expect(mark(text, text.length, 'whisper').text).toBe('I was fine. You never called me. [whisper] We spoke later.');
  });

  it('keeps the caret in the same place in the words', () => {
    const text = 'I was fine. You never called me.';
    expect(mark(text, 20, 'sad')).toEqual({ text: 'I was fine. [sad] You never called me.', caret: 26 });
    expect(mark(text, 3, 'sad').caret).toBe(9);
  });

  it('puts the caret after the mark when the text ends there, so typing goes on', () => {
    expect(mark('', 0, 'sad')).toEqual({ text: '[sad] ', caret: 6 });
    expect(mark('Done. ', 6, 'angry')).toEqual({ text: 'Done. [angry] ', caret: 14 });
  });

  it('replaces the marks that already open the sentence, whichever they are', () => {
    expect(mark('[sad] You never called.', 10, 'angry')).toEqual({ text: '[angry] You never called.', caret: 12 });
    expect(mark('Hi. [sad] [Whisper]  Come here.', 15, 'as set').text).toBe('Hi. [as set] Come here.');
    expect(mark('[sad] Hello', 0, 'sad')).toEqual({ text: '[sad] Hello', caret: 0 });
  });

  it('leaves a bracketed word that is not a mark alone', () => {
    expect(mark('[1] Hello', 4, 'sad').text).toBe('[sad] [1] Hello');
    expect(mark('[laughs] Hello', 4, 'sad').text).toBe('[sad] [laughs] Hello');
  });

  it('keeps a line break after the mark', () => {
    expect(mark('Hello.\n[sad]\nWorld', 12, 'angry').text).toBe('Hello.\n[angry]\nWorld');
  });

  it('writes every served mark in the shape the server reads: 2 to 20 letters and spaces, in brackets, then a space', () => {
    for (const id of marks) {
      expect(mark('One. Two.', 6, id).text).toMatch(new RegExp(`^One\\. \\[${id}\\] Two\\.$`));
      expect(id).toMatch(/^[A-Za-z ]{2,20}$/);
    }
  });

  it('starts at the first word when the text opens with white space', () => {
    expect(mark('\n\n  Hello there', 8, 'sad').text).toBe('\n\n  [sad] Hello there');
  });
});
