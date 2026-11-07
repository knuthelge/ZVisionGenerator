import { describe, expect, it } from 'vitest';

import { insertAt, sentenceStart, wordsAround } from './caretInsert';

describe('insertAt', () => {
  it('puts the snippet in as it is, replacing the range, with the caret after it', () => {
    expect(insertAt('A $ba rest', 2, 5, '$bag')).toEqual({ text: 'A $bag rest', caret: 6 });
    expect(insertAt('abc', 1, 1, 'X')).toEqual({ text: 'aXbc', caret: 2 });
  });

  it('spaces a snippet off the word before it, and not after whitespace or at the start', () => {
    expect(insertAt('Well', 4, 4, 'haha', { before: true })).toEqual({ text: 'Well haha', caret: 9 });
    expect(insertAt('Well ', 5, 5, 'haha', { before: true }).text).toBe('Well haha');
    expect(insertAt('', 0, 0, 'haha', { before: true })).toEqual({ text: 'haha', caret: 4 });
  });

  it('spaces a snippet off the word after it, and not before punctuation or white space', () => {
    expect(insertAt('Hello there', 5, 5, '…', { after: true })).toEqual({ text: 'Hello… there', caret: 6 });
    expect(insertAt('Hello, there', 5, 5, '…', { after: true }).text).toBe('Hello…, there');
    expect(insertAt('Hello there', 6, 6, '…', { after: true })).toEqual({ text: 'Hello … there', caret: 8 });
    expect(insertAt('Hello 2', 5, 6, 'x', { after: true }).text).toBe('Hellox 2');
  });

  it('replaces a selection', () => {
    expect(insertAt('say wrong thing', 4, 9, 'haha', { before: true, after: true })).toEqual({ text: 'say haha thing', caret: 8 });
  });
});

describe('sentenceStart', () => {
  const text = 'First one. Second one! Third\nFourth “quoted.” Fifth';

  it.each([
    [0, 0],
    [5, 0],
    [9, 0],
    [10, 0],
    [11, 11],
    [15, 11],
    [22, 11],
    [23, 23],
    [26, 23],
    [29, 29],
    [34, 29],
  ])('puts the sentence start of caret %i at %i', (caret, expected) => {
    expect(sentenceStart(text, caret)).toBe(expected);
  });

  it('starts the sentence after a closing quote and its space', () => {
    expect(sentenceStart(text, text.indexOf('Fifth') + 2)).toBe(text.indexOf('Fifth'));
  });

  it('skips the white space that opens the text, and ends at the text for an empty last sentence', () => {
    expect(sentenceStart('  \n Hello there', 8)).toBe(4);
    expect(sentenceStart('Done. ', 6)).toBe(6);
    expect(sentenceStart('', 0)).toBe(0);
    expect(sentenceStart('   ', 3)).toBe(3);
  });

  it('does not end a sentence at a decimal point or an abbreviation without a space', () => {
    expect(sentenceStart('It costs 3.5 euros.', 15)).toBe(0);
  });
});

describe('wordsAround', () => {
  it('shows up to five words each side of the caret and marks what is cut', () => {
    expect(wordsAround('one two three four five six seven eight nine ten eleven twelve', 28)).toBe('…two three four five six | seven eight nine ten eleven…');
  });

  it('shows a short text whole, and a caret at either end', () => {
    expect(wordsAround('Hello there', 5)).toBe('Hello | there');
    expect(wordsAround('Hello there', 0)).toBe('| Hello there');
    expect(wordsAround('Hello there', 11)).toBe('Hello there |');
  });

  it('is empty without words', () => {
    expect(wordsAround('  \n', 1)).toBe('');
  });
});
