import { describe, expect, it } from 'vitest';

import { EMPTY_REWRITES, editRewrite, pushRewrite, recordJson, rewriteStepper, scratchRewrite, sentText, stepRewrite, clearRewrites, rewriteStaleReason } from './audioRewrites';
import { VERSION_LIMIT } from './versions';
import type { AudioRewrite, AudioRewriteSource, AudioRewrites } from '$lib/types';

const source = (input = 'rain', model = 'sa3'): AudioRewriteSource => ({ input, model, task: 'sound' });
const rewrite = (text: string, from: AudioRewriteSource | null = source()): AudioRewrite => ({ text, from });

/** The box after the rewrites *texts*, each made from the same text. */
function pushed(...texts: string[]): AudioRewrites {
  return texts.reduce((rewrites, text) => pushRewrite(rewrites, rewrite(text)), EMPTY_REWRITES);
}

describe('pushRewrite', () => {
  it('shows the new rewrite and keeps it as the newest version', () => {
    const rewrites = pushed('Heavy rain.', 'Soft rain.');

    expect(rewrites).toMatchObject({ text: 'Soft rain.', from: source() });
    expect(rewrites.history.map((entry) => entry.text)).toEqual(['Heavy rain.', 'Soft rain.']);
  });

  it(`keeps the last ${VERSION_LIMIT} versions`, () => {
    const rewrites = pushed('1', '2', '3', '4', '5', '6', '7');

    expect(rewrites.history.map((entry) => entry.text)).toEqual(['3', '4', '5', '6', '7']);
  });

  it.each([
    ['own text', source('thunder')],
    ['audio model', source('rain', 'sa4')],
  ])('keeps the earlier versions made from another %s', (_name, other) => {
    const rewrites = pushRewrite(pushed('Heavy rain.', 'Soft rain.'), rewrite('Distant thunder.', other));

    expect(rewrites.history.map((entry) => entry.text)).toEqual(['Heavy rain.', 'Soft rain.', 'Distant thunder.']);
    expect(rewriteStepper(rewrites)).toEqual({ index: 3, count: 3 });
  });

  it('forgets which version was scratched once a new one is written', () => {
    expect(pushRewrite(scratchRewrite(pushed('1', '2')), rewrite('3')).scratched).toBeNull();
  });

  it('moves a Reuse of an earlier version to the newest place, so the stepper goes back to the other version and ends', () => {
    const reused = pushRewrite(pushed('A', 'B'), rewrite('A'));

    expect(reused.history.map((entry) => entry.text)).toEqual(['B', 'A']);
    expect(rewriteStepper(reused)).toEqual({ index: 2, count: 2 });
    const back = stepRewrite(reused, 1);
    expect(back.text).toBe('B');
    expect(rewriteStepper(back)).toEqual({ index: 1, count: 2 });
  });

  it('adds no duplicate when the newest version is reused', () => {
    const reused = pushRewrite(pushed('A', 'B'), rewrite('B'));

    expect(reused.history.map((entry) => entry.text)).toEqual(['A', 'B']);
    expect(rewriteStepper(reused)).toEqual({ index: 2, count: 2 });
  });
});

describe('clearRewrites', () => {
  it('has no Enhanced text and no versions, as after a recent-prompt pick', () => {
    expect(clearRewrites()).toEqual({ text: '', from: null, history: [] });
    expect(rewriteStepper(clearRewrites())).toBeNull();
  });
});

describe('a Music style rewritten for one model', () => {
  const style = (model: string): AudioRewriteSource => ({ input: 'warm folk', model, task: 'style' });
  const written = pushRewrite(EMPTY_REWRITES, rewrite('Folk, warm, acoustic guitar, intimate', style('yue2')));

  it('is out of date once the Music model is switched, and current again when it is switched back', () => {
    expect(rewriteStaleReason(written, 'warm folk', 'yue2', 'Style')).toBeNull();
    expect(rewriteStaleReason(written, 'warm folk', 'sa3', 'Style')).not.toBeNull();
    expect(rewriteStaleReason(written, 'warm folk', 'yue2', 'Style')).toBeNull();
  });

  it('still runs when out of date, with the style it was made from, and keeps the stepper', () => {
    expect(sentText(written, 'warm folk')).toMatchObject({ text: 'Folk, warm, acoustic guitar, intimate', input: 'warm folk' });
    const two = pushRewrite(written, rewrite('Folk, warm, fingerpicked guitar', style('yue2')));
    expect(rewriteStaleReason(two, 'warm folk', 'sa3', 'Style')).not.toBeNull();
    expect(rewriteStepper(two)).toEqual({ index: 2, count: 2 });
  });
});

describe('rewriteStaleReason', () => {
  const rewrites = pushed('Heavy rain.');

  it('says nothing while the rewrite is up to date, for hand-typed text and for no text', () => {
    expect(rewriteStaleReason(rewrites, 'rain', 'sa3', 'Prompt')).toBeNull();
    expect(rewriteStaleReason({ ...rewrites, from: null }, 'other', 'sa4', 'Prompt')).toBeNull();
    expect(rewriteStaleReason(scratchRewrite(rewrites), 'other', 'sa4', 'Prompt')).toBeNull();
    expect(rewriteStaleReason(pushRewrite(EMPTY_REWRITES, rewrite('Dreamy lo-fi.', source(''))), '', 'sa3', 'Prompt')).toBeNull();
  });

  it.each([
    ['the own text changed', 'rain, loud', 'sa3', 'Your Style changed since this was written.'],
    ['the model changed', 'rain', 'sa4', 'The model changed since this was written.'],
    ['both changed', 'rain, loud', 'sa4', 'Your Style and the model changed since this was written.'],
  ])('names what changed when %s', (_name, own, model, reason) => {
    expect(rewriteStaleReason(rewrites, own, model, 'Style')).toBe(reason);
  });
});

describe('rewriteStepper and stepRewrite', () => {
  it('has no stepper with one version', () => {
    expect(rewriteStepper(pushed('1'))).toBeNull();
  });

  it('places the shown text among the versions, and steps to another', () => {
    const rewrites = pushed('1', '2', '3');

    expect(rewriteStepper(rewrites)).toEqual({ index: 3, count: 3 });
    const back = stepRewrite(rewrites, 1);
    expect(back).toMatchObject({ text: '1', from: source() });
    expect(rewriteStepper(back)).toEqual({ index: 1, count: 3 });
    expect(back.history).toBe(rewrites.history);
  });

  it('stays visible while the text is out of date (the stale state is not asked)', () => {
    expect(rewriteStepper(pushed('1', '2'))).toEqual({ index: 2, count: 2 });
  });

  it('hides while Enhance runs, and when the text is hand-typed rather than a version', () => {
    const rewrites = pushed('1', '2');

    expect(rewriteStepper(rewrites, true)).toBeNull();
    expect(rewriteStepper({ ...rewrites, text: 'my own words', from: null })).toBeNull();
  });

  it('changes nothing for a version that does not exist', () => {
    const rewrites = pushed('1', '2');

    expect(stepRewrite(rewrites, 0)).toBe(rewrites);
    expect(stepRewrite(rewrites, 3)).toBe(rewrites);
  });
});

describe('scratchRewrite', () => {
  it('empties the Enhanced text and keeps the versions', () => {
    const rewrites = pushed('1', '2', '3');

    expect(scratchRewrite(rewrites)).toMatchObject({ text: '', from: null, history: rewrites.history, scratched: 2 });
  });

  it('shows "– of N" while the Enhanced text is empty, and ‹ brings back the version that was scratched', () => {
    const middle = stepRewrite(pushed('1', '2', '3'), 2);
    const scratched = scratchRewrite(middle);

    expect(rewriteStepper(scratched)).toEqual({ index: null, count: 3, back: 2 });
    expect(stepRewrite(scratched, 2)).toMatchObject({ text: '2', scratched: null });
    expect(rewriteStepper(stepRewrite(scratched, 2))).toEqual({ index: 2, count: 3 });
  });

  it('shows "– of 1" after Scratch with one version', () => {
    expect(rewriteStepper(scratchRewrite(pushed('only')))).toEqual({ index: null, count: 1, back: 1 });
  });

  it('shows "– of N" only while the text is empty, and not for a box that was never scratched', () => {
    expect(rewriteStepper(pushed('1', '2'))?.index).toBe(2);
    expect(rewriteStepper({ ...pushed('1', '2'), text: '', from: null })).toBeNull();
  });

  it('cannot bring back what was typed by hand, so it has no stepper', () => {
    const typed = editRewrite({ ...pushed('1', '2'), text: '', from: null }, 'by hand');

    expect(rewriteStepper(scratchRewrite(typed))).toBeNull();
  });

  it('hides "– of N" while Enhance runs', () => {
    expect(rewriteStepper(scratchRewrite(pushed('1', '2')), true)).toBeNull();
  });
});

describe('editRewrite', () => {
  const record = { settings: { space: 'forest' }, model: 'owner/llm', format: 'text', task: 'sound' } as const;
  const written = pushRewrite(EMPTY_REWRITES, rewrite('Heavy rain.', { ...source(), record }));

  it('drops the record of a rewrite edited by hand, but keeps what it was made from', () => {
    const edited = editRewrite(written, 'Heavy rain, louder.');

    expect(edited.text).toBe('Heavy rain, louder.');
    expect(edited.from).toEqual(source());
    expect(edited.from).not.toHaveProperty('record');
  });

  it('replaces the shown version\'s text in the history, found by its text before the edit, and drops its record there too', () => {
    const rewrites = pushRewrite(pushRewrite(EMPTY_REWRITES, rewrite('One.', { ...source(), record })), rewrite('Two.', { ...source(), record }));
    const viewing = stepRewrite(rewrites, 1);

    const edited = editRewrite(viewing, 'One, but mine.');

    expect(edited.history.map((entry) => entry.text)).toEqual(['One, but mine.', 'Two.']);
    expect(edited.history[0].from).toEqual(source());
    expect(edited.history[0].from).not.toHaveProperty('record');
    expect(edited.history[1].from).toHaveProperty('record');
    expect(rewriteStepper(edited)).toEqual({ index: 1, count: 2 });
    // The next key finds the same version again.
    expect(editRewrite(edited, 'One, but mine!').history.map((entry) => entry.text)).toEqual(['One, but mine!', 'Two.']);
  });

  it('is a Scratch when the box is emptied by hand: the versions stay and ‹ brings the version back', () => {
    const emptied = editRewrite(pushed('1', '2'), '  ');

    expect(emptied).toMatchObject({ text: '', from: null, scratched: 1 });
    expect(emptied.history).toHaveLength(2);
  });

  it('keeps typed-in text without a source, outside the versions', () => {
    const typed = editRewrite({ ...EMPTY_REWRITES }, 'typed');

    expect(typed).toMatchObject({ text: 'typed', from: null, history: [] });
  });

  it('leaves the versions alone when the text typed over is not one of them', () => {
    const rewrites = { ...pushed('1', '2'), text: 'by hand', from: null };

    expect(editRewrite(rewrites, 'by hand, more').history.map((entry) => entry.text)).toEqual(['1', '2']);
  });
});

describe('sentText', () => {
  const record = { settings: {}, model: 'owner/llm', format: 'text', task: 'sound' } as const;

  it('sends the own text alone when there is no Enhanced text', () => {
    expect(sentText(EMPTY_REWRITES, 'rain')).toEqual({ text: 'rain', input: null, record: null });
  });

  it('sends the rewrite with the text it was made from, even when the own text changed since', () => {
    const rewrites = pushRewrite(EMPTY_REWRITES, rewrite('Heavy rain.', { ...source('rain'), record }));

    expect(sentText(rewrites, 'rain, loud')).toEqual({ text: 'Heavy rain.', input: 'rain', record });
  });

  it('sends the own text it was made from when it was empty', () => {
    expect(sentText(pushRewrite(EMPTY_REWRITES, rewrite('Dreamy lo-fi.', source(''))), 'typed since')).toMatchObject({ text: 'Dreamy lo-fi.', input: '' });
  });

  it('sends text typed in the Enhanced tab beside the current own text, with no record', () => {
    expect(sentText({ ...EMPTY_REWRITES, text: 'typed' }, 'rain')).toEqual({ text: 'typed', input: 'rain', record: null });
  });
});

describe('recordJson', () => {
  it('writes the keys in the order the server records them, leaving out a tier that is not known', () => {
    expect(recordJson({ settings: { space: 'forest' }, model: 'm', format: 'text', task: 'sound' })).toBe('{"settings":{"space":"forest"},"model":"m","format":"text","task":"sound"}');
    expect(recordJson({ settings: {}, model: 'm', tier: 'custom', format: 'text', task: 'style' })).toBe('{"settings":{},"model":"m","tier":"custom","format":"text","task":"style"}');
  });
});
