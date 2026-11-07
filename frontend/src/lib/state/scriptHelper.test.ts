import { describe, expect, it } from 'vitest';

import { AUDIO_SETTINGS } from '../../test-utils/audio';
import { RECORDING_NOTE, SCRIPT_SECONDS_MAX, SCRIPT_SECONDS_MIN, defaultFitSeconds, parseFitSeconds, orderedActions, scriptRecord, scriptRequest, scriptRunLabel, scriptSettings, scriptView } from './scriptHelper';

describe('scriptView', () => {
  it('offers Add delivery and Fit to time for a built-in voice, and Fit alone for a recording', () => {
    expect(scriptView('Hello there.', 'normal', false, AUDIO_SETTINGS).actions).toEqual(['fit', 'delivery']);
    expect(scriptView('Hello there.', 'normal', true, AUDIO_SETTINGS).actions).toEqual(['fit']);
    expect(RECORDING_NOTE).toContain('reads as recorded');
  });

  it.each(['', '   ', '[sad]', '[sad] … [as set]', '…', '. . .'])('finds nothing to work on in %j and points to Describe it', (text) => {
    const view = scriptView(text, 'normal', false, AUDIO_SETTINGS);

    expect(view.problem).toContain('Describe it');
  });

  it('works on text that holds words, and counts the characters that are read with marks left out', () => {
    const view = scriptView('[sad] ' + 'x'.repeat(279), 'normal', false, AUDIO_SETTINGS);

    expect(view.problem).toBeNull();
    expect(view.characters).toBe(280);
    expect(view.seconds).toBeCloseTo(20, 5);
  });

  it('reads at the pace', () => {
    const slow = scriptView('x'.repeat(280), 'slow', false, AUDIO_SETTINGS);

    expect(slow.seconds).toBeCloseTo(20 / 0.81, 5);
  });

  it('keeps a bracketed word that is not a mark as text', () => {
    expect(scriptView('Press [Enter] now', 'normal', false, AUDIO_SETTINGS).characters).toBe(17);
  });

  it('tells no time without the vocabulary', () => {
    expect(scriptView('Hello there.', 'normal', false, undefined).seconds).toBe(0);
  });
});

describe('defaultFitSeconds', () => {
  it.each([
    [47, 45],
    [48, 50],
    [61.7, 60],
    [0, SCRIPT_SECONDS_MIN],
    [1, SCRIPT_SECONDS_MIN],
    [10_000, SCRIPT_SECONDS_MAX],
  ])('turns %d seconds of reading into %d', (reading, expected) => {
    expect(defaultFitSeconds(reading)).toBe(expected);
  });
});

describe('parseFitSeconds', () => {
  it.each([['30', 30], [' 45 ', 45], ['3', 3], ['600', 600]])('reads %j as %d', (value, expected) => {
    expect(parseFitSeconds(value)).toBe(expected);
  });

  it.each(['', '  ', 'abc', '2', '601', '12.5', '-5', '1e1x'])('reads %j as none', (value) => {
    expect(parseFitSeconds(value)).toBeNull();
  });
});

describe('the request', () => {
  it('names Add delivery with no more than the actions', () => {
    expect(scriptSettings(['delivery'], 30, 'slow')).toEqual({ actions: ['delivery'] });
    expect(scriptRequest('qwen3_tts', 'Hello.', { actions: ['delivery'] })).toEqual({ mode: 'audio', task: 'script', model: 'qwen3_tts', kind: 'speech', prompt: 'Hello.', settings: { actions: ['delivery'] } });
  });

  it('names Fit with its time, and the pace only when it is not Normal', () => {
    expect(scriptSettings(['fit'], 20, 'normal')).toEqual({ actions: ['fit'], seconds: 20 });
    expect(scriptSettings(['fit'], 20, 'very-slow')).toEqual({ actions: ['fit'], seconds: 20, pace: 'very-slow' });
  });

  it('asks for both in one request, fit first whatever order they were picked in', () => {
    expect(scriptSettings(['delivery', 'fit'], 20, 'slow')).toEqual({ actions: ['fit', 'delivery'], seconds: 20, pace: 'slow' });
    expect(scriptRequest('qwen3_tts', 'Hello.', scriptSettings(['delivery', 'fit'], 20, 'normal')).settings).toEqual({ actions: ['fit', 'delivery'], seconds: 20 });
    expect(orderedActions(['delivery', 'fit', 'delivery'])).toEqual(['fit', 'delivery']);
  });

  it('records what the helper did, one record for both, under the script task', () => {
    expect(scriptRecord('owner/llm', { actions: ['fit', 'delivery'], seconds: 20 })).toEqual({ settings: { actions: ['fit', 'delivery'], seconds: 20 }, model: 'owner/llm', format: 'text', task: 'script' });
  });

  it('names the picks on the run button', () => {
    expect(scriptRunLabel(['fit'])).toBe('Fit to time');
    expect(scriptRunLabel(['delivery'])).toBe('Add delivery');
    expect(scriptRunLabel(['delivery', 'fit'])).toBe('Fit and add delivery');
  });
});
