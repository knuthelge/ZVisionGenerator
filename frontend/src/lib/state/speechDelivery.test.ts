import { describe, expect, it } from 'vitest';

import speechCharacterCounts from '../../../../tests/fixtures/speech_character_counts.json';
import speechUnknownMarks from '../../../../tests/fixtures/speech_unknown_marks.json';
import { AUDIO_SETTINGS } from '../../test-utils/audio';
import { deliveryLabel, deliverySummary, formatReadingTime, readingSeconds, sameDelivery, stripMarks, unknownMarks, vibeDetail, vibeOf } from './speechDelivery';

const { vibes, delivery_defaults: defaults, reading } = AUDIO_SETTINGS;
const markIds = AUDIO_SETTINGS.marks.map((mark) => mark.id);
const vibe = (id: string) => vibes.find((item) => item.id === id)!;

describe('vibeOf', () => {
  it('finds the Vibe whose five values are all the given ones', () => {
    expect(vibeOf({ ...vibe('storyteller').delivery }, vibes)?.id).toBe('storyteller');
    expect(vibeOf({ ...vibe('whispered-secret').delivery }, vibes)?.label).toBe('Whispered secret');
  });

  it('finds none once one value differs, and none at the defaults', () => {
    expect(vibeOf({ ...vibe('storyteller').delivery, pitch: 'high' }, vibes)).toBeNull();
    expect(vibeOf(defaults, vibes)).toBeNull();
  });

  it('knows every served Vibe by its own values', () => {
    expect(vibes.map((item) => vibeOf(item.delivery, vibes)?.id)).toEqual(vibes.map((item) => item.id));
  });
});

describe('sameDelivery', () => {
  it('compares the five values, not the objects', () => {
    expect(sameDelivery({ ...defaults }, defaults)).toBe(true);
    expect(sameDelivery({ ...defaults, pace: 'fast' }, defaults)).toBe(false);
  });
});

describe('deliverySummary', () => {
  it('names the matching Vibe, Natural at the defaults, and Custom for anything else', () => {
    expect(deliverySummary(vibe('excited-host').delivery, AUDIO_SETTINGS)).toBe('Excited host');
    expect(deliverySummary({ ...defaults }, AUDIO_SETTINGS)).toBe('Natural');
    expect(deliverySummary({ ...defaults, emotion: 'angry' }, AUDIO_SETTINGS)).toBe('Custom');
    expect(deliverySummary({ ...vibe('announcer').delivery, manner: 'shout' }, AUDIO_SETTINGS)).toBe('Custom');
  });
});

describe('vibeDetail', () => {
  it('lists the emotion, the manner and the pace that are not the default', () => {
    expect(vibeDetail(vibe('calm-narrator'), AUDIO_SETTINGS)).toBe('Thoughtful · slow');
    expect(vibeDetail(vibe('whispered-secret'), AUDIO_SETTINGS)).toBe('Whisper · slow');
    expect(vibeDetail(vibe('announcer'), AUDIO_SETTINGS)).toBe('Determined');
  });
});

describe('deliveryLabel', () => {
  it('reads the label from the setting\'s own list and falls back to the id', () => {
    expect(deliveryLabel(AUDIO_SETTINGS, 'pace', 'very-slow')).toBe('Very slow');
    expect(deliveryLabel(AUDIO_SETTINGS, 'pitch', 'high')).toBe('High');
    expect(deliveryLabel(AUDIO_SETTINGS, 'emotion', 'giddy')).toBe('giddy');
  });
});

describe('stripMarks', () => {
  it('takes out the marks the vocabulary lists, in any case and spacing', () => {
    expect(stripMarks('[sad] Goodbye. [Whisper]Be quiet. [AS  SET] Done.', markIds)).toBe(' Goodbye. Be quiet.  Done.');
  });

  it('leaves brackets that name no mark, and everything without a vocabulary', () => {
    expect(stripMarks('A [sadd] note, [1] and [x].', markIds)).toBe('A [sadd] note, [1] and [x].');
    expect(stripMarks('[sad] Hi', [])).toBe('[sad] Hi');
  });
});

describe('stripMarks against the server', () => {
  // tests/audio/test_speech_parts.py reads the same file: the count shown and the count the limit uses are one number.
  it.each(speechCharacterCounts)('counts $characters characters in $text', ({ text, characters }) => {
    expect([...stripMarks(text, markIds)]).toHaveLength(characters);
  });
});

describe('unknownMarks', () => {
  it('lists the bracketed words that are not marks, as written', () => {
    expect(unknownMarks('[sad] Hi [laughs] and [Sadd].', markIds)).toEqual(['[laughs]', '[Sadd]']);
  });

  it('reports nothing without a vocabulary', () => {
    expect(unknownMarks('[laughs]', [])).toEqual([]);
  });
});

describe('unknownMarks against the server', () => {
  // tests/audio/test_speech_parts.py reads the same file: Generate is off for exactly the texts the server refuses.
  it.each(speechUnknownMarks)('finds $unknown in $text', ({ text, unknown }) => {
    expect(unknownMarks(text, markIds)).toEqual(unknown);
  });
});

describe('readingSeconds', () => {
  it('divides the characters by the characters a second at Normal times the Pace factor', () => {
    expect(readingSeconds(1_400, 'normal', reading)).toBeCloseTo(100);
    expect(readingSeconds(1_400, 'slow', reading)).toBeCloseTo(100 / reading.pace_factor.slow);
    expect(readingSeconds(1_400, 'very-fast', reading)).toBeLessThan(100);
  });

  it('is 0 for no text and reads an unknown pace at Normal', () => {
    expect(readingSeconds(0, 'slow', reading)).toBe(0);
    expect(readingSeconds(1_400, 'sideways', reading)).toBeCloseTo(100);
  });
});

describe('formatReadingTime', () => {
  it('writes minutes and seconds, rounded to the second', () => {
    expect(formatReadingTime(84)).toBe('1:24');
    expect(formatReadingTime(8.4)).toBe('0:08');
    expect(formatReadingTime(59.6)).toBe('1:00');
    expect(formatReadingTime(3_605)).toBe('60:05');
  });
});
