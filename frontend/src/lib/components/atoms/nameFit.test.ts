import { describe, expect, it } from 'vitest';

import { fitName, nameParts } from './nameFit';

// One unit per character.
const measure = (text: string) => text.length;

describe('nameParts', () => {
  it('keeps the time of an app-named file as the tail', () => {
    expect(nameParts('web_2026-11-14_09-12-33.png')).toEqual({ start: 'web_2026-11-14', tail: '_09-12-33.png' });
  });

  it('keeps the counter of a counted name as the tail', () => {
    expect(nameParts('web_2026-11-14_09-12-33_2.png')).toEqual({ start: 'web_2026-11-14_09-12-33', tail: '_2.png' });
  });

  it('keeps only the extension of a name with no segment, and nothing of a name with no extension', () => {
    expect(nameParts('holiday-scan.png')).toEqual({ start: 'holiday-scan', tail: '.png' });
    expect(nameParts('README')).toEqual({ start: 'README', tail: '' });
  });
});

describe('fitName', () => {
  it('returns a name that fits whole', () => {
    expect(fitName('web_2026-11-14_09-12-33.png', 40, measure)).toBe('web_2026-11-14_09-12-33.png');
  });

  it('cuts the middle and keeps the tail, in as many characters as fit', () => {
    expect(fitName('web_2026-11-14_09-12-33.png', 23, measure)).toBe('web_2026-…_09-12-33.png');
    expect(fitName('web_2026-11-14_09-12-33.png', 20, measure)).toBe('web_20…_09-12-33.png');
    expect(fitName('web_2026-11-14_09-12-33.png', 16, measure).endsWith('…_09-12-33.png')).toBe(true);
  });

  it('shows the tail alone when not even one character of the start fits', () => {
    expect(fitName('web_2026-11-14_09-12-33.png', 5, measure)).toBe('…_09-12-33.png');
  });
});
