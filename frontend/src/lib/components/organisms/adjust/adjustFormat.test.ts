import { describe, expect, it } from 'vitest';

import { NEUTRAL_LIGHT } from '$lib/state/adjustMath';
import { cropSummary, centredFill, lightSummary, mixSummary, signedLabel, signedValueText } from './adjustFormat';

describe('adjustFormat', () => {
  it('writes a signed label with a true minus sign', () => {
    expect(signedLabel(24)).toBe('+24');
    expect(signedLabel(-8)).toBe('−8');
    expect(signedLabel(0)).toBe('0');
  });

  it('writes a signed spoken value with an ASCII minus sign', () => {
    expect(signedValueText(24)).toBe('+24');
    expect(signedValueText(-8)).toBe('-8');
    expect(signedValueText(0)).toBe('0');
  });

  it('fills a centre-zero range from the middle to the thumb', () => {
    expect(centredFill(0)).toEqual({ lo: 50, hi: 50 });
    expect(centredFill(100)).toEqual({ lo: 50, hi: 100 });
    expect(centredFill(40)).toEqual({ lo: 50, hi: 70 });
    expect(centredFill(-100)).toEqual({ lo: 0, hi: 50 });
    expect(centredFill(-20)).toEqual({ lo: 40, hi: 50 });
  });

  it('keeps the fill inside the track for a value outside the range', () => {
    expect(centredFill(500)).toEqual({ lo: 50, hi: 100 });
    expect(centredFill(-500)).toEqual({ lo: 0, hi: 50 });
  });

  describe('section summaries', () => {
    const free = { preset: 'free', ratio: null } as const;
    const plain = { rotate: 0, flip: false, crop: null } as const;

    it('says Full picture for no crop, and gives the shape and size of a crop', () => {
      expect(cropSummary(free, plain, { width: 1290, height: 860 }, false)).toEqual({ text: 'Full picture', set: false });
      expect(cropSummary(free, plain, { width: 1290, height: 860 }, true)).toEqual({ text: 'Free · 1290 × 860', set: true });
      expect(cropSummary({ preset: '4:5', ratio: 0.8 }, plain, { width: 737, height: 922 }, true)).toEqual({ text: '4:5 · 737 × 922', set: true });
    });

    it('gives the size alone for a turn or a flip without a crop', () => {
      expect(cropSummary(free, { rotate: 90, flip: false, crop: null }, { width: 860, height: 1290 }, false)).toEqual({ text: '860 × 1290', set: true });
    });

    it('counts the changed light sliders, and the mixer bands do not count there', () => {
      expect(lightSummary(NEUTRAL_LIGHT)).toEqual({ text: 'No changes', set: false });
      expect(lightSummary({ ...NEUTRAL_LIGHT, exposure: 10, warmth: -5, mix_red: 50 })).toEqual({ text: '2 changed', set: true });
    });

    it('names the one band that is set, and counts several', () => {
      expect(mixSummary(NEUTRAL_LIGHT)).toEqual({ text: 'No changes', set: false });
      expect(mixSummary({ ...NEUTRAL_LIGHT, mix_blue: 30 })).toEqual({ text: 'Blue +30', set: true });
      expect(mixSummary({ ...NEUTRAL_LIGHT, mix_aqua: -8 })).toEqual({ text: 'Aqua −8', set: true });
      expect(mixSummary({ ...NEUTRAL_LIGHT, mix_blue: 30, mix_red: 1, exposure: 9 })).toEqual({ text: '2 colours changed', set: true });
    });
  });
});
