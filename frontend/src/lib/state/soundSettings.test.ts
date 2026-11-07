import { describe, expect, it } from 'vitest';

import { AUDIO_SETTINGS } from '../../test-utils/audio';
import { autoSecondsText, autoSoundSeconds, soundSummary, soundTypeLabel } from './soundSettings';

describe('the Sound Type', () => {
  it('is named as the control writes it', () => {
    expect(['auto', 'ambience', 'effect', 'one-shot'].map((type) => soundTypeLabel(type as 'auto', AUDIO_SETTINGS))).toEqual(['Auto', 'Ambience', 'Effect', 'One-shot']);
  });

  it('gives Auto the length of its type: 12 s for Auto and Ambience, 5 for an Effect and 2 for a One-shot', () => {
    expect(['auto', 'ambience', 'effect', 'one-shot'].map((type) => autoSoundSeconds(type as 'auto', AUDIO_SETTINGS))).toEqual([12, 12, 5, 2]);
    expect(autoSecondsText('effect', AUDIO_SETTINGS)).toBe('≈ 5 s');
  });

  it('summarises as the type and the length Auto will use, or the length that is set', () => {
    expect(soundSummary({ soundType: 'ambience', audioSeconds: 0 }, AUDIO_SETTINGS)).toBe('Ambience · ≈ 12 s');
    expect(soundSummary({ soundType: 'one-shot', audioSeconds: 0 }, AUDIO_SETTINGS)).toBe('One-shot · ≈ 2 s');
    expect(soundSummary({ soundType: 'effect', audioSeconds: 8 }, AUDIO_SETTINGS)).toBe('Effect · 8 s');
  });
});
