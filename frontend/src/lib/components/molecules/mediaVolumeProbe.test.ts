import { describe, expect, it } from 'vitest';

import { canSetVolume, holdsVolume } from './mediaVolumeProbe';

describe('holdsVolume', () => {
  it('is true for an element that keeps the level it is given', () => {
    const probe = { volume: 1 };

    expect(holdsVolume(probe)).toBe(true);
    expect(probe.volume).toBe(0.5);
  });

  it('is false for an element that reads 1 whatever it is given, as on iPhone and iPad', () => {
    const probe = { get volume(): number { return 1; }, set volume(_: number) { /* only the device's buttons set the level */ } };

    expect(holdsVolume(probe)).toBe(false);
  });

  it('is false for an element that refuses the level', () => {
    const probe = { get volume(): number { return 1; }, set volume(_: number) { throw new DOMException('no', 'NotSupportedError'); } };

    expect(holdsVolume(probe)).toBe(false);
  });
});

describe('canSetVolume', () => {
  it('asks the browser\'s own audio element, and answers the same every time', () => {
    expect(typeof canSetVolume()).toBe('boolean');
    expect(canSetVolume()).toBe(canSetVolume());
  });
});
