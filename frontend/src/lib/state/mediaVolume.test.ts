import { beforeEach, describe, expect, it } from 'vitest';

import { mediaVolume } from './mediaVolume.svelte';

describe('mediaVolume', () => {
  beforeEach(() => mediaVolume.reset());

  it('starts at full volume, unmuted', () => {
    expect(mediaVolume.volume).toBe(1);
    expect(mediaVolume.muted).toBe(false);
  });

  it('keeps the volume between 0 and 1', () => {
    mediaVolume.setVolume(3);
    expect(mediaVolume.volume).toBe(1);
    mediaVolume.setVolume(-1);
    expect(mediaVolume.volume).toBe(0);
  });

  it('unmutes when the volume is raised, and stays muted when it is lowered to silence', () => {
    mediaVolume.setMuted(true);
    mediaVolume.setVolume(0);
    expect(mediaVolume.muted).toBe(true);

    mediaVolume.setVolume(0.4);
    expect(mediaVolume.muted).toBe(false);
  });
});
