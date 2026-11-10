import { describe, expect, it } from 'vitest';

import { FINE_LIFT_PX, FINE_MAX_RATE, fineRate, moveScrub, startScrub } from './mediaScrub';

const TRACK = { width: 400, lastFrame: 100 };

describe('fineRate', () => {
  it('is an eighth of the normal speed on a short clip', () => {
    expect(fineRate({ width: 360, lastFrame: 119 })).toBe(FINE_MAX_RATE);
  });

  it('never goes faster than a frame per six pixels on a long clip', () => {
    const track = { width: 1000, lastFrame: 2000 };
    const rate = fineRate(track);

    expect(rate).toBeLessThan(FINE_MAX_RATE);
    // Frames per pixel = (lastFrame / width) * rate.
    expect((track.lastFrame / track.width) * rate).toBeCloseTo(1 / 6);
  });

  it('is the ceiling for audio, which has no frames', () => {
    expect(fineRate({ width: 400, lastFrame: null })).toBe(FINE_MAX_RATE);
  });
});

describe('moveScrub', () => {
  it('points at the fraction of the press, and follows the pointer one to one', () => {
    const press = startScrub(100, 500, 0.25);
    const moved = moveScrub(press, 160, 500, TRACK);

    expect(press.fraction).toBe(0.25);
    expect(moved.fraction).toBeCloseTo(0.25 + 60 / 400);
    expect(moved.fine).toBe(false);
  });

  it('keeps the fraction between 0 and 1', () => {
    const press = startScrub(100, 500, 0.9);

    expect(moveScrub(press, 900, 500, TRACK).fraction).toBe(1);
    expect(moveScrub(press, -900, 500, TRACK).fraction).toBe(0);
  });

  it('slows only once the pointer is more than 48 px above where it pressed', () => {
    const press = startScrub(100, 500, 0.5);

    expect(moveScrub(press, 100, 500 - FINE_LIFT_PX, TRACK).fine).toBe(false);
    expect(moveScrub(press, 100, 500 - FINE_LIFT_PX - 1, TRACK).fine).toBe(true);
    // Below the press is not a lift.
    expect(moveScrub(press, 100, 700, TRACK).fine).toBe(false);
  });

  it('moves an eighth as far per pixel when slowed, on any pointer', () => {
    const slowed = moveScrub(startScrub(100, 500, 0.5), 100, 400, TRACK);
    const moved = moveScrub(slowed, 140, 400, TRACK);

    expect(moved.fine).toBe(true);
    expect(moved.fraction).toBeCloseTo(0.5 + (40 / 400) * FINE_MAX_RATE);
  });

  it('does not jump when the speed changes, in either direction', () => {
    let scrub = startScrub(100, 500, 0.2);
    scrub = moveScrub(scrub, 200, 500, TRACK);
    const before = scrub.fraction;

    scrub = moveScrub(scrub, 200, 430, TRACK);
    expect(scrub.fine).toBe(true);
    expect(scrub.fraction).toBeCloseTo(before);

    scrub = moveScrub(scrub, 240, 430, TRACK);
    const slowed = scrub.fraction;
    scrub = moveScrub(scrub, 240, 500, TRACK);
    expect(scrub.fine).toBe(false);
    expect(scrub.fraction).toBeCloseTo(slowed);

    // Back at full speed, the pointer moves the playhead one to one from there.
    expect(moveScrub(scrub, 280, 500, TRACK).fraction).toBeCloseTo(slowed + 40 / 400);
  });

  it('counts from the press, so going back to the press point returns to the press fraction', () => {
    const slowed = moveScrub(startScrub(100, 500, 0.5), 300, 400, TRACK);
    const back = moveScrub(slowed, 100, 400, TRACK);

    expect(back.fraction).toBeCloseTo(0.5);
  });

  it('holds the fraction on a track that has no width yet', () => {
    expect(moveScrub(startScrub(10, 10, 0.4), 80, 10, { width: 0, lastFrame: null }).fraction).toBe(0.4);
  });
});
