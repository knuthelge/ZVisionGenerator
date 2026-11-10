import { describe, expect, it } from 'vitest';

import { CUT_STOP_REASON, extendCutView, extendCuttable, extendFromAt, keepsLabel, settledExtendFrom } from './extendCut';
import { sourceInfo } from './extend.fixtures';

// A 144-frame clip at 24 fps (6.0 s) whose first frame is at 0.5 s of the container clock.
const clip = (overrides = {}) => sourceInfo({ fps: 24, frame_count: 144, video_start: 0.5, extend_frames: 48, ...overrides });

describe('extendFromAt', () => {
  const facts = { fps: 24, frames: 144, start: 0.5 };

  it('names the last kept frame by its time on the original clock', () => {
    expect(extendFromAt(77, facts)).toBeCloseTo(0.5 + 76 / 24, 6);
  });

  it('is null for the whole clip', () => {
    expect(extendFromAt(144, facts)).toBeNull();
  });

  it('never goes below the least a cut may keep', () => {
    expect(extendFromAt(3, facts)).toBeCloseTo(0.5 + 24 / 24, 6);
  });
});

describe('extendCutView', () => {
  it('is the whole clip without a point', () => {
    const view = extendCutView(clip(), null);
    expect(view).toMatchObject({ keepFrames: 144, whole: true, total: { frames: 192, seconds: expect.closeTo(8, 6) }, clipSeconds: 6 });
  });

  it('keeps the frames up to and including the point, and the total follows', () => {
    const point = 0.5 + 76 / 24;
    const view = extendCutView(clip(), point);
    expect(view).toMatchObject({ keepFrames: 77, whole: false, total: { frames: 77 + 48, seconds: expect.any(Number) } });
    expect(view?.keepSeconds).toBeCloseTo(76 / 24, 6);
    expect(view?.thumbAt).toBeCloseTo(point, 6);
  });

  it('reads a point as the server does, by the frame it lies in', () => {
    const inside = extendCutView(clip(), 0.5 + 76 / 24 + 0.02);
    expect(inside?.keepFrames).toBe(77);
  });

  it('keeps the whole clip for a point at or past the last frame', () => {
    expect(extendCutView(clip(), 0.5 + 143 / 24)?.whole).toBe(true);
    expect(extendCutView(clip(), 99)?.whole).toBe(true);
  });

  it('stops at 25 frames for a point before the 25th', () => {
    expect(extendCutView(clip(), 0.5)?.keepFrames).toBe(25);
    expect(extendCutView(clip(), 0)?.keepFrames).toBe(25);
  });

  it('uses the new frames it is given for the total', () => {
    expect(extendCutView(clip(), null, 16)?.total?.frames).toBe(160);
    expect(extendCutView(clip({ extend_frames: null }), null)?.total).toBeNull();
  });

  it('is null when the frames are not known', () => {
    expect(extendCutView(clip({ frame_count: null }), 1)).toBeNull();
  });

  it('moves the end of a track back by the removed frames, never below 0', () => {
    const track = { audio_track_path: '/audio/song.mp3', audio_track_end: 10 };
    expect(extendCutView(clip(track), 0.5 + 76 / 24)?.trackStart).toBeCloseTo(10 - 67 / 24, 6);
    expect(extendCutView(clip({ ...track, audio_track_end: 1 }), 0.5 + 76 / 24)?.trackStart).toBe(0);
  });

  it('leaves the end of a track as the source reported it for the whole clip', () => {
    expect(extendCutView(clip({ audio_track_path: '/audio/song.mp3', audio_track_end: 6.6500001 }), null)?.trackStart).toBe(6.6500001);
  });

  it('has no track start for a source that followed none', () => {
    expect(extendCutView(clip(), 0.5 + 76 / 24)?.trackStart).toBeNull();
  });
});

describe('settledExtendFrom', () => {
  it('turns a point at the end into the whole clip', () => {
    expect(settledExtendFrom(clip(), 99)).toBeNull();
  });

  it('moves a point before the 25th frame to it', () => {
    expect(settledExtendFrom(clip(), 0.6)).toBeCloseTo(0.5 + 24 / 24, 6);
  });

  it('keeps a good point as it was given', () => {
    expect(settledExtendFrom(clip(), 3.17)).toBe(3.17);
  });

  it('leaves a point alone when the frames are unknown', () => {
    expect(settledExtendFrom(clip({ fps: null }), 3.17)).toBe(3.17);
  });

  it('keeps a clip no longer than the minimum whole', () => {
    expect(settledExtendFrom(clip({ frame_count: 25 }), 0.7)).toBeNull();
  });
});

describe('labels and limits', () => {
  it('reads Keeps with one decimal, and the whole clip at the end', () => {
    expect(keepsLabel(extendCutView(clip(), 0.5 + 76 / 24)!)).toBe('Keeps 3.2 s of 6.0 s');
    expect(keepsLabel(extendCutView(clip(), null)!)).toBe('Keeps the whole clip');
  });

  it('names the 25-frame stop', () => {
    expect(CUT_STOP_REASON).toContain('25 frames');
  });

  it('keeps all of a clip shorter than 25 frames', () => {
    expect(extendCutView(clip({ frame_count: 10 }), 0.5 + 3 / 24)).toMatchObject({ keepFrames: 10, whole: true });
    expect(extendCutView(clip(), 0.5 + 3 / 24)).toMatchObject({ keepFrames: 25, whole: false });
  });

  it('can cut only a clip longer than the minimum', () => {
    expect(extendCuttable(clip())).toBe(true);
    expect(extendCuttable(clip({ frame_count: 25 }))).toBe(false);
    expect(extendCuttable(null)).toBe(false);
  });
});
