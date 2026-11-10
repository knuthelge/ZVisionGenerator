import { describe, expect, it } from 'vitest';

import frameCases from '../../../../tests/fixtures/frame_point_cases.json';
import labelCases from '../../../../tests/fixtures/point_label_cases.json';
import { EXTEND_MIN_FRAMES, EXTEND_SHORT_REASON, frameAt, framePoint, pointLabel, stepTarget, videoFactsOf, type MediaSnapshot, type VideoFacts } from './framePoint';

// 6 s at 24 fps, with a start offset of 0.
const FACTS: VideoFacts = { fps: 24, frames: 144, start: 0 };
const LAST = FACTS.frames - 1;

function snapshot(overrides: Partial<MediaSnapshot> = {}): MediaSnapshot {
  return { paused: true, ended: false, time: 0, error: false, duration: 6, ...overrides };
}

// tests/fixtures/frame_point_cases.json is also checked by pytest against `frame_at` in utils/video_point.py: both pick one frame.
describe('frameAt', () => {
  it.each(frameCases)('picks the frame the server picks: $name', ({ seconds, start, fps, frames, frame }) => {
    expect(frameAt(seconds, { fps, frames, start })).toBe(frame);
  });

  it('lands on the frame a step target was made for', () => {
    const facts = { fps: 30000 / 1001, frames: 300, start: 0.021 };

    for (let frame = 0; frame < facts.frames; frame += 1) {
      expect(frameAt(stepTarget(frame, facts), facts)).toBe(frame);
    }
  });
});

describe('videoFactsOf', () => {
  it('reads the frame facts of a clip', () => {
    expect(videoFactsOf(24, 144, 0.5)).toEqual({ fps: 24, frames: 144, start: 0.5 });
  });

  it.each([
    [null, 144, 0.5], [0, 144, 0.5], [24, null, 0.5], [24, 0, 0.5], [24, 144, null], [24, 144, Number.NaN], [undefined, 144, 0.5],
  ])('has none when the rate, frames or start is unusable: %j %j %j', (fps, frames, start) => {
    expect(videoFactsOf(fps, frames, start)).toBeNull();
  });
});

describe('stepTarget', () => {
  it('is the middle of the frame, after the clip start', () => {
    expect(stepTarget(0, { fps: 25, frames: 100, start: 0.04 })).toBeCloseTo(0.06);
    expect(stepTarget(10, FACTS)).toBeCloseTo(10.5 / 24);
  });
});

// One case per row of the player state table in the PRD.
describe('framePoint', () => {
  it('row 1: a video that has not played, at its start, selects nothing', () => {
    expect(framePoint(snapshot({ time: 0 }), FACTS)).toBeNull();
  });

  it('row 1: pausing back on frame 0 selects nothing, even partway into the frame', () => {
    expect(framePoint(snapshot({ time: 0.5 / 24 }), FACTS)).toBeNull();
  });

  it('row 2: a playing video selects nothing', () => {
    expect(framePoint(snapshot({ paused: false, time: 3.2 }), FACTS)).toBeNull();
  });

  it('row 3: paused mid-clip selects that frame, and the label is the frame time', () => {
    const point = framePoint(snapshot({ time: 3.2 }), { fps: 25, frames: 200, start: 0 });

    expect(point).toEqual({ time: 3.2, seconds: 3.2, tooShort: false, atEnd: false });
    expect(point && pointLabel(point)).toBe('3.2 s');
  });

  it('row 3: the point keeps the raw time, while the label comes from the frame', () => {
    const point = framePoint(snapshot({ time: 3 + 0.4 / 24 }), FACTS);

    expect(point?.time).toBeCloseTo(3 + 0.4 / 24);
    expect(point?.seconds).toBe(3);
  });

  it('row 3: a clip that starts late counts frames from its start', () => {
    const point = framePoint(snapshot({ time: 1.04 }), { fps: 25, frames: 125, start: 0.04 });

    expect(point?.seconds).toBe(1);
  });

  it('row 4: within the first 25 frames is a selection that cannot extend', () => {
    const point = framePoint(snapshot({ time: 0.4 }), FACTS);

    expect(point).toMatchObject({ seconds: 9 / 24, tooShort: true, atEnd: false });
    expect(point && pointLabel(point)).toBe('0.4 s');
  });

  it('keeps the 25-frame boundary exact: 25 kept frames extend, 24 do not', () => {
    expect(EXTEND_MIN_FRAMES).toBe(25);
    expect(framePoint(snapshot({ time: stepTarget(EXTEND_MIN_FRAMES - 2, FACTS) }), FACTS)).toMatchObject({ tooShort: true });
    expect(framePoint(snapshot({ time: stepTarget(EXTEND_MIN_FRAMES - 1, FACTS) }), FACTS)).toMatchObject({ tooShort: false });
  });

  it('the first frame after frame 0 is a selection of two frames', () => {
    expect(framePoint(snapshot({ time: stepTarget(1, FACTS) }), FACTS)).toMatchObject({ seconds: 1 / 24, tooShort: true });
  });

  it('row 5: paused on the last frame is the end, the middle of that frame, never too short', () => {
    const point = framePoint(snapshot({ time: stepTarget(LAST, FACTS) }), FACTS);

    expect(point).toEqual({ time: stepTarget(LAST, FACTS), seconds: LAST / 24, tooShort: false, atEnd: true });
  });

  it('row 5: sitting anywhere on the last frame, or past it, is the end', () => {
    expect(framePoint(snapshot({ time: LAST / 24 + 0.001 }), FACTS)?.atEnd).toBe(true);
    expect(framePoint(snapshot({ time: 6 }), FACTS)).toMatchObject({ atEnd: true, seconds: LAST / 24 });
  });

  it('row 5: the frame before the last is not the end', () => {
    expect(framePoint(snapshot({ time: stepTarget(LAST - 1, FACTS) }), FACTS)).toMatchObject({ seconds: (LAST - 1) / 24, atEnd: false });
  });

  it('row 6: ended is the end even when the time is short of the last frame', () => {
    const point = framePoint(snapshot({ ended: true, time: 5.5 }), FACTS);

    expect(point).toMatchObject({ seconds: LAST / 24, atEnd: true, tooShort: false, time: stepTarget(LAST, FACTS) });
  });

  it('row 6: ended wins even if the element still says it is not paused', () => {
    expect(framePoint(snapshot({ ended: true, paused: false, time: 6 }), FACTS)?.atEnd).toBe(true);
  });

  it('row 7: a seek in flight reads as the target time', () => {
    expect(framePoint(snapshot({ time: 3.2 }), FACTS)).toMatchObject({ seconds: 76 / 24, atEnd: false });
    expect(framePoint(snapshot({ time: 0 }), FACTS)).toBeNull();
    expect(framePoint(snapshot({ time: 6 }), FACTS)?.atEnd).toBe(true);
  });

  it('row 8: without frame facts or metadata nothing is selected', () => {
    expect(framePoint(snapshot({ time: 3.2 }), null)).toBeNull();
    expect(framePoint(snapshot({ time: 3.2, duration: 0 }), FACTS)).toBeNull();
    expect(framePoint(snapshot({ time: 3.2, duration: Number.NaN }), FACTS)).toBeNull();
  });

  it('row 9: a media error selects nothing, even at the end', () => {
    expect(framePoint(snapshot({ error: true, time: 3.2 }), FACTS)).toBeNull();
    expect(framePoint(snapshot({ error: true, ended: true, time: 6 }), FACTS)).toBeNull();
  });

  it('row 11: a clip under 25 frames is too short anywhere inside it', () => {
    const facts = { fps: 24, frames: 10, start: 0 };

    expect(framePoint(snapshot({ time: stepTarget(4, facts), duration: 10 / 24 }), facts)).toMatchObject({ seconds: 4 / 24, tooShort: true, atEnd: false });
    expect(framePoint(snapshot({ time: stepTarget(9, facts), duration: 10 / 24 }), facts)).toMatchObject({ atEnd: true, tooShort: false });
  });

  it('row 11: a one-frame clip selects nothing until it has played, then it is the end', () => {
    const facts = { fps: 24, frames: 1, start: 0 };

    expect(framePoint(snapshot({ time: 0, duration: 1 / 24 }), facts)).toBeNull();
    expect(framePoint(snapshot({ ended: true, time: 1 / 24, duration: 1 / 24 }), facts)).toMatchObject({ seconds: 0, atEnd: true, tooShort: false });
  });
});

describe('pointLabel', () => {
  it('rounds the frame time to one decimal', () => {
    expect(pointLabel({ seconds: 3.24 })).toBe('3.2 s');
    expect(pointLabel({ seconds: 0.05 })).toBe('0.1 s');
    expect(pointLabel({ seconds: 6 })).toBe('6.0 s');
  });

  // tests/fixtures/point_label_cases.json is also checked by pytest against `frame_file_name` in utils/video_frame.py: the label and the saved file's name read one time.
  it.each(labelCases)('reads the time as the saved file name does: $name', ({ seconds, label }) => {
    expect(pointLabel({ seconds })).toBe(label);
  });
});

describe('EXTEND_SHORT_REASON', () => {
  it('names the limit that EXTEND_MIN_FRAMES sets', () => {
    expect(EXTEND_SHORT_REASON).toContain(String(EXTEND_MIN_FRAMES));
  });
});
