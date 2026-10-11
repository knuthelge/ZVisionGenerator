import { describe, expect, it } from 'vitest';
import { normalizePromptText } from './extend';
import { extendFramesFromSeconds, snapExtendFrames } from './videoDuration';

// Expected values below were produced by the Python functions of the same names
// (zvisiongenerator.utils.prompt_compose.normalize_prompt_text, zvisiongenerator.utils.video_extend.*).
// Left out: U+0085, U+001C..U+001F and U+FEFF, which Python's split() and JavaScript's \s classify differently
// (U+0085 and the separators split in Python only, U+FEFF in JavaScript only). The warning they feed never blocks a run.
const TEXT_CASES: [string, string][] = [["", ""], ["  ", ""], ["A  dog\nruns.", "A dog runs."], ["\tA\tdog\t", "A dog"], ["a\r\nb\r\n\r\nc", "a b c"], ["x\u00a0y", "x y"], ["x\u2003y", "x y"], ["x\u3000y", "x y"], ["x\u200by", "x\u200by"], ["x\u2028y", "x y"], [" lead and trail ", "lead and trail"], ["a\u000bb\fc", "a b c"]];
const SECONDS_CASES: [number, number, number][] = [[0.1, 24, 8], [0.25, 24, 8], [0.5, 24, 16], [1, 24, 24], [1.5, 24, 32], [2, 24, 48], [2.5, 24, 64], [3, 24, 72], [3.5, 24, 80], [4, 24, 96], [5, 24, 120], [5.5, 24, 128], [7.5, 24, 176], [10, 24, 240], [0.3333, 24, 8], [0.1, 30, 8], [0.25, 30, 8], [0.5, 30, 16], [1, 30, 32], [1.5, 30, 48], [2, 30, 64], [2.5, 30, 72], [3, 30, 88], [3.5, 30, 104], [4, 30, 120], [5, 30, 152], [5.5, 30, 168], [7.5, 30, 224], [10, 30, 304], [0.3333, 30, 8], [0.1, 25, 8], [0.25, 25, 8], [0.5, 25, 16], [1, 25, 24], [1.5, 25, 40], [2, 25, 48], [2.5, 25, 64], [3, 25, 72], [3.5, 25, 88], [4, 25, 96], [5, 25, 128], [5.5, 25, 136], [7.5, 25, 184], [10, 25, 248], [0.3333, 25, 8], [0.1, 29.97002997002997, 8], [0.25, 29.97002997002997, 8], [0.5, 29.97002997002997, 16], [1, 29.97002997002997, 32], [1.5, 29.97002997002997, 48], [2, 29.97002997002997, 56], [2.5, 29.97002997002997, 72], [3, 29.97002997002997, 88], [3.5, 29.97002997002997, 104], [4, 29.97002997002997, 120], [5, 29.97002997002997, 152], [5.5, 29.97002997002997, 168], [7.5, 29.97002997002997, 224], [10, 29.97002997002997, 296], [0.3333, 29.97002997002997, 8], [0.1, 23.976, 8], [0.25, 23.976, 8], [0.5, 23.976, 8], [1, 23.976, 24], [1.5, 23.976, 32], [2, 23.976, 48], [2.5, 23.976, 56], [3, 23.976, 72], [3.5, 23.976, 80], [4, 23.976, 96], [5, 23.976, 120], [5.5, 23.976, 128], [7.5, 23.976, 176], [10, 23.976, 240], [0.3333, 23.976, 8], [0.1, 60, 8], [0.25, 60, 16], [0.5, 60, 32], [1, 60, 64], [1.5, 60, 88], [2, 60, 120], [2.5, 60, 152], [3, 60, 176], [3.5, 60, 208], [4, 60, 240], [5, 60, 304], [5.5, 60, 328], [7.5, 60, 448], [10, 60, 600], [0.3333, 60, 16], [0.1, 29.97, 8], [0.25, 29.97, 8], [0.5, 29.97, 16], [1, 29.97, 32], [1.5, 29.97, 48], [2, 29.97, 56], [2.5, 29.97, 72], [3, 29.97, 88], [3.5, 29.97, 104], [4, 29.97, 120], [5, 29.97, 152], [5.5, 29.97, 168], [7.5, 29.97, 224], [10, 29.97, 296], [0.3333, 29.97, 8]];
const SNAP_CASES: [number, number][] = [[0, 8], [1, 8], [3, 8], [4, 8], [5, 8], [8, 8], [11, 8], [12, 16], [12.5, 16], [20, 16], [28, 32], [36, 32], [44, 48], [52, 48], [60, 64], [100, 96], [1000, 1000], [1000000000.0, 1000000000]];

describe('Python parity of the extend helpers', () => {
  it.each(TEXT_CASES)('normalizePromptText(%j) matches normalize_prompt_text', (input, expected) => {
    expect(normalizePromptText(input)).toBe(expected);
  });

  it.each(SECONDS_CASES)('extendFramesFromSeconds(%d s, %d fps) matches extend_frames_from_seconds', (seconds, fps, expected) => {
    expect(extendFramesFromSeconds(seconds, fps)).toBe(expected);
  });

  it.each(SNAP_CASES)('snapExtendFrames(%d) matches snap_extend_frames', (frames, expected) => {
    expect(snapExtendFrames(frames)).toBe(expected);
  });
});
