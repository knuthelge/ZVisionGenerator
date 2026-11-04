import { describe, expect, it } from 'vitest';

import { waveBars } from '$lib/components/atoms/Waveform.svelte';
import type { AudioProgress } from '$lib/types';
import { PEAKS } from '../../../test-utils/audio';
import {
  AURORA_BANDS,
  AURORA_STAGES,
  auroraHead,
  auroraProgress,
  auroraStage,
  easeToward,
  landedBars,
  landingGrow,
  landingProgress,
  LANDING_MS,
  outlineHalf,
  parseCssColor,
  peakOutline,
  readAuroraColors,
  ribbonPoint,
  planTarget,
  skyBoost,
  smoothstep,
  stageLook,
  stageMixOf,
  stageWeights,
  standIn,
  STAGE_FADE_S,
  stepStageMix,
  tightnessAt,
  WAVE_INSET,
  type StageMix,
} from './audioAurora';

const at = (phase: string, step: number, total: number | null, part = 1, parts = 1): AudioProgress => ({ phase, step, total, part, parts });

describe('the estimated progress of a take', () => {
  it('runs music through its phases, each inside its share of the take', () => {
    const steps = [at('Plan', 0, null), at('Plan', 600, null), at('Song tokens', 0, null), at('Song tokens', 2000, null), at('Song tokens', 6000, null), at('Synthesis', 0, 128), at('Synthesis', 64, 128), at('Synthesis', 128, 128), at('Decode', 0, 1), at('Decode', 1, 1)];
    const fractions = steps.map((step) => auroraProgress('music', step));

    expect(fractions[0]).toBe(0);
    expect([...fractions].sort((a, b) => a - b)).toEqual(fractions);
    expect(Math.max(...fractions)).toBeCloseTo(1);
    expect(Math.max(...fractions)).toBeLessThanOrEqual(1);
  });

  it('keeps a song with an unknown length short of the end until it renders', () => {
    expect(auroraProgress('music', at('Song tokens', 25 * 600, null))).toBeLessThan(0.72);
  });

  it('puts a rendering with no step count halfway through its share', () => {
    expect(auroraProgress('music', at('Synthesis', 0, null))).toBeCloseTo(0.72 + 0.24 / 2);
  });

  it('counts a speech part by its frames against the part\'s own total, and the parts by their number', () => {
    expect(auroraProgress('speech', at('Speech tokens', 0, 700, 1, 2))).toBe(0);
    expect(auroraProgress('speech', at('Speech tokens', 350, 700, 1, 2))).toBeCloseTo(0.95 / 4);
    expect(auroraProgress('speech', at('Speech tokens', 700, 700, 2, 2))).toBeCloseTo((1 + 0.95) / 2);
  });

  it('lets a speech take reach 1 at the end of its last decode, and not before', () => {
    expect(auroraProgress('speech', at('Speech tokens', 700, 700, 2, 2))).toBeLessThan(1);
    expect(auroraProgress('speech', at('Decode', 0, 1, 2, 2))).toBeLessThan(1);
    expect(auroraProgress('speech', at('Decode', 1, 1, 2, 2))).toBeCloseTo(1);
  });

  it('counts a sound by its steps out of their total, then by its decode', () => {
    expect(auroraProgress('sound', at('Sampling', 0, 8))).toBe(0);
    expect(auroraProgress('sound', at('Sampling', 4, 8))).toBe(0.5);
    expect(auroraProgress('sound', at('Decode', 0, 1))).toBe(1);
  });

  it('is 0 before the first event and for a phase it does not know', () => {
    expect(auroraProgress('music', null)).toBe(0);
    expect(auroraProgress('music', at('Elsewhere', 5, 10))).toBe(0);
  });
});

describe('the playhead', () => {
  it('is at the end for a sound, which is made all at once', () => {
    expect(auroraHead('sound', at('Sampling', 1, 8))).toBe(1);
    expect(auroraHead('sound', null)).toBe(1);
  });

  it('follows the written part of a song and reaches the end when it renders', () => {
    const early = auroraHead('music', at('Song tokens', 25 * 10, null));
    const later = auroraHead('music', at('Song tokens', 25 * 100, null));

    expect(0 < early && early < later && later < 1).toBe(true);
    expect(auroraHead('music', at('Synthesis', 3, 128))).toBe(1);
  });

  it('follows the parts of speech', () => {
    expect(auroraHead('speech', at('Speech tokens', 350, 700, 2, 2))).toBeGreaterThan(auroraHead('speech', at('Speech tokens', 350, 700, 1, 2)));
  });
});

describe('easing', () => {
  it('moves toward the target without passing it and holds still with no time', () => {
    expect(easeToward(0, 1, 0, 2.5)).toBe(0);
    const step = easeToward(0, 1, 0.5, 2.5);
    expect(step > 0 && step < 1).toBe(true);
    expect(easeToward(0.9, 1, 10, 2.5)).toBeCloseTo(1);
  });
});

describe('the colours', () => {
  it('reads each role from its CSS custom property', () => {
    const asked: string[] = [];
    const colors = readAuroraColors((name) => {
      asked.push(name);
      return name === '--color-primary-main' ? ' #2dd4bf ' : name === '--color-accent-coral' ? '#fb8f7c' : '#ffffff';
    });

    expect(asked).toEqual(expect.arrayContaining(['--color-primary-main', '--color-accent-amber', '--color-accent-coral', '--color-zinc-500']));
    expect(colors.primary).toEqual([45, 212, 191]);
    expect(colors.coral).toEqual([251, 143, 124]);
    expect(colors.amber).toEqual([255, 255, 255]);
  });

  it('draws a dim grey for a token it cannot read', () => {
    const colors = readAuroraColors(() => '');

    expect(colors.primary[0]).toBe(colors.primary[1] - 4);
  });

  it('parses a hex token and refuses the rest', () => {
    expect(parseCssColor('#2DD4BF')).toEqual([45, 212, 191]);
    expect(parseCssColor('teal')).toBeNull();
  });
});

describe('the ribbons', () => {
  const state = (kind: 'music' | 'speech' | 'sound', over: Partial<{ time: number; head: number; level: number }> = {}) => ({ kind, time: 0, head: 1, level: 1, ...over });

  it('gives each kind its own colours', () => {
    const colors = (kind: 'music' | 'speech' | 'sound') => AURORA_BANDS[kind].map((band) => band.color);

    expect(colors('music')).toEqual(['amber', 'primary', 'coral']);
    expect(colors('speech')).toEqual(['coral', 'primary']);
    expect(colors('sound')).toEqual(['primary', 'amber']);
  });

  it('glows behind the playhead and is faint ahead of it', () => {
    const band = AURORA_BANDS.music[1]!;
    const written = ribbonPoint(band, 0.2, state('music', { head: 0.6 })).half;
    const ahead = ribbonPoint(band, 0.9, state('music', { head: 0.6 })).half;

    expect(ahead).toBeLessThan(written / 2);
  });

  it('brightens a sound with its level and never goes dark', () => {
    const band = AURORA_BANDS.sound[0]!;
    const dim = ribbonPoint(band, 0.5, state('sound', { level: 0 })).half;
    const bright = ribbonPoint(band, 0.5, state('sound', { level: 1 })).half;

    expect(dim > 0 && bright > dim * 2).toBe(true);
  });

  it('drifts slowly: every ribbon takes 10 to 30 seconds for a cycle', () => {
    for (const band of Object.values(AURORA_BANDS).flat()) {
      const cycleSeconds = (2 * Math.PI) / Math.abs(band.speed) / 1000;

      expect(cycleSeconds).toBeGreaterThanOrEqual(10);
      expect(cycleSeconds).toBeLessThanOrEqual(30);
    }
  });

  it('has speech gaps between words in the stand-in shape', () => {
    const shape = Array.from({ length: 100 }, (_, i) => standIn('speech', i));

    expect(Math.min(...shape)).toBeLessThan(0.1);
    expect(Math.max(...shape)).toBeGreaterThan(0.5);
  });
});

describe('the landing', () => {
  it('takes the landing time from the moment the take is ready', () => {
    expect(landingProgress(0)).toBe(0);
    expect(landingProgress(LANDING_MS / 2)).toBeCloseTo(0.5);
    expect(landingProgress(LANDING_MS * 3)).toBe(1);
  });

  it('ends exactly on the bars the Waveform draws for the same peaks', () => {
    expect(landedBars(PEAKS, 1)).toEqual(waveBars(PEAKS));
    expect(landedBars([], 1)).toEqual(waveBars([]));
  });

  it('begins with no bars and opens them from the centre outward', () => {
    expect(landedBars(PEAKS, 0).every((bar) => bar.height === 0)).toBe(true);

    const half = landedBars(PEAKS, 0.55);
    const final = waveBars(PEAKS);
    const opened = half.map((bar, i) => bar.height / final[i]!.height);

    expect(opened[50]).toBeGreaterThan(opened[10]!);
    expect(opened[49]).toBeGreaterThan(opened[90]!);
    expect(landingGrow(0, 100, 1)).toBe(1);
    expect(landingGrow(99, 100, 1)).toBe(1);
  });

  it('keeps every opening bar centred on the strip and never taller than its final height', () => {
    const final = waveBars(PEAKS);

    landedBars(PEAKS, 0.7).forEach((bar, i) => {
      expect(bar.height).toBeLessThanOrEqual(final[i]!.height);
      if (bar.height > 0) expect(bar.y + bar.height / 2).toBeCloseTo(50);
    });
  });

  it('collapses the ribbons onto the outline of the take, which is zero outside the waveform', () => {
    const outline = peakOutline(PEAKS);
    const middle = outlineHalf(outline, 0.5);

    expect(outlineHalf(outline, 0.01)).toBe(0);
    expect(outlineHalf(outline, 0.99)).toBe(0);
    expect(middle).toBeCloseTo((Math.max(...waveBars(PEAKS).map((bar) => bar.height)) / 100 / 2) * WAVE_INSET.h, 1);
  });

  it('eases its phases smoothly from 0 to 1', () => {
    expect(smoothstep(0.5, 1, 0.4)).toBe(0);
    expect(smoothstep(0.5, 1, 1)).toBe(1);
  });
});

describe('the stage of the sky', () => {
  it('is waiting until the first progress event', () => {
    expect(auroraStage(null)).toBe('wait');
  });

  it('takes a song through waiting, plan, writing, synthesis and decode', () => {
    const phases = [at('Plan', 10, null), at('Song tokens', 10, null), at('Synthesis', 10, 128), at('Decode', 0, 1)];

    expect(phases.map((phase) => auroraStage(phase))).toEqual(['plan', 'write', 'synth', 'decode']);
  });

  it('takes speech through writing and decode, again for each part', () => {
    const parts = [at('Speech tokens', 100, 700, 1, 2), at('Decode', 0, 1, 1, 2), at('Speech tokens', 100, 700, 2, 2), at('Decode', 0, 1, 2, 2)];

    expect(parts.map((part) => auroraStage(part))).toEqual(['write', 'decode', 'write', 'decode']);
  });

  it('takes a sound through its sampling, which is writing, and its decode', () => {
    expect(auroraStage(at('Sampling', 3, 8))).toBe('write');
    expect(auroraStage(at('Decode', 0, 1))).toBe('decode');
  });

  it('treats a phase it does not know as writing', () => {
    expect(auroraStage(at('Something new', 3, null))).toBe('write');
  });

  it('brings a song\'s stars out with its plan, and no other kind\'s', () => {
    const plan = (kind: 'music' | 'speech', progress: ReturnType<typeof at> | null) => planTarget(kind, auroraStage(progress), progress);

    expect(plan('music', null)).toBe(0);
    expect(plan('music', at('Plan', 300, null))).toBeCloseTo(0.5);
    expect(plan('music', at('Plan', 9000, null))).toBe(1);
    expect(plan('music', at('Song tokens', 1, null))).toBe(1);
    expect(plan('speech', at('Speech tokens', 1, 700))).toBe(0);
  });
});

describe('the cross-fade between stages', () => {
  const sum = (mix: StageMix) => AURORA_STAGES.reduce((total, id) => total + mix[id], 0);

  it('starts on one stage and moves toward the next by real time', () => {
    let mix = stageMixOf('plan');
    mix = stepStageMix(mix, 'write', STAGE_FADE_S / 2);

    expect(mix.plan).toBeCloseTo(0.5);
    expect(mix.write).toBeCloseTo(0.5);
  });

  it('takes the same time whatever the frame rate', () => {
    const run = (frames: number) => {
      let mix = stageMixOf('wait');
      for (let i = 0; i < frames; i += 1) mix = stepStageMix(mix, 'plan', STAGE_FADE_S / frames);
      return mix;
    };

    for (const frames of [3, 22, 45]) {
      const mix = run(frames);
      expect(mix.plan).toBeCloseTo(1);
      expect(mix.wait).toBeCloseTo(0);
    }
  });

  it('lands on the new stage after its time and stays there', () => {
    const mix = stepStageMix(stepStageMix(stageMixOf('synth'), 'decode', STAGE_FADE_S), 'decode', 5);

    expect(mix).toEqual(stageMixOf('decode'));
    expect(stepStageMix(mix, 'wait', 100)).toEqual(stageMixOf('wait'));
  });

  it('gives weights that add up to one and ease at both ends', () => {
    const halfway = stepStageMix(stageMixOf('plan'), 'write', STAGE_FADE_S / 2);
    const early = stepStageMix(stageMixOf('plan'), 'write', STAGE_FADE_S * 0.1);

    expect(sum(stageWeights(halfway))).toBeCloseTo(1);
    expect(stageWeights(halfway).plan).toBeCloseTo(0.5);
    expect(stageWeights(early).plan).toBeGreaterThan(0.9);
    expect(stageWeights(stageMixOf('synth')).synth).toBe(1);
  });

  it('does not pulse: the weight of a stage only falls while it fades out and only rises while it fades in', () => {
    let mix = stageMixOf('write');
    let previous = stageWeights(mix);
    for (let i = 0; i < 40; i += 1) {
      mix = stepStageMix(mix, 'synth', 1 / 30);
      const weights = stageWeights(mix);
      expect(weights.write).toBeLessThanOrEqual(previous.write + 1e-9);
      expect(weights.synth).toBeGreaterThanOrEqual(previous.synth - 1e-9);
      previous = weights;
    }
  });
});

describe('the look of the sky', () => {
  const input = { kind: 'music' as const, head: 0, level: 0, plan: 1 };

  it('differs by stage in the sky, not only in the ribbons', () => {
    const look = (stage: Parameters<typeof stageMixOf>[0]) => stageLook(stageMixOf(stage), input);

    expect(look('wait').haze).toBeGreaterThan(look('write').haze);
    expect(look('wait').star).toBe(0);
    expect(look('synth').depth).toBeGreaterThan(look('write').depth);
    expect(look('synth').horizon).toBeGreaterThan(look('write').horizon);
    expect(look('decode').line).toBeGreaterThan(0);
    expect(look('write').line).toBe(0);
    expect(look('wait').rest).toBe(1);
  });

  it('brings the stars out with the plan', () => {
    expect(stageLook(stageMixOf('plan'), { ...input, plan: 0.25 }).star).toBe(0.25);
    expect(stageLook(stageMixOf('plan'), { ...input, plan: 1 }).star).toBe(1);
  });

  it('raises the horizon as the take is written, by its playhead or, for a sound, its brightness', () => {
    const write = stageMixOf('write');

    expect(stageLook(write, { ...input, head: 0.8 }).horizon).toBeGreaterThan(stageLook(write, { ...input, head: 0.1 }).horizon);
    expect(stageLook(write, { kind: 'sound', head: 1, level: 0.9, plan: 0 }).horizon).toBeGreaterThan(stageLook(write, { kind: 'sound', head: 1, level: 0.1, plan: 0 }).horizon);
  });

  it('blends two stages by their weights', () => {
    const mid = stageWeights(stepStageMix(stageMixOf('wait'), 'write', STAGE_FADE_S / 2));
    const wait = stageLook(stageMixOf('wait'), input).haze;
    const write = stageLook(stageMixOf('write'), input).haze;

    expect(stageLook(mid, input).haze).toBeCloseTo((wait + write) / 2);
  });

  it('holds the parts of a speech that are done, tightens the one decoding, and leaves the rest', () => {
    expect(tightnessAt(0.2, 0.5, 2, 3)).toBeGreaterThan(0);
    expect(tightnessAt(0.5, 0.9, 2, 3)).toBe(0.9);
    expect(tightnessAt(0.9, 0.9, 2, 3)).toBe(0);
    expect(tightnessAt(0.9, 0.9, 1, 1)).toBe(0.9);
  });

  it('draws the sky stronger on a narrow slot, up to a limit', () => {
    expect(skyBoost(632)).toBe(1);
    expect(skyBoost(520)).toBe(1);
    expect(skyBoost(366)).toBeGreaterThan(1.5);
    expect(skyBoost(300)).toBe(skyBoost(366));
    expect(skyBoost(440)).toBeGreaterThan(skyBoost(500));
  });
});

describe('the ribbons at rest', () => {
  const band = AURORA_BANDS.music;

  it('fold into one thinner teal ribbon while the model loads', () => {
    const state = { kind: 'music' as const, time: 0, head: 0, level: 0 };
    const calm = { ...state, rest: 1 };

    expect(ribbonPoint(band[0]!, 0.5, calm).half).toBeCloseTo(0);
    expect(ribbonPoint(band[2]!, 0.5, calm).half).toBeCloseTo(0);
    expect(ribbonPoint(band[1]!, 0.5, calm).half).toBeGreaterThan(0);
    expect(ribbonPoint(band[1]!, 0.5, calm).half).toBeLessThan(ribbonPoint(band[1]!, 0.5, state).half);
  });
});
