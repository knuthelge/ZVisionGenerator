import { describe, expect, it } from 'vitest';

import type { ActiveJobState, JobStatus } from '$lib/types';
import { homePlacement, mascotMood, mascotProgress, mascotSpot, mascotTool, type MascotSpotSignals } from './mascot';

function job(status: JobStatus, currentStep = 0, totalSteps = 0, overrides: Partial<ActiveJobState> = {}): ActiveJobState {
  return { status, currentStep, totalSteps, stageName: '', outputs: [], ...overrides } as ActiveJobState;
}

describe('mascotMood', () => {
  it('is idle without a job or activity', () => {
    expect(mascotMood({ job: null })).toBe('idle');
  });

  it('thinks while queued or before denoising steps arrive', () => {
    expect(mascotMood({ job: job('queued') })).toBe('thinking');
    expect(mascotMood({ job: job('pending') })).toBe('thinking');
    expect(mascotMood({ job: job('running', 0, 0) })).toBe('thinking');
  });

  it('creates once step progress is reported', () => {
    expect(mascotMood({ job: job('running', 3, 20) })).toBe('creating');
  });

  it('keeps creating between the images of a batch', () => {
    const output = { id: 'a' } as ActiveJobState['outputs'][number];
    expect(mascotMood({ job: job('running', 0, 0, { outputs: [output] }) })).toBe('creating');
    expect(mascotMood({ job: job('running', 0, 0, { promptNumber: 2, promptCount: 3 }) })).toBe('creating');
  });

  it('reads while prompts are being enhanced', () => {
    expect(mascotMood({ job: job('running', 2, 4, { stageName: 'enhancing_prompts' }) })).toBe('reading');
  });

  it('adds the finishing touches in the last stretch of steps', () => {
    expect(mascotMood({ job: job('running', 16, 20) })).toBe('creating');
    expect(mascotMood({ job: job('running', 17, 20) })).toBe('finishing');
    expect(mascotMood({ job: job('running', 3, 3) })).toBe('creating');
  });

  it('waits with its own mood while a job is paused', () => {
    expect(mascotMood({ job: job('paused', 3, 20), drowsy: true })).toBe('paused');
  });

  it('lets reactions override everything but a painting job', () => {
    expect(mascotMood({ job: job('completed'), reaction: 'cheerful' })).toBe('cheerful');
    expect(mascotMood({ job: job('failed'), reaction: 'sad' })).toBe('sad');
    expect(mascotMood({ job: job('queued'), reaction: 'surprised', typing: true })).toBe('surprised');
    expect(mascotMood({ job: job('running', 0, 0), reaction: 'cheerful' })).toBe('cheerful');
  });

  it('cuts a reaction short once the next job starts painting', () => {
    expect(mascotMood({ job: job('running', 3, 20), reaction: 'cheerful' })).toBe('creating');
    expect(mascotMood({ job: job('running', 19, 20), reaction: 'sad' })).toBe('finishing');
  });

  it('nods over a painting job when one is added to the queue', () => {
    expect(mascotMood({ job: job('running', 3, 20), reaction: 'nodding' })).toBe('nodding');
  });

  it('keeps the active job ahead of user activity', () => {
    expect(mascotMood({ job: job('running', 3, 20), greeting: true, typing: true, drowsy: true })).toBe('creating');
  });

  it('looks to the user while a render waits for approval', () => {
    expect(mascotMood({ job: job('running', 3, 20, { review: { iteration: 1 } }) })).toBe('curious');
  });

  it('is sad when the workspace failed to load', () => {
    expect(mascotMood({ job: null, loadError: true, greeting: true })).toBe('sad');
  });

  it('thinks while the latest output is loading', () => {
    expect(mascotMood({ job: null, loading: true, greeting: true })).toBe('thinking');
  });

  it('ranks greeting, then typing, then drowsiness for an idle workspace', () => {
    expect(mascotMood({ job: null, greeting: true, typing: true, drowsy: true })).toBe('waving');
    expect(mascotMood({ job: job('completed'), typing: true, drowsy: true })).toBe('curious');
    expect(mascotMood({ job: job('cancelled'), drowsy: true })).toBe('sleeping');
  });
});

describe('mascotMood for an audio take', () => {
  const take = (overrides: Partial<ActiveJobState> = {}) => job('running', 0, 0, { media_type: 'audio', workflow: 'music', ...overrides });
  const phase = (name: string, step: number, total: number | null) => ({ audio: { phase: name, step, total, part: 1, parts: 1 } });

  it('thinks until the model reports progress', () => {
    expect(mascotMood({ job: take() })).toBe('thinking');
    expect(mascotMood({ job: take({ audio: null }) })).toBe('thinking');
  });

  it('creates once progress arrives, whatever the phase', () => {
    expect(mascotMood({ job: take(phase('Plan', 12, null)) })).toBe('creating');
    expect(mascotMood({ job: take(phase('Song tokens', 400, null)) })).toBe('creating');
    expect(mascotMood({ job: take({ workflow: 'speech', ...phase('Speech', 0, 1) }) })).toBe('creating');
  });

  it('keeps creating between the takes of a batch', () => {
    expect(mascotMood({ job: take({ promptNumber: 2, promptCount: 3 }) })).toBe('creating');
  });

  it('adds the finishing touches on the decode', () => {
    expect(mascotMood({ job: take(phase('Decode', 1, 4)) })).toBe('finishing');
    expect(mascotMood({ job: take({ workflow: 'speech', ...phase('Decode', 0, null) }) })).toBe('finishing');
  });

  it('finishes a sound on its last sampling step, not before', () => {
    expect(mascotMood({ job: take({ workflow: 'sound', ...phase('Sampling', 7, 8) }) })).toBe('creating');
    expect(mascotMood({ job: take({ workflow: 'sound', ...phase('Sampling', 8, 8) }) })).toBe('finishing');
    expect(mascotMood({ job: take({ workflow: 'music', ...phase('Synthesis', 8, 8) }) })).toBe('creating');
  });

  it('pauses as any job does', () => {
    expect(mascotMood({ job: take({ status: 'paused', ...phase('Song tokens', 400, null) }) })).toBe('paused');
  });

  it('drives no canvas from steps it never reports', () => {
    expect(mascotProgress(take({ ...phase('Sampling', 4, 8), currentStep: 4, totalSteps: 8 }))).toBeUndefined();
  });
});

describe('mascotTool', () => {
  it('wears headphones for every kind of audio', () => {
    expect(mascotTool('music')).toBe('audio');
    expect(mascotTool('speech')).toBe('audio');
    expect(mascotTool('sound')).toBe('audio');
  });

  it('holds a clapperboard for video and a brush otherwise', () => {
    expect(mascotTool('txt2vid')).toBe('clapper');
    expect(mascotTool('img2vid')).toBe('clapper');
    expect(mascotTool('extend')).toBe('clapper');
    expect(mascotTool('txt2img')).toBe('brush');
    expect(mascotTool('upscale')).toBe('brush');
    expect(mascotTool(undefined)).toBe('brush');
  });
});

describe('mascotSpot', () => {
  const at = (overrides: Partial<MascotSpotSignals>) => mascotSpot({
    view: 'latest', running: false, reviewing: false, reacting: false, lookingForHistory: false, welcoming: false, ...overrides,
  });

  it.each<[string, Partial<MascotSpotSignals>, ReturnType<typeof mascotSpot>]>([
    ['stays home while the latest output shows', {}, null],
    ['works on the job while it runs', { view: 'job', running: true }, 'job'],
    ['works on the job while reacting to its end', { view: 'job', reacting: true }, 'job'],
    ['leaves an idle job view alone', { view: 'job' }, null],
    ['waits beside a review instead of the job card', { view: 'job', running: true, reviewing: true }, 'review'],
    ['presents finished outputs while reacting', { view: 'outputs', reacting: true }, 'outputs'],
    ['leaves finished outputs alone otherwise', { view: 'outputs' }, null],
    ['explains a real empty stage', { view: 'empty' }, 'empty'],
    ['waits while still looking for history', { view: 'empty', lookingForHistory: true }, null],
    ['greets a first-time user in the welcome', { view: 'empty', welcoming: true }, 'welcome'],
    ['keeps the welcome while looking for history', { view: 'empty', welcoming: true, lookingForHistory: true }, 'welcome'],
    ['explains a load error', { view: 'error', reviewing: true }, 'error'],
  ])('%s', (_name, overrides, expected) => {
    expect(at(overrides)).toBe(expected);
  });
});

describe('homePlacement', () => {
  it('takes the Workspace placement while it is open', () => {
    const workspace = { away: true, mood: 'creating' as const, tool: 'clapper' as const };
    expect(homePlacement(workspace, null, 'txt2img')).toBe(workspace);
  });

  it('is away off the Workspace while a job is in flight', () => {
    expect(homePlacement(null, job('running', 3, 20), 'txt2img').away).toBe(true);
    expect(homePlacement(null, job('queued'), 'txt2img').away).toBe(true);
  });

  it('is home off the Workspace once the job ended, reacting to it', () => {
    expect(homePlacement(null, job('completed'), 'txt2img', 'cheerful')).toEqual({ away: false, mood: 'cheerful', tool: 'brush' });
    expect(homePlacement(null, null, 'txt2vid')).toEqual({ away: false, mood: 'idle', tool: 'clapper' });
  });
});

describe('mascotProgress', () => {
  it('is the share of the job\'s steps done', () => {
    expect(mascotProgress(job('running', 5, 20))).toBe(0.25);
    expect(mascotProgress(job('running', 20, 20))).toBe(1);
  });

  it('is unknown without a job, before the steps are known, and in the stages around generating', () => {
    expect(mascotProgress(null)).toBeUndefined();
    expect(mascotProgress(job('running', 0, 0))).toBeUndefined();
    expect(mascotProgress(job('running', 5, 20, { phase: 'load' }))).toBeUndefined();
    expect(mascotProgress(job('running', 5, 20, { stageName: 'upscale' }))).toBeUndefined();
  });
});
