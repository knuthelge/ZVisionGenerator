import { describe, expect, it } from 'vitest';

import type { ActiveJobState, JobStatus } from '$lib/types';
import { homePlacement, mascotMood, mascotSpot, mascotTool, type MascotSpotSignals } from './mascot';

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

describe('mascotTool', () => {
  it('holds a clapperboard for video and a brush otherwise', () => {
    expect(mascotTool('txt2vid')).toBe('clapper');
    expect(mascotTool('img2vid')).toBe('clapper');
    expect(mascotTool('txt2img')).toBe('brush');
    expect(mascotTool('upscale')).toBe('brush');
    expect(mascotTool(undefined)).toBe('brush');
  });
});

describe('mascotTool for extend', () => {
  it('holds the clapperboard for an extension', () => {
    expect(mascotTool('extend')).toBe('clapper');
    expect(mascotTool('txt2img')).toBe('brush');
  });
});

describe('mascotMood during a review', () => {
  it('looks to the user while a render waits for approval', () => {
    expect(mascotMood({ job: job('running', 3, 20, { review: { iteration: 1 } }) })).toBe('curious');
  });
});

describe('mascotSpot', () => {
  const at = (overrides: Partial<MascotSpotSignals>) => mascotSpot({
    view: 'latest', running: false, reviewing: false, reacting: false, lookingForHistory: false, welcoming: false, ...overrides,
  });

  it('stays home while the latest output shows', () => {
    expect(at({})).toBeNull();
  });

  it('works on the job while it runs and while reacting to its end', () => {
    expect(at({ view: 'job', running: true })).toBe('job');
    expect(at({ view: 'job', reacting: true })).toBe('job');
    expect(at({ view: 'job' })).toBeNull();
  });

  it('waits beside a review instead of the job card', () => {
    expect(at({ view: 'job', running: true, reviewing: true })).toBe('review');
  });

  it('presents finished outputs only while reacting', () => {
    expect(at({ view: 'outputs', reacting: true })).toBe('outputs');
    expect(at({ view: 'outputs' })).toBeNull();
  });

  it('explains a real empty stage, not one still looking for history', () => {
    expect(at({ view: 'empty' })).toBe('empty');
    expect(at({ view: 'empty', lookingForHistory: true })).toBeNull();
  });

  it('greets a first-time user in the welcome, in place of the empty stage', () => {
    expect(at({ view: 'empty', welcoming: true })).toBe('welcome');
    expect(at({ view: 'empty', welcoming: true, lookingForHistory: true })).toBe('welcome');
  });

  it('explains a load error', () => {
    expect(at({ view: 'error', reviewing: true })).toBe('error');
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
