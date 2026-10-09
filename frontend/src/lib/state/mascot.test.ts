import { describe, expect, it } from 'vitest';

import type { ActiveJobState, JobStatus } from '$lib/types';
import { mascotMood } from './mascot';

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

  it('thinks while prompts are being enhanced', () => {
    expect(mascotMood({ job: job('running', 2, 4, { stageName: 'enhancing_prompts' }) })).toBe('thinking');
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
