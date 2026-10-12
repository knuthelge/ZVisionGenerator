import { describe, expect, it } from 'vitest';

import type { ActiveJobState } from '$lib/types';
import { epochSeconds, initialPhase, nextPhase, phaseForEvent, phaseSeconds, settleControl, trackJobEvent } from './jobPhases';

function job(overrides: Partial<ActiveJobState> = {}): ActiveJobState {
  return {
    job_id: 'j', workflow: 'txt2img', prompt: '', model: 'zit', runs: 1, created_at: '', status: 'running',
    currentStep: 0, totalSteps: 0, elapsed: 0, remaining: 0, stageName: '', stageIndex: 0, batchLabel: '', batchIndex: 0,
    paused: false, message: '', outputs: [], previewUrl: null, ...overrides,
  };
}

describe('jobPhases', () => {
  it.each([
    ['job_submitted', {}, 'queue'],
    ['job_started', {}, 'load'],
    ['preflight_started', { total_rewrites: 2 }, 'enhance'],
    ['preflight_started', { total_rewrites: 0 }, null],
    ['prompts_enhancing', {}, 'enhance'],
    ['model_loading', {}, 'load'],
    ['batch_started', {}, 'generate'],
    ['step_progress', {}, 'generate'],
    ['control_queued', {}, null],
  ] as const)('maps %s %j to %s', (type, event, phase) => {
    expect(phaseForEvent(type, event)).toBe(phase);
  });

  it('never moves back, except into enhancement before the model starts loading', () => {
    expect(nextPhase({ phase: 'generate' }, 'load')).toBe('generate');
    expect(nextPhase({ phase: 'load', phaseMarks: {} }, 'enhance')).toBe('enhance');
    expect(nextPhase({ phase: 'load', phaseMarks: { load: { start: 5 } } }, 'enhance')).toBe('load');
    expect(nextPhase({ phase: 'enhance' }, 'load')).toBe('load');
    expect(nextPhase({ phase: undefined }, 'load')).toBe('load');
  });

  it('times a phase only once both of its ends are known', () => {
    expect(phaseSeconds({ load: { start: 10, end: 25 } }, 'load')).toBe(15);
    expect(phaseSeconds({ load: { start: 10 } }, 'load')).toBeUndefined();
    expect(phaseSeconds(undefined, 'queue')).toBeUndefined();
  });

  it('settles each queued control on the event that shows it took effect', () => {
    expect(settleControl('pause', 'step_progress')).toBe('pause');
    expect(settleControl('pause', 'job_paused')).toBeNull();
    expect(settleControl('next', 'generation_finished')).toBeNull();
    expect(settleControl('next', 'prompt_enhance_failed')).toBeNull();
    expect(settleControl('repeat', 'generation_repeat')).toBeNull();
    expect(settleControl('quit', 'generation_finished')).toBe('quit');
  });

  it('keeps a queued pause through the pause a review starts, and settles it on a plain pause', () => {
    expect(settleControl('pause', 'job_paused', { review: { iteration: 1 } })).toBe('pause');
    expect(settleControl('pause', 'job_paused', {})).toBeNull();
    expect(settleControl('pause', 'job_paused')).toBeNull();
  });

  it('keeps a pending pause, next or repeat through a review and its decision', () => {
    for (const queued of ['pause', 'next', 'repeat'] as const) {
      let state = job({ controlQueued: queued });
      state = trackJobEvent(state, 'job_paused', { review: { iteration: 1 } }, 0);
      expect(state.controlQueued).toBe(queued);
      state = trackJobEvent(state, 'control_queued', { action: 'approve' }, 0);
      expect(state.controlQueued).toBe(queued);
      state = trackJobEvent(state, 'control_queued', { action: 'discard' }, 0);
      expect(state.controlQueued).toBe(queued);
      state = trackJobEvent(state, 'job_resumed', {}, 0);
      expect(state.controlQueued).toBe(queued);
    }
  });

  it('closes the phase a job ended in, so its duration still shows', () => {
    const loading = trackJobEvent(job({ phase: 'load', phaseMarks: { load: { start: 10 } } }), 'job_failed', { timestamp: 40 }, 0);
    expect(phaseSeconds(loading.phaseMarks, 'load')).toBe(30);
  });

  it('measures the server clock against the client clock', () => {
    expect(trackJobEvent(job(), 'step_progress', { timestamp: 1010 }, 1_000_000).serverOffset).toBe(10);
  });

  it('reads created_at as epoch seconds or an ISO date', () => {
    expect(epochSeconds('100')).toBe(100);
    expect(epochSeconds('1970-01-01T00:01:40Z')).toBe(100);
    expect(epochSeconds('')).toBeUndefined();
  });

  it('starts a new job in the queue and a reconnected one in the phase of its last event', () => {
    expect(initialPhase('queued', '100')).toEqual({ phase: 'queue', phaseMarks: { queue: { start: 100 } } });
    expect(initialPhase('running', '100', { type: 'model_loading' }).phase).toBe('load');
    expect(initialPhase('running', '100', { type: 'control_queued' }).phase).toBeUndefined();
  });
});
