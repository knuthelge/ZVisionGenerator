import { describe, expect, it } from 'vitest';

import type { ActiveJobState } from '$lib/types';
import { downloadFromEvent, downloadStopFromEvent, epochSeconds, initialPhase, nextPhase, phaseForEvent, phaseSeconds, settleControl, trackJobEvent } from './jobPhases';

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
    ['model_downloading', { role: 'model' }, 'load'],
    ['model_downloading', { role: 'enhancer' }, 'enhance'],
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

  it('follows a model build and the load after it', () => {
    let state = job({ phase: 'load' });
    state = trackJobEvent(state, 'model_loading', { phase: 'building', model: 'ideo' }, 0);
    expect(state.loadPhase).toBe('building');

    state = trackJobEvent(state, 'model_loading', { phase: 'loading', model: 'ideo' }, 0);
    expect(state.loadPhase).toBe('loading');
  });

  it('follows the newest model_loading phase and whether the copy replaces the download', () => {
    let state = job({ phase: 'load' });
    state = trackJobEvent(state, 'model_loading', { phase: 'downloading', model: 'SeedVR2 7B q8' }, 0);
    expect(state.loadPhase).toBe('downloading');
    expect(state.loadQuant).toBeUndefined();
    expect(state.loadReplacesDownload).toBeUndefined();

    state = trackJobEvent(state, 'model_loading', { phase: 'saving_quant', quantize: 8, replaces_download: true }, 0);
    expect(state).toMatchObject({ loadPhase: 'saving_quant', loadQuant: 8, loadReplacesDownload: true });

    expect(trackJobEvent(job({ phase: 'load' }), 'model_loading', { phase: 'loading' }, 0).loadPhase).toBe('loading');
  });

  it('keeps the load phase when a model_loading event names none', () => {
    const state = trackJobEvent(job({ phase: 'load', loadPhase: 'loading' }), 'model_loading', { mode: 'image' }, 0);
    expect(state.loadPhase).toBe('loading');
  });

  describe('model downloads', () => {
    const progress = { bytes_done: 5, bytes_total: 10, bytes_per_sec: 2, download_eta_secs: 3, repo: 'org/repo', part: 1, parts: 2, shared: false };
    const dlEvent = (role: 'model' | 'enhancer') => ({ ...progress, role, model: 'klein9b' });

    it('reads the byte progress and role of a model_downloading event', () => {
      expect(downloadFromEvent(dlEvent('enhancer'))).toEqual({ ...progress, role: 'enhancer', model: 'klein9b' });
      expect(downloadFromEvent({ role: 'model', bytes_done: 1, model: 'm' })).toMatchObject({ bytes_total: null, bytes_per_sec: null, download_eta_secs: null, part: 1, parts: 1, shared: false });
      expect(downloadFromEvent({ role: 'model' })).toBeNull();
    });

    it('sets the download and moves the job into the row of its role', () => {
      const model = trackJobEvent(job({ phase: 'load' }), 'model_downloading', dlEvent('model'), 0);
      expect(model).toMatchObject({ phase: 'load', download: { role: 'model', bytes_done: 5 } });

      const enhancer = trackJobEvent(job({ phase: 'enhance', enhanceTotal: 2 }), 'model_downloading', dlEvent('enhancer'), 0);
      expect(enhancer).toMatchObject({ phase: 'enhance', download: { role: 'enhancer' } });
    });

    it('replaces the download with each newer event', () => {
      const first = trackJobEvent(job({ phase: 'load' }), 'model_downloading', dlEvent('model'), 0);
      const next = trackJobEvent(first, 'model_downloading', { ...dlEvent('model'), bytes_done: 9 }, 0);
      expect(next.download?.bytes_done).toBe(9);
    });

    it('clears a model download when the model starts loading, and an enhancer download when the enhancer does', () => {
      const model = job({ phase: 'load', download: downloadFromEvent(dlEvent('model')) });
      expect(trackJobEvent(model, 'model_loading', { phase: 'loading' }, 0).download).toBeNull();
      expect(trackJobEvent(model, 'enhancer_loading', {}, 0).download).not.toBeNull();

      const enhancer = job({ phase: 'enhance', download: downloadFromEvent(dlEvent('enhancer')) });
      for (const type of ['enhancer_loading', 'prompts_enhancing']) expect(trackJobEvent(enhancer, type, {}, 0).download).toBeNull();
      expect(trackJobEvent(enhancer, 'model_loading', {}, 0).download).not.toBeNull();
    });

    it('clears the download once generation starts', () => {
      const state = job({ phase: 'load', download: downloadFromEvent(dlEvent('model')) });
      expect(trackJobEvent(state, 'batch_started', {}, 0).download).toBeNull();
    });

    it('records how a cancelled job left its download', () => {
      const state = job({ phase: 'load', download: downloadFromEvent(dlEvent('model')) });
      expect(trackJobEvent(state, 'job_cancelled', { download: 'stopped' }, 0).downloadStop).toBe('stopped');
      expect(trackJobEvent(state, 'job_cancelled', { download: 'detached' }, 0).downloadStop).toBe('detached');
      expect(trackJobEvent(state, 'job_cancelled', {}, 0).downloadStop).toBeUndefined();
      expect(downloadStopFromEvent({ download: 'other' })).toBeUndefined();
    });

    it('times the load step from the download, so its duration covers it', () => {
      let state = trackJobEvent(job({ phase: 'queue' }), 'job_started', { timestamp: 100 }, 0);
      expect(state.phaseMarks?.load?.start).toBeUndefined();
      state = trackJobEvent(state, 'model_downloading', { ...dlEvent('model'), timestamp: 110 }, 0);
      expect(state.phaseMarks?.load?.start).toBe(110);
      state = trackJobEvent(state, 'model_downloading', { ...dlEvent('model'), timestamp: 130 }, 0);
      expect(state.phaseMarks?.load?.start).toBe(110);
      state = trackJobEvent(state, 'model_loading', { timestamp: 160 }, 0);
      state = trackJobEvent(state, 'batch_started', { timestamp: 170 }, 0);
      expect(phaseSeconds(state.phaseMarks, 'load')).toBe(60);
    });

    it('does not open the load step for an enhancer download', () => {
      const state = trackJobEvent(job({ phase: 'load' }), 'model_downloading', { ...dlEvent('enhancer'), timestamp: 100 }, 0);
      expect(state.phaseMarks?.load?.start).toBeUndefined();
    });

    it('takes the phase of a reconnected job from its last model_downloading event', () => {
      expect(initialPhase('running', '100', { type: 'model_downloading', role: 'model' }).phase).toBe('load');
      expect(initialPhase('running', '100', { type: 'model_downloading', role: 'enhancer' }).phase).toBe('enhance');
    });
  });
});
