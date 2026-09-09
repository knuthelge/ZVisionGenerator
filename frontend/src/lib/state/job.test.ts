import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { readActiveJobId } from './activeJobStorage';
import { jobStore } from './job.svelte';

describe('jobStore reconnect contract', () => {
  beforeEach(() => {
    sessionStorage.clear();
    jobStore.clearJob();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('updates paused state from pause and resume SSE events', () => {
    jobStore.startJob({
      job_id: 'job-1',
      workflow: 'txt2img',
      prompt: 'Test prompt',
      model: 'zit',
      runs: 1,
      created_at: '2026-04-22T00:00:00Z',
      supported_controls: ['pause', 'resume'],
    });

    const mockEventSource = (globalThis.EventSource as unknown as {
      lastInstance: { emit: (type: string, data: unknown) => void; close: () => void };
    }).lastInstance;

    mockEventSource.emit('job_paused', { type: 'job_paused', job_id: 'job-1' });
    expect(jobStore.current?.paused).toBe(true);

    mockEventSource.emit('job_resumed', { type: 'job_resumed', job_id: 'job-1' });
    expect(jobStore.current?.paused).toBe(false);

    mockEventSource.close();
  });

  it('tracks prompts across YAML groups and repeat runs, retaining context during steps', () => {
    jobStore.startJob({ job_id: 'prompts', workflow: 'txt2img', prompt: 'First', model: 'zit', runs: 2, created_at: '' });
    const source = (globalThis.EventSource as unknown as { lastInstance: { emit: (type: string, data: unknown) => void } }).lastInstance;
    source.emit('prompt_started', { type: 'prompt_started', prompt: 'Third', run_index: 0, total_runs: 2, ran_iterations: 3, total_iterations: 6, prompt_index: 0, total_prompts: 1 });
    expect(jobStore.current).toMatchObject({ prompt: 'Third', promptNumber: 3, promptCount: 3, batchIndex: 0, currentStep: 0, totalSteps: 0 });
    source.emit('step_progress', { type: 'step_progress', current_step: 4, total_steps: 20, elapsed_secs: 1 });
    expect(jobStore.current).toMatchObject({ prompt: 'Third', promptNumber: 3, currentStep: 4 });
    source.emit('prompt_started', { type: 'prompt_started', prompt: 'First again', run_index: 1, total_runs: 2, ran_iterations: 4, total_iterations: 6 });
    expect(jobStore.current).toMatchObject({ prompt: 'First again', promptNumber: 1, promptCount: 3, batchIndex: 1, currentStep: 0, totalSteps: 0 });
  });

  it('restores the current prompt and counter from a step snapshot', async () => {
    await jobStore.reconnectActiveJob({ snapshot: {
      id: 'restore-prompts', job_id: 'restore-prompts', workflow: 'txt2img', job_type: 'txt2img', status: 'running',
      prompt: 'Original', model: 'zit', runs: 2, created_at: '', event_count: 10, paused: false,
      last_event: { type: 'step_progress', prompt: 'Current', total_runs: 2, total_iterations: 6, ran_iterations: 5, run_index: 1, current_step: 2, total_steps: 20 },
    } });
    expect(jobStore.current).toMatchObject({ prompt: 'Current', promptNumber: 2, promptCount: 3, batchIndex: 1, currentStep: 2 });
  });

  it('clears stored continuity when reconnect finds a terminal snapshot', async () => {
    sessionStorage.setItem('ziv-active-job-id-v1', 'job-terminal');
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        id: 'job-terminal',
        job_id: 'job-terminal',
        workflow: 'txt2img',
        job_type: 'Text to Image',
        status: 'completed',
        created_at: '2026-04-22T00:00:00Z',
        completed_at: '2026-04-22T00:00:05Z',
        event_count: 4,
        last_event: { type: 'job_completed' },
        supported_controls: [],
        paused: false,
        result_path: '/tmp/output.png',
        prompt: 'Finished prompt',
        model: 'zit',
        runs: 1,
      }),
    });
    vi.stubGlobal('fetch', fetchMock);

    await expect(jobStore.reconnectActiveJob()).resolves.toBe(false);

    expect(readActiveJobId()).toBeNull();
    expect(jobStore.current).toBeNull();
  });

  it('reconnects a stored active job from snapshot lookup and preserves continuity state', async () => {
    sessionStorage.setItem('ziv-active-job-id-v1', 'job-reconnect');
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        id: 'job-reconnect',
        job_id: 'job-reconnect',
        workflow: 'txt2img',
        job_type: 'Text to Image',
        status: 'running',
        created_at: '2026-04-22T00:00:00Z',
        completed_at: null,
        event_count: 2,
        last_event: null,
        supported_controls: ['next', 'pause', 'resume', 'repeat', 'quit'],
        paused: false,
        result_path: null,
        prompt: 'Recovered prompt',
        model: 'zit',
        runs: 2,
      }),
    });
    vi.stubGlobal('fetch', fetchMock);

    await expect(jobStore.reconnectActiveJob()).resolves.toBe(true);

    expect(fetchMock).toHaveBeenCalledWith('/jobs/job-reconnect', { method: 'GET', headers: {}, body: undefined });
    expect(jobStore.current?.job_id).toBe('job-reconnect');
    expect(jobStore.current?.prompt).toBe('Recovered prompt');
    expect(readActiveJobId()).toBe('job-reconnect');

    const mockEventSource = (globalThis.EventSource as unknown as {
      lastInstance: { url: string };
    }).lastInstance;
    expect(mockEventSource.url).toBe('/jobs/job-reconnect/events');
  });

  it('restores an active job directly from a backend bootstrap snapshot', async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);

    const outputAsset = {
      id: 'outputs/result.png',
      url: '/media/outputs/result.png',
      thumbnail_url: '/media/outputs/result.png',
      filename: 'result.png',
      created_at: '2026-04-22T00:00:04Z',
      workflow: 'txt2vid' as const,
      prompt: 'Recovered video prompt',
      model: 'ltx-8',
      media_type: 'image' as const,
      reuse_workspace_url: '#/workspace?workflow=txt2vid',
    };

    await expect(jobStore.reconnectActiveJob({
      snapshot: {
        id: 'job-bootstrap',
        job_id: 'job-bootstrap',
        workflow: 'txt2vid',
        job_type: 'Text to Video',
        status: 'running',
        created_at: '2026-04-22T00:00:00Z',
        completed_at: null,
        event_count: 3,
        last_event: {
          type: 'step_progress',
          current_step: 2,
          total_steps: 8,
          elapsed_secs: 5,
          eta_secs: 12,
        },
        supported_controls: [],
        paused: false,
        result_path: null,
        outputs: [outputAsset],
        prompt: 'Recovered video prompt',
        model: 'ltx-8',
        runs: 1,
      },
    })).resolves.toBe(true);

    expect(fetchMock).not.toHaveBeenCalled();
    expect(jobStore.current?.job_id).toBe('job-bootstrap');
    expect(jobStore.current?.currentStep).toBe(2);
    expect(jobStore.current?.totalSteps).toBe(8);
    expect(jobStore.current?.remaining).toBe(12);
    expect(jobStore.current?.outputs).toEqual([outputAsset]);
    expect(readActiveJobId()).toBe('job-bootstrap');
  });

  it('reconnects batch progress from the backend iteration-shaped batch_completed snapshot', async () => {
    await expect(jobStore.reconnectActiveJob({
      snapshot: {
        id: 'job-batch-reconnect',
        job_id: 'job-batch-reconnect',
        workflow: 'txt2img',
        job_type: 'Text to Image',
        status: 'running',
        created_at: '2026-04-22T00:00:00Z',
        completed_at: null,
        event_count: 6,
        last_event: {
          type: 'batch_completed',
          mode: 'image',
          completed_iterations: 12,
          total_iterations: 20,
        },
        supported_controls: [],
        paused: false,
        result_path: null,
        outputs: [],
        prompt: 'Recovered batch prompt',
        model: 'zit',
        runs: 2,
      },
    })).resolves.toBe(true);

    expect(jobStore.current?.batchLabel).toBe('12 / 20 iterations');
    expect(jobStore.current?.message).toBe('Batch completed: 12 of 20 iterations.');
  });

  it('keeps one transport while lifecycle ownership moves from a detached consumer to a new consumer', () => {
    const ownerA = vi.fn();
    const ownerB = vi.fn();
    const detachA = jobStore.subscribeLifecycle({ onComplete: ownerA });

    jobStore.startJob({
      job_id: 'job-lifecycle',
      workflow: 'txt2img',
      prompt: 'Test prompt',
      model: 'zit',
      runs: 1,
      created_at: '2026-04-22T00:00:00Z',
    });

    const MockEventSource = globalThis.EventSource as unknown as {
      instances: Array<{ emit: (type: string, data: unknown) => void; closeCalls: number }>;
      lastInstance: { emit: (type: string, data: unknown) => void; closeCalls: number };
    };
    const constructionCount = MockEventSource.instances.length;
    const source = MockEventSource.lastInstance;

    detachA();
    detachA();
    const detachB = jobStore.subscribeLifecycle({ onComplete: ownerB });
    expect(jobStore.isRunning).toBe(true);

    source.emit('job_completed', { type: 'job_completed', job_id: 'job-lifecycle', total_runs: 1, outputs: [] });

    expect(ownerA).not.toHaveBeenCalled();
    expect(ownerB).toHaveBeenCalledOnce();
    expect(jobStore.current?.status).toBe('completed');
    expect(source.closeCalls).toBe(1);
    expect(MockEventSource.instances).toHaveLength(constructionCount);
    detachB();
  });

  it('isolates lifecycle sync throws and async rejections while terminalizing and closing', async () => {
    const reportError = vi.spyOn(console, 'error').mockImplementation(() => undefined);
    const healthy = vi.fn();
    const detachThrowing = jobStore.subscribeLifecycle({
      onComplete: () => { throw new Error('sync consumer failure'); },
    });
    const detachRejecting = jobStore.subscribeLifecycle({
      onComplete: async () => { throw new Error('async consumer failure'); },
    });
    const detachHealthy = jobStore.subscribeLifecycle({ onComplete: healthy });

    jobStore.startJob({
      job_id: 'job-isolation',
      workflow: 'txt2img',
      prompt: 'Test prompt',
      model: 'zit',
      runs: 1,
      created_at: '2026-04-22T00:00:00Z',
    });
    const source = (globalThis.EventSource as unknown as {
      lastInstance: { emit: (type: string, data: unknown) => void; closeCalls: number };
    }).lastInstance;

    source.emit('job_completed', { type: 'job_completed', job_id: 'job-isolation', total_runs: 1, outputs: [] });
    await Promise.resolve();
    await Promise.resolve();

    expect(healthy).toHaveBeenCalledOnce();
    expect(jobStore.current?.status).toBe('completed');
    expect(source.closeCalls).toBe(1);
    expect(reportError).toHaveBeenCalledTimes(2);

    detachThrowing();
    detachRejecting();
    detachHealthy();
    reportError.mockRestore();
  });

  it('retains its subscription after a transient EventSource error and clears it only after a real close', async () => {
    jobStore.startJob({
      job_id: 'job-retry',
      workflow: 'txt2img',
      prompt: 'Test prompt',
      model: 'zit',
      runs: 1,
      created_at: '2026-04-22T00:00:00Z',
    });
    const MockEventSource = globalThis.EventSource as unknown as {
      instances: Array<{ emitError: () => void; closeCalls: number }>;
      lastInstance: { emitError: () => void; closeCalls: number };
    };
    const source = MockEventSource.lastInstance;
    const constructionCount = MockEventSource.instances.length;

    source.emitError();
    await expect(jobStore.reconnectActiveJob()).resolves.toBe(true);

    expect(MockEventSource.instances).toHaveLength(constructionCount);
    expect(source.closeCalls).toBe(0);
    jobStore.clearJob();
    expect(source.closeCalls).toBe(1);
  });

  it('shows successful generation assets immediately, ignores informational/failed events, then accepts terminal order', () => {
    jobStore.startJob({
      job_id: 'job-progressive', workflow: 'txt2img', prompt: 'Test prompt', model: 'zit', runs: 3,
      created_at: '2026-04-22T00:00:00Z',
    });
    const source = (globalThis.EventSource as unknown as {
      lastInstance: { emit: (type: string, data: unknown) => void };
    }).lastInstance;
    const first = makeOutput('first');
    const second = makeOutput('second');
    const terminalOnly = makeOutput('terminal');

    source.emit('generation_finished', { type: 'generation_finished', job_id: 'job-progressive', status: 'success', run_index: 0, asset: first });
    source.emit('generation_finished', { type: 'generation_finished', job_id: 'job-progressive', status: 'success', run_index: 1, asset: second });
    source.emit('generation_finished', { type: 'generation_finished', job_id: 'job-progressive', status: 'success', run_index: 1, asset: first });
    expect(jobStore.current?.outputs.map((asset) => asset.id)).toEqual([first.id, second.id]);

    source.emit('generation_finished', { type: 'generation_finished', job_id: 'job-progressive', status: 'success', filename: 'no-asset.png' });
    expect(jobStore.current?.outputs.map((asset) => asset.id)).toEqual([first.id, second.id]);
    source.emit('generation_finished', { type: 'generation_finished', job_id: 'job-progressive', status: 'failed', filename: 'failed.png' });
    expect(jobStore.current?.message).toBe('Generation failed for failed.png.');
    source.emit('generation_finished', { type: 'generation_finished', job_id: 'job-progressive', status: 'skipped', filename: 'skipped.png' });
    expect(jobStore.current?.message).toBe('Skipped skipped.png.');
    source.emit('batch_completed', { type: 'batch_completed', job_id: 'job-progressive', completed_iterations: 2, total_iterations: 3 });
    expect(jobStore.current?.outputs.map((asset) => asset.id)).toEqual([first.id, second.id]);
    expect(jobStore.current?.message).toBe('Batch completed: 2 of 3 iterations.');

    source.emit('job_completed', {
      type: 'job_completed', job_id: 'job-progressive', total_runs: 3,
      outputs: [second, terminalOnly, second, first],
    });
    expect(jobStore.current?.status).toBe('completed');
    expect(jobStore.current?.outputs.map((asset) => asset.id)).toEqual([second.id, terminalOnly.id, first.id]);
  });

  it('preserves progressive assets for malformed or missing terminal output lists and for failed/cancelled jobs', () => {
    const terminalCases: Array<{ terminal: 'job_completed' | 'job_failed' | 'job_cancelled'; outputs?: unknown }> = [
      { terminal: 'job_completed' },
      { terminal: 'job_completed', outputs: [{ id: 'not-a-gallery-asset' }] },
      { terminal: 'job_failed' },
      { terminal: 'job_cancelled' },
    ];

    for (const [index, testCase] of terminalCases.entries()) {
      jobStore.clearJob();
      const jobId = `job-preserve-${index}`;
      jobStore.startJob({ job_id: jobId, workflow: 'txt2img', prompt: 'Test prompt', model: 'zit', runs: 1, created_at: '2026-04-22T00:00:00Z' });
      const source = (globalThis.EventSource as unknown as {
        lastInstance: { emit: (type: string, data: unknown) => void };
      }).lastInstance;
      const output = makeOutput(`preserved-${index}`);
      source.emit('generation_finished', { type: 'generation_finished', job_id: jobId, status: 'success', asset: output });
      source.emit(testCase.terminal, { type: testCase.terminal, job_id: jobId, total_runs: 1, ...(testCase.outputs === undefined ? {} : { outputs: testCase.outputs }) });

      expect(jobStore.current?.outputs).toEqual([output]);
      expect(jobStore.current?.status).toBe(testCase.terminal.replace('job_', ''));
    }
  });
});

function makeOutput(id: string) {
  return {
    id: `outputs/${id}.png`, url: `/media/${id}.png`, thumbnail_url: `/media/${id}.png`, filename: `${id}.png`,
    created_at: '2026-04-22T00:00:00Z', workflow: 'txt2img' as const, prompt: 'Test prompt', model: 'zit',
    media_type: 'image' as const, reuse_workspace_url: '#/workspace?workflow=txt2img',
  };
}
