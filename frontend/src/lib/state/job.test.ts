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
});