import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { JobSnapshot, MemoryReading } from '$lib/types';
import { jobStore } from './job.svelte';
import { lastEventSource } from '../../test-utils/eventSource';

const reading = (used: number): MemoryReading => ({ kind: 'unified', used_gb: used, budget_gb: 21.3 });
const usage = (id: number, used: number) => ({ type: 'memory_usage', event_id: id, job_id: 'job-mem', memory: reading(used) });

function snapshot(extra: Partial<JobSnapshot> = {}): JobSnapshot {
  return {
    id: 'job-mem', job_id: 'job-mem', workflow: 'txt2img', job_type: 'txt2img', status: 'running',
    prompt: 'P', model: 'zit', runs: 1, created_at: '', event_count: 5, paused: false,
    last_event: { type: 'step_progress', event_id: 4, current_step: 3, total_steps: 8, elapsed_secs: 12, eta_secs: 20 },
    ...extra,
  };
}

describe('jobStore memory readings', () => {
  beforeEach(() => {
    sessionStorage.clear();
    jobStore.clearJob();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('starts a job with no reading', () => {
    jobStore.startJob({ job_id: 'job-mem', workflow: 'txt2img', prompt: 'P', model: 'zit', runs: 1, created_at: '' });
    expect(jobStore.current).toMatchObject({ memory: null });
  });

  it('keeps only the newest reading', () => {
    jobStore.startJob({ job_id: 'job-mem', workflow: 'txt2img', prompt: 'P', model: 'zit', runs: 1, created_at: '' });
    const source = lastEventSource();
    source.emit('memory_usage', usage(2, 10));
    source.emit('memory_usage', usage(3, 12.5));
    expect(jobStore.current?.memory).toEqual(reading(12.5));
  });

  it('leaves step, ETA, status and phase alone', () => {
    jobStore.startJob({ job_id: 'job-mem', workflow: 'txt2img', prompt: 'P', model: 'zit', runs: 1, created_at: '' });
    const source = lastEventSource();
    source.emit('step_progress', { type: 'step_progress', event_id: 1, current_step: 3, total_steps: 8, elapsed_secs: 12, eta_secs: 20 });
    const before = { ...jobStore.current! };
    source.emit('memory_usage', usage(2, 10));
    expect(jobStore.current?.memory).toEqual(reading(10));
    const { memory: _after, ...after } = jobStore.current!;
    const { memory: _before, ...rest } = before;
    expect(after).toEqual(rest);
    expect(jobStore.current).toMatchObject({ status: 'running', currentStep: 3, totalSteps: 8, elapsed: 12, remaining: 20 });
  });

  it('ignores a reading that arrives after the job ended', () => {
    jobStore.startJob({ job_id: 'job-mem', workflow: 'txt2img', prompt: 'P', model: 'zit', runs: 1, created_at: '' });
    const source = lastEventSource();
    source.emit('memory_usage', usage(2, 10));
    source.emit('job_failed', { type: 'job_failed', job_id: 'job-mem', error: 'boom' });
    source.emit('memory_usage', usage(9, 99));
    expect(jobStore.current?.memory).toEqual(reading(10));
  });

  describe('reconnecting the stream', () => {
    const step = (id: number) => ({ type: 'step_progress', event_id: id, current_step: id, total_steps: 20, elapsed_secs: id, eta_secs: 5 });
    let stopWatching: () => void;

    function setHidden(hidden: boolean): void {
      Object.defineProperty(document, 'hidden', { configurable: true, value: hidden });
    }

    function leaveAndReturn(): void {
      setHidden(true);
      document.dispatchEvent(new Event('visibilitychange'));
      setHidden(false);
      document.dispatchEvent(new Event('visibilitychange'));
    }

    beforeEach(() => {
      stopWatching = jobStore.watchVisibility(document);
      jobStore.startJob({ job_id: 'job-mem', workflow: 'txt2img', prompt: 'P', model: 'zit', runs: 1, created_at: '' });
    });

    afterEach(() => {
      stopWatching();
      setHidden(false);
    });

    it('resumes after the newest memory reading, and a real event after it is still applied', () => {
      const old = lastEventSource();
      old.emit('step_progress', step(1));
      old.emit('memory_usage', usage(2, 10));
      old.emit('memory_usage', usage(3, 11));

      leaveAndReturn();

      const reopened = lastEventSource();
      expect(reopened).not.toBe(old);
      expect(reopened.url).toMatch(/^\/jobs\/job-mem\/events\?after=3$/);
      reopened.emit('step_progress', step(4));
      expect(jobStore.current).toMatchObject({ currentStep: 4, memory: reading(11) });
    });

    it('does not lower the resume point when a reading overtakes a real event', () => {
      const old = lastEventSource();
      old.emit('memory_usage', usage(5, 10));
      old.emit('step_progress', step(4));

      leaveAndReturn();

      expect(lastEventSource().url).toMatch(/after=5$/);
    });

    it('a snapshot restore replays the stream from the start, so no step before a reading is skipped', async () => {
      jobStore.clearJob();
      await jobStore.reconnectActiveJob({ snapshot: snapshot({ memory: reading(14.2) }) });
      expect(lastEventSource().url).toBe('/jobs/job-mem/events');
    });

    it('a replayed reading and the steps around it rebuild the same state as the live stream', async () => {
      jobStore.clearJob();
      await jobStore.reconnectActiveJob({ snapshot: snapshot({ memory: reading(14.2) }) });
      const source = lastEventSource();
      source.emit('step_progress', step(4));
      source.emit('memory_usage', usage(6, 15));
      source.emit('step_progress', step(7));
      expect(jobStore.current).toMatchObject({ currentStep: 7, remaining: 5, memory: reading(15), status: 'running' });
    });
  });

  describe('restored from a snapshot', () => {
    it('takes the reading from the snapshot, and the step state from the last event', async () => {
      await jobStore.reconnectActiveJob({ snapshot: snapshot({ memory: reading(14.2) }) });
      expect(jobStore.current).toMatchObject({ memory: reading(14.2), currentStep: 3, totalSteps: 8, elapsed: 12, remaining: 20, status: 'running' });
    });

    it('reads a snapshot without a memory field as no reading', async () => {
      await jobStore.reconnectActiveJob({ snapshot: snapshot() });
      expect(jobStore.current).toMatchObject({ memory: null });
    });

    it('lets the replayed history refresh the reading', async () => {
      await jobStore.reconnectActiveJob({ snapshot: snapshot({ memory: reading(14.2) }) });
      lastEventSource().emit('memory_usage', usage(6, 15));
      expect(jobStore.current?.memory).toEqual(reading(15));
    });
  });
});
