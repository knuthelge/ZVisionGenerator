import { describe, expect, it } from 'vitest';

import type { MemoryReading } from '$lib/types';
import { memoryReadout } from './memoryReadout';

const unified = (used: number, budget = 21.3): MemoryReading => ({ kind: 'unified', used_gb: used, budget_gb: budget });
const discrete = (gpu: number, system: number): MemoryReading => ({ kind: 'discrete', gpu_gb: gpu, gpu_budget_gb: 10, system_gb: system, system_budget_gb: 30.3 });

describe('memoryReadout while the job runs', () => {
  it('shows the unified reading as one plain figure', () => {
    expect(memoryReadout({ status: 'running', memory: unified(18.2) })).toEqual({
      parts: [{ label: 'Memory', figure: '18.2', budget: '21.3' }],
      text: 'Memory 18.2 / 21.3 GB',
    });
  });

  it('shows GPU and RAM on a discrete GPU, dropping a trailing .0', () => {
    const readout = memoryReadout({ status: 'running', memory: discrete(7.1, 21.4) });
    expect(readout?.text).toBe('GPU 7.1 / 10 GB · RAM 21.4 / 30.3 GB');
    expect(readout?.parts.map((part) => part.label)).toEqual(['GPU', 'RAM']);
  });

  it('shows a dash for the GPU until its use is known, and still gives the RAM', () => {
    const readout = memoryReadout({ status: 'running', memory: { kind: 'discrete', gpu_gb: null, gpu_budget_gb: 10, system_gb: 21.4, system_budget_gb: 30.3 } });
    expect(readout?.text).toBe('GPU — / 10 GB · RAM 21.4 / 30.3 GB');
    expect(readout?.parts[0]).toEqual({ label: 'GPU', figure: '—', unknown: true, budget: '10' });
    expect(readout?.parts[1]).not.toHaveProperty('unknown');
  });

  it('rejects a GPU figure that is neither a number nor null', () => {
    const bad = { kind: 'discrete', gpu_gb: Number.NaN, gpu_budget_gb: 10, system_gb: 21.4, system_budget_gb: 30.3 } as MemoryReading;
    expect(memoryReadout({ status: 'running', memory: bad })).toBeNull();
  });

  it('carries no status word or colour, however full the figures are', () => {
    const readout = memoryReadout({ status: 'running', memory: discrete(10.4, 40) });
    expect(readout?.text).toBe('GPU 10.4 / 10 GB · RAM 40 / 30.3 GB');
    expect(readout).not.toHaveProperty('status');
    expect(readout?.parts.every((part) => !('status' in part))).toBe(true);
  });

  it('counts a queued or paused job as running', () => {
    expect(memoryReadout({ status: 'paused', memory: unified(1) })?.text).toBe('Memory 1 / 21.3 GB');
    expect(memoryReadout({ status: 'queued', memory: unified(1) })?.text).toBe('Memory 1 / 21.3 GB');
  });

  it('is null before the first reading', () => {
    expect(memoryReadout({ status: 'running', memory: null })).toBeNull();
    expect(memoryReadout({ status: 'running', memory: undefined })).toBeNull();
  });
});

describe('memoryReadout once the job has ended', () => {
  it('shows nothing, not even the last live reading', () => {
    expect(memoryReadout({ status: 'completed', memory: unified(18.2) })).toBeNull();
    expect(memoryReadout({ status: 'failed', memory: unified(18.2) })).toBeNull();
  });

  it('is null for a job that is not tracked', () => {
    expect(memoryReadout({ status: 'unknown', memory: null })).toBeNull();
  });
});
