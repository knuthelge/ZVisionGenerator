import { readFileSync } from 'node:fs';

import { beforeEach, describe, expect, it } from 'vitest';

import { jobStore } from '$lib/state/job.svelte';

function readSource(relativePath: string): string {
  return readFileSync(new URL(relativePath, import.meta.url), 'utf-8');
}

describe('fix-cycle-3 job card parity', () => {
  beforeEach(() => {
    jobStore.clearJob();
  });

  it('uses step_progress SSE naming across frontend and backend', () => {
    const sseSource = readSource('../lib/api/sse.ts');
    const runnerSource = readSource('../../../zvisiongenerator/image_runner.py');

    expect(sseSource).toContain("case 'step_progress'");
    expect(sseSource).toContain("'step_progress'");
    expect(runnerSource).toContain('"step_progress"');
  });

  it('captures eta_secs from step progress events in active job state', () => {
    jobStore.startJob({
      job_id: 'job-eta',
      workflow: 'txt2img',
      prompt: 'Parity test prompt',
      model: 'zit',
      runs: 3,
      created_at: '2026-04-22T00:00:00Z'
    });

    const mockEventSource = (globalThis.EventSource as unknown as {
      lastInstance: { emit: (type: string, data: unknown) => void; close: () => void };
    }).lastInstance;

    mockEventSource.emit('step_progress', {
      type: 'step_progress',
      job_id: 'job-eta',
      current_step: 4,
      total_steps: 20,
      elapsed_secs: 12,
      eta_secs: 48,
      workflow_stage_name: 'denoise',
      workflow_stage_index: 1,
      run_index: 1,
      total_runs: 3,
    });

    expect(jobStore.current?.currentStep).toBe(4);
    expect(jobStore.current?.totalSteps).toBe(20);
    expect(jobStore.current?.remaining).toBe(48);
    expect(jobStore.current?.batchIndex).toBe(1);

    mockEventSource.close();
  });

  it('keeps the batch counter and remaining time wiring in the job card', () => {
    const jobCardSource = readSource('../lib/components/molecules/JobCard.svelte');

    expect(jobCardSource).toContain('const batchMeta  = $derived(`${job.batchIndex + 1} / ${job.runs}`);');
    expect(jobCardSource).toContain('const remainingStr = $derived(job.remaining > 0 ? formatDuration(job.remaining) : \'--:--\');');
    expect(jobCardSource).toContain('remaining={remainingStr}');
  });

  it('keeps repeat control wiring and status message rendering', () => {
    const jobCardSource = readSource('../lib/components/molecules/JobCard.svelte');
    const workspaceSource = readSource('../features/workspace/WorkspacePage.svelte');
    const typesSource = readSource('../lib/types/index.ts');

    expect(jobCardSource).toContain('Repeat');
    expect(jobCardSource).toContain('onrepeat?.(job.job_id)');
    expect(workspaceSource).toContain('onrepeat={(id) => fetch(`/jobs/${id}/controls/repeat`, { method: \'POST\' })}');
    expect(typesSource).toContain('message: string;');
    expect(jobCardSource).toContain('{job.message}');
  });
});