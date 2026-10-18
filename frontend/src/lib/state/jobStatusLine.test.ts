import { describe, expect, it } from 'vitest';

import type { ActiveJobState } from '$lib/types';
import { jobStatusLine } from './jobStatusLine';

function job(overrides: Partial<ActiveJobState> = {}): ActiveJobState {
  return {
    job_id: 'j1', workflow: 'txt2img', prompt: 'p', model: 'm', runs: 1, created_at: '', supported_controls: [],
    status: 'running', phase: 'generate', currentStep: 12, totalSteps: 32, elapsed: 0, remaining: 0,
    stageName: 'text_to_image', stageIndex: 0, batchLabel: '', batchIndex: 0, paused: false, message: '', outputs: [], previewUrl: null,
    ...overrides,
  } as ActiveJobState;
}

describe('jobStatusLine', () => {
  it('is Ready with nothing running and nothing queued', () => {
    expect(jobStatusLine(null, 0)).toEqual({ text: 'Ready', tone: 'idle' });
  });

  it('counts the queue when idle', () => {
    expect(jobStatusLine(null, 2).text).toBe('2 queued');
  });

  it('names the step while generating, with the item position for a batch', () => {
    expect(jobStatusLine(job(), 0)).toEqual({ text: 'Step 12 of 32', tone: 'running' });
    expect(jobStatusLine(job({ runs: 3, batchIndex: 1 }), 0).text).toBe('2/3 · Step 12 of 32');
  });

  it('names the phase before the steps', () => {
    expect(jobStatusLine(job({ phase: 'load' }), 0).text).toBe('Loading model…');
    expect(jobStatusLine(job({ phase: 'enhance' }), 0).text).toBe('Enhancing…');
    expect(jobStatusLine(job({ phase: 'queue', status: 'queued' }), 0).text).toBe('Queued');
  });

  it('says Review while a render waits', () => {
    expect(jobStatusLine(job({ status: 'paused', paused: true, review: { iteration: 1, decisions: ['approve', 'discard'] } as ActiveJobState['review'] }), 0)).toEqual({ text: 'Review', tone: 'review' });
  });
});
