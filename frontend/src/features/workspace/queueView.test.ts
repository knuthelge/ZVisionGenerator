import { describe, expect, it } from 'vitest';

import type { JobSnapshot } from '$lib/types';
import { QUEUE_ROW_PX, queueMetaParts, shouldFoldQueue } from './queueView';

function queued(overrides: Partial<JobSnapshot> = {}): JobSnapshot {
  return {
    id: 'q', job_id: 'q', job_type: 'Text to Image', workflow: 'txt2img', prompt: 'a fox', model: 'zit', runs: 2, created_at: '', status: 'queued', event_count: 1, paused: false, settings: {}, ...overrides,
  };
}

describe('shouldFoldQueue', () => {
  it('folds only when a whole queue row would be hidden', () => {
    expect(shouldFoldQueue(0)).toBe(false);
    expect(shouldFoldQueue(-40)).toBe(false);
    expect(shouldFoldQueue(QUEUE_ROW_PX)).toBe(false);
    expect(shouldFoldQueue(QUEUE_ROW_PX + 1)).toBe(true);
  });

  it('lets a smaller overflow scroll: 11 px for a notice, 23 px for a two-line alert', () => {
    expect(shouldFoldQueue(11)).toBe(false);
    expect(shouldFoldQueue(29)).toBe(false);
  });
});

describe('queueMetaParts', () => {
  it('reads workflow, runs, model, then the size and steps, so the end is cut first', () => {
    expect(queueMetaParts(queued({ model_label: 'Z-Image Turbo', meta: '3:2 · M · 8 steps' }))).toEqual(['Text to image', '2 runs', 'Z-Image Turbo', '3:2', 'M', '8 steps']);
  });

  it('never shows a raw workflow id', () => {
    const [kind] = queueMetaParts(queued({ workflow: 'img2vid' }));

    expect(kind).not.toBe('img2vid');
    expect(kind).toBeTruthy();
  });

  it('falls back to the model id and singularises one run', () => {
    expect(queueMetaParts(queued({ runs: 1 }))).toEqual(['Text to image', '1 run', 'zit']);
  });

  it('gives a design job its voices instead of runs', () => {
    expect(queueMetaParts(queued({ workflow: 'voice_design', runs: 3, model_label: 'Qwen3-TTS 1.7B VoiceDesign', meta: '3 voices' }))).toEqual(['New voice', 'Qwen3-TTS 1.7B VoiceDesign', '3 voices']);
  });
});
