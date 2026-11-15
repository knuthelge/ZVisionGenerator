import { beforeEach, describe, expect, it } from 'vitest';

import { jobStore } from './job.svelte';
import { ringView } from './jobRing';
import { headline, iterations, outcome } from './jobView';
import { lastEventSource } from '../../test-utils/eventSource';

const asset = (id: string) => ({ id, url: `/media/${id}`, thumbnail_url: `/media/${id}`, filename: id, created_at: '', project: null, workflow: 'txt2img', prompt: 'p', model: 'zit', media_type: 'image', reuse_workspace_url: '' });

/** A four-image batch: image 1 saved, image 2 stopped at step 3 of 8, as the runner reports it (`generation_finished` skipped, then `job_cancelled`). */
function replayStop(): (type: string, data?: Record<string, unknown>) => void {
  jobStore.clearJob();
  jobStore.startJob({ job_id: 'stopped-mid-image', workflow: 'txt2img', prompt: 'a fox', model: 'zit', runs: 4, created_at: '' });
  const source = lastEventSource();
  const emit = (type: string, data: Record<string, unknown> = {}) => source.emit(type, { type, timestamp: 100, ...data });
  emit('job_started');
  emit('batch_started', { mode: 'image' });
  emit('prompt_started', { prompt: 'a fox', run_index: 0, total_runs: 4, ran_iterations: 0, total_iterations: 4 });
  emit('step_progress', { current_step: 8, total_steps: 8, elapsed_secs: 4, workflow_stage_name: 'text_to_image', preview_version: 2 });
  emit('generation_finished', { status: 'success', asset: asset('one.png'), run_index: 0 });
  emit('prompt_started', { prompt: 'a fox', run_index: 1, total_runs: 4, ran_iterations: 1, total_iterations: 4 });
  emit('step_progress', { current_step: 3, total_steps: 8, elapsed_secs: 6, workflow_stage_name: 'text_to_image', preview_version: 3 });
  return emit;
}

describe('a job stopped inside an image, replayed event by event', () => {
  beforeEach(() => sessionStorage.clear());

  it('counts only what was saved, and says the same in the headline and in the save line', () => {
    const emit = replayStop();
    emit('generation_finished', { status: 'skipped', run_index: 1 });
    emit('job_cancelled', {});
    const job = jobStore.current!;

    expect(job.status).toBe('cancelled');
    expect(iterations(job).done).toBe(1);
    expect(headline(job)).toBe('Stopped after 1 of 4 images');
    expect(outcome(job)?.saved).toBe('1 of 4 saved to the Gallery · 3 not run');
  });

  it('draws the saved image teal, the interrupted one grey at the step it reached, and the rest as track', () => {
    const emit = replayStop();
    emit('generation_finished', { status: 'skipped', run_index: 1 });
    emit('job_cancelled', {});

    expect(ringView(jobStore.current!).arcs).toEqual([
      { fill: 1, ink: 'done' }, { fill: 3 / 8, ink: 'stopped' }, { fill: 0, ink: 'done' }, { fill: 0, ink: 'done' },
    ]);
  });

  it('keeps the last frame, so the stage can show it as the last preview', () => {
    const emit = replayStop();
    expect(jobStore.current?.previewUrl).toBe('/jobs/stopped-mid-image/preview?v=3');

    emit('generation_finished', { status: 'skipped', run_index: 1 });
    emit('job_cancelled', {});

    expect(jobStore.current?.previewUrl).toBe('/jobs/stopped-mid-image/preview?v=3');
  });

  it('drops the preview of an image that finished, and clears a kept one when the next image starts', () => {
    const emit = replayStop();
    emit('generation_finished', { status: 'skipped', run_index: 1 });
    emit('prompt_started', { prompt: 'a fox', run_index: 2, total_runs: 4, ran_iterations: 2, total_iterations: 4 });

    expect(jobStore.current?.previewUrl).toBeNull();
    emit('step_progress', { current_step: 1, total_steps: 8, elapsed_secs: 8, workflow_stage_name: 'text_to_image', preview_version: 4 });
    emit('generation_finished', { status: 'success', asset: asset('three.png'), run_index: 2 });
    expect(jobStore.current?.previewUrl).toBeNull();
  });

  it('does not count a skipped image as done while the job keeps running', () => {
    const emit = replayStop();
    emit('generation_finished', { status: 'skipped', run_index: 1 });

    expect(iterations(jobStore.current!)).toMatchObject({ index: 1, done: 1 });
    expect(ringView(jobStore.current!).arcs.map((arc) => arc.fill)).toEqual([1, 0, 0, 0]);
  });

  it('counts a finished image as done once its generation succeeded', () => {
    replayStop();
    const emit = lastEventSource();
    emit.emit('generation_finished', { type: 'generation_finished', status: 'success', asset: asset('two.png'), run_index: 1 });

    expect(iterations(jobStore.current!)).toMatchObject({ index: 1, done: 2 });
  });
});
