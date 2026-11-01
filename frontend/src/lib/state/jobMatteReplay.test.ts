import { beforeEach, describe, expect, it } from 'vitest';

import { jobStore } from './job.svelte';
import { activeDownload, finishingLabel, headline, itemActivity, jobBar, phaseRows } from './jobView';
import { lastEventSource } from '../../test-utils/eventSource';

const MODEL = { type: 'model_downloading', mode: 'image', role: 'model', model: 'atlas', repo: 'org/atlas', part: 1, parts: 1, bytes_total: 1000, bytes_per_sec: 100, download_eta_secs: 1, shared: false };
const MATTE = { type: 'model_downloading', mode: 'image', role: 'matte', model: 'Background removal', bytes_total: 600, bytes_per_sec: 100, download_eta_secs: 1, shared: false };

function phaseRow(key: string) {
  return phaseRows(jobStore.current!).find((row) => row.key === key)!;
}

describe('a transparent image job, replayed event by event', () => {
  beforeEach(() => {
    sessionStorage.clear();
    jobStore.clearJob();
    jobStore.startJob({ job_id: 'matte-replay', workflow: 'txt2img', prompt: 'a fox', model: 'atlas', model_label: 'Atlas', runs: 1, created_at: '' });
  });

  it('names the matte download, then the loading, the steps and the cut-out', () => {
    const source = lastEventSource();
    const emit = (type: string, data: Record<string, unknown> = {}) => source.emit(type, { type, timestamp: 100, ...data });
    const current = () => jobStore.current!;

    emit('job_started');
    emit('model_downloading', { ...MODEL, bytes_done: 500 });
    expect(headline(current())).toBe('Downloading Atlas');
    expect(jobBar(current())).toMatchObject({ kind: 'download', role: 'model' });

    emit('model_downloading', { ...MODEL, bytes_done: 1000 });
    emit('model_downloading', { ...MATTE, repo: 'org/birefnet', part: 1, parts: 2, bytes_done: 100 });
    expect(headline(current())).toBe('Downloading background removal');
    expect(jobBar(current())).toMatchObject({ kind: 'download', role: 'matte' });
    expect(phaseRow('load').label).toBe('Background removal download');
    expect(current().message).toContain('Downloading background removal');
    expect(current().message).not.toContain('Atlas');

    emit('model_downloading', { ...MATTE, repo: 'org/vitmatte', part: 2, parts: 2, bytes_done: 600 });
    expect(headline(current())).toBe('Downloading background removal');
    expect(activeDownload(current())).toMatchObject({ role: 'matte', part: 2, parts: 2 });

    emit('model_loading', { mode: 'image', model: 'atlas', phase: 'loading' });
    expect(current().download).toBeNull();
    expect(activeDownload(current())).toBeNull();
    expect(headline(current())).not.toContain('background');
    expect(phaseRow('load')).toMatchObject({ label: 'Load model', status: 'current' });

    emit('batch_started', { mode: 'image' });
    emit('prompt_started', { prompt: 'a fox', run_index: 0, total_runs: 1, ran_iterations: 0, total_iterations: 1 });
    emit('workflow_started', { stage_names: ['text_to_image', 'matte', 'save'] });
    emit('workflow_stage_started', { stage_name: 'text_to_image' });
    emit('step_progress', { current_step: 8, total_steps: 8, elapsed_secs: 5, workflow_stage_name: 'text_to_image', phase: 'image_generation' });
    expect(itemActivity(current())).toBe('finishing');
    expect(headline(current())).toBe('Generating image');

    emit('workflow_stage_started', { stage_name: 'matte' });
    expect(current().message).toBe('Removing background.');
    expect(finishingLabel(current())).toBe('Removing background…');
    expect(itemActivity(current())).toBe('finishing');

    emit('step_progress', { current_step: 1, total_steps: 1, elapsed_secs: 6, workflow_stage_name: 'matte', phase: 'image_matte' });
    expect(itemActivity(current())).toBe('finishing');
    expect(finishingLabel(current())).toBe('Removing background…');
    expect(phaseRow('generate').status).toBe('current');

    emit('workflow_stage_started', { stage_name: 'save' });
    expect(finishingLabel(current())).toBe('Finishing up…');
    emit('generation_finished', { status: 'success' });
    emit('job_completed', { total_runs: 1, outputs: [] });

    const rows = phaseRows(current());
    expect(rows.map((row) => row.status)).toEqual(['done', 'done', 'done']);
    expect(rows.map((row) => row.label)).toEqual(['Queued', 'Load model', 'Generate']);
  });

  it('shows a matte download alone when the model is already on disk, and stopped text when stopped', () => {
    const source = lastEventSource();
    source.emit('job_started', { type: 'job_started', timestamp: 100 });
    source.emit('model_downloading', { ...MATTE, repo: 'org/birefnet', part: 1, parts: 2, bytes_done: 0, timestamp: 101 });
    expect(headline(jobStore.current!)).toBe('Downloading background removal');
    source.emit('job_cancelled', { type: 'job_cancelled', download: 'stopped', timestamp: 102 });
    expect(headline(jobStore.current!)).toBe('Stopped while downloading background removal');
  });
});
