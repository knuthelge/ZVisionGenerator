import { describe, it, expect, vi } from 'vitest';
import { connectJobSSE } from './sse';

describe('connectJobSSE', () => {
  it('calls onStep handler when step event received', () => {
    const onStep = vi.fn();
    const subscription = connectJobSSE('test-job', { onStep });
    const mockES = (globalThis.EventSource as unknown as { lastInstance: { emit: (type: string, data: unknown) => void } }).lastInstance;
    mockES.emit('step_progress', {
      type: 'step_progress',
      job_id: 'test-job',
      current_step: 1,
      total_steps: 20,
      elapsed_secs: 0.5,
      eta_secs: 9.5,
      workflow_stage_name: 'denoise',
      workflow_stage_index: 0,
      run_index: 0,
      total_runs: 1,
    });
    expect(onStep).toHaveBeenCalledOnce();
    subscription.close();
  });

  it('does not close connection on batch_completed for multi-run job', () => {
    const onClose = vi.fn();
    const subscription = connectJobSSE('test-job', { onClose });
    const mockES = (globalThis.EventSource as unknown as { lastInstance: { emit: (type: string, data: unknown) => void } }).lastInstance;
    // batch_completed mid-run should NOT trigger onClose
    mockES.emit('batch_completed', {
      type: 'batch_completed',
      job_id: 'test-job',
      run_index: 0,
      total_runs: 3,
      ran_iterations: 20,
      output_path: '/out/img.png',
      asset: { path: '/out/img.png', url: '/out/img.png', thumbnail_url: '/thumb.png', filename: 'img.png', created_at: '', workflow: 'txt2img', prompt: '', model: '', media_type: 'image', reuse_workspace_url: '' }
    });
    expect(onClose).not.toHaveBeenCalled();
    subscription.close();
    // only from manual close()
    expect(onClose).toHaveBeenCalledOnce();
  });

  it('closes connection on job_completed', () => {
    const onClose = vi.fn();
    const onJobCompleted = vi.fn();
    const subscription = connectJobSSE('test-job', { onJobCompleted, onClose });
    // Before terminal event, onClose not called
    expect(onClose).not.toHaveBeenCalled();
    const mockES = (globalThis.EventSource as unknown as { lastInstance: { emit: (type: string, data: unknown) => void } }).lastInstance;
    mockES.emit('job_completed', { type: 'job_completed', job_id: 'test-job', total_runs: 1, outputs: [] });
    expect(onJobCompleted).toHaveBeenCalledOnce();
    expect(onClose).toHaveBeenCalledOnce();
    // Prevent double-close warning — subscription already closed by terminal event
    void subscription;
  });
});
