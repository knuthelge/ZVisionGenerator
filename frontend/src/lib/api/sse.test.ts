import { describe, it, expect, vi } from 'vitest';
import { connectJobSSE } from './sse';

type MockEventSource = {
  emit: (type: string, data: unknown) => void;
  emitError: () => void;
  close: () => void;
  closeCalls: number;
  readyState: number;
};

function latestEventSource(): MockEventSource {
  return (globalThis.EventSource as unknown as { lastInstance: MockEventSource }).lastInstance;
}

describe('connectJobSSE', () => {
  it('calls onStep handler when step event received', () => {
    const onStep = vi.fn();
    const subscription = connectJobSSE('test-job', { onStep });
    const mockES = latestEventSource();
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
    const mockES = latestEventSource();
    // batch_completed mid-run should NOT trigger onClose
    mockES.emit('batch_completed', {
      type: 'batch_completed',
      job_id: 'test-job',
      completed_iterations: 20,
      total_iterations: 60,
    });
    expect(onClose).not.toHaveBeenCalled();
    subscription.close();
    // only from manual close()
    expect(onClose).toHaveBeenCalledOnce();
  });

  it('routes progressive assets from generation_finished while keeping batch completion informational', () => {
    const onGenerationFinished = vi.fn();
    const onBatchCompleted = vi.fn();
    const onStatus = vi.fn();
    const subscription = connectJobSSE('test-job', { onGenerationFinished, onBatchCompleted, onStatus });
    const mockES = latestEventSource();
    const asset = {
      id: 'outputs/first.png', url: '/media/first.png', thumbnail_url: '/media/first.png', filename: 'first.png',
      created_at: '', workflow: 'txt2img', prompt: '', model: '', media_type: 'image', reuse_workspace_url: '',
    };

    mockES.emit('generation_finished', {
      type: 'generation_finished', job_id: 'test-job', status: 'success', run_index: 0, asset,
    });
    mockES.emit('generation_finished', {
      type: 'generation_finished', job_id: 'test-job', status: 'failed', run_index: 1, filename: 'failed.png',
    });
    mockES.emit('batch_completed', {
      type: 'batch_completed', job_id: 'test-job', completed_iterations: 2, total_iterations: 3,
    });

    expect(onGenerationFinished).toHaveBeenCalledTimes(2);
    expect(onGenerationFinished).toHaveBeenNthCalledWith(1, expect.objectContaining({ status: 'success', asset }));
    expect(onGenerationFinished).toHaveBeenNthCalledWith(2, expect.objectContaining({ status: 'failed' }));
    expect(onBatchCompleted).toHaveBeenCalledWith(expect.objectContaining({ completed_iterations: 2, total_iterations: 3 }));
    expect(onStatus).toHaveBeenNthCalledWith(1, 'generation_finished', expect.objectContaining({ asset }));
    expect(onStatus).toHaveBeenNthCalledWith(2, 'generation_finished', expect.objectContaining({ status: 'failed' }));
    expect(onStatus).not.toHaveBeenCalledWith('batch_completed', expect.anything());
    expect(mockES.closeCalls).toBe(0);
    subscription.close();
  });

  it('closes connection on job_completed', () => {
    const onClose = vi.fn();
    const onJobCompleted = vi.fn();
    const subscription = connectJobSSE('test-job', { onJobCompleted, onClose });
    // Before terminal event, onClose not called
    expect(onClose).not.toHaveBeenCalled();
    const mockES = latestEventSource();
    mockES.emit('job_completed', { type: 'job_completed', job_id: 'test-job', total_runs: 1, outputs: [] });
    expect(onJobCompleted).toHaveBeenCalledOnce();
    expect(onClose).toHaveBeenCalledOnce();
    expect(mockES.closeCalls).toBe(1);
    subscription.close();
    expect(onClose).toHaveBeenCalledOnce();
    expect(mockES.closeCalls).toBe(1);
  });

  it('closes connection on job_cancelled', () => {
    const onClose = vi.fn();
    const onJobCancelled = vi.fn();
    const subscription = connectJobSSE('test-job', { onJobCancelled, onClose });
    const mockES = latestEventSource();
    mockES.emit('job_cancelled', { type: 'job_cancelled', job_id: 'test-job' });
    expect(onJobCancelled).toHaveBeenCalledOnce();
    expect(onClose).toHaveBeenCalledOnce();
    expect(mockES.closeCalls).toBe(1);
    subscription.close();
    expect(onClose).toHaveBeenCalledOnce();
    expect(mockES.closeCalls).toBe(1);
  });

  it('leaves the EventSource open and does not report a close on a transient error', () => {
    const onClose = vi.fn();
    const subscription = connectJobSSE('test-job', { onClose });
    const mockES = latestEventSource();

    mockES.emitError();

    expect(mockES.closeCalls).toBe(0);
    expect(onClose).not.toHaveBeenCalled();
    subscription.close();
    expect(mockES.closeCalls).toBe(1);
    expect(onClose).toHaveBeenCalledOnce();
  });

  it('closes exactly once when a terminal handler throws', () => {
    const onClose = vi.fn();
    const subscription = connectJobSSE('test-job', {
      onJobFailed: () => { throw new Error('consumer failure'); },
      onClose,
    });
    const mockES = latestEventSource();

    mockES.emit('job_failed', { type: 'job_failed', job_id: 'test-job' });

    expect(mockES.closeCalls).toBe(1);
    expect(onClose).toHaveBeenCalledOnce();
    subscription.close();
    expect(mockES.closeCalls).toBe(1);
    expect(onClose).toHaveBeenCalledOnce();
  });

  it('makes repeated manual close calls idempotent', () => {
    const onClose = vi.fn();
    const subscription = connectJobSSE('test-job', { onClose });
    const mockES = latestEventSource();

    subscription.close();
    subscription.close();

    expect(mockES.closeCalls).toBe(1);
    expect(onClose).toHaveBeenCalledOnce();
  });
});
