import type { ActiveJobState, JobContext, GalleryAsset, StepEvent } from '$lib/types';
import { connectJobSSE } from '$lib/api/sse';

let _job = $state<ActiveJobState | null>(null);

function makeInitialJobState(ctx: JobContext): ActiveJobState {
  return {
    ...ctx,
    status: 'running',
    currentStep: 0,
    totalSteps: 0,
    elapsed: 0,
    remaining: 0,
    stageName: '',
    stageIndex: 0,
    batchLabel: '',
    batchIndex: 0,
    paused: false,
    message: 'Waiting for worker allocation...',
    outputs: []
  };
}

export const jobStore = {
  get current(): ActiveJobState | null { return _job; },
  get isRunning(): boolean { return _job?.status === 'running'; },

  startJob(ctx: JobContext, onComplete?: (outputs: GalleryAsset[]) => void, onFailed?: () => void): void {
    _job = makeInitialJobState(ctx);

    connectJobSSE(ctx.job_id, {
      onStep(event) {
        if (!_job) return;
        const ev = event as unknown as StepEvent;
        _job = {
          ..._job,
          currentStep: ev.current_step,
          totalSteps: ev.total_steps,
          elapsed: ev.elapsed_secs,
          remaining: ev.eta_secs ?? _job.remaining,
          stageName: ev.workflow_stage_name ?? _job.stageName,
          stageIndex: ev.workflow_stage_index ?? _job.stageIndex,
          batchIndex: ev.run_index ?? _job.batchIndex,
        };
      },
      onBatchCompleted(event) {
        if (!_job) return;
        _job = {
          ..._job,
          outputs: [..._job.outputs, event.asset],
          batchLabel: `Run ${event.run_index + 1} / ${event.total_runs}`,
          batchIndex: event.run_index,
          message: 'Batch completed.',
        };
      },
      onJobCompleted(event) {
        if (!_job) return;
        const ev = event as { outputs: GalleryAsset[] };
        _job = { ..._job, status: 'completed', outputs: ev.outputs ?? _job.outputs, message: 'Job completed.' };
        onComplete?.(_job.outputs);
      },
      onJobFailed(_event) {
        if (!_job) return;
        _job = { ..._job, status: 'failed', message: 'Job failed.' };
        onFailed?.();
      },
      onJobCancelled() {
        if (!_job) return;
        _job = { ..._job, status: 'cancelled', message: 'Job stopped.' };
      },
      onJobPaused() {
        if (!_job) return;
        _job = { ..._job, paused: true, message: 'Job paused. Resume to continue.' };
      },
      onJobResumed() {
        if (!_job) return;
        _job = { ..._job, paused: false, message: 'Job resumed.' };
      },
      onStatus(type, event) {
        if (!_job) return;
        const data = event as unknown as Record<string, unknown>;
        let msg = '';
        if (type === 'model_loading') {
          msg = `Loading ${(data.model as string | undefined) ?? 'model'}...`;
        } else if (type === 'batch_started') {
          msg = 'Starting generation...';
        } else if (type === 'workflow_stage_started') {
          const name = data.stage_name as string | undefined;
          msg = name ? `Running ${name.replaceAll('_', ' ')}.` : 'Running workflow.';
        } else if (type === 'workflow_stage_completed') {
          const name = data.stage_name as string | undefined;
          msg = name ? `Finished ${name.replaceAll('_', ' ')}.` : 'Stage complete.';
        } else if (type === 'generation_finished') {
          const filename = data.filename as string | undefined;
          msg = filename ? `Wrote ${filename}.` : 'Generation finished.';
        }
        if (msg) _job = { ..._job, message: msg };
      },
      onClose() {
        // SSE closed (terminal event or error)
      }
    });
  },

  clearJob(): void {
    _job = null;
  }
};
