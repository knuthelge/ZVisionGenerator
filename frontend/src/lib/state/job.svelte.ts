import type { ActiveJobState, JobContext, JobSnapshot, GalleryAsset, StepEvent, SSEEvent } from '$lib/types';
import { connectJobSSE } from '$lib/api/sse';
import type { SSESubscription } from '$lib/api/sse';
import { getJobSnapshot } from '$lib/api/workspace';
import { clearActiveJobId, readActiveJobId, writeActiveJobId } from './activeJobStorage';

let _job = $state<ActiveJobState | null>(null);
let _subscription: SSESubscription | null = null;

type JobCallbacks = {
  onComplete?: (outputs: GalleryAsset[]) => void;
  onFailed?: () => void;
  onCancelled?: () => void;
};

type ReconnectJobOptions = JobCallbacks & {
  snapshot?: JobSnapshot | null;
};

function eventFieldNumber(event: Record<string, unknown> | null | undefined, key: string): number {
  const value = event?.[key];
  return typeof value === 'number' ? value : 0;
}

function eventFieldString(event: Record<string, unknown> | null | undefined, key: string): string {
  const value = event?.[key];
  return typeof value === 'string' ? value : '';
}

function statusMessageForEvent(type: string | undefined, event: Record<string, unknown> | null | undefined): string {
  if (type === 'model_loading') {
    const model = eventFieldString(event, 'model');
    return `Loading ${model || 'model'}...`;
  }
  if (type === 'batch_started') {
    return 'Starting generation...';
  }
  if (type === 'workflow_stage_started') {
    const name = eventFieldString(event, 'stage_name');
    return name ? `Running ${name.replaceAll('_', ' ')}.` : 'Running workflow.';
  }
  if (type === 'workflow_stage_completed') {
    const name = eventFieldString(event, 'stage_name');
    return name ? `Finished ${name.replaceAll('_', ' ')}.` : 'Stage complete.';
  }
  if (type === 'generation_finished') {
    const filename = eventFieldString(event, 'filename');
    return filename ? `Wrote ${filename}.` : 'Generation finished.';
  }
  return '';
}

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

function makeJobStateFromSnapshot(snapshot: JobSnapshot): ActiveJobState {
  const lastEvent = snapshot.last_event ?? null;
  const isPaused = snapshot.paused || snapshot.status === 'paused';
  const batchIndex = eventFieldNumber(lastEvent, 'run_index');
  const totalRuns = eventFieldNumber(lastEvent, 'total_runs');
  const batchLabel = lastEvent?.type === 'batch_completed' && totalRuns > 0
    ? `Run ${batchIndex + 1} / ${totalRuns}`
    : '';
  const statusMessage = statusMessageForEvent(typeof lastEvent?.type === 'string' ? lastEvent.type : undefined, lastEvent);
  return {
    job_id: snapshot.job_id ?? snapshot.id,
    id: snapshot.id,
    workflow: snapshot.workflow,
    prompt: snapshot.prompt,
    model: snapshot.model,
    runs: snapshot.runs,
    created_at: String(snapshot.created_at),
    supported_controls: snapshot.supported_controls ?? [],
    status: snapshot.status,
    currentStep: eventFieldNumber(lastEvent, 'current_step'),
    totalSteps: eventFieldNumber(lastEvent, 'total_steps'),
    elapsed: eventFieldNumber(lastEvent, 'elapsed_secs'),
    remaining: eventFieldNumber(lastEvent, 'eta_secs'),
    stageName: eventFieldString(lastEvent, 'workflow_stage_name'),
    stageIndex: eventFieldNumber(lastEvent, 'workflow_stage_index'),
    batchLabel,
    batchIndex,
    paused: isPaused,
    message: isPaused ? 'Job paused. Resume to continue.' : (statusMessage || 'Reconnected to active job.'),
    outputs: snapshot.outputs ?? []
  };
}

function isTerminalStatus(status: string): boolean {
  return status === 'completed' || status === 'failed' || status === 'cancelled';
}

function closeSubscription(): void {
  const subscription = _subscription;
  _subscription = null;
  subscription?.close();
}

function connectSnapshot(snapshot: JobSnapshot, callbacks: JobCallbacks): true {
  _job = makeJobStateFromSnapshot(snapshot);
  writeActiveJobId(snapshot.job_id ?? snapshot.id);
  attachJobEvents(snapshot.job_id ?? snapshot.id, callbacks);
  return true;
}

function applyStatusEvent(type: string, event: SSEEvent): void {
  if (!_job) return;
  const data = event as unknown as Record<string, unknown>;
  const msg = statusMessageForEvent(type, data);
  if (msg) _job = { ..._job, message: msg };
}

function attachJobEvents(jobId: string, callbacks: JobCallbacks = {}): void {
  closeSubscription();
  _subscription = connectJobSSE(jobId, {
    onStep(event) {
      if (!_job) return;
      const ev = event as unknown as StepEvent;
      _job = {
        ..._job,
        status: _job.status === 'paused' ? 'paused' : 'running',
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
      const hasRunIndex = typeof event.run_index === 'number';
      const hasTotalRuns = typeof event.total_runs === 'number';
      _job = {
        ..._job,
        outputs: event.asset ? [..._job.outputs, event.asset] : _job.outputs,
        batchLabel: hasRunIndex && hasTotalRuns ? `Run ${event.run_index + 1} / ${event.total_runs}` : 'Batch completed',
        batchIndex: hasRunIndex ? event.run_index : _job.batchIndex,
        message: 'Batch completed.',
      };
    },
    onJobCompleted(event) {
      if (!_job) return;
      const ev = event as { outputs?: GalleryAsset[] };
      _job = { ..._job, status: 'completed', paused: false, outputs: ev.outputs ?? _job.outputs, message: 'Job completed.' };
      clearActiveJobId(_job.job_id);
      callbacks.onComplete?.(_job.outputs);
    },
    onJobFailed() {
      if (!_job) return;
      _job = { ..._job, status: 'failed', paused: false, message: 'Job failed.' };
      clearActiveJobId(_job.job_id);
      callbacks.onFailed?.();
    },
    onJobCancelled() {
      if (!_job) return;
      _job = { ..._job, status: 'cancelled', paused: false, message: 'Job stopped.' };
      clearActiveJobId(_job.job_id);
      callbacks.onCancelled?.();
    },
    onJobPaused() {
      if (!_job) return;
      _job = { ..._job, status: 'paused', paused: true, message: 'Job paused. Resume to continue.' };
    },
    onJobResumed() {
      if (!_job) return;
      _job = { ..._job, status: 'running', paused: false, message: 'Job resumed.' };
    },
    onStatus: applyStatusEvent,
    onClose() {
      _subscription = null;
    }
  });
}

export const jobStore = {
  get current(): ActiveJobState | null { return _job; },
  get isRunning(): boolean { return _job?.status === 'queued' || _job?.status === 'running' || _job?.status === 'paused'; },

  startJob(ctx: JobContext, onComplete?: (outputs: GalleryAsset[]) => void, onFailed?: () => void, onCancelled?: () => void): void {
    _job = makeInitialJobState(ctx);
    writeActiveJobId(ctx.job_id);
    attachJobEvents(ctx.job_id, { onComplete, onFailed, onCancelled });
  },

  async reconnectActiveJob(options: ReconnectJobOptions = {}): Promise<boolean> {
    const { snapshot = null, ...callbacks } = options;
    if (_job && !isTerminalStatus(_job.status)) return true;
    const snapshotJobId = snapshot ? (snapshot.job_id ?? snapshot.id) : null;
    if (snapshot) {
      if (isTerminalStatus(snapshot.status)) {
        if (snapshotJobId) clearActiveJobId(snapshotJobId);
      } else {
        return connectSnapshot(snapshot, callbacks);
      }
    }
    const jobId = readActiveJobId();
    if (!jobId) return false;
    try {
      const snapshot = await getJobSnapshot(jobId);
      if (isTerminalStatus(snapshot.status)) {
        clearActiveJobId(jobId);
        return false;
      }
      return connectSnapshot(snapshot, callbacks);
    } catch {
      clearActiveJobId(jobId);
      return false;
    }
  },

  clearJob(): void {
    if (_job) clearActiveJobId(_job.job_id);
    closeSubscription();
    _job = null;
  }
};
