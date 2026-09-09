import type { ActiveJobState, JobContext, JobSnapshot, GalleryAsset, StepEvent, SSEEvent } from '$lib/types';
import { connectJobSSE } from '$lib/api/sse';
import type { SSESubscription } from '$lib/api/sse';
import { getJobSnapshot } from '$lib/api/workspace';
import { clearActiveJobId, readActiveJobId, writeActiveJobId } from './activeJobStorage';

let _job = $state<ActiveJobState | null>(null);
let _subscription: SSESubscription | null = null;

export type JobLifecycleCallbacks = {
  onComplete?: (outputs: GalleryAsset[]) => void | Promise<void>;
  onFailed?: () => void | Promise<void>;
  onCancelled?: () => void | Promise<void>;
};

type ReconnectJobOptions = {
  snapshot?: JobSnapshot | null;
};

type LifecycleRegistration = {
  callbacks: JobLifecycleCallbacks;
};

const _lifecycleSubscribers = new Set<LifecycleRegistration>();

function eventFieldNumber(event: Record<string, unknown> | null | undefined, key: string): number {
  const value = event?.[key];
  return typeof value === 'number' ? value : 0;
}

function eventFieldString(event: Record<string, unknown> | null | undefined, key: string): string {
  const value = event?.[key];
  return typeof value === 'string' ? value : '';
}

function promptProgress(event: Record<string, unknown> | null | undefined): Partial<ActiveJobState> {
  const runs = eventFieldNumber(event, 'total_runs');
  const total = eventFieldNumber(event, 'total_iterations');
  const iteration = eventFieldNumber(event, 'ran_iterations');
  const count = total / runs;
  const progress: Partial<ActiveJobState> = {};
  // Iterations span all YAML groups; prompt_index is only local to one group.
  if (Number.isInteger(count) && count > 0 && Number.isInteger(iteration) && iteration > 0 && iteration <= total) {
    progress.promptCount = count;
    progress.promptNumber = ((iteration - 1) % count) + 1;
  }
  if (typeof event?.prompt === 'string') progress.prompt = event.prompt;
  if (typeof event?.run_index === 'number') progress.batchIndex = event.run_index;
  return progress;
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
    const status = eventFieldString(event, 'status');
    const filename = eventFieldString(event, 'filename');
    if (status === 'success') return filename ? `Wrote ${filename}.` : 'Generation finished.';
    if (status === 'failed') return filename ? `Generation failed for ${filename}.` : 'Generation failed.';
    if (status === 'skipped') return filename ? `Skipped ${filename}.` : 'Generation skipped.';
    return 'Generation finished.';
  }
  if (type === 'batch_completed') {
    const completed = eventFieldNumber(event, 'completed_iterations');
    const total = eventFieldNumber(event, 'total_iterations');
    return total > 0 ? `Batch completed: ${completed} of ${total} iterations.` : 'Batch completed.';
  }
  return '';
}

function dedupeOutputs(outputs: GalleryAsset[]): GalleryAsset[] {
  const seen = new Set<string>();
  return outputs.filter((output) => {
    if (seen.has(output.id)) return false;
    seen.add(output.id);
    return true;
  });
}

function isGalleryAsset(value: unknown): value is GalleryAsset {
  if (!value || typeof value !== 'object') return false;
  const asset = value as Partial<GalleryAsset>;
  return typeof asset.id === 'string' && asset.id.length > 0
    && typeof asset.url === 'string'
    && typeof asset.thumbnail_url === 'string'
    && typeof asset.filename === 'string'
    && (asset.media_type === 'image' || asset.media_type === 'video');
}

function validTerminalOutputs(value: unknown): GalleryAsset[] | null {
  if (!Array.isArray(value) || !value.every(isGalleryAsset)) return null;
  return dedupeOutputs(value);
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
  const completedIterations = eventFieldNumber(lastEvent, 'completed_iterations');
  const totalIterations = eventFieldNumber(lastEvent, 'total_iterations');
  const batchLabel = lastEvent?.type === 'batch_completed' && totalIterations > 0
    ? `${completedIterations} / ${totalIterations} iterations`
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
    ...promptProgress(lastEvent),
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

function connectSnapshot(snapshot: JobSnapshot): true {
  _job = makeJobStateFromSnapshot(snapshot);
  writeActiveJobId(snapshot.job_id ?? snapshot.id);
  attachJobEvents(snapshot.job_id ?? snapshot.id);
  return true;
}

function applyStatusEvent(type: string, event: SSEEvent): void {
  if (!_job) return;
  const data = event as unknown as Record<string, unknown>;
  const msg = statusMessageForEvent(type, data);
  _job = {
    ..._job,
    ...promptProgress(data),
    ...(msg ? { message: msg } : {}),
    ...(type === 'prompt_started' ? { currentStep: 0, totalSteps: 0, stageName: '', stageIndex: 0, message: 'Preparing generation.' } : {}),
    ...(type === 'workflow_stage_started' ? { stageName: eventFieldString(data, 'stage_name') } : {}),
  };
}

function reportLifecycleError(callbackName: keyof JobLifecycleCallbacks, error: unknown): void {
  console.error(`Job lifecycle subscriber ${callbackName} failed`, error);
}

function notifyLifecycle(
  callbackName: keyof JobLifecycleCallbacks,
  outputs: GalleryAsset[] = []
): void {
  for (const registration of Array.from(_lifecycleSubscribers)) {
    try {
      const result = callbackName === 'onComplete'
        ? registration.callbacks.onComplete?.(outputs)
        : callbackName === 'onFailed'
          ? registration.callbacks.onFailed?.()
          : registration.callbacks.onCancelled?.();
      if (result) {
        void Promise.resolve(result).catch((error: unknown) => {
          reportLifecycleError(callbackName, error);
        });
      }
    } catch (error) {
      reportLifecycleError(callbackName, error);
    }
  }
}

function attachJobEvents(jobId: string): void {
  closeSubscription();
  _subscription = connectJobSSE(jobId, {
    onStep(event) {
      if (!_job) return;
      const ev = event as unknown as StepEvent;
      _job = {
        ..._job,
        status: _job.status === 'paused' ? 'paused' : 'running',
        ...promptProgress(event as unknown as Record<string, unknown>),
        currentStep: ev.current_step,
        totalSteps: ev.total_steps,
        elapsed: ev.elapsed_secs,
        remaining: ev.eta_secs ?? _job.remaining,
        stageName: ev.workflow_stage_name ?? _job.stageName,
        stageIndex: ev.workflow_stage_index ?? _job.stageIndex,
        batchIndex: ev.run_index ?? _job.batchIndex,
      };
    },
    onGenerationFinished(event) {
      if (!_job) return;
      const asset = event.status === 'success' && isGalleryAsset(event.asset) ? event.asset : null;
      const outputs = asset
        ? dedupeOutputs([..._job.outputs, asset])
        : _job.outputs;
      _job = {
        ..._job,
        outputs,
        batchIndex: typeof event.run_index === 'number' ? event.run_index : _job.batchIndex,
      };
    },
    onBatchCompleted(event) {
      if (!_job) return;
      const completed = typeof event.completed_iterations === 'number' ? event.completed_iterations : null;
      const total = typeof event.total_iterations === 'number' ? event.total_iterations : null;
      _job = {
        ..._job,
        batchLabel: completed !== null && total !== null ? `${completed} / ${total} iterations` : 'Batch completed',
        message: completed !== null && total !== null ? `Batch completed: ${completed} of ${total} iterations.` : 'Batch completed.',
      };
    },
    onJobCompleted(event) {
      if (!_job) return;
      const ev = event as { outputs?: unknown };
      const terminalOutputs = validTerminalOutputs(ev.outputs);
      _job = { ..._job, status: 'completed', paused: false, outputs: terminalOutputs ?? dedupeOutputs(_job.outputs), message: 'Job completed.' };
      clearActiveJobId(_job.job_id);
      notifyLifecycle('onComplete', _job.outputs);
    },
    onJobFailed() {
      if (!_job) return;
      _job = { ..._job, status: 'failed', paused: false, message: 'Job failed.' };
      clearActiveJobId(_job.job_id);
      notifyLifecycle('onFailed');
    },
    onJobCancelled() {
      if (!_job) return;
      _job = { ..._job, status: 'cancelled', paused: false, message: 'Job stopped.' };
      clearActiveJobId(_job.job_id);
      notifyLifecycle('onCancelled');
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

  subscribeLifecycle(callbacks: JobLifecycleCallbacks): () => void {
    const registration = { callbacks };
    let subscribed = true;
    _lifecycleSubscribers.add(registration);
    return () => {
      if (!subscribed) return;
      subscribed = false;
      _lifecycleSubscribers.delete(registration);
    };
  },

  startJob(ctx: JobContext): void {
    _job = makeInitialJobState(ctx);
    writeActiveJobId(ctx.job_id);
    attachJobEvents(ctx.job_id);
  },

  async reconnectActiveJob(options: ReconnectJobOptions = {}): Promise<boolean> {
    const { snapshot = null } = options;
    if (_job && !isTerminalStatus(_job.status)) return true;
    const snapshotJobId = snapshot ? (snapshot.job_id ?? snapshot.id) : null;
    if (snapshot) {
      if (isTerminalStatus(snapshot.status)) {
        if (snapshotJobId) clearActiveJobId(snapshotJobId);
      } else {
        return connectSnapshot(snapshot);
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
      return connectSnapshot(snapshot);
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
