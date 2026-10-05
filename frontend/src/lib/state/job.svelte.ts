import type { ActiveJobState, EnhanceStatus, JobContext, JobSnapshot, GalleryAsset, StepEvent, SSEEvent } from '$lib/types';
import { connectJobSSE } from '$lib/api/sse';
import type { SSESubscription } from '$lib/api/sse';
import { ApiError } from '$lib/api/client';
import { cancelJob, clearQueue, getJobSnapshot, jobPreviewUrl, listJobs } from '$lib/api/workspace';
import { clearActiveJobId, readActiveJobId, writeActiveJobId } from './activeJobStorage';

let _job = $state<ActiveJobState | null>(null);
// Generation jobs waiting behind the active one, oldest first (from the server; shared by every tab).
let _queued = $state<JobSnapshot[]>([]);
// Only the newest job-list request applies, so a slow older response cannot undo a newer one.
let _listRequest = 0;
// Event id the reconnect snapshot was taken at: replayed SSE history up to it must not change the live preview,
// which the snapshot already reflects.
let _previewEventFloor = 0;
let _subscription: SSESubscription | null = null;
// Consecutive stream-loss recoveries for the current job; reset whenever an event arrives.
let _recoveryAttempts = 0;
let _recoveryTimer: ReturnType<typeof setTimeout> | null = null;

const RECOVERY_BASE_DELAY_MS = 1000;
const RECOVERY_MAX_DELAY_MS = 30000;
/** How often a visible tab re-reads the job list while jobs are active, and while idle. */
export const QUEUE_SYNC_ACTIVE_MS = 4000;
export const QUEUE_SYNC_IDLE_MS = 15000;
/** `reason` of a `job_cancelled` event for a job removed from the queue before it started. */
const REMOVED_REASON = 'removed';

export type JobLifecycleCallbacks = {
  onComplete?: (outputs: GalleryAsset[]) => void | Promise<void>;
  onFailed?: () => void | Promise<void>;
  onCancelled?: () => void | Promise<void>;
  /** The server no longer knows the job (pruned after finishing, or restarted); its outcome is unknown. */
  onLost?: (outputs: GalleryAsset[]) => void | Promise<void>;
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

const ENHANCE_STATUSES: ReadonlySet<string> = new Set(['off', 'enhanced', 'failed', 'skipped']);

function eventEnhanceStatus(event: Record<string, unknown> | null | undefined): EnhanceStatus | undefined {
  const value = event?.enhance_status;
  return typeof value === 'string' && ENHANCE_STATUSES.has(value) ? (value as EnhanceStatus) : undefined;
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
  if (typeof event?.enhanced_prompt === 'string') progress.enhancedPrompt = event.enhanced_prompt;
  const enhanceStatus = eventEnhanceStatus(event);
  if (enhanceStatus) progress.enhanceStatus = enhanceStatus;
  if (typeof event?.run_index === 'number') progress.batchIndex = event.run_index;
  return progress;
}

function statusMessageForEvent(type: string | undefined, event: Record<string, unknown> | null | undefined): string {
  if (type === 'model_loading') {
    const model = eventFieldString(event, 'model');
    if (eventFieldString(event, 'phase') === 'saving_quant') {
      const bits = eventFieldNumber(event, 'quantize');
      return `Saving a q${bits} copy of ${model || 'the model'} for faster loading (first use only)...`;
    }
    return `Loading ${model || 'model'}...`;
  }
  if (type === 'batch_started') {
    return 'Starting generation...';
  }
  if (type === 'job_started') {
    return 'Starting...';
  }
  if (type === 'enhancer_loading') {
    const phase = eventFieldString(event, 'phase');
    if (phase === 'downloading') return 'Downloading prompt enhancer (first use only)...';
    if (phase === 'cpu') return 'Prompt enhancer runs on the CPU (no GPU found); each prompt may take a few minutes.';
    return 'Loading prompt enhancer...';
  }
  if (type === 'prompt_enhanced') {
    return 'Prompt enhanced.';
  }
  if (type === 'prompts_enhancing') {
    return `Enhancing prompt ${eventFieldNumber(event, 'index')} of ${eventFieldNumber(event, 'total')}...`;
  }
  if (type === 'prompt_enhance_failed') {
    return `Could not enhance prompt ${eventFieldNumber(event, 'index')} of ${eventFieldNumber(event, 'total')}; it uses the original prompt.`;
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

function previewUrlFor(jobId: string, version: number | undefined): string | null {
  return typeof version === 'number' && version > 0 ? jobPreviewUrl(jobId, version) : null;
}

function isReplayedBeforeSnapshot(event: Record<string, unknown>): boolean {
  return _previewEventFloor > 0 && eventFieldNumber(event, 'event_id') <= _previewEventFloor;
}

// Outputs the user deleted during this job; later events (e.g. the terminal output list) must not bring them back.
const _removedOutputIds = new Set<string>();

/** Drop duplicate outputs and outputs the user deleted. */
function dedupeOutputs(outputs: GalleryAsset[]): GalleryAsset[] {
  const seen = new Set<string>();
  return outputs.filter((output) => {
    if (seen.has(output.id) || _removedOutputIds.has(output.id)) return false;
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
    status: ctx.queue_position ? 'queued' : 'running',
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
    outputs: [],
    previewUrl: null
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
    notices: snapshot.notices ?? [],
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
    outputs: dedupeOutputs(snapshot.outputs ?? []),
    previewUrl: previewUrlFor(snapshot.job_id ?? snapshot.id, snapshot.preview_version)
  };
}

function isTerminalStatus(status: string): boolean {
  return status === 'completed' || status === 'failed' || status === 'cancelled' || status === 'unknown';
}

function cancelRecovery(): void {
  if (_recoveryTimer !== null) clearTimeout(_recoveryTimer);
  _recoveryTimer = null;
  _recoveryAttempts = 0;
}

function closeSubscription(): void {
  if (_recoveryTimer !== null) clearTimeout(_recoveryTimer);
  _recoveryTimer = null;
  const subscription = _subscription;
  _subscription = null;
  subscription?.close();
}

function connectSnapshot(snapshot: JobSnapshot): true {
  _job = makeJobStateFromSnapshot(snapshot);
  _previewEventFloor = eventFieldNumber(snapshot.last_event, 'event_id');
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
    ...(type === 'prompt_started' ? { currentStep: 0, totalSteps: 0, stageName: '', stageIndex: 0, message: 'Preparing generation.', enhancedPrompt: undefined, enhanceStatus: eventEnhanceStatus(data) } : {}),
    ...(type === 'prompts_enhancing' ? { stageName: 'enhancing_prompts', currentStep: Math.max(0, eventFieldNumber(data, 'index') - 1), totalSteps: eventFieldNumber(data, 'total') } : {}),
    ...(type === 'preflight_finished' ? { currentStep: 0, totalSteps: 0, stageName: '' } : {}),
    ...(type === 'workflow_stage_started' ? { stageName: eventFieldString(data, 'stage_name') } : {}),
    ...((type === 'prompt_started' || type === 'workflow_stage_started') && !isReplayedBeforeSnapshot(data) ? { previewUrl: null } : {}),
    ...(type === 'job_started' ? { status: 'running' as const } : {}),
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
        : callbackName === 'onLost'
          ? registration.callbacks.onLost?.(outputs)
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
      _recoveryAttempts = 0;
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
        previewUrl: isReplayedBeforeSnapshot(event as unknown as Record<string, unknown>)
          ? _job.previewUrl
          : previewUrlFor(_job.job_id, ev.preview_version) ?? _job.previewUrl,
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
        previewUrl: isReplayedBeforeSnapshot(event as unknown as Record<string, unknown>) ? _job.previewUrl : null,
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
      finishCompleted((event as { outputs?: unknown }).outputs);
    },
    onJobFailed() {
      finishFailed('Job failed.');
    },
    onJobCancelled(event) {
      if ((event as { reason?: unknown }).reason === REMOVED_REASON) finishRemoved();
      else finishCancelled();
    },
    onJobPaused() {
      if (!_job) return;
      _job = { ..._job, status: 'paused', paused: true, message: 'Job paused. Resume to continue.' };
    },
    onJobResumed() {
      if (!_job) return;
      _job = { ..._job, status: 'running', paused: false, message: 'Job resumed.' };
    },
    onStatus(type, event) {
      _recoveryAttempts = 0;
      applyStatusEvent(type, event);
    },
    onClose() {
      _subscription = null;
    },
    onStreamLost() {
      scheduleRecovery(jobId);
    }
  });
}

function finishCompleted(outputs: unknown): void {
  if (!_job) return;
  const terminalOutputs = validTerminalOutputs(outputs);
  _job = { ..._job, status: 'completed', paused: false, outputs: dedupeOutputs(terminalOutputs ?? _job.outputs), message: 'Job completed.' };
  clearActiveJobId(_job.job_id);
  notifyLifecycle('onComplete', _job.outputs);
  void refreshJobs();
}

function finishFailed(message: string): void {
  if (!_job) return;
  _job = { ..._job, status: 'failed', paused: false, message };
  clearActiveJobId(_job.job_id);
  notifyLifecycle('onFailed');
  void refreshJobs();
}

function finishCancelled(): void {
  if (!_job) return;
  _job = { ..._job, status: 'cancelled', paused: false, message: 'Job stopped.' };
  clearActiveJobId(_job.job_id);
  notifyLifecycle('onCancelled');
  void refreshJobs();
}

/** The followed job was taken off the queue before it started: not a stop, so no lifecycle callbacks. */
function finishRemoved(): void {
  if (!_job) return;
  _job = { ..._job, status: 'cancelled', paused: false, message: 'Removed from the queue.' };
  clearActiveJobId(_job.job_id);
  void refreshJobs();
}

function wasRemoved(snapshot: JobSnapshot): boolean {
  return snapshot.status === 'cancelled' && snapshot.last_event?.reason === REMOVED_REASON;
}

/** Follow a job the server reports as active, unless this tab already follows a live job. */
function followActive(active: JobSnapshot | null): void {
  if (!active || (_job && !isTerminalStatus(_job.status))) return;
  cancelRecovery();
  _removedOutputIds.clear();
  connectSnapshot(active);
}

/** Re-read the job list: refresh the queue and follow the next active job once the followed one has ended. */
async function refreshJobs(): Promise<void> {
  const request = ++_listRequest;
  try {
    const jobs = await listJobs();
    if (request !== _listRequest) return;
    _queued = jobs.queued_jobs ?? [];
    followActive(jobs.active_job ?? null);
  } catch {
    // Transient: the next sync, focus or job event tries again.
  }
}

function isCurrentLiveJob(jobId: string): boolean {
  return _job !== null && _job.job_id === jobId && !isTerminalStatus(_job.status);
}

/** Retry recovery with exponential backoff; the first attempt after a healthy stream runs immediately. */
function scheduleRecovery(jobId: string): void {
  const delay = _recoveryAttempts === 0 ? 0 : Math.min(RECOVERY_BASE_DELAY_MS * 2 ** (_recoveryAttempts - 1), RECOVERY_MAX_DELAY_MS);
  _recoveryAttempts += 1;
  if (_recoveryTimer !== null) clearTimeout(_recoveryTimer);
  _recoveryTimer = setTimeout(() => {
    _recoveryTimer = null;
    void recoverLostStream(jobId);
  }, delay);
}

function finishLost(): void {
  if (!_job) return;
  // The server keeps the final status of pruned jobs, so a 404 means it restarted: the outcome is unknown.
  _job = {
    ..._job,
    status: 'unknown',
    paused: false,
    outputs: dedupeOutputs(_job.outputs),
    message: 'Outcome unknown: the server restarted and no longer has this job. Check the gallery for its results.',
  };
  clearActiveJobId(_job.job_id);
  notifyLifecycle('onLost', _job.outputs);
}

/** Resolve the job's real state after its event stream closed without a terminal event. */
async function recoverLostStream(jobId: string): Promise<void> {
  if (!isCurrentLiveJob(jobId)) return;
  let snapshot: JobSnapshot;
  try {
    snapshot = await getJobSnapshot(jobId);
  } catch (error) {
    // Ignore if the store moved on (new job, cleared, or already terminal) while the snapshot was loading.
    if (!isCurrentLiveJob(jobId)) return;
    if (error instanceof ApiError && error.status === 404) {
      finishLost();
    } else {
      // Transient (network, 5xx, ...): keep the job and its stored id so a later attempt or reload can reattach.
      _job = { ..._job!, message: 'Connection to the job lost. Retrying...' };
      scheduleRecovery(jobId);
    }
    return;
  }
  if (!isCurrentLiveJob(jobId)) return;
  if (snapshot.status === 'completed') {
    finishCompleted(snapshot.outputs);
  } else if (wasRemoved(snapshot)) {
    finishRemoved();
  } else if (snapshot.status === 'cancelled') {
    finishCancelled();
  } else if (snapshot.status === 'failed') {
    finishFailed('Job failed.');
  } else {
    // Still running: reattach. The attempt count persists until an event arrives, so a stream that keeps failing backs off.
    connectSnapshot(snapshot);
  }
}

export const jobStore = {
  get current(): ActiveJobState | null { return _job; },
  get isRunning(): boolean { return _job?.status === 'queued' || _job?.status === 'running' || _job?.status === 'paused'; },
  /** Generation jobs waiting behind the active one, oldest first. */
  get queuedJobs(): JobSnapshot[] { return _queued; },
  /** Whether a job is running or waiting: new submissions join the queue and on-demand enhancement is unavailable. */
  get jobsActive(): boolean { return this.isRunning || _queued.length > 0; },

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

  /** Remove deleted assets from the current job's outputs. */
  removeOutputs(ids: Iterable<string>): void {
    for (const id of ids) _removedOutputIds.add(id);
    if (_job) _job = { ..._job, outputs: dedupeOutputs(_job.outputs) };
  },

  /** Take the response of a submit: follow the job when it is the active one, otherwise add it to the queue. */
  jobSubmitted(ctx: JobContext): void {
    if (ctx.queue_position) {
      void refreshJobs();
      return;
    }
    this.startJob(ctx);
    void refreshJobs();
  },

  /** Seed the queue from the workspace payload. */
  seedQueue(jobs: JobSnapshot[]): void {
    _queued = jobs;
  },

  refreshJobs,

  /** Remove one queued job (it never runs). */
  async removeQueued(jobId: string): Promise<void> {
    _queued = _queued.filter((job) => (job.job_id ?? job.id) !== jobId);
    try {
      await cancelJob(jobId);
    } finally {
      await refreshJobs();
    }
  },

  /** Remove every queued job; the active job keeps running. */
  async clearQueue(): Promise<void> {
    _queued = [];
    try {
      await clearQueue();
    } finally {
      await refreshJobs();
    }
  },

  /** Keep the job list current while this tab is visible: on focus, and on a timer (faster while jobs are active). */
  startSync(): () => void {
    let lastSync = 0;
    const sync = (): void => {
      lastSync = Date.now();
      void refreshJobs();
    };
    const onVisible = (): void => {
      if (!document.hidden) sync();
    };
    const timer = setInterval(() => {
      if (document.hidden) return;
      const interval = this.jobsActive ? QUEUE_SYNC_ACTIVE_MS : QUEUE_SYNC_IDLE_MS;
      if (Date.now() - lastSync >= interval - 50) sync();
    }, QUEUE_SYNC_ACTIVE_MS);
    window.addEventListener('focus', sync);
    document.addEventListener('visibilitychange', onVisible);
    sync();
    return () => {
      clearInterval(timer);
      window.removeEventListener('focus', sync);
      document.removeEventListener('visibilitychange', onVisible);
    };
  },

  startJob(ctx: JobContext): void {
    cancelRecovery();
    _removedOutputIds.clear();
    _job = makeInitialJobState(ctx);
    _previewEventFloor = 0;
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
    } catch (error) {
      // Only forget the job when the server says it does not exist; keep it on transient failures so a later reload can reattach.
      if (error instanceof ApiError && error.status === 404) clearActiveJobId(jobId);
      return false;
    }
  },

  clearJob(): void {
    if (_job) clearActiveJobId(_job.job_id);
    cancelRecovery();
    closeSubscription();
    _previewEventFloor = 0;
    _job = null;
    _queued = [];
    _listRequest += 1;
  }
};
