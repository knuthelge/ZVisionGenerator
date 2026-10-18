import type { ActiveJobState, JobControlAction, JobPhase, PhaseMarks } from '$lib/types';

type EventData = Record<string, unknown> | null | undefined;

/** Enhancement and model load are both setup: a job that started may still turn out to enhance first. */
const PHASE_RANK: Record<JobPhase, number> = { queue: 0, enhance: 1, load: 1, generate: 2 };
const GENERATE_EVENTS: ReadonlySet<string> = new Set([
  'batch_started', 'prompt_started', 'workflow_started', 'workflow_stage_started', 'step_progress', 'generation_finished',
]);
const CONTROL_ACTIONS: ReadonlySet<string> = new Set(['pause', 'resume', 'next', 'repeat', 'approve', 'discard', 'quit']);
/** Controls whose control_queued event does not become the card's pending label: Resume acts at once, a review decision is the overlay's. */
const UNQUEUED_ACTIONS: ReadonlySet<string> = new Set(['resume', 'approve', 'discard']);
const TERMINAL_EVENTS: ReadonlySet<string> = new Set(['job_completed', 'job_failed', 'job_cancelled']);

/** Events after which a queued control has taken effect. Quit ends the job, so it never settles here. */
const SETTLED_BY: Record<JobControlAction, ReadonlySet<string>> = {
  pause: new Set(['job_paused']),
  resume: new Set(['job_resumed']),
  next: new Set(['prompt_started', 'generation_finished', 'prompts_enhancing', 'prompt_enhance_failed', 'preflight_finished']),
  repeat: new Set(['prompt_started', 'generation_finished', 'generation_repeat']),
  // A review decision is never queued here: it must not replace the pause, repeat or next still waiting to apply.
  approve: new Set(),
  discard: new Set(),
  quit: new Set(),
};

function numberField(event: EventData, key: string): number | undefined {
  const value = event?.[key];
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined;
}

/** The phase an event moves the job into, or null when the event does not say. */
export function phaseForEvent(type: string, event: EventData): JobPhase | null {
  if (type === 'job_submitted') return 'queue';
  // A started job loads its model next, unless prompt enhancement comes first (see nextPhase).
  if (type === 'job_started' || type === 'model_loading') return 'load';
  if (type === 'preflight_started') return (numberField(event, 'total_rewrites') ?? 0) > 0 ? 'enhance' : null;
  if (type === 'enhancer_loading' || type === 'prompts_enhancing') return 'enhance';
  if (GENERATE_EVENTS.has(type)) return 'generate';
  return null;
}

/**
 * The phase after an event. Phases only move forward (a replayed or late event never moves the job back), except that
 * a job that started but has not begun loading its model may still enhance prompts first.
 */
export function nextPhase(job: Pick<ActiveJobState, 'phase' | 'phaseMarks'>, target: JobPhase | null): JobPhase | undefined {
  if (target === null) return job.phase;
  if (!job.phase || PHASE_RANK[target] > PHASE_RANK[job.phase]) return target;
  if (job.phase === 'enhance' && target === 'load') return 'load';
  if (job.phase === 'load' && target === 'enhance' && job.phaseMarks?.load?.start === undefined) return 'enhance';
  return job.phase;
}

/**
 * Close the phase that ended and open the next one at *at* (epoch seconds).
 *
 * A phase only gets an end once it has a start. `job_started` closes the queue without starting the load: the load is
 * timed from its own `model_loading` event, since prompt enhancement may come first.
 */
export function markPhase(marks: PhaseMarks, from: JobPhase | undefined, to: JobPhase, at: number | undefined, opensTarget = true): PhaseMarks {
  if (at === undefined || from === to) return marks;
  const next: PhaseMarks = { ...marks };
  if (from && next[from]?.start !== undefined) next[from] = { ...next[from], end: next[from]?.end ?? at };
  if (opensTarget) next[to] = { ...next[to], start: next[to]?.start ?? at };
  return next;
}

/** Seconds a setup phase took, or undefined when either end is unknown (e.g. after a reload). */
export function phaseSeconds(marks: PhaseMarks | undefined, phase: JobPhase): number | undefined {
  const mark = marks?.[phase];
  if (mark?.start === undefined || mark.end === undefined) return undefined;
  return Math.max(0, mark.end - mark.start);
}

/**
 * Drop a queued control once the event shows it took effect.
 *
 * The pause a review starts (a `job_paused` that carries a `review`) is not the pause the user queued: that one waits
 * for the image to finish.
 */
export function settleControl(queued: JobControlAction | null | undefined, type: string, event?: EventData): JobControlAction | null {
  if (!queued) return null;
  if (queued === 'pause' && type === 'job_paused' && event?.review) return queued;
  return SETTLED_BY[queued].has(type) ? null : queued;
}

/**
 * Fold the phase, phase timing, stage plan and queued-control bookkeeping of one event into the job.
 *
 * Pure: the store applies it to every event before its own handling.
 */
export function trackJobEvent(job: ActiveJobState, type: string, event: EventData, receivedAt: number): ActiveJobState {
  const at = numberField(event, 'timestamp');
  const phase = nextPhase(job, phaseForEvent(type, event));
  let phaseMarks = job.phaseMarks ?? {};
  if (phase && phase !== job.phase) phaseMarks = markPhase(phaseMarks, job.phase, phase, at, type !== 'job_started');
  if (type === 'model_loading' && at !== undefined && phaseMarks.load?.start === undefined) {
    phaseMarks = { ...phaseMarks, load: { ...phaseMarks.load, start: at } };
  }

  // A terminal event closes the phase the job ended in, so its duration can still be shown.
  if (TERMINAL_EVENTS.has(type) && job.phase && at !== undefined) {
    phaseMarks = { ...phaseMarks, [job.phase]: { ...phaseMarks[job.phase], end: phaseMarks[job.phase]?.end ?? at } };
  }

  const next: ActiveJobState = {
    ...job,
    phase,
    phaseMarks,
    connection: 'live',
    lastEventAt: receivedAt,
    serverOffset: at !== undefined ? at - receivedAt / 1000 : job.serverOffset,
    controlQueued: settleControl(job.controlQueued, type, event),
  };

  if (type === 'control_queued') {
    const action = event?.action;
    if (typeof action === 'string' && CONTROL_ACTIONS.has(action) && !UNQUEUED_ACTIONS.has(action)) next.controlQueued = action as JobControlAction;
  }
  if (type === 'preflight_started') next.enhanceTotal = numberField(event, 'total_rewrites') ?? 0;
  if (type === 'model_loading') {
    const loadPhase = event?.phase;
    if (loadPhase === 'building' || loadPhase === 'downloading' || loadPhase === 'loading' || loadPhase === 'saving_quant') next.loadPhase = loadPhase;
    if (event?.replaces_download === true) next.loadReplacesDownload = true;
    if (loadPhase === 'saving_quant') next.loadQuant = numberField(event, 'quantize') ?? 8;
  }
  if (type === 'generation_finished') next.itemFinished = true;
  if (type === 'prompt_started' || type === 'generation_repeat') next.itemFinished = false;
  if (type === 'workflow_started' && Array.isArray(event?.stage_names)) {
    next.stageNames = event.stage_names.filter((name): name is string => typeof name === 'string');
    next.stageSteps = {};
  }
  if (type === 'prompt_started') next.stageSteps = {};
  if (type === 'step_progress') {
    const stage = event?.workflow_stage_name;
    const total = numberField(event, 'total_steps');
    if (typeof stage === 'string' && total !== undefined) next.stageSteps = { ...job.stageSteps, [stage]: total };
    if (typeof event?.phase === 'string') next.stepPhase = event.phase;
  }
  return next;
}

/** Epoch seconds of a `created_at` that is either epoch seconds (snapshots) or an ISO date (submit responses). */
export function epochSeconds(value: string | number | undefined): number | undefined {
  if (value === undefined || value === '') return undefined;
  const numeric = Number(value);
  if (Number.isFinite(numeric)) return numeric;
  const parsed = Date.parse(String(value));
  return Number.isFinite(parsed) ? parsed / 1000 : undefined;
}

/**
 * Phase bookkeeping for a job this tab starts following.
 *
 * A just-submitted job starts in the queue (its job_started event closes it). A reconnected job takes the phase of
 * the last event the server kept; the replayed event history then fills in the timings it still has.
 */
export function initialPhase(status: string, createdAt: string | number | undefined, lastEvent?: EventData): Pick<ActiveJobState, 'phase' | 'phaseMarks'> {
  const created = epochSeconds(createdAt);
  const phaseMarks: PhaseMarks = created !== undefined ? { queue: { start: created } } : {};
  const type = typeof lastEvent?.type === 'string' ? lastEvent.type : undefined;
  const fromEvent = type ? phaseForEvent(type, lastEvent) : null;
  // Unknown for a running job: the first event that names a phase sets it.
  return { phase: fromEvent ?? (status === 'queued' || !lastEvent ? 'queue' : undefined), phaseMarks };
}
