import type { ActiveJobState, JobControlAction, JobPhase } from '$lib/types';
import { jobReviewUrl } from '$lib/api/workspace';
import { phaseSeconds } from './jobPhases';
import { checked } from './jobSettings';

/** What the job card says about a job, derived from its state. Every function here is pure. */

export type JobTone = 'queued' | 'preparing' | 'running' | 'paused' | 'stale' | 'completed' | 'failed' | 'stopped' | 'unknown';
export type PhaseRowStatus = 'done' | 'current' | 'paused' | 'pending' | 'skipped' | 'failed' | 'stopped' | 'unknown';

export interface Iterations {
  /** Items in the job (runs × prompts). */
  total: number;
  /** 0-based position of the current (or last) item. */
  index: number;
  /** Items finished, counting the current one once its generation finished. */
  done: number;
}

export interface PhaseRow {
  key: JobPhase;
  label: string;
  detail: string;
  status: PhaseRowStatus;
  /** Seconds the phase took; undefined while it runs or when its timing is unknown. */
  secs?: number;
  /** Server epoch seconds the running phase started at, for a live duration. */
  startedAt?: number;
}

export interface StageBar {
  name: string;
  label: string;
  state: 'done' | 'current' | 'next';
  current: number;
  total: number;
}

export type JobBarSegment = { state: 'done' | 'now' | 'failed' | 'pending'; runBreak: boolean; fill: number };
export type JobBar =
  | { kind: 'segments'; segments: JobBarSegment[] }
  | { kind: 'bar'; fraction: number };

export interface JobControls {
  /** Job-scoped controls for the header: Pause or Resume, then Cancel. */
  job: JobControlAction[];
  /** Item-scoped controls for the current task's line: Skip, Repeat. */
  item: JobControlAction[];
}

/** Up to this many items the job bar is one segment per item; above it, one plain bar. */
export const SEGMENT_LIMIT = 48;

/** Workflow stages that run denoising steps, with the label their bar gets. */
const STEPPED_STAGES: Record<string, string> = {
  text_to_image: 'Generate', upscale: 'Upscale', text_to_video: 'Generate', image_to_video: 'Generate', extend_video: 'Generate',
};

/** What an item shows after its last step, for stages that name their work; others read "Finishing up…". */
const FINISHING_LABELS: Record<string, string> = { join_clips: 'Joining clips…' };

/** The label for the stretch after the last step: what the current stage is doing, or "Finishing up…". */
export function finishingLabel(job: ActiveJobState): string {
  return FINISHING_LABELS[job.stageName] ?? 'Finishing up…';
}

/** The same label without its ellipsis, for a progress bar's accessible value text. */
export function finishingText(job: ActiveJobState): string {
  return finishingLabel(job).replace(/…$/, '');
}

/** Whether a workflow stage runs denoising steps (as opposed to e.g. resolving the prompt or saving the file). */
export function isSteppedStage(name: string): boolean {
  return name in STEPPED_STAGES;
}

/**
 * What the current item is doing: getting ready for its first step (or for the next pass), running steps,
 * finishing up after its last step (decoding, post-processing and writing the file), or waiting for the user to
 * review its small render. A waiting review wins over finishing, which its `review` stage would otherwise give.
 */
export function itemActivity(job: ActiveJobState): 'starting' | 'stepping' | 'finishing' | 'reviewing' {
  if (job.review) return 'reviewing';
  const stepped = (job.stageNames ?? []).filter(isSteppedStage);
  const stepsRan = Object.keys(job.stageSteps ?? {}).length > 0;
  // A later non-stepped stage (contrast, sharpen, save) after the denoising passes.
  if (job.stageName && !isSteppedStage(job.stageName) && job.stageName !== 'enhancing_prompts' && stepsRan) return 'finishing';
  if (!(job.totalSteps > 0)) return 'starting';
  if (job.currentStep < job.totalSteps) return 'stepping';
  // All steps of this pass ran. Another pass may follow: the upscale stage, or the second stage of an upscaled video.
  const lastPass = !stepped.length || stepped.indexOf(job.stageName) === stepped.length - 1;
  return lastPass && job.stepPhase !== 'video_upscale_stage_1' ? 'finishing' : 'starting';
}
const ENDED_PHASE_WORDS: Record<JobPhase, string> = {
  queue: 'before starting',
  enhance: 'while enhancing prompts',
  load: 'while loading the model',
  generate: 'while generating',
};

export function isActive(job: ActiveJobState): boolean {
  return job.status === 'queued' || job.status === 'running' || job.status === 'paused';
}

export function isPaused(job: ActiveJobState): boolean {
  return isActive(job) && (job.paused || job.status === 'paused');
}

export function isVideo(job: ActiveJobState): boolean {
  return job.workflow === 'txt2vid' || job.workflow === 'img2vid' || job.workflow === 'extend';
}

export function itemNoun(job: ActiveJobState): 'image' | 'video' {
  return isVideo(job) ? 'video' : 'image';
}

export function jobPhase(job: ActiveJobState): JobPhase {
  return job.phase ?? (job.status === 'queued' ? 'queue' : 'generate');
}

export function jobTone(job: ActiveJobState): JobTone {
  if (job.status === 'completed') return 'completed';
  if (job.status === 'failed') return 'failed';
  if (job.status === 'cancelled') return 'stopped';
  if (job.status === 'unknown') return 'unknown';
  if (job.connection === 'reconnecting') return 'stale';
  if (isPaused(job)) return 'paused';
  if (job.status === 'queued' || jobPhase(job) === 'queue') return 'queued';
  return jobPhase(job) === 'generate' ? 'running' : 'preparing';
}

function promptsPerRun(job: ActiveJobState): number {
  return Number.isInteger(job.promptCount) && job.promptCount! > 1 ? job.promptCount! : 1;
}

export function iterations(job: ActiveJobState): Iterations {
  const perRun = promptsPerRun(job);
  const total = Math.max(1, Number.isFinite(job.runs) ? Math.floor(job.runs) : 1) * perRun;
  const run = Number.isFinite(job.batchIndex) ? Math.max(0, Math.floor(job.batchIndex)) : 0;
  const prompt = perRun > 1 && Number.isInteger(job.promptNumber) ? Math.max(1, job.promptNumber!) : 1;
  const index = jobPhase(job) === 'generate' ? Math.min(total - 1, run * perRun + prompt - 1) : 0;
  const done = jobPhase(job) === 'generate' ? index + (job.itemFinished ? 1 : 0) : 0;
  return { total, index, done };
}

/** The current item's step progress in [0, 1]. */
export function stepFraction(job: ActiveJobState): number {
  if (jobPhase(job) !== 'generate' || !(job.totalSteps > 0) || !Number.isFinite(job.currentStep)) return 0;
  return Math.min(1, Math.max(0, job.currentStep / job.totalSteps));
}

/** One bar per stepped stage when the item has more than one (generate, then upscale); otherwise none. */
export function stageBars(job: ActiveJobState): StageBar[] {
  const names = (job.stageNames ?? []).filter((name) => name in STEPPED_STAGES);
  if (names.length < 2) return [];
  const currentIndex = names.indexOf(job.stageName);
  return names.map((name, index) => {
    const label = STEPPED_STAGES[name];
    const steps = job.stageSteps?.[name] ?? 0;
    if (index === currentIndex && !job.itemFinished) return { name, label, state: 'current', current: job.currentStep, total: job.totalSteps };
    const finished = job.itemFinished || (currentIndex >= 0 ? index < currentIndex : steps > 0);
    return finished ? { name, label, state: 'done', current: steps, total: steps } : { name, label, state: 'next', current: 0, total: steps };
  });
}

/** The current item's progress in [0, 1], counting each stage of a multi-stage item equally. */
export function itemFraction(job: ActiveJobState): number {
  if (job.itemFinished || itemActivity(job) === 'finishing') return 1;
  const bars = stageBars(job);
  if (!bars.length) return stepFraction(job);
  const done = bars.filter((bar) => bar.state === 'done').length;
  const current = bars.find((bar) => bar.state === 'current');
  const part = current && current.total > 0 ? Math.min(1, current.current / current.total) : 0;
  return (done + part) / bars.length;
}

/** The whole-job bar: one segment per item, the current one filling as it runs, or a plain bar for long jobs. */
export function jobBar(job: ActiveJobState): JobBar {
  const { total, index, done } = iterations(job);
  const active = isActive(job);
  const generating = jobPhase(job) === 'generate';
  const failedAt = job.status === 'failed' && generating ? index : -1;
  const live = active && generating && !job.itemFinished;
  if (total === 1) {
    const fraction = failedAt === 0 ? 1 : active ? (generating ? itemFraction(job) : 0) : done;
    return { kind: 'bar', fraction };
  }
  if (total > SEGMENT_LIMIT) {
    const finished = active ? done : Math.max(done, job.outputs.length);
    return { kind: 'bar', fraction: Math.min(1, (finished + (live ? itemFraction(job) : 0)) / total) };
  }
  const perRun = promptsPerRun(job);
  const segments = Array.from({ length: total }, (_, k): JobBarSegment => {
    const runBreak = perRun > 1 && k > 0 && k % perRun === 0;
    if (k === failedAt) return { state: 'failed', runBreak, fill: 1 };
    if (k < done) return { state: 'done', runBreak, fill: 1 };
    if (live && k === index && !isPaused(job)) return { state: 'now', runBreak, fill: itemFraction(job) };
    return { state: 'pending', runBreak, fill: 0 };
  });
  return { kind: 'segments', segments };
}

/** The one sentence that says what is happening, or how the job ended and where. */
export function headline(job: ActiveJobState): string {
  const { total, index, done } = iterations(job);
  const noun = itemNoun(job);
  const phase = jobPhase(job);
  const generating = phase === 'generate';
  const tone = jobTone(job);
  if (tone === 'failed') {
    if (!generating) return `Failed ${ENDED_PHASE_WORDS[phase]}`;
    return total > 1 ? `Failed on ${noun} ${index + 1} of ${total}` : `The ${noun} failed`;
  }
  if (tone === 'stopped') {
    if (phase === 'queue' && job.message.startsWith('Removed')) return 'Removed from the queue';
    if (!generating) return `Stopped ${ENDED_PHASE_WORDS[phase]}`;
    return total > 1 ? `Stopped after ${done} of ${total} ${noun}s` : 'Stopped';
  }
  if (tone === 'unknown') return 'Lost track of this job';
  if (tone === 'completed') return total > 1 ? `Finished ${total} ${noun}s` : 'Finished';
  if (tone === 'stale') return 'Reconnecting…';
  if (tone === 'paused') {
    if (job.review) return 'Waiting for your review';
    if (!generating) return phase === 'enhance' ? 'Paused while enhancing prompts' : 'Paused';
    if (total === 1) return 'Paused';
    return job.itemFinished ? `Paused after ${noun} ${index + 1} of ${total}` : `Paused before ${noun} ${index + 1} of ${total}`;
  }
  if (phase === 'queue') return 'Waiting to start';
  if (phase === 'enhance') {
    const rewrite = enhanceProgress(job);
    return rewrite ? `Enhancing prompt ${rewrite.current} of ${rewrite.total}` : 'Loading the prompt enhancer';
  }
  if (phase === 'load') {
    if (job.loadPhase === 'downloading') return `Downloading ${job.model}`;
    return job.loadQuant ? 'Preparing the model' : 'Loading the model';
  }
  if (job.workflow === 'upscale') return 'Upscaling';
  const verb = job.stageName === 'upscale' && !job.itemFinished ? 'Upscaling' : 'Generating';
  return total > 1 ? `${verb} ${noun} ${index + 1} of ${total}` : `${verb} ${noun}`;
}

/** The rewrite being worked on (1-based) while prompts are enhanced. */
export function enhanceProgress(job: ActiveJobState): { current: number; done: number; total: number } | null {
  if (job.stageName !== 'enhancing_prompts' || !(job.totalSteps > 0)) return null;
  const done = Math.max(0, Math.min(job.totalSteps, job.currentStep));
  return { current: Math.min(job.totalSteps, done + 1), done, total: job.totalSteps };
}

function supports(job: ActiveJobState, action: JobControlAction): boolean {
  const controls = job.supported_controls ?? [];
  return controls.includes(action) || (action === 'quit' && controls.includes('cancel'));
}

/**
 * Controls the server honours right now, split by scope.
 *
 * Pause waits for the current image (or rewrite) and is queued during model load; Skip interrupts the current image or
 * rewrite; Repeat waits for the current image. A queued job offers only Cancel; a paused one only Resume and Cancel.
 */
export function jobControls(job: ActiveJobState): JobControls {
  if (!isActive(job)) return { job: [], item: [] };
  if (job.review) return { job: supports(job, 'quit') ? ['quit'] : [], item: [] };
  const phase = jobPhase(job);
  const started = job.status !== 'queued' && phase !== 'queue';
  const jobScope: JobControlAction[] = [];
  if (isPaused(job)) {
    if (supports(job, 'resume')) jobScope.push('resume');
  } else if (started && supports(job, 'pause')) {
    jobScope.push('pause');
  }
  if (supports(job, 'quit')) jobScope.push('quit');

  const item: JobControlAction[] = [];
  if (started && !isPaused(job)) {
    if (phase === 'enhance' && enhanceProgress(job) && supports(job, 'next')) item.push('next');
    if (phase === 'generate') {
      if (iterations(job).total > 1 && supports(job, 'next')) item.push('next');
      if (supports(job, 'repeat')) item.push('repeat');
    }
  }
  return { job: jobScope, item };
}

/** Whether the job is waiting for the user to approve or discard a small render. */
export function isReviewing(job: ActiveJobState | null | undefined): boolean {
  return !!job && isActive(job) && !!job.review;
}

/** The decisions the user can make now about the render under review, in display order; empty outside a review. */
export function reviewDecisions(job: ActiveJobState): ('approve' | 'discard')[] {
  if (!isReviewing(job)) return [];
  return (['approve', 'discard'] as const).filter((action) => supports(job, action));
}

/** Pixel size of a render, as shown in the review line. */
export interface ReviewSize {
  width: number;
  height: number;
}

/** The factor an image job upscales by, read from the settings it was submitted with. */
export function reviewFactor(job: ActiveJobState): number | null {
  const value = job.settings?.upscale;
  const factor = Number(Array.isArray(value) ? value[0] : value);
  return Number.isFinite(factor) && factor > 0 ? factor : null;
}

/** The video under review, served by the job's review route; null for an image, whose render is the job preview. */
export function reviewVideoUrl(job: ActiveJobState): string | null {
  if (!isReviewing(job) || !isVideo(job)) return null;
  return jobReviewUrl(job.job_id, job.review?.iteration ?? 0);
}

/** Whether the video under review keeps its sound: only when the job was submitted with Audio on. */
export function reviewHasSound(job: ActiveJobState): boolean {
  return !!job.settings && job.settings.audio !== undefined && checked(job.settings, 'audio');
}

/**
 * The line under the render being reviewed: its position in the job, then the small size and the size it upscales to.
 *
 * A size that is not known yet is left out, so the line never shows a guess.
 */
export function reviewLine(job: ActiveJobState, size: ReviewSize | null): string {
  const position = `Review ${job.review?.iteration ?? iterations(job).index + 1} of ${iterations(job).total}`;
  if (!size || !(size.width > 0) || !(size.height > 0)) return position;
  const factor = reviewFactor(job);
  const small = `${size.width}×${size.height}`;
  return factor ? `${position} · ${small} → ${size.width * factor}×${size.height * factor}` : `${position} · ${small}`;
}

/** The browser tab title: the review marker first while a render waits, otherwise the base title. */
export function documentTitle(base: string, reviewing: boolean): string {
  return reviewing ? `⏸ Review · ${base}` : base;
}

function generateLabel(job: ActiveJobState): string {
  if (job.workflow === 'upscale') return 'Upscale';
  return isVideo(job) ? 'Generate video' : 'Generate';
}

/** The run log's rows in server order: queued, prompt enhancement (only when the job rewrites prompts), model load, generation. */
export function phaseRows(job: ActiveJobState): PhaseRow[] {
  const phase = jobPhase(job);
  const order: JobPhase[] = ['queue', 'enhance', 'load', 'generate'];
  const at = order.indexOf(phase);
  const active = isActive(job);
  const enhances = (job.enhanceTotal ?? 0) > 0;
  const { total } = iterations(job);
  const endedStatus: PhaseRowStatus = job.status === 'failed' ? 'failed' : job.status === 'unknown' ? 'unknown' : job.status === 'completed' ? 'done' : 'stopped';
  const rows: PhaseRow[] = [];
  for (const [position, key] of order.entries()) {
    // Only a job that rewrites prompts has an enhancement step; gallery upscales never do.
    if (key === 'enhance' && (job.workflow === 'upscale' || (!enhances && phase !== 'enhance'))) continue;
    let status: PhaseRowStatus;
    if (key === 'enhance' && !enhances && position !== at) status = 'skipped';
    else if (position < at) status = 'done';
    else if (position > at) status = 'pending';
    else status = active ? (isPaused(job) ? 'paused' : 'current') : endedStatus;
    rows.push({
      key,
      label: key === 'queue' ? 'Queued' : key === 'enhance' ? 'Enhance prompts' : key === 'load' ? loadLabel(job) : generateLabel(job),
      detail: rowDetail(job, key, status, total),
      status,
      secs: status === 'pending' || status === 'skipped' || status === 'current' || status === 'paused' ? undefined : phaseSeconds(job.phaseMarks, key),
      startedAt: status === 'current' || status === 'paused' ? job.phaseMarks?.[key]?.start : undefined,
    });
  }
  return rows;
}

/** The explanation under the model-load row: what the load is doing and why it only happens the first time. */
export function loadBody(job: ActiveJobState): string {
  if (job.loadPhase === 'downloading') return 'Downloading the model files. Only the first time; later upscales start from the saved copy.';
  if (job.loadQuant && job.loadReplacesDownload) return 'Saving an 8-bit copy and removing the downloaded files to save disk. Only the first time.';
  if (job.loadQuant) return 'Saving a quantized copy so later jobs load in seconds. Only the first time you use this model.';
  return `Loading ${job.model} into memory.`;
}

function loadLabel(job: ActiveJobState): string {
  if (job.loadPhase === 'downloading') return 'Download model';
  return job.loadQuant ? 'Prepare model' : 'Load model';
}

function rowDetail(job: ActiveJobState, key: JobPhase, status: PhaseRowStatus, total: number): string {
  if (status === 'pending' && !isActive(job)) return 'Not started';
  if (key === 'enhance') {
    if (status === 'skipped') return 'Off';
    const rewrite = enhanceProgress(job);
    if (rewrite && status !== 'done') return `${rewrite.done} of ${rewrite.total} rewritten`;
    return job.enhanceTotal ? `${job.enhanceTotal} prompt${job.enhanceTotal === 1 ? '' : 's'}` : '';
  }
  if (key === 'load') return job.loadQuant ? `q${job.loadQuant} copy · first use only` : job.model;
  if (key === 'generate') return total > 1 ? `${total} ${itemNoun(job)}s` : '';
  return '';
}

/** The folded "Ready" line once generation runs: each setup phase and how long it took. */
export function readySummary(job: ActiveJobState): { parts: string[]; secs: number | undefined; warn: boolean } {
  const parts: string[] = [];
  let secs: number | undefined = 0;
  const add = (phase: JobPhase, label: string): void => {
    const took = phaseSeconds(job.phaseMarks, phase);
    parts.push(took === undefined ? label : `${label} ${formatClock(took)}`);
    secs = took === undefined || secs === undefined ? undefined : secs + took;
  };
  add('queue', 'queued');
  // The enhancer failing for this image takes the enhancement's place in the line.
  const warn = job.enhanceStatus === 'failed';
  if ((job.enhanceTotal ?? 0) > 0 && !warn) add('enhance', `${job.enhanceTotal} prompt${job.enhanceTotal === 1 ? '' : 's'} enhanced`);
  add('load', job.loadQuant ? 'model prepared' : 'model loaded');
  return { parts, secs, warn };
}

/** Where the current item sits in the batch, or what runs next when paused between items. */
export function positionLabel(job: ActiveJobState): string {
  if (job.workflow === 'upscale') return 'Source';
  const perRun = promptsPerRun(job);
  const runs = Math.max(1, Math.floor(job.runs || 1));
  const { total, index } = iterations(job);
  if (total === 1) return 'Prompt';
  const next = isPaused(job) && job.itemFinished;
  const position = Math.min(total - 1, index + (next ? 1 : 0));
  const parts: string[] = [];
  if (perRun > 1) parts.push(`prompt ${(position % perRun) + 1} of ${perRun}`);
  if (runs > 1) parts.push(`run ${Math.floor(position / perRun) + 1} of ${runs}`);
  const text = parts.join(' · ');
  return next ? `Next: ${text}` : text.charAt(0).toUpperCase() + text.slice(1);
}

/** The estimate for the rest of the job, or why there is none yet. Single-item jobs never get one. */
export function etaLabel(job: ActiveJobState): string {
  if (!isActive(job) || isPaused(job) || jobPhase(job) !== 'generate' || job.connection === 'reconnecting') return '';
  if (iterations(job).total < 2) return '';
  if (job.remaining > 0) return timeLeft(job.remaining);
  return `Estimate after the first ${itemNoun(job)}`;
}

export function timeLeft(secs: number): string {
  if (secs < 60) return 'Under a minute left';
  const minutes = Math.round(secs / 60);
  if (minutes < 60) return `About ${minutes} min left`;
  return `About ${Math.floor(minutes / 60)} h ${minutes % 60} min left`;
}

/** How the job ended, and what it saved. Null while it runs. */
export function outcome(job: ActiveJobState): { text: string; error: boolean; saved: string } | null {
  if (isActive(job)) return null;
  const { total, done } = iterations(job);
  const saved = job.outputs.length;
  const failedItem = job.status === 'failed' && jobPhase(job) === 'generate' ? 1 : 0;
  const notRun = Math.max(0, total - Math.max(done, saved) - failedItem);
  let text = job.message;
  if (job.status === 'cancelled' && !job.message.startsWith('Removed')) text = 'You stopped the job.';
  if (job.status === 'unknown') text = 'The server restarted and no longer has this job.';
  const savedText = saved === 0 ? 'Nothing saved' : total > 1 ? `${saved} of ${total} saved to History` : 'Saved to History';
  return { text, error: job.status === 'failed', saved: total > 1 && notRun > 0 ? `${savedText} · ${notRun} not run` : savedText };
}

/** m:ss, or h:mm:ss from an hour. */
export function formatClock(secs: number): string {
  const rounded = Math.max(0, Math.round(Number.isFinite(secs) ? secs : 0));
  const h = Math.floor(rounded / 3600);
  const m = Math.floor((rounded % 3600) / 60);
  const s = rounded % 60;
  return h > 0 ? `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}` : `${m}:${String(s).padStart(2, '0')}`;
}
