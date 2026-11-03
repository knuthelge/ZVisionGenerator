import type { MascotMood, MascotTool } from '$lib/components/atoms/Mascot.svelte';
import { PHASE_DECODE } from '$lib/components/molecules/audioAurora';
import type { ActiveJobState, JobWorkflow } from '$lib/types';
import { isAudioJob, jobPhase, stepFraction } from './jobView';
import { isAudioWorkflow } from './workspaceTab';

/**
 * A short-lived reaction: to a job finishing, failing, being stopped, or getting
 * lost, or a nod when the user adds a job to the queue.
 */
export type MascotReaction = 'cheerful' | 'sad' | 'surprised' | 'nodding';

export const REACTION_DURATION_MS: Record<MascotReaction, number> = {
  cheerful: 4000,
  sad: 5000,
  surprised: 2500,
  nodding: 1200,
};
/** Share of the denoising steps after which the mascot adds the finishing touches. */
export const FINISHING_FRACTION = 0.85;
/** Fewer steps than this pass too quickly to show a separate finishing pose. */
export const FINISHING_MIN_STEPS = 4;
export const GREETING_DURATION_MS = 2500;
export const TYPING_DURATION_MS = 1500;
export const DROWSY_AFTER_MS = 60_000;

export interface MascotSignals {
  job: ActiveJobState | null;
  reaction?: MascotReaction | null;
  loadError?: boolean;
  loading?: boolean;
  greeting?: boolean;
  typing?: boolean;
  drowsy?: boolean;
}

/**
 * Report whether a running job is painting. Once the first image has started,
 * the gaps between batch images still count, so the mascot does not flicker
 * back to thinking at every prompt boundary.
 */
function isPainting(job: ActiveJobState): boolean {
  if (job.totalSteps > 0 && job.currentStep > 0) return true;
  return job.outputs.length > 0 || (job.promptNumber ?? 1) > 1;
}

/**
 * Report whether a running audio take has started. Audio jobs report no denoising steps, so the first `audio_progress`
 * event is what shows the model at work; the gaps between the takes of a batch still count.
 */
function isMakingTake(job: ActiveJobState): boolean {
  return job.audio != null || job.outputs.length > 0 || (job.promptNumber ?? 1) > 1;
}

/** Report whether a running take is in its last stretch: the decode, or for a sound the last sampling step. */
function isFinishingTake(job: ActiveJobState): boolean {
  const progress = job.audio;
  if (!progress) return false;
  if (progress.phase === PHASE_DECODE) return true;
  return job.workflow === 'sound' && progress.total !== null && progress.total > 0 && progress.step >= progress.total;
}

/** Report whether a running job is in the last stretch of its denoising steps. */
function isFinishing(job: ActiveJobState): boolean {
  return job.totalSteps >= FINISHING_MIN_STEPS && job.currentStep / job.totalSteps >= FINISHING_FRACTION;
}

/** Pick the mood for an active job, or null when no job is in flight. */
function jobMood(job: ActiveJobState | null): MascotMood | null {
  switch (job?.status) {
    case 'queued':
    case 'pending':
      return 'thinking';
    case 'running':
      // A render waiting for approval: the mascot looks to the user.
      if (job.review) return 'curious';
      if (job.stageName === 'enhancing_prompts') return 'reading';
      if (isAudioJob(job)) {
        if (!isMakingTake(job)) return 'thinking';
        return isFinishingTake(job) ? 'finishing' : 'creating';
      }
      if (!isPainting(job)) return 'thinking';
      return isFinishing(job) ? 'finishing' : 'creating';
    case 'paused':
      return 'paused';
    default:
      return null;
  }
}

/**
 * Pick the mascot mood from workspace signals. A nod always shows; other
 * reactions win until the next job starts painting. Then comes the active job,
 * then workspace state, then the user's own activity.
 */
export function mascotMood({ job, reaction, loadError, loading, greeting, typing, drowsy }: MascotSignals): MascotMood {
  const active = jobMood(job);
  const painting = active === 'creating' || active === 'finishing';
  if (reaction === 'nodding' || (reaction && !painting)) return reaction;
  if (active) return active;
  if (loadError) return 'sad';
  if (loading) return 'thinking';
  if (greeting) return 'waving';
  if (typing) return 'curious';
  if (drowsy) return 'sleeping';
  return 'idle';
}

/**
 * Report how far the running job's generate steps have come (0 to 1), or undefined while it reports none: before its
 * steps are known, between the items of a batch, and in the stages around generating, including the upscale, whose
 * own steps would otherwise read as the creating canvas starting over.
 */
export function mascotProgress(job: ActiveJobState | null): number | undefined {
  return job && !isAudioJob(job) && jobPhase(job) === 'generate' && job.stageName !== 'upscale' && job.totalSteps > 0 ? stepFraction(job) : undefined;
}

/** Pick what the mascot wears or holds: headphones for audio, a clapperboard for video workflows, otherwise a brush. */
export function mascotTool(workflow: JobWorkflow | undefined): MascotTool {
  if (isAudioWorkflow(workflow)) return 'audio';
  return workflow === 'txt2vid' || workflow === 'img2vid' || workflow === 'extend' ? 'clapper' : 'brush';
}

/** The Workspace spots the mascot can be called out to; elsewhere he is at home in the logo. */
export type MascotSpotId = 'welcome' | 'job' | 'outputs' | 'review' | 'empty' | 'error';

export interface MascotSpotSignals {
  view: 'error' | 'outputs' | 'job' | 'latest' | 'empty';
  running: boolean;
  reviewing: boolean;
  reacting: boolean;
  lookingForHistory: boolean;
  /** The first-run welcome is open. */
  welcoming: boolean;
}

/**
 * Pick the one Workspace spot the mascot works at, or null to send him home.
 * He stays on a job while it runs and while he reacts to its end, presents
 * finished outputs only while reacting, greets a first-time user in the welcome,
 * and explains a real empty or error stage.
 */
export function mascotSpot({ view, running, reviewing, reacting, lookingForHistory, welcoming }: MascotSpotSignals): MascotSpotId | null {
  if (view === 'error') return 'error';
  if (welcoming) return 'welcome';
  if (reviewing) return 'review';
  if (view === 'empty') return lookingForHistory ? null : 'empty';
  if (view === 'outputs') return reacting ? 'outputs' : null;
  if (view === 'job') return running || reacting ? 'job' : null;
  return null;
}

/** Where the mascot is and how it looks: out on a Workspace spot, or at home in the logo. */
export interface MascotPlacement {
  away: boolean;
  mood: MascotMood;
  tool: MascotTool;
}

/**
 * Place the mascot for the logo. The Workspace decides while it is open; on
 * other pages he is away while a job is in flight and reacts at home to its end.
 */
export function homePlacement(
  workspace: MascotPlacement | null,
  job: ActiveJobState | null,
  workflow: JobWorkflow | undefined,
  reaction: MascotReaction | null = null,
): MascotPlacement {
  if (workspace) return workspace;
  return { away: jobMood(job) !== null, mood: reaction ?? 'idle', tool: mascotTool(job?.workflow ?? workflow) };
}
