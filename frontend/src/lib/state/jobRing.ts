import type { ActiveJobState } from '$lib/types';
import {
  activeDownload, downloadFraction, downloadView, enhanceProgress, headline, isActive, isAudioJob, isPaused, isVoiceDesignJob, itemActivity, itemFraction,
  itemNoun, iterations, jobPhase, jobTone, stageBars, stepFraction, stepUnit,
} from './jobView';
import { headAngles, RING_SPECS, type RingSize } from './ringGeom';

/**
 * What the job ring draws for a job: its arcs, the stamp, the status mark and the signs of work. Every function here is
 * pure; the ring itself is `ProgressRing`, and `RingHero` places the stamp and the mark.
 */

/** The colour role of an arc's filled part. */
export type RingInk = 'work' | 'paused' | 'done' | 'failed' | 'stopped';

/** One arc of the ring: `fill` is 0 to 1 of the arc. */
export interface RingArc {
  fill: number;
  ink: RingInk;
}

/** The step in progress, drawn as a ghost: on arc `arc`, from its solid fill `from`, one step's `share` of the arc long. */
export interface RingGhost {
  arc: number;
  from: number;
  share: number;
}

/** `rest`: only the track. `busy`: a length the job cannot give, shown as a sweep. `arc`: a real fraction. */
export type RingKind = 'rest' | 'busy' | 'arc';

/** The status mark that takes the stamp's place when work stops. */
export type MarkKind = 'paused' | 'stale' | 'review' | 'completed' | 'failed' | 'stopped' | 'unknown';

export interface RingView {
  kind: RingKind;
  arcs: RingArc[];
  /** The whole job's share done, floored so it reads 100 only when the job is done: the ring's fill, `aria-valuenow` and the stamp. */
  percent: number;
  /** The stamp at 6 o'clock while work moves, or null (also for a song, whose fill is an estimate). */
  stamp: number | null;
  mark: MarkKind | null;
  /** The ring's light: teal while work runs, grey for a fault, none at rest. */
  pool: 'work' | 'dim' | null;
  /** Every arc drawn muted: the stream is stale. */
  faded: boolean;
  /** The ring says nothing to assistive technology: it is at rest, or the job has ended. */
  decorative: boolean;
  valuetext: string;
  /** Work runs right now: the spark shows (leading the sweep, or at the head of the working arc). */
  live: boolean;
  /** The arc whose end is the working head (where the spark sits), or null. */
  head: number | null;
  /** The step in progress, or null without timing, between steps, while paused and while reconnecting. */
  ghost: RingGhost | null;
}

/** The ring's own number: floored, so it reads 100 only when the job is done. */
export function jobPercent(fraction: number): number {
  return Math.min(99, Math.max(0, Math.floor(fraction * 100)));
}

function oneArc(fill: number, ink: RingInk): RingArc[] {
  return [{ fill, ink }];
}

/** An ended ring's mark, from the job's tone. */
function endedMark(tone: ReturnType<typeof jobTone>): MarkKind {
  return tone === 'failed' ? 'failed' : tone === 'unknown' ? 'unknown' : tone === 'completed' ? 'completed' : 'stopped';
}

/** What the working head's step is, as a share of one arc: one step of the stage (or of both stages of a two-pass item) over the arc's length. */
function stepShare(job: ActiveJobState, passes: number): number {
  return job.totalSteps > 0 ? 1 / (job.totalSteps * passes) : 0;
}

/** What one phase draws, before the state every phase shares (reconnecting, paused, ended) is applied. */
interface Shape {
  kind: RingKind;
  arcs: RingArc[];
  percent: number;
  valuetext: string;
  /** The arc whose end is the working head, if the phase has one. */
  head: number | null;
  /** The step in progress, if the phase times its steps. */
  ghost: RingGhost | null;
  /** The percentage is a number the job can show as the stamp (not a song's estimate, and not a sweep). */
  numbered: boolean;
}

const sweep = (valuetext: string): Shape => ({ kind: 'busy', arcs: [], percent: 0, valuetext, head: null, ghost: null, numbered: false });
const resting = (valuetext: string, percent = 0): Shape => ({ kind: 'rest', arcs: [], percent, valuetext, head: null, ghost: null, numbered: false });

/** Decide what the phase the job is in draws: its kind, its arcs and its percent. */
function phaseShape(job: ActiveJobState, stepMs: number | null): Shape {
  const tone = jobTone(job);
  const phase = jobPhase(job);
  const { total, index, done } = iterations(job);
  const active = isActive(job);
  const paused = isPaused(job);
  const liveInk: RingInk = paused ? 'paused' : 'work';

  // Downloads: the ring is the download; an unknown size sweeps.
  const download = activeDownload(job);
  if (download) {
    const fraction = downloadFraction(download);
    const view = downloadView(download);
    if (fraction === null) return sweep(view.valuetext);
    return { kind: 'arc', arcs: oneArc(fraction, liveInk), percent: jobPercent(fraction), valuetext: view.valuetext, head: 0, ghost: null, numbered: true };
  }

  if (tone === 'queued' || (active && phase === 'queue')) return resting('Waiting to start');

  // Setup phases with no fraction sweep. Prompt rewrites are one arc each, and the rewrite in progress gets the step ghost.
  if (active && phase === 'enhance') {
    const rewrite = enhanceProgress(job);
    if (!rewrite) return sweep('Loading the prompt enhancer');
    const arcs = Array.from({ length: rewrite.total }, (_, k): RingArc => ({ fill: k < rewrite.done ? 1 : 0, ink: liveInk }));
    const head = Math.min(rewrite.done, rewrite.total - 1);
    return {
      kind: 'arc', arcs, percent: jobPercent(rewrite.done / rewrite.total), valuetext: `${rewrite.done} of ${rewrite.total} prompts rewritten`, head,
      ghost: stepMs && rewrite.done < rewrite.total ? { arc: head, from: 0, share: 1 } : null, numbered: true,
    };
  }
  if (active && phase === 'load') return paused ? resting('Paused') : sweep(headline(job));

  // Ended before generating: no arc to freeze, only the outcome mark.
  if (!active && phase !== 'generate') return resting(headline(job), tone === 'completed' ? 100 : 0);

  // Generating, or ended while generating.
  // A song's fill is Aurora's estimate of an unknown length: no number on it.
  const estimated = job.workflow === 'music';
  // The image that was interrupted, or skipped as the job stopped, keeps the fraction it reached.
  const stopFill = !active && (!job.itemFinished || job.itemSkipped) ? stepFraction(job) : 0;
  const passCount = Math.max(1, stageBars(job).length);

  let arcs: RingArc[];
  let fraction: number;
  let head = 0;
  let share = 0;
  if (total > 1) {
    // A batch: one arc per item at any count. The gap rule shrinks the gaps as the arcs get shorter.
    head = Math.min(index, total - 1);
    share = stepShare(job, passCount);
    arcs = Array.from({ length: total }, (_, k): RingArc => {
      if (tone === 'completed') return { fill: 1, ink: 'done' };
      if (tone === 'failed') {
        if (k < done) return { fill: 1, ink: 'done' };
        return k === index ? { fill: Math.max(0.04, stopFill), ink: 'failed' } : { fill: 0, ink: 'done' };
      }
      if (tone === 'stopped' || tone === 'unknown') {
        if (k < done) return { fill: 1, ink: 'done' };
        return k === index ? { fill: stopFill, ink: 'stopped' } : { fill: 0, ink: 'done' };
      }
      if (k < done) return { fill: 1, ink: liveInk };
      // The item in progress keeps its fill while paused; paused between items leaves the next arc empty.
      if (k === index && active && !job.itemFinished) return { fill: isVoiceDesignJob(job) ? 0 : itemFraction(job), ink: liveInk };
      return { fill: 0, ink: liveInk };
    });
    fraction = tone === 'completed' ? 1 : arcs.reduce((sum, arc) => sum + arc.fill, 0) / total;
  } else {
    // One item: its passes as arcs when it has two, otherwise one arc.
    const passes = active ? stageBars(job) : [];
    if (passes.length > 1) {
      head = Math.max(0, passes.findIndex((pass) => pass.state === 'current'));
      share = stepShare(job, 1);
      arcs = passes.map((pass): RingArc => ({ fill: pass.state === 'done' ? 1 : pass.state === 'current' && pass.total > 0 ? Math.min(1, pass.current / pass.total) : 0, ink: liveInk }));
      fraction = itemFraction(job);
    } else {
      fraction = tone === 'completed' ? 1 : active ? itemFraction(job) : done;
      const ink: RingInk = tone === 'completed' ? 'done' : tone === 'failed' ? 'failed' : tone === 'stopped' || tone === 'unknown' ? 'stopped' : liveInk;
      const frozen = tone === 'failed' ? Math.max(0.04, stopFill) : tone === 'stopped' || tone === 'unknown' ? stopFill : fraction;
      arcs = oneArc(frozen, ink);
      share = stepShare(job, passCount);
    }
  }

  const percent = tone === 'completed' ? 100 : jobPercent(fraction);
  const stepping = !job.itemFinished && !isAudioJob(job) && !isVoiceDesignJob(job) && itemActivity(job) === 'stepping' && share > 0;
  const headArc = arcs[head];
  const ghost: RingGhost | null = stepping && stepMs && headArc ? { arc: head, from: headArc.fill, share: Math.min(share, 1 - headArc.fill) } : null;

  const parts: string[] = [];
  if (total > 1) {
    const noun = itemNoun(job);
    parts.push(`${noun.charAt(0).toUpperCase()}${noun.slice(1)} ${index + 1} of ${total}`);
  }
  if (job.totalSteps > 0 && active) parts.push(`${stepUnit(job).toLowerCase()} ${job.currentStep} of ${job.totalSteps}`);
  // The stamp says the same number, so it is read out here while work moves.
  const working = active && !paused && job.connection !== 'reconnecting' && !job.review;
  if (working && !estimated) parts.push(`${percent} percent of the job`);

  return { kind: 'arc', arcs, percent, valuetext: parts.join(', ') || headline(job), head, ghost, numbered: !estimated };
}

/**
 * Decide what the ring draws.
 *
 * Each phase gives its kind, arcs and percent; the state every phase shares is applied once on top, so a job looks the
 * same in every phase when it is paused, reconnecting or ended: its mark, the muted arcs, the glow, and whether the
 * spark, the stamp and the step ghost show. *stepMs* is how long the last step took (`trackStepTiming`); without it the
 * step ghost is not drawn.
 */
export function ringView(job: ActiveJobState, stepMs: number | null = null): RingView {
  const tone = jobTone(job);
  const active = isActive(job);
  const stale = tone === 'stale';
  const working = active && !isPaused(job) && !stale && !job.review;
  const shape = phaseShape(job, stepMs);
  const moving = working && shape.kind !== 'rest';
  const mark: MarkKind | null = !active ? endedMark(tone) : stale ? 'stale' : job.review ? 'review' : isPaused(job) ? 'paused' : null;

  return {
    kind: shape.kind,
    arcs: shape.arcs,
    percent: shape.percent,
    stamp: moving && shape.numbered ? shape.percent : null,
    mark,
    pool: tone === 'running' || tone === 'preparing' ? 'work' : tone === 'failed' ? 'dim' : null,
    faded: stale,
    decorative: !active || shape.kind === 'rest',
    valuetext: shape.valuetext,
    live: moving,
    head: moving ? shape.head : null,
    ghost: moving ? shape.ghost : null,
  };
}

/** Changes with every step of every item, so the ring restarts its ghost and its spark. */
export function stepKey(job: ActiveJobState): string {
  return `${job.job_id}:${job.stageName}:${job.batchIndex}:${job.promptNumber ?? 0}:${job.currentStep}`;
}

/**
 * Where Blob looks: degrees clockwise from straight up at the ring's working head, the spark or the front of the ghost
 * that the spark rides to. Undefined when no head moves (queued, paused, ended, reconnecting, a sweep with no fraction),
 * so Blob behaves as he does without a ring. *ghost* is the ghost drawn on the ring, if any.
 */
export function ringGaze(view: RingView, size: Exclude<RingSize, 'XS'>, ghost: RingGhost | null): number | undefined {
  if (!view.live || view.head === null || view.kind !== 'arc') return undefined;
  const { box, stroke } = RING_SPECS[size];
  const angles = headAngles(box, stroke, view.arcs, view.head, ghost);
  return angles ? (ghost ? angles.to : angles.from) % 360 : undefined;
}
