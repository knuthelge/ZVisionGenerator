import type { ActiveJobState, JobControlAction, JobInput } from '$lib/types';
import { jobStatusLine } from './jobStatusLine';
import {
  activeDownload, audioRows, downloadView, enhanceProgress, finishingLabel, headline, isActive, isAudioJob, isPaused, isVoiceDesignJob, itemActivity, itemNoun, iterations,
  jobPhase, jobTone, loadBody, outcome, stageBars, stepUnit, voiceDesignProgress,
} from './jobView';

/** The words beside the job ring: the "now" caption, the headline and the time line. Every function here is pure. */

/** From this many items the headline drops "Generating", so the count is never cut. */
export const SHORT_HEADLINE_FROM = 100;

/**
 * What is happening this second: a lead, then an optional raw line in mono. Each is a list of parts that read joined by
 * "·" and break only between each other. A `wrap` lead is one sentence, shown over two lines.
 */
export interface Caption {
  lead: string[];
  line: string[];
  wrap?: boolean;
}

const sentence = (text: string): Caption => ({ lead: [text], line: [], wrap: true });
const parts = (lead: string[], line: string[] = []): Caption => ({ lead, line });

/** The "now" caption beside the ring, or null when the headline and the mark already say it (Done). */
export function caption(job: ActiveJobState): Caption | null {
  const phase = jobPhase(job);
  const tone = jobTone(job);
  if (!isActive(job)) {
    const ended = outcome(job);
    if (!ended || tone === 'completed') return null;
    if (tone === 'failed') return phase === 'generate' && job.totalSteps > 0 ? parts([`Stopped at ${stepUnit(job).toLowerCase()} ${job.currentStep} of ${job.totalSteps}`]) : null;
    return sentence(ended.text);
  }
  if (job.review) return sentence('Approve or discard the small render');
  const download = activeDownload(job);
  if (download) {
    // The part and the time left are what a person waits for; the bytes and the speed are the raw line.
    const view = downloadView(download);
    return parts([view.part ?? 'Downloading', view.eta].filter((part) => part !== null), [view.amount, view.speed].filter((part) => part !== null));
  }
  if (phase === 'queue') return sentence('The worker picks it up as soon as it’s free.');
  if (phase === 'enhance') {
    const rewrite = enhanceProgress(job);
    return parts([rewrite ? `${rewrite.done} of ${rewrite.total} rewritten` : 'Loading the prompt enhancer']);
  }
  if (phase === 'load') return sentence(loadBody(job));
  if (isPaused(job) && job.itemFinished) return sentence(`Resumes with the next ${itemNoun(job)}.`);
  if (isVoiceDesignJob(job)) {
    const voices = voiceDesignProgress(job);
    return parts([job.audio?.phase ?? 'Sampling'], [`${voices.done} of ${voices.total} voices made`]);
  }
  if (isAudioJob(job)) {
    const rows = audioRows(job);
    const current = rows.find((row) => row.state === 'current');
    const others = rows.filter((row) => row !== current).map((row) => `${row.text}${row.state === 'done' ? ' ✓' : ''}`);
    return parts([current?.text ?? 'Starting…'], rows.length > 1 ? others : []);
  }
  const activity = itemActivity(job);
  if (activity === 'finishing') return parts([finishingLabel(job)]);
  const passes = stageBars(job);
  if (passes.length > 1) {
    const current = passes.find((pass) => pass.state === 'current');
    const rest = passes.filter((pass) => pass !== current).map((pass) => (pass.state === 'done' ? `${pass.label} · ${pass.total} steps ✓` : `${pass.label} next`));
    // A pass that has not reported its steps yet is starting; before any pass is current, so is the job.
    const lead = !current ? 'Starting…' : current.total > 0 ? `${current.label} · step ${current.current} of ${current.total}` : `${current.label} · starting`;
    return parts([lead], rest);
  }
  // No raw line for batches: the prompt and run position are in the phase log.
  if (activity === 'starting') return parts(['First step coming up']);
  return parts([`${stepUnit(job)} ${job.currentStep} of ${job.totalSteps}`]);
}

/**
 * The line a job leads with in a bar or a queue row: the enhanced prompt when there is one, or what stands in for an empty
 * one (a prompt file for a job that runs one). An upscale has no prompt of its own to lead with, so it leads with the name of its source file.
 */
export function promptText(job: Pick<ActiveJobState, 'prompt' | 'enhancedPrompt' | 'workflow' | 'settings'> & { inputs?: readonly JobInput[] }): string {
  if (job.workflow === 'upscale' && job.inputs?.length) return job.inputs[0].name;
  const shown = job.enhancedPrompt || job.prompt;
  if (shown) return shown;
  // An empty prompt is a prompt file only when the job really runs one.
  return job.settings?.prompt_source === 'file' ? 'Prompt file' : 'No prompt';
}

/** What a control that waits for the job's next step says while it waits. */
export interface PendingNotice {
  text: string;
  tone: 'info' | 'warning';
}

/** The notice for a pause, skip, repeat or cancel that was sent and has not taken effect yet; null for any other control. */
export function pendingNotice(job: ActiveJobState, action: JobControlAction | null): PendingNotice | null {
  const phase = jobPhase(job);
  const noun = itemNoun(job);
  switch (action) {
    case 'pause':
      return { text: phase === 'load' ? 'Pauses once the model is loaded' : phase === 'enhance' ? 'Pausing after this rewrite' : `Pausing after this ${noun}`, tone: 'warning' };
    case 'next':
      return { text: phase === 'enhance' ? 'Skipping this rewrite' : 'Skipping after this step', tone: 'info' };
    case 'repeat':
      return { text: `Repeating after this ${noun}`, tone: 'info' };
    case 'quit':
      return { text: 'Stopping at the next step', tone: 'warning' };
    default:
      return null;
  }
}

/** "Generating image 33 of 100" as "Image 33 of 100" (also video, upscaling and takes are left alone). */
export function shortHeadline(text: string): string {
  return text.replace(/^Generating (image|video) (\d+ of \d+)$/, (_match, noun: string, count: string) => `${noun.charAt(0).toUpperCase()}${noun.slice(1)} ${count}`);
}

/** The card's headline: the job's own, in the short form from `SHORT_HEADLINE_FROM` items. */
export function cardHeadline(job: ActiveJobState): string {
  const text = headline(job);
  return iterations(job).total >= SHORT_HEADLINE_FROM ? shortHeadline(text) : text;
}

/** A headline in runs of text, with the model's name as one run of its own that never breaks inside itself. */
export interface HeadlinePart {
  text: string;
  model: boolean;
}

export function splitModelName(text: string, name: string | undefined): HeadlinePart[] {
  const at = name ? text.indexOf(name) : -1;
  if (!name || at < 0) return [{ text, model: false }];
  return [{ text: text.slice(0, at), model: false }, { text: name, model: true }, { text: text.slice(at + name.length), model: false }].filter((part) => part.text !== '');
}

/** One part of the time line; the parts are joined by "·" and break only between each other. */
export interface TimePart {
  text: string;
  /** `eta` is the time left, in bold; `estimate` a placeholder for it, muted. */
  kind: 'eta' | 'estimate' | 'plain';
}

export interface TimeState {
  active: boolean;
  offline: boolean;
  /** The ETA label (`etaLabel`), or an empty string. */
  eta: string;
  /** The elapsed clock, "1:14". */
  elapsed: string;
  /** Seconds since the last event, for a lost connection. */
  sinceUpdate: number;
}

/** The time line under the headline. */
export function timeParts({ active, offline, eta, elapsed, sinceUpdate }: TimeState): TimePart[] {
  if (offline) return [{ text: `Last update ${sinceUpdate} s ago`, kind: 'plain' }];
  if (!active) return [{ text: `${elapsed} total`, kind: 'plain' }];
  const elapsedPart: TimePart = { text: `${elapsed} elapsed`, kind: 'plain' };
  if (!eta) return [elapsedPart];
  return [{ text: eta, kind: eta.startsWith('Estimate') ? 'estimate' : 'eta' }, elapsedPart];
}

/** The phone bar's second line: the ETA (shown bold), then what is moving. */
export interface BarSubline {
  eta: string | null;
  text: string;
}

export function barSubline(job: ActiveJobState, eta: string, elapsed: string): BarSubline {
  if (job.connection === 'reconnecting') return { eta: null, text: 'Reconnecting · controls come back soon' };
  if (!isActive(job)) return { eta: null, text: `${elapsed} total` };
  const now = caption(job);
  return { eta: eta || null, text: now && !now.wrap ? now.lead.join(' · ') : jobStatusLine(job, 0).text };
}
