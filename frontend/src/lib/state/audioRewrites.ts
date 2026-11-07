import type { AudioBox, AudioEnhanceRecord, AudioRewrite, AudioRewrites, AudioWorkflow } from '$lib/types';
import { editedHistory, pushVersion, scratchedPosition, shownPosition, versionStepper, withoutRecord, type StepperState, type VersionReader } from './versions';

/** Audio versions hold a string: two are the same text when they are equal. */
const READER: VersionReader<string, AudioRewrite> = { valueOf: (entry) => entry.text, same: (a, b) => a === b };

/** A box with no Enhanced text and no earlier versions. */
export const EMPTY_REWRITES: AudioRewrites = { text: '', from: null, history: [] };

/** The Enhanced text of every box of a fresh draft. */
export function emptyAudioRewrites(): Record<AudioBox, AudioRewrites> {
  return { style: EMPTY_REWRITES, sound: EMPTY_REWRITES, text: EMPTY_REWRITES };
}

/** The box each kind writes its text in. */
export const AUDIO_BOX_OF_KIND: Record<AudioWorkflow, AudioBox> = { music: 'style', sound: 'sound', speech: 'text' };

/** Whether the box has Enhanced text, which Generate then sends in place of the own text. */
export function rewriteUsed(rewrites: AudioRewrites): boolean {
  return rewrites.text.trim() !== '';
}

/** Show *rewrite* (a new one or an earlier version) as the Enhanced text, keeping the history. */
function shown(rewrites: AudioRewrites, rewrite: AudioRewrite): AudioRewrites {
  return { ...rewrites, text: rewrite.text, from: rewrite.from, scratched: null };
}

/** Add a new rewrite as the Enhanced text and as the newest version, keeping the earlier ones (whatever text they were made from) up to `VERSION_LIMIT`. */
export function pushRewrite(rewrites: AudioRewrites, rewrite: AudioRewrite): AudioRewrites {
  return { ...shown(rewrites, rewrite), history: pushVersion(rewrites.history, rewrite, READER) };
}

/** Drop every version and the Enhanced text: what a recent-prompt pick does, as the versions belong to the text that was there. */
export function clearRewrites(): AudioRewrites {
  return EMPTY_REWRITES;
}

/**
 * Why the Enhanced text is out of date, or null while it is not: it was made from other own text (named *label*, such as "Style") or for another
 * model than the box has now. Text typed in the Enhanced tab has no source and is never out of date.
 */
export function rewriteStaleReason(rewrites: AudioRewrites, ownText: string, model: string, label: string): string | null {
  const source = rewrites.from;
  if (source === null || !rewriteUsed(rewrites)) return null;
  const text = source.input !== ownText;
  const other = source.model !== model;
  if (!text && !other) return null;
  return `${text && other ? `Your ${label} and the model` : text ? `Your ${label}` : 'The model'} changed since this was written.`;
}

/** What the version stepper shows for the box, or null for none (it is hidden then); see `versionStepper`. Hidden while *running*. */
export function rewriteStepper(rewrites: AudioRewrites, running = false): StepperState | null {
  if (running) return null;
  return versionStepper(rewrites.history.length, rewriteUsed(rewrites), shownPosition(rewrites.history, rewrites.text, READER), rewrites.scratched ?? null);
}

/** Show version *index* (1-based) of the history as the Enhanced text; an index outside it changes nothing. */
export function stepRewrite(rewrites: AudioRewrites, index: number): AudioRewrites {
  const entry = rewrites.history[index - 1];
  return entry === undefined ? rewrites : shown(rewrites, entry);
}

/** Empty the Enhanced text, so Generate sends the own text; the versions stay, and the stepper remembers which one was emptied so ‹ brings it back. */
export function scratchRewrite(rewrites: AudioRewrites): AudioRewrites {
  return { ...rewrites, text: '', from: null, scratched: scratchedPosition(rewrites.history, rewrites.text, rewriteUsed(rewrites), READER) };
}

/**
 * Apply an edit of the Enhanced text. Text typed by hand is no longer what the enhancer wrote, so the record is dropped (the take must not record the
 * enhancer's options for it). Typing in a shown version replaces that version's text, found by its text before the edit, and drops its record
 * there too; an emptied box is a Scratch.
 */
export function editRewrite(rewrites: AudioRewrites, text: string): AudioRewrites {
  if (text.trim() === '') return scratchRewrite(rewrites);
  const history = editedHistory(rewrites.history, shownPosition(rewrites.history, rewrites.text, READER), (entry) => ({ text, from: withoutRecord(entry.from) }));
  return { ...rewrites, text, from: withoutRecord(rewrites.from), history, scratched: null };
}

/** The record as the server's form field holds it: settings, model, tier when known, format, task. */
export function recordJson(record: AudioEnhanceRecord): string {
  const { settings, model, tier, format, task } = record;
  return JSON.stringify({ settings, model, ...(tier === undefined ? {} : { tier }), format, task });
}

/** What Generate sends for a box: the text that runs, and, when it is a rewrite, the own text it was made from and how it was written. */
export interface SentText {
  text: string;
  /** The own text the rewrite was made from (`prompt_input`); null when the own text runs. */
  input: string | null;
  /** How the rewrite was written (`enhance_record`); null for the own text and for a rewrite edited by hand. */
  record: AudioEnhanceRecord | null;
}

/**
 * The text a box sends. A rewrite runs when the box has one; its `prompt_input` is the text it was made from, even when the own text changed
 * since (an out-of-date rewrite still runs), and the current own text for a rewrite typed by hand.
 */
export function sentText(rewrites: AudioRewrites, ownText: string): SentText {
  if (!rewriteUsed(rewrites)) return { text: ownText, input: null, record: null };
  return { text: rewrites.text, input: rewrites.from?.input ?? ownText, record: rewrites.from?.record ?? null };
}
