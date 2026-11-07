import type { ScriptEnhanceRequest } from '$lib/api/promptEnhance';
import type { AudioEnhanceRecord, AudioSettingsVocabulary, ScriptAction, ScriptSettings } from '$lib/types';
import { formatReadingTime, readingSeconds, stripMarks } from './speechDelivery';

/** The shortest and longest reading time Fit takes, in seconds; the server keeps the same. */
export const SCRIPT_SECONDS_MIN = 3;
export const SCRIPT_SECONDS_MAX = 600;

/** The ways to edit a script, in the order the panel lists them and a combined run applies them (fit first), and the words that name them. */
export const SCRIPT_ACTIONS: readonly { action: ScriptAction; label: string }[] = [
  { action: 'fit', label: 'Fit to time' },
  { action: 'delivery', label: 'Add delivery' },
];

const NOTHING_TO_WORK_ON = 'Nothing to work on yet. A script is written from your idea in Describe it; this helper marks up and fits the words that are there.';

/** The note a recording gets in place of Add delivery. */
export const RECORDING_NOTE = 'A voice from a recording reads as recorded, so delivery marks are left out.';

/** What the panel offers and says for a script. */
export interface ScriptView {
  /** The actions the panel offers: Add delivery is left out for a voice from a recording, which is read as recorded. */
  actions: readonly ScriptAction[];
  /** Why there is nothing to run, in the words the panel shows; null when it can run. */
  problem: string | null;
  /** The characters that are read (marks excluded) and about how long they take at the pace. */
  characters: number;
  seconds: number;
}

/** What the panel shows for *text* read at *pace*; *recording* is a voice that follows a recording. */
export function scriptView(text: string, pace: string, recording: boolean, vocabulary: Pick<AudioSettingsVocabulary, 'marks' | 'reading'> | undefined): ScriptView {
  const characters = [...stripMarks(text, vocabulary?.marks.map((mark) => mark.id) ?? [])].length;
  const words = stripMarks(text, vocabulary?.marks.map((mark) => mark.id) ?? []).replace(/…/g, ' ');
  return {
    actions: recording ? SCRIPT_ACTIONS.filter((item) => item.action === 'fit').map((item) => item.action) : SCRIPT_ACTIONS.map((item) => item.action),
    problem: /[\p{L}\p{N}]/u.test(words) ? null : NOTHING_TO_WORK_ON,
    characters,
    seconds: vocabulary ? readingSeconds(characters, pace, vocabulary.reading) : 0,
  };
}

/** The time Fit starts at: the script's reading time rounded to the nearest 5 seconds, within what Fit takes. */
export function defaultFitSeconds(readingSecondsNow: number): number {
  return Math.min(SCRIPT_SECONDS_MAX, Math.max(SCRIPT_SECONDS_MIN, Math.round(readingSecondsNow / 5) * 5));
}

/** Read the seconds field: the whole number it holds, or null when it is blank, not whole, or outside what Fit takes. */
export function parseFitSeconds(value: string): number | null {
  const seconds = value.trim() === '' ? Number.NaN : Number(value);
  return Number.isInteger(seconds) && seconds >= SCRIPT_SECONDS_MIN && seconds <= SCRIPT_SECONDS_MAX ? seconds : null;
}

/** The run button's words for the picks: `Fit to time`, `Add delivery` or `Fit and add delivery`; with nothing picked, the words of both. */
export function scriptRunLabel(actions: readonly ScriptAction[]): string {
  const fit = actions.includes('fit');
  const delivery = actions.includes('delivery');
  return fit && delivery ? 'Fit and add delivery' : fit ? 'Fit to time' : delivery ? 'Add delivery' : 'Fit and add delivery';
}

/** The picks that apply, in the order a run applies them (fit first), without repeats. */
export function orderedActions(picked: readonly ScriptAction[]): ScriptAction[] {
  return SCRIPT_ACTIONS.map((item) => item.action).filter((action) => picked.includes(action));
}

/** The settings of a request and of the record of its result: the actions in run order; a Fit adds the time and a pace that is not Normal. */
export function scriptSettings(picked: readonly ScriptAction[], seconds: number, pace: string): ScriptSettings {
  const actions = orderedActions(picked);
  return actions.includes('fit') ? { actions, seconds, ...(pace === 'normal' ? {} : { pace }) } : { actions };
}

/** The request for *settings* on *text*: one request, whichever actions it holds. */
export function scriptRequest(model: string, text: string, settings: ScriptSettings): ScriptEnhanceRequest {
  return { mode: 'audio', task: 'script', model, kind: 'speech', prompt: text, settings };
}

/** The record Generate sends for a script the enhancer *model* wrote with *settings*. */
export function scriptRecord(model: string, settings: ScriptSettings): AudioEnhanceRecord {
  return { settings, model, format: 'text', task: 'script' };
}

/** What a finished run hands back: the whole script, the enhancer's notice about it, the enhancer that wrote it (`REPO[@REVISION]`) and what it was asked. */
export interface ScriptResult {
  text: string;
  warning: string | null;
  model: string | null;
  settings: ScriptSettings;
}
