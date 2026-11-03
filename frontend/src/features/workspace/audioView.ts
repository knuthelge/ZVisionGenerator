import { fileName } from '$lib/components/molecules/assetDetails';
import { AUDIO_TEXT_KEYS, audioModelById } from '$lib/state/audioDraft';
import { formatTimestamp } from '$lib/state/audioTrack';
import { trackKey, trackLengths } from '$lib/state/audioTrackLength.svelte';
import { draft } from '$lib/state/draft.svelte';
import { heldUploads } from '$lib/state/heldUploads.svelte';
import { hasPath } from '$lib/state/imageSlots';
import { musicMode, type MusicMode } from '$lib/state/musicMode';
import { voiceConsent } from '$lib/state/voiceConsent.svelte';
import type { AudioModelOption, AudioWorkflow, DraftState, WorkspaceContext } from '$lib/types';
import { workspaceCapabilities, type WorkspaceCapabilities } from './capabilities';

/** The voice the Speech tab follows: a recording and its transcript, or none (the model picks a voice). */
export interface VoiceView {
  /** The voice follows a recording: *From a recording* is chosen on a model that can. */
  recording: boolean;
  /** A recording is chosen, as a path or as an upload. */
  set: boolean;
  path: string | null;
  /** The file held for the recording, or null. */
  file: File | null;
  /** The name to show: the held file's, the path's file name, else the name an upload was stored under. */
  name: string;
  /** The recording's length in seconds, once it is known. */
  length: number | null;
  /** The recording is an upload whose file did not come back from a reload. */
  gone: boolean;
  /** What is wrong with the recording itself (its length, or a file that did not come back), shown under its slot; null when nothing. */
  error: string | null;
  text: string;
  /** The speaker's consent is asked for: a recording is chosen and it is not a take the model chose the voice of. */
  needsConsent: boolean;
  ticked: boolean;
  /** What a submit sends for the voice: null unless the voice follows a recording that has a path or a file. */
  sent: { path: string | null; file: File | null; text: string } | null;
  /** The Voice section's header summary: "Any", the file name, or what is wrong. */
  summary: { text: string; tone: 'muted' | 'active' | 'error' };
  /** The phone's Voice chip: "Any", the file name, or "none" while a recording is still to be chosen. */
  chip: string;
}

/**
 * The one answer to what the Audio tab sends and why Generate is off. It is derived from the draft, the open model, the held
 * recording and the speaker's tick; the boxes render it, Generate reads `reason` and the submit reads `sent`.
 */
export interface AudioView {
  /** The Audio tab is open. */
  active: boolean;
  kind: AudioWorkflow | null;
  model: AudioModelOption | null;
  /** The text of the open kind: the style, the text to speak, or the sound's description. */
  text: string;
  /** Music: whether the Lyrics make a song or an instrumental; null off the music kind or without a section list. */
  music: MusicMode | null;
  /** Speech: how many characters the text has (as Python counts them), and the model's cap. */
  characters: number;
  maxCharacters: number | null;
  overLimit: boolean;
  voice: VoiceView;
  /** Generate's one reason when the take cannot be made yet, or null. */
  reason: string | null;
}

/** What `audioView` reads from the draft, the open model and the screen. */
export interface AudioViewInput {
  state: DraftState;
  caps: Pick<WorkspaceCapabilities, 'isAudio' | 'audioKind' | 'showVoice'>;
  model: AudioModelOption | null;
  /** The held voice recording (`heldUploads.voice`), and whether the held uploads have been read back. */
  file: File | null;
  restored: boolean;
  /** The length in seconds of the recording under a key (`trackKey`), or null while unknown. */
  lengthOf: (key: string | null) => number | null;
  consent: { ticked: boolean; exempt: boolean };
}

const NO_VOICE: VoiceView = {
  recording: false, set: false, path: null, file: null, name: '', length: null, gone: false, error: null,
  text: '', needsConsent: false, ticked: false, sent: null, summary: { text: 'Any', tone: 'muted' }, chip: 'Any',
};

/** Format a range of seconds for a message: "3 to 30 seconds". */
function rangeText(range: { min: number; max: number }): string {
  return `${range.min} to ${range.max} seconds`;
}

/** The voice, and the reason (in reading order) a recording blocks Generate, or null. */
function voiceView(input: AudioViewInput): { voice: VoiceView; reason: string | null } {
  const { state, caps, model, file, restored, lengthOf, consent } = input;
  if (!caps.showVoice) return { voice: NO_VOICE, reason: null };
  const range = { min: model?.limits.min_voice_seconds ?? 3, max: model?.limits.max_voice_seconds ?? 30 };
  const path = hasPath(state.voicePath) ? state.voicePath : null;
  const set = path !== null || state.voiceUpload !== null;
  const recording = state.voiceMode === 'recording';
  const name = file?.name ?? (path !== null ? fileName(path) : (state.voiceUpload ?? ''));
  const gone = recording && file === null && path === null && state.voiceUpload !== null && restored;
  const length = recording && set ? lengthOf(trackKey(file === null ? path : null, file)) : null;
  const outOfRange = length !== null && (length < range.min || length > range.max);
  const error = gone ? 'Choose the recording again.' : outOfRange ? `Use a recording of ${rangeText(range)}. This one is ${formatTimestamp(length)}.` : null;
  const text = state.voiceText;
  const needsConsent = recording && set && !consent.exempt;
  const sentPath = file === null ? path : null;
  const sent = recording && (file !== null || sentPath !== null) ? { path: sentPath, file, text } : null;
  const summary: VoiceView['summary'] = !recording ? { text: 'Any', tone: 'muted' } : error !== null ? { text: error.split('.')[0], tone: 'error' } : set ? { text: name, tone: 'active' } : { text: 'Choose a recording', tone: 'muted' };
  const chip = !recording ? 'Any' : set ? name : 'none';
  const reason = !recording ? null
    : !set ? 'Choose a recording to clone.'
    : gone ? error
    : error !== null ? `Use a recording of ${rangeText(range)}.`
    : text.trim() === '' ? 'Type what the recording says.'
    : needsConsent && !consent.ticked ? 'Tick the box to confirm you may clone this voice.'
    : null;
  return { voice: { recording, set, path, file, name, length, gone, error, text, needsConsent, ticked: consent.ticked, sent, summary, chip }, reason };
}

/** Derive what the open Audio tab sends and the one reason, if any, that Generate is off. Reasons come in the order the boxes are read. */
export function audioView(input: AudioViewInput): AudioView {
  const { state, caps, model } = input;
  const kind = caps.audioKind;
  const { voice, reason: voiceReason } = voiceView(input);
  if (!caps.isAudio || kind === null) {
    return { active: false, kind: null, model, text: '', music: null, characters: 0, maxCharacters: null, overLimit: false, voice, reason: null };
  }
  const text = state[AUDIO_TEXT_KEYS[kind]];
  const maxCharacters = kind === 'speech' ? (model?.limits.max_text_chars ?? null) : null;
  // Counted as Python's len counts: a character outside the BMP is one.
  const characters = kind === 'speech' ? [...text].length : 0;
  const overLimit = maxCharacters !== null && characters > maxCharacters;
  const music = kind === 'music' && model?.instrumental_sections ? musicMode(state.lyrics, model.instrumental_sections) : null;
  const minutes = maxCharacters === null ? 0 : Math.round(maxCharacters / 1000);
  const reason =
    text.trim() === '' ? { music: 'Write the style.', speech: 'Write the text to speak.', sound: 'Describe the sound.' }[kind]
    : voiceReason !== null ? voiceReason
    : overLimit ? `Shorten the text: it's over ${maxCharacters!.toLocaleString('en-US')} characters, about ${minutes} minutes of speech.`
    : null;
  return { active: true, kind, model, text, music, characters, maxCharacters, overLimit, voice, reason };
}

/** The view of the open tab, read from the draft, the held recording and the tick. */
export function currentAudioView(context: WorkspaceContext): AudioView {
  const state = draft.state;
  const caps = workspaceCapabilities(context, state);
  return audioView({
    state,
    caps,
    model: audioModelById(context, state.model),
    file: heldUploads.voice,
    restored: heldUploads.restored,
    lengthOf: (key) => trackLengths.lengthOf(key),
    consent: voiceConsent,
  });
}

/**
 * The fields an audio Generate sends: the ones Copy CLI reads, and only those of the open kind. The tick goes as
 * `voice_consent` (a hidden "false" first, "on" when ticked), so an unticked box sends nothing true; the recording goes as
 * `voice_file` (an upload, which wins) or `voice_path`.
 */
export function audioFormData(view: AudioView, state: DraftState): FormData {
  const form = new FormData();
  const kind = view.kind!;
  form.set('mode', 'audio');
  form.set('workflow', kind);
  form.set('model', state.model);
  form.set('prompt', view.text);
  // A model that takes no lyrics or no length is not sent them: the server refuses a field the model does not take.
  if (kind === 'music' && view.model?.capabilities.lyrics !== false) form.set('lyrics', state.lyrics);
  if (kind === 'sound' && view.model?.capabilities.duration !== false) form.set('seconds', String(state.audioSeconds));
  if (kind === 'speech' && view.voice.sent !== null) {
    const { file, path, text } = view.voice.sent;
    if (file !== null) form.set('voice_file', file);
    else if (path !== null) form.set('voice_path', path);
    form.set('voice_text', text);
    form.append('voice_consent', 'false');
    if (view.voice.ticked) form.append('voice_consent', 'on');
  }
  form.set('runs', String(state.runs));
  if (state.seed !== null) form.set('seed', String(state.seed));
  return form;
}

/** The Voice section's and popover's shared words: the two modes, the Any hint, the transcript's placeholder and the consent line. */
export const VOICE_WORDS = {
  modes: [{ value: 'any', label: 'Any' }, { value: 'recording', label: 'From a recording' }] as { value: 'any' | 'recording'; label: string }[],
  anyHint: 'The model picks a voice, and it changes from take to take. To keep one, use a take as the voice: Gallery, Use as → Voice.',
  placeholder: 'The words spoken in the recording',
  consent: "This is my voice, or I have the speaker's permission to clone it.",
};
