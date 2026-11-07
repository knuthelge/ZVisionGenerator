import { fileName } from '$lib/components/molecules/assetDetails';
import { AUDIO_TEXT_KEYS, DELIVERY_FIELDS, audioModelById, voicesOf } from '$lib/state/audioDraft';
import { hasDescribe } from '$lib/state/audioFill';
import { AUDIO_BOX_OF_KIND, EMPTY_REWRITES, recordJson, rewriteStaleReason, sentText, type SentText } from '$lib/state/audioRewrites';
import { formatTimestamp } from '$lib/state/audioTrack';
import { textWriting } from '$lib/state/textWriting.svelte';
import { trackKey, trackLengths } from '$lib/state/audioTrackLength.svelte';
import { draft } from '$lib/state/draft.svelte';
import { heldUploads } from '$lib/state/heldUploads.svelte';
import { hasPath } from '$lib/state/imageSlots';
import { musicMode, type MusicMode } from '$lib/state/musicMode';
import { readingSeconds, stripMarks, unknownMarkMessage, unknownMarks } from '$lib/state/speechDelivery';
import { voiceConsent } from '$lib/state/voiceConsent.svelte';
import type { AudioBox, AudioModelOption, AudioRewrites, AudioSettingsVocabulary, AudioVoice, AudioWorkflow, DraftState, WorkspaceContext } from '$lib/types';
import { workspaceCapabilities, type WorkspaceCapabilities } from './capabilities';

/** The voice the Speech tab reads in: one of the model's built-in voices, or a recording and its transcript. */
export interface VoiceView {
  /** The voice follows a recording: *From a recording* is chosen on a model that can. */
  recording: boolean;
  /** The Voice select's options: each built-in voice as "Ryan · English", then *From a recording*. */
  options: { value: string; label: string }[];
  /** The Voice select's value: the id of the built-in voice, or `recording`. */
  choice: string;
  /** The built-in voice the take reads in (the draft's, else the model's default); empty on a model without any. */
  speaker: string;
  /** The built-in voice's name, such as "Ryan"; empty on a model without any. */
  speakerLabel: string;
  /** The id of the model's default built-in voice; empty on a model without any. */
  defaultSpeaker: string;
  /** *From a recording* is chosen and the files cloning needs are not downloaded yet: the hint says they download with the first take. */
  cloneDownload: boolean;
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
  /** The Voice section's header summary: the built-in voice's name, the file name, or what is wrong. */
  summary: { text: string; tone: 'muted' | 'active' | 'error' };
  /** The phone's Voice chip: the built-in voice's name, the file name, or "none" while a recording is still to be chosen. */
  chip: string;
}

/** The controls the open (model, kind) shows, whose settings a submit sends: those of its workflow that its model's row does not hide. */
export type AudioShown = Pick<WorkspaceCapabilities, 'showLyrics' | 'showLength' | 'showTempo' | 'showKey' | 'showDelivery' | 'showSoundType'>;

const NOT_SHOWN: AudioShown = { showLyrics: false, showLength: false, showTempo: false, showKey: false, showDelivery: false, showSoundType: false };

/**
 * The one answer to what the Audio tab sends and why Generate is off. It is derived from the draft, the open model, the held
 * recording and the speaker's tick; the boxes render it, Generate reads `reason` and the submit reads `sent`.
 */
export interface AudioView {
  /** The Audio tab is open. */
  active: boolean;
  kind: AudioWorkflow | null;
  model: AudioModelOption | null;
  /** The own text of the open kind: the style, the text to speak, or the sound's description. */
  text: string;
  /** The box that holds the text, with its Enhanced text; null off the Audio tab. */
  box: AudioBox | null;
  rewrites: AudioRewrites;
  /** The box has Enhanced text, which Generate sends in place of the own text. */
  enhanced: boolean;
  /** The Enhanced text was made from other own text, or for another model, than the box has now. */
  stale: boolean;
  /** Why it is out of date, in a sentence ("Your Style changed since this was written."); null while it is not. */
  staleReason: string | null;
  /** What Generate sends for the text: the own text, or the Enhanced text with the own text and the record of how it was written. */
  sent: SentText;
  /** Music: whether the Lyrics make a song or an instrumental; null off the music kind or without a section list. */
  music: MusicMode | null;
  /** Speech: how many characters the text has (as Python counts them), and the model's cap. */
  characters: number;
  /** Speech: about how many seconds the text takes to read at the open Pace; null without the vocabulary or off the speech kind. */
  readingSeconds: number | null;
  /** Speech: the Pace the text is read at; a voice that follows a recording is read as recorded, so Normal. */
  pace: string;
  maxCharacters: number | null;
  overLimit: boolean;
  voice: VoiceView;
  /** The settings the open row takes; the submit sends these and no others. */
  shown: AudioShown;
  /** Generate's one reason when the take cannot be made yet, or null. */
  reason: string | null;
  /** The reason comes from the voice (its recording, transcript or tick), which the Voice chip and section show. */
  reasonFromVoice: boolean;
}

/** What `audioView` reads from the draft, the open model and the screen. */
export interface AudioViewInput {
  state: DraftState;
  caps: Pick<WorkspaceCapabilities, 'isAudio' | 'audioKind' | 'showVoice'> & AudioShown;
  model: AudioModelOption | null;
  /** The held voice recording (`heldUploads.voice`), and whether the held uploads have been read back. */
  file: File | null;
  restored: boolean;
  /** The length in seconds of the recording under a key (`trackKey`), or null while unknown. */
  lengthOf: (key: string | null) => number | null;
  consent: { ticked: boolean; exempt: boolean };
  /** The marks a speech text may hold, which are not read and so not counted, and how fast speech reads (`audio_settings`); without it nothing is stripped and no time is told. */
  vocabulary?: Pick<AudioSettingsVocabulary, 'marks' | 'reading'>;
  /** The Describe it box is served (an enhancer is): an empty Style or Text with an idea written then points to Fill in. */
  describes?: boolean;
  /** Why Generate waits for a run that is writing the text it sends (`textWriting.reason`), or null. It comes before every other reason. */
  writing?: string | null;
}

const NO_VOICE: VoiceView = {
  recording: false, options: [], choice: '', speaker: '', speakerLabel: '', defaultSpeaker: '', cloneDownload: false, set: false, path: null, file: null, name: '', length: null, gone: false, error: null,
  text: '', needsConsent: false, ticked: false, sent: null, summary: { text: '', tone: 'muted' }, chip: '',
};

/** The value of the Voice select's *From a recording* option. */
export const RECORDING_CHOICE = 'recording';

/** Apply a pick of the Voice select: *From a recording*, or a built-in voice (the model's default is kept as no choice, so it is not shown as changed). */
export function chooseVoice(voice: Pick<VoiceView, 'defaultSpeaker'>, choice: string): void {
  if (choice === RECORDING_CHOICE) draft.update('voiceMode', 'recording');
  else draft.patch({ voiceMode: 'builtin', speechSpeaker: choice === voice.defaultSpeaker ? null : choice });
}

/** The Voice select's options for a model: its built-in voices with their languages, then *From a recording*. */
export function voiceOptions(voices: readonly AudioVoice[]): { value: string; label: string }[] {
  return [...voices.map((voice) => ({ value: voice.id, label: `${voice.label} · ${voice.language}` })), { value: RECORDING_CHOICE, label: VOICE_WORDS.recording }];
}

/** Format the voice range of a model for a message: "3 to 30 seconds". */
function rangeText(range: { min: number; max: number } | null): string {
  return range === null ? '' : `${range.min} to ${range.max} seconds`;
}

/** The voice, and the reason (in reading order) a recording blocks Generate, or null. */
function voiceView(input: AudioViewInput): { voice: VoiceView; reason: string | null } {
  const { state, caps, model, file, restored, lengthOf, consent } = input;
  if (!caps.showVoice) return { voice: NO_VOICE, reason: null };
  const range = model?.kind === 'speech' ? { min: model.limits.min_voice_seconds, max: model.limits.max_voice_seconds } : null;
  const path = hasPath(state.voicePath) ? state.voicePath : null;
  const set = path !== null || state.voiceUpload !== null;
  const recording = state.voiceMode === 'recording';
  const voices = voicesOf(model);
  const builtIn = voices.find((voice) => voice.id === (state.speechSpeaker ?? (model?.kind === 'speech' ? model.default_voice : null))) ?? voices[0] ?? null;
  const speaker = recording ? '' : (builtIn?.id ?? '');
  const cloneDownload = recording && model?.kind === 'speech' && model.voice_download?.downloaded === false;
  const name = file?.name ?? (path !== null ? fileName(path) : (state.voiceUpload ?? ''));
  const gone = recording && file === null && path === null && state.voiceUpload !== null && restored;
  const length = recording && set ? lengthOf(trackKey(file === null ? path : null, file)) : null;
  const outOfRange = range !== null && length !== null && (length < range.min || length > range.max);
  const error = gone ? 'Choose the recording again.' : outOfRange ? `Use a recording of ${rangeText(range)}. This one is ${formatTimestamp(length)}.` : null;
  const text = state.voiceText;
  const needsConsent = recording && set && !consent.exempt;
  const sentPath = file === null ? path : null;
  const sent = recording && (file !== null || sentPath !== null) ? { path: sentPath, file, text } : null;
  const summary: VoiceView['summary'] = !recording ? { text: builtIn?.label ?? '', tone: 'muted' } : error !== null ? { text: error.split('.')[0], tone: 'error' } : set ? { text: name, tone: 'active' } : { text: 'Choose a recording', tone: 'muted' };
  const chip = !recording ? (builtIn?.label ?? '') : set ? name : 'none';
  const reason = !recording ? null
    : !set ? 'Choose a recording to clone.'
    : gone ? error
    : error !== null ? `Use a recording of ${rangeText(range)}.`
    : text.trim() === '' ? 'Type what the recording says.'
    : needsConsent && !consent.ticked ? 'Tick the box to confirm you may clone this voice.'
    : null;
  return { voice: { recording, options: voiceOptions(voices), choice: recording ? RECORDING_CHOICE : (builtIn?.id ?? RECORDING_CHOICE), speaker, speakerLabel: builtIn?.label ?? '', defaultSpeaker: model?.kind === 'speech' ? (model.default_voice ?? '') : '', cloneDownload, set, path, file, name, length, gone, error, text, needsConsent, ticked: consent.ticked, sent, summary, chip }, reason };
}

/** Derive what the open Audio tab sends and the one reason, if any, that Generate is off. Reasons come in the order the boxes are read. */
export function audioView(input: AudioViewInput): AudioView {
  const { state, caps, model } = input;
  const kind = caps.audioKind;
  const { voice, reason: voiceReason } = voiceView(input);
  if (!caps.isAudio || kind === null) {
    return { active: false, kind: null, model, text: '', box: null, rewrites: EMPTY_REWRITES, enhanced: false, stale: false, staleReason: null, sent: sentText(EMPTY_REWRITES, ''), music: null, characters: 0, readingSeconds: null, pace: 'normal', maxCharacters: null, overLimit: false, voice, shown: NOT_SHOWN, reason: null, reasonFromVoice: false };
  }
  const text = state[AUDIO_TEXT_KEYS[kind]];
  const box = AUDIO_BOX_OF_KIND[kind];
  const rewrites = state.audioRewrites[box];
  const sent = sentText(rewrites, text);
  const maxCharacters = model?.kind === 'speech' ? model.limits.max_text_chars : null;
  // Counted as Python's len counts what is read: a character outside the BMP is one, and a mark is none.
  const marks = input.vocabulary?.marks.map((mark) => mark.id) ?? [];
  const read = kind === 'speech' ? stripMarks(sent.text, marks) : sent.text;
  const characters = kind === 'speech' ? [...read].length : 0;
  const unknown = kind === 'speech' ? unknownMarks(sent.text, marks) : [];
  // A voice that follows a recording is read as recorded, so its Pace does not apply.
  const pace = voice.recording ? 'normal' : state.speechDelivery.pace;
  const seconds = kind === 'speech' && input.vocabulary ? readingSeconds(characters, pace, input.vocabulary.reading) : null;
  const overLimit = maxCharacters !== null && characters > maxCharacters;
  const music = kind === 'music' && model?.instrumental_sections ? musicMode(state.lyrics, model.instrumental_sections) : null;
  const minutes = maxCharacters === null ? 0 : Math.round(maxCharacters / 1000);
  // A text of marks alone has nothing to read, as on the server.
  const empty = read.trim() === '';
  const idea = input.describes === true && hasDescribe(kind) && state.audioDescribe[kind].text.trim() !== '';
  const textReason =
    empty ? (idea ? { music: 'Press Fill in, or write the style.', speech: 'Press Fill in, or write the text to speak.' }[kind as 'music' | 'speech'] : { music: 'Write the style.', speech: 'Write the text to speak.', sound: 'Describe the sound.' }[kind])
    : unknown.length > 0 ? unknownMarkMessage(unknown[0])
    : null;
  const writing = input.writing ?? null;
  const reason =
    writing !== null ? writing
    : textReason !== null ? textReason
    : voiceReason !== null ? voiceReason
    : overLimit ? `Shorten the text: it's over ${maxCharacters!.toLocaleString('en-US')} characters, about ${minutes} minutes of speech.`
    : null;
  const shown: AudioShown = { showLyrics: caps.showLyrics, showLength: caps.showLength, showTempo: caps.showTempo, showKey: caps.showKey, showDelivery: caps.showDelivery, showSoundType: caps.showSoundType };
  const staleReason = rewriteStaleReason(rewrites, text, state.model, { music: 'Style', sound: 'Prompt', speech: 'Text' }[kind]);
  return { active: true, kind, model, text, box, rewrites, enhanced: sent.input !== null, stale: staleReason !== null, staleReason, sent, music, characters, readingSeconds: seconds, pace, maxCharacters, overLimit, voice, shown, reason, reasonFromVoice: writing === null && textReason === null && voiceReason !== null };
}

/** The view of the open tab, read from the draft, the held recording and the tick. */
export function currentAudioView(context: WorkspaceContext): AudioView {
  const state = draft.state;
  const caps = workspaceCapabilities(context, state);
  return audioView({
    state,
    caps,
    model: caps.audioKind === null ? null : audioModelById(context, state.model, caps.audioKind),
    file: heldUploads.voice,
    restored: heldUploads.restored,
    lengthOf: (key) => trackLengths.lengthOf(key),
    consent: voiceConsent,
    vocabulary: context.audio_settings,
    describes: context.prompt_enhancer != null,
    writing: textWriting.reason,
  });
}

/**
 * The fields an audio Generate sends: the ones Copy CLI reads, and only those of the open kind. The tick goes as
 * `voice_consent` (a hidden "false" first, "on" when ticked), so an unticked box sends nothing true; the recording goes as
 * `voice_file` (an upload, which wins) or `voice_path`. A built-in voice goes as `speaker` and a recording has none.
 *
 * A setting goes only when the open (model, kind) shows its control (`view.shown`): the Lyrics, the Tempo and Key (blank is Auto),
 * the length (blank is Auto, or the model's default), the Sound Type, and the five delivery values of a built-in voice. A value
 * the draft keeps for a hidden control is not sent.
 */
export function audioFormData(view: AudioView, state: DraftState): FormData {
  const form = new FormData();
  const kind = view.kind!;
  form.set('mode', 'audio');
  form.set('workflow', kind);
  form.set('model', state.model);
  form.set('prompt', view.sent.text);
  // A rewrite runs; the own text it was made from and how it was written go beside it, so the take records both.
  if (view.sent.input !== null) form.set('prompt_input', view.sent.input);
  if (view.sent.record !== null) form.set('enhance_record', recordJson(view.sent.record));
  const { shown } = view;
  if (kind === 'music' && shown.showLyrics) form.set('lyrics', state.lyrics);
  if (kind === 'music' && shown.showTempo) form.set('tempo', state.audioTempo === null ? '' : String(state.audioTempo));
  if (kind === 'music' && shown.showKey) form.set('key', state.audioKey ?? '');
  if (kind !== 'speech' && shown.showLength) {
    const seconds = kind === 'music' ? state.musicSeconds : state.audioSeconds;
    form.set('seconds', seconds > 0 ? String(seconds) : '');
  }
  if (kind === 'sound' && shown.showSoundType) form.set('sound_type', state.soundType);
  if (kind === 'speech' && view.voice.speaker !== '') form.set('speaker', view.voice.speaker);
  if (kind === 'speech' && shown.showDelivery && !view.voice.recording) {
    for (const field of DELIVERY_FIELDS) form.set(field, state.speechDelivery[field]);
  }
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

/** The Voice section's and popover's shared words: the recording option, the cloning download hint, the transcript's placeholder and the consent line. */
export const VOICE_WORDS = {
  recording: 'From a recording',
  downloadHint: 'Voice cloning downloads about 4.9 GB the first time you generate with a recording.',
  placeholder: 'The words spoken in the recording',
  consent: "This is my voice, or I have the speaker's permission to clone it.",
};
