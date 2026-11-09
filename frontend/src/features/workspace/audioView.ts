import { browseMediaUrl } from '$lib/api/browse';
import { voiceAudioUrl } from '$lib/api/voices';
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
import { voicesStore } from '$lib/state/voices.svelte';
import type { AudioBox, AudioModelOption, AudioRewrites, AudioSettingsVocabulary, AudioVoice, AudioWorkflow, DraftState, SavedVoice, WorkspaceContext } from '$lib/types';
import { workspaceCapabilities, type WorkspaceCapabilities } from './capabilities';

/** One option of the Voice select; `group` names the group it sits in. */
export interface VoiceOption {
  value: string;
  label: string;
  group: string;
}

/** What a submit sends for the voice: the slug of a saved voice, or a one-off recording with its transcript. */
export type VoiceSent = { saved: string } | { path: string | null; file: File | null; text: string };

/** What the Play button next to the Voice select plays: a clip the server holds (`src`), a recording held in the browser (`file`), or nothing yet. */
export interface VoicePlay {
  src: string | null;
  file: File | null;
  /** What the button is named after: "Play Morgan", "Play a preview of Ryan", "Play the recording". */
  label: string;
}

/** The preview clip shipped for a default voice, served from the app's public assets. */
const voicePreviewUrl = (id: string): string => `/app-static/voices/${id}.mp3`;

/** The voice the Speech tab reads in: one of the model's default voices, a saved voice, or a one-off recording and its transcript. */
export interface VoiceView {
  /** The voice follows a one-off recording, made in the Voices window or reused from a take. */
  recording: boolean;
  /** The voice is a saved one (`voiceMode` is `saved`). */
  saved: boolean;
  /** The voice is cloned from a reference, a one-off recording or a saved voice: it is read at Normal pace, with no delivery and no built-in voice. */
  cloned: boolean;
  /** The library has loaded and has no voice under the draft's slug: it was deleted (here, in another tab, or before a reload). */
  deleted: boolean;
  /** The Voice select's options: each saved voice by name (group *Your voices*, left out while there are none), then each default voice as "Ryan · English" (group *Default voices*). A one-off recording has no option. */
  options: VoiceOption[];
  /** The Voice select's value: `saved:<slug>`, the id of the built-in voice, or empty while a one-off recording is the voice. */
  choice: string;
  /** What the Play button plays. */
  play: VoicePlay;
  /** How many designed voices wait to be picked (made and not saved since the last save or Clear voices). */
  pending: number;
  /** The built-in voice the take reads in (the draft's, else the model's default); empty on a model without any. */
  speaker: string;
  /** The built-in voice's name, such as "Ryan"; empty on a model without any. */
  speakerLabel: string;
  /** The id of the model's default built-in voice; empty on a model without any. */
  defaultSpeaker: string;
  /** A one-off recording or a saved voice is the voice and the files cloning needs are not downloaded yet: the hint says they download with the first take. */
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
  /** What a submit sends for the voice: the saved voice's slug, or a recording with a path or a file; null otherwise. */
  sent: VoiceSent | null;
  /** The Voice section's header summary: the built-in voice's name, the saved voice's name, the file name, or what is wrong; with the count of designed voices waiting to be picked. */
  summary: { text: string; tone: 'muted' | 'active' | 'error' };
  /** The phone's Voice chip: the built-in or saved voice's name (`deleted` once it is gone), the file name, or "none" while a recording is still to be chosen. */
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
  /** Speech: the Pace the text is read at; a cloned voice (a recording or a saved voice) is read as recorded, so Normal. */
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
  /** The saved voices (`voicesStore`): whether the list has been read, the voices it holds, and how many designed voices wait to be picked. Without it nothing is known about them. */
  voices?: { loaded: boolean; list: readonly SavedVoice[]; unpicked?: number };
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
  recording: false, saved: false, cloned: false, deleted: false, options: [], choice: '', play: { src: null, file: null, label: '' }, pending: 0, speaker: '', speakerLabel: '', defaultSpeaker: '', cloneDownload: false, set: false, path: null, file: null, name: '', length: null, gone: false, error: null,
  text: '', needsConsent: false, ticked: false, sent: null, summary: { text: '', tone: 'muted' }, chip: '',
};

/** What a saved voice's option value starts with, followed by its slug. */
const SAVED_PREFIX = 'saved:';

/** The Voice select's value for the saved voice *slug*. */
export function savedChoice(slug: string): string {
  return `${SAVED_PREFIX}${slug}`;
}

/** Apply a pick of the Voice select: a saved voice or a built-in voice (the model's default is kept as no choice, so it is not shown as changed). Either drops a one-off recording. */
export function chooseVoice(voice: Pick<VoiceView, 'defaultSpeaker'>, choice: string): void {
  if (choice.startsWith(SAVED_PREFIX)) draft.setSavedVoice(choice.slice(SAVED_PREFIX.length));
  else draft.setBuiltinVoice(choice === voice.defaultSpeaker ? null : choice);
}

/** The Voice select's options for a model: the saved voices by name, then its built-in voices with their languages. */
export function voiceOptions(voices: readonly AudioVoice[], saved: readonly SavedVoice[] = []): VoiceOption[] {
  return [
    ...saved.map((voice) => ({ value: savedChoice(voice.slug), label: voice.name, group: VOICE_WORDS.savedGroup })),
    ...voices.map((voice) => ({ value: voice.id, label: `${voice.label} · ${voice.language}`, group: VOICE_WORDS.builtInGroup })),
  ];
}

/** Format the voice range of a model for a message: "3 to 30 seconds". */
function rangeText(range: { min: number; max: number } | null): string {
  return range === null ? '' : `${range.min} to ${range.max} seconds`;
}

/**
 * The sentence for a recording whose length is outside the model's range. The length is in whole seconds, rounded toward the
 * side of the range it misses (2.6 s of 3 to 30 reads 0:02), so it never reads as a length that fits.
 */
export function recordingLengthError(length: number, range: { min: number; max: number }): string {
  const whole = length < range.min ? Math.floor(length) : Math.ceil(length);
  return `Use a recording of ${rangeText(range)}. This one is ${formatTimestamp(whole)}.`;
}

/** The voice, and the reason (in reading order) a recording blocks Generate, or null. */
function voiceView(input: AudioViewInput): { voice: VoiceView; reason: string | null } {
  const { state, caps, model, file, restored, lengthOf, consent } = input;
  const library = input.voices ?? { loaded: false, list: [] };
  if (!caps.showVoice) return { voice: NO_VOICE, reason: null };
  const range = model?.kind === 'speech' ? { min: model.limits.min_voice_seconds, max: model.limits.max_voice_seconds } : null;
  const path = hasPath(state.voicePath) ? state.voicePath : null;
  const set = path !== null || state.voiceUpload !== null;
  const recording = state.voiceMode === 'recording';
  const saved = state.voiceMode === 'saved';
  const cloned = recording || saved;
  const voices = voicesOf(model);
  const builtIn = voices.find((voice) => voice.id === (state.speechSpeaker ?? (model?.kind === 'speech' ? model.default_voice : null))) ?? voices[0] ?? null;
  const speaker = cloned ? '' : (builtIn?.id ?? '');
  const record = saved ? (library.list.find((voice) => voice.slug === state.savedVoice) ?? null) : null;
  // Nothing is known about the library before its first load, so a voice is not called deleted until then.
  const deleted = saved && library.loaded && record === null;
  const cloneDownload = cloned && model?.kind === 'speech' && model.voice_download?.downloaded === false;
  const name = file?.name ?? (path !== null ? fileName(path) : (state.voiceUpload ?? ''));
  const gone = recording && file === null && path === null && state.voiceUpload !== null && restored;
  const length = recording && set ? lengthOf(trackKey(file === null ? path : null, file)) : null;
  const outOfRange = range !== null && length !== null && (length < range.min || length > range.max);
  const error = gone ? 'Choose the recording again.' : outOfRange ? recordingLengthError(length, range) : null;
  const text = state.voiceText;
  const needsConsent = recording && set && !consent.exempt;
  const sentPath = file === null ? path : null;
  const savedName = record?.name ?? state.savedVoice ?? '';
  const sent: VoiceSent | null = saved ? (state.savedVoice === null ? null : { saved: state.savedVoice }) : recording && (file !== null || sentPath !== null) ? { path: sentPath, file, text } : null;
  const pending = library.unpicked ?? 0;
  const waiting = (text: string): string => (pending > 0 ? `${text} · ${pending} to pick from` : text);
  const summary: VoiceView['summary'] = saved ? (deleted ? { text: VOICE_WORDS.deletedSummary, tone: 'error' } : { text: waiting(savedName), tone: 'active' }) : !recording ? { text: waiting(builtIn?.label ?? ''), tone: 'muted' } : error !== null ? { text: error.split('.')[0], tone: 'error' } : set ? { text: waiting(name), tone: 'active' } : { text: waiting('Choose a recording'), tone: 'muted' };
  const chip = saved ? (deleted ? VOICE_WORDS.deletedChip : savedName) : !recording ? (builtIn?.label ?? '') : set ? name : 'none';
  // Play: a saved voice's recording, the one-off recording (held in the browser, or a file on the host), or the preview clip shipped for a default voice.
  const play: VoicePlay = saved ? { src: record === null ? null : voiceAudioUrl(record.slug), file: null, label: savedName }
    : recording ? { src: file === null && path !== null ? browseMediaUrl('audio_track', path) : null, file, label: 'the recording' }
    : { src: builtIn === null ? null : voicePreviewUrl(builtIn.id), file: null, label: `a preview of ${builtIn?.label ?? 'the voice'}` };
  const reason = saved ? (deleted ? VOICE_WORDS.deleted : null)
    : !recording ? null
    : !set ? 'Choose a recording to clone.'
    : gone ? error
    : error !== null ? `Use a recording of ${rangeText(range)}.`
    : text.trim() === '' ? 'Type what the recording says.'
    : needsConsent && !consent.ticked ? 'Tick the box to confirm you may clone this voice.'
    : null;
  return { voice: { recording, saved, cloned, deleted, options: voiceOptions(voices, library.list), choice: saved ? savedChoice(state.savedVoice ?? '') : recording ? '' : (builtIn?.id ?? ''), play, pending, speaker, speakerLabel: builtIn?.label ?? '', defaultSpeaker: model?.kind === 'speech' ? (model.default_voice ?? '') : '', cloneDownload, set, path, file, name, length, gone, error, text, needsConsent, ticked: consent.ticked, sent, summary, chip }, reason };
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
  // A cloned voice (a recording or a saved voice) is read as recorded, so its Pace does not apply.
  const pace = voice.cloned ? 'normal' : state.speechDelivery.pace;
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
    voices: voicesStore,
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
  if (kind === 'speech' && shown.showDelivery && !view.voice.cloned) {
    for (const field of DELIVERY_FIELDS) form.set(field, state.speechDelivery[field]);
  }
  if (kind === 'speech' && view.voice.sent !== null) {
    const sent = view.voice.sent;
    if ('saved' in sent) {
      // The record holds the transcript and the consent, so a saved voice sends neither.
      form.set('saved_voice', sent.saved);
    } else {
      if (sent.file !== null) form.set('voice_file', sent.file);
      else if (sent.path !== null) form.set('voice_path', sent.path);
      form.set('voice_text', sent.text);
      form.append('voice_consent', 'false');
      if (view.voice.ticked) form.append('voice_consent', 'on');
    }
  }
  form.set('runs', String(state.runs));
  if (state.seed !== null) form.set('seed', String(state.seed));
  return form;
}

/** The Voice section's and popover's shared words: the groups of the select, the cloning download hint, the deleted-voice reason, the transcript's placeholder and the consent line. */
export const VOICE_WORDS = {
  recording: 'From recording',
  builtInGroup: 'Default voices',
  savedGroup: 'Your voices',
  downloadHint: 'Voice cloning downloads its model the first time you generate with a recording or a saved voice.',
  deleted: 'This saved voice was deleted. Choose another voice.',
  deletedPlaceholder: 'Deleted voice',
  deletedSummary: 'Voice deleted',
  deletedChip: 'deleted',
  placeholder: 'The words spoken in the recording',
};
