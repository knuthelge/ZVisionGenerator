import { parseAudioRecord, rewriteTaskOf } from './audioEnhance';
import { AUDIO_BOX_OF_KIND, emptyAudioRewrites, pushRewrite, scratchRewrite } from './audioRewrites';
import { isAudioWorkflow } from './workspaceTab';
import { AUDIO_KINDS, SOUND_TYPES, type AudioDescribe, type AudioEnhanceSettings, type AudioModelOption, type AudioVoice, type AudioWorkflow, type DraftState, type SpeechDeliveryValues, type WorkspaceContext, type WorkspacePrefill } from '$lib/types';

/** A Describe it box nobody has used: no idea, no filled field, nothing to undo, folded while the Lyrics or Text have words. */
const EMPTY_DESCRIBE: AudioDescribe = { text: '', marked: [], filled: {}, undo: null, folded: null };

/**
 * The text fields of the Audio tab: the Music style and lyrics, the Speech text and the Sound prompt, the Enhanced text of the boxes
 * that have one, the lyrics helper's versions, the Enhance options of each task, and the Describe it box of Music and of Speech.
 */
const AUDIO_TEXT_DEFAULTS = {
  audioStyle: '',
  lyrics: '',
  speechText: '',
  soundPrompt: '',
  audioRewrites: emptyAudioRewrites(),
  lyricsVersions: [],
  lyricsCurrent: 0,
  enhanceSettingsAudio: { sound: null, style: null },
  audioDescribe: { music: EMPTY_DESCRIBE, speech: EMPTY_DESCRIBE },
} as const satisfies Partial<DraftState>;

/** The delivery of a built-in voice nobody steered: the id of each of the five settings at its middle. */
export const DEFAULT_SPEECH_DELIVERY: SpeechDeliveryValues = { emotion: 'neutral', manner: 'natural', pace: 'normal', pitch: 'normal', expressiveness: 'normal' };

/** The five delivery settings, in the order of the form, a saved take and a Reuse link. */
export const DELIVERY_FIELDS = ['emotion', 'manner', 'pace', 'pitch', 'expressiveness'] as const satisfies readonly (keyof SpeechDeliveryValues)[];

/**
 * The other fields the Audio tab keeps in the draft: the settings of each kind (a sound's length and type, a song's tempo, key and
 * length, a speech's delivery) and the voice (a built-in voice, null for the model's default, or a recording). A sound's or a song's
 * length of 0 is Auto (the form sends it blank, and a take that uses it saves the number it picked), and a tempo or key of null is Auto.
 */
const AUDIO_SETTING_DEFAULTS = {
  audioSeconds: 0,
  audioTempo: null,
  audioKey: null,
  musicSeconds: 0,
  soundType: 'auto',
  speechDelivery: DEFAULT_SPEECH_DELIVERY,
  voiceMode: 'builtin',
  speechSpeaker: null,
  savedVoice: null,
  voicePath: null,
  voiceUpload: null,
  voiceText: '',
} as const satisfies Partial<DraftState>;

/** The fields the Audio tab keeps in the draft. The kind and the model of each kind are kept apart. */
export const AUDIO_FIELD_DEFAULTS = { ...AUDIO_TEXT_DEFAULTS, ...AUDIO_SETTING_DEFAULTS } as const satisfies Partial<DraftState>;

/** The settings that belong to one kind. A Reuse of a kind puts back its own and leaves the others' as the form has them. */
export const AUDIO_KIND_SETTINGS: Record<AudioWorkflow, readonly (keyof DraftState)[]> = {
  music: ['audioTempo', 'audioKey', 'musicSeconds'],
  sound: ['audioSeconds', 'soundType'],
  speech: ['speechDelivery'],
};

/** The draft field the text box of each kind writes to. */
export const AUDIO_TEXT_KEYS = { music: 'audioStyle', speech: 'speechText', sound: 'soundPrompt' } as const satisfies Record<AudioWorkflow, keyof DraftState>;

/** The control id (of the workflow contract) of each kind's text box. */
const TEXT_CONTROLS: Record<AudioWorkflow, string> = { music: 'music_style', speech: 'speech_text', sound: 'sound_prompt' };

/** The audio models a context offers for a kind, in config order. */
export function audioModelsOf(ctx: WorkspaceContext, kind: AudioWorkflow): AudioModelOption[] {
  return (ctx.audio_models ?? []).filter((model) => model.kind === kind);
}

/** The kinds a context can run: those the workflow contract lists, in the Kind control's order. */
export function availableKinds(ctx: WorkspaceContext): AudioWorkflow[] {
  return AUDIO_KINDS.filter((kind) => ctx.workflow_contract.values.includes(kind));
}

/** The row of the audio model with this id as a *kind* model, or null: a model that makes several kinds has one row per kind, and what it takes differs between them. */
export function audioModelById(ctx: WorkspaceContext | null, id: string, kind: AudioWorkflow): AudioModelOption | null {
  return ctx?.audio_models?.find((model) => model.id === id && model.kind === kind) ?? null;
}

/** Whether *id* is an audio model of any kind (it has a row for at least one). */
export function isAudioModelId(ctx: WorkspaceContext | null, id: string): boolean {
  return ctx?.audio_models?.some((model) => model.id === id) ?? false;
}

/** The lengths a row lets the person choose: a sound row, and a music row of a model that sets its length (it lists a shortest and a default length). Null for any other row. */
function lengthLimits(model: AudioModelOption | null): { min: number; max: number; fallback: number } | null {
  if (model?.kind === 'sound') return { min: model.limits.min_seconds, max: model.limits.max_seconds, fallback: model.limits.default_seconds };
  if (model?.kind === 'music' && model.limits.min_seconds !== undefined) {
    return { min: model.limits.min_seconds, max: model.limits.max_seconds, fallback: model.limits.default_seconds ?? model.limits.min_seconds };
  }
  return null;
}

/** The range a length can take on a row, from its shortest to its longest; null for a row that does not set a length. */
export function secondsRange(model: AudioModelOption | null): { min: number; max: number } | null {
  const limits = lengthLimits(model);
  return limits === null ? null : { min: limits.min, max: limits.max };
}

/** Move a length into the range of a row that sets one (a length that is not a number becomes the row's default); another row's length is left as it is. */
export function clampSeconds(seconds: number, model: AudioModelOption | null): number {
  const limits = lengthLimits(model);
  if (limits === null) return seconds;
  return Math.min(limits.max, Math.max(limits.min, Number.isFinite(seconds) ? seconds : limits.fallback));
}

/** The length a row's take has when its length is left at the model's default: a song's 0 is its row's default length (null for a row that sets no length). */
export function musicSecondsShown(seconds: number, model: AudioModelOption | null): number | null {
  const limits = model?.kind === 'music' ? lengthLimits(model) : null;
  return limits === null ? null : seconds > 0 ? seconds : limits.fallback;
}

/** A song's length as the draft keeps it: the length moved into the row's range, and 0 (the model's default) when it is the default, so a length nobody changed is not shown as changed. */
export function settleMusicSeconds(seconds: number, model: AudioModelOption | null): number {
  const limits = model?.kind === 'music' ? lengthLimits(model) : null;
  if (limits === null) return seconds;
  const clamped = clampSeconds(seconds, model);
  return clamped === limits.fallback ? 0 : clamped;
}

/** The built-in voices of a speech model; none for any other model. */
export function voicesOf(model: AudioModelOption | null): AudioVoice[] {
  return model?.kind === 'speech' ? model.voices : [];
}

/**
 * The voice id the draft keeps for a built-in voice of *model*: null for the model's default voice (so a take nobody changed is not
 * shown as changed) and for one the model does not have; otherwise the id.
 */
export function settledSpeaker(speaker: string | null, model: AudioModelOption | null): string | null {
  if (speaker === null || model?.kind !== 'speech' || speaker === model.default_voice) return null;
  return model.voices.some((voice) => voice.id === speaker) ? speaker : null;
}

/** A length moved into the range of a row that sets one; 0 stays Auto, and a row without a length keeps what the draft holds (unsent). */
function settledLength(seconds: number, model: AudioModelOption | null): number {
  return seconds > 0 ? clampSeconds(seconds, model) : seconds;
}

/**
 * Return the state settled on a kind and one of its models: the kind falls to the first one the machine runs when it runs
 * none of the state's, the model is the preferred one, else the state's, else the one last chosen for the kind, else the
 * kind's first, and the length of the kind (a sound's, or a song's on a model that sets it) moves into the model's range (a
 * length left on Auto stays 0).
 */
export function hydrateAudio(ctx: WorkspaceContext, state: DraftState, preferredModel: string | null = null): DraftState {
  const kinds = availableKinds(ctx);
  const open = isAudioWorkflow(state.workflow) ? state.workflow : state.audioKind;
  const kind = kinds.includes(open) ? open : (kinds[0] ?? open);
  const models = audioModelsOf(ctx, kind);
  const candidates = preferredModel !== null ? [preferredModel] : [state.model, state.audioModels[kind]];
  const model = candidates.find((id) => id !== '' && models.some((option) => option.id === id)) ?? models[0]?.id ?? '';
  const chosen = models.find((option) => option.id === model) ?? null;
  return {
    ...state,
    workflow: kind,
    audioKind: kind,
    model,
    audioModels: { ...state.audioModels, [kind]: model },
    audioSeconds: kind === 'sound' ? settledLength(state.audioSeconds, chosen) : state.audioSeconds,
    musicSeconds: kind === 'music' ? settledLength(state.musicSeconds, chosen) : state.musicSeconds,
    speechSpeaker: kind === 'speech' ? settledSpeaker(state.speechSpeaker, chosen) : state.speechSpeaker,
  };
}

/**
 * Whether Workspace prefill params replace the Speech voice: a Use as → Voice naming a recording, a Reuse of a take (it names
 * the prompt, and a take without a recording is in a built-in voice) or any params that name a saved voice, a recording or a built-in voice.
 */
export function replacesVoice(params: Record<string, string | undefined>, kind: AudioWorkflow): boolean {
  if (kind !== 'speech') return false;
  if (params.use_as === 'voice') return Boolean(params.voice_path);
  return params.prompt !== undefined || params.saved_voice !== undefined || params.voice_path !== undefined || params.speaker !== undefined;
}

/** The fields of the draft a Reuse of Enhanced text builds on: the box's earlier versions and the Enhance options. */
type EnhanceDraft = Pick<DraftState, 'audioRewrites' | 'enhanceSettingsAudio'>;

/** What a prefill's audio params change in the draft, and what they say about the voice's consent. */
interface AudioPrefill {
  patch: WorkspacePrefill;
  /** The voice was replaced: the tick is cleared, and the voice needs none when it is a take the model chose the voice of. */
  voice: { exempt: boolean } | null;
}

/** Read a number of seconds or beats: the number a param names, or null for a blank or broken one. */
function positiveNumber(value: string): number | null {
  const parsed = value.trim() === '' ? Number.NaN : Number(value);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : null;
}

/**
 * Read the settings of a kind from Workspace prefill params: a sound's length and type, a song's tempo, key and length (a blank
 * one is Auto, or 0 for a length), and the delivery of a built-in voice. A take that followed a recording had no delivery, so its
 * params set none. A param the kind does not show, or that is not named, sets nothing.
 */
function settingsPatch(params: Record<string, string | undefined>, kind: AudioWorkflow, shows: (control: string) => boolean): WorkspacePrefill {
  const patch: WorkspacePrefill = {};
  if (kind !== 'speech' && params.seconds !== undefined && shows('length')) {
    patch[kind === 'music' ? 'musicSeconds' : 'audioSeconds'] = positiveNumber(params.seconds) ?? 0;
  }
  if (kind === 'music') {
    if (params.tempo !== undefined && shows('tempo')) {
      const tempo = positiveNumber(params.tempo);
      patch.audioTempo = tempo === null ? null : Math.round(tempo);
    }
    if (params.key !== undefined && shows('key')) patch.audioKey = params.key.trim() || null;
  }
  if (kind === 'sound' && params.sound_type !== undefined && shows('sound_type')) {
    const type = SOUND_TYPES.find((id) => id === params.sound_type);
    if (type !== undefined) patch.soundType = type;
  }
  if (kind === 'speech' && !params.voice_path && !params.saved_voice && shows('delivery') && DELIVERY_FIELDS.some((field) => params[field] !== undefined)) {
    patch.speechDelivery = { ...DEFAULT_SPEECH_DELIVERY, ...Object.fromEntries(DELIVERY_FIELDS.flatMap((field) => (params[field] ? [[field, params[field]]] : []))) };
  }
  return patch;
}

/**
 * Read the Enhanced text of a Reuse (`enhanced_prompt`, made from the `prompt` param, with the record in `enhance`): it goes in the Enhanced tab of the
 * kind's box with its options and enhancer, and is not out of date. A Reuse that names none empties the box's Enhanced text (a Scratch), so a rewrite left
 * from another take does not run. The own text's rewrites stay, as they are the box's earlier versions.
 */
function enhancedPatch(params: Record<string, string | undefined>, kind: AudioWorkflow, current: EnhanceDraft): WorkspacePrefill {
  const box = AUDIO_BOX_OF_KIND[kind];
  const text = params.enhanced_prompt ?? '';
  if (text.trim() === '') return { audioRewrites: { ...current.audioRewrites, [box]: scratchRewrite(current.audioRewrites[box]) } };
  const record = parseAudioRecord(params.enhance);
  const task = record?.task ?? rewriteTaskOf(kind);
  const from = { input: params.prompt ?? '', model: params.model ?? '', task, ...(record === null ? {} : { record }) };
  const patch: WorkspacePrefill = { audioRewrites: { ...current.audioRewrites, [box]: pushRewrite(current.audioRewrites[box], { text, from }) } };
  if (record === null) return patch;
  if (record.task === 'sound' || record.task === 'style') patch.enhanceSettingsAudio = { ...current.enhanceSettingsAudio, [record.task]: record.settings as Partial<AudioEnhanceSettings> };
  if (record.tier === 'fast' || record.tier === 'smart') patch.enhanceModel = record.tier;
  return patch;
}

/**
 * Read the audio fields of Workspace prefill params (a Reuse, a Use as, a link, a job's settings) for the kind they open.
 *
 * A Use as → Voice (`use_as=voice`) changes the voice alone: the recording and its transcript, and it needs no consent when
 * the take was made in the model's own voice (no `voice_clone`). Any other prefill sets the fields it names, and a Reuse
 * (it names the prompt) sets the whole kind: a song without lyrics is an instrumental, speech without a recording is in the
 * built-in voice it names (the model's default when it names none). A param whose control the kind does not show is skipped, unless *ungated*. The consent is never a param.
 */
export function audioPrefill(params: Record<string, string | undefined>, ctx: WorkspaceContext | null, kind: AudioWorkflow, ungated = false, current: EnhanceDraft = AUDIO_FIELD_DEFAULTS): AudioPrefill {
  const visible = ctx ? new Set(ctx.workflow_contract.definitions[kind]?.visible_controls ?? []) : null;
  const shows = (control: string): boolean => ungated || visible === null || visible.has(control);
  const patch: WorkspacePrefill = {};
  let voice: AudioPrefill['voice'] = null;

  const setVoice = (): void => {
    const saved = params.saved_voice;
    const path = saved ? undefined : params.voice_path;
    if (saved && shows('voice')) Object.assign(patch, { voiceMode: 'saved', savedVoice: saved, voicePath: null, voiceUpload: null });
    else if (path && shows('voice_path')) Object.assign(patch, { voiceMode: 'recording', savedVoice: null, voicePath: path, voiceUpload: null });
    else if (shows('voice')) Object.assign(patch, { voiceMode: 'builtin', savedVoice: null, speechSpeaker: params.speaker ?? null, voicePath: null, voiceUpload: null });
    if (params.voice_text !== undefined && shows('voice_text') && !saved) patch.voiceText = params.voice_text;
    else if (!path) patch.voiceText = '';
    voice = { exempt: params.voice_clone !== 'true' && params.use_as === 'voice' };
  };

  if (params.use_as === 'voice') {
    if (replacesVoice(params, kind)) setVoice();
    return { patch, voice };
  }
  // The kind's model, when this machine runs it; the draft settles on it with `hydrateAudio`.
  if (params.model && (ctx === null || audioModelsOf(ctx, kind).some((model) => model.id === params.model))) patch.model = params.model;
  const textKey = AUDIO_TEXT_KEYS[kind];
  const reuses = params.prompt !== undefined;
  if (reuses && shows(TEXT_CONTROLS[kind])) {
    patch[textKey] = params.prompt;
    Object.assign(patch, enhancedPatch(params, kind, current));
  }
  if (kind === 'music' && shows('lyrics') && (params.lyrics !== undefined || reuses)) patch.lyrics = params.lyrics ?? '';
  Object.assign(patch, settingsPatch(params, kind, shows));
  if (replacesVoice(params, kind)) setVoice();
  if (params.seed && shows('seed')) patch.seed = Number(params.seed);
  if (params.runs && shows('runs')) patch.runs = Math.max(1, Number(params.runs) || 1);
  return { patch, voice };
}
