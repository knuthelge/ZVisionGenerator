import { isAudioWorkflow } from './workspaceTab';
import type { AudioModelOption, AudioWorkflow, DraftState, WorkspaceContext, WorkspacePrefill } from '$lib/types';

/** The kinds of the Audio tab, in the order the Kind control lists them. */
const AUDIO_KINDS: readonly AudioWorkflow[] = ['music', 'speech', 'sound'];

/** The length of a sound when none is chosen, and the shortest one a model makes (the server's `DEFAULT_SOUND_SECONDS` and `MIN_SOUND_SECONDS`). */
const DEFAULT_SOUND_SECONDS = 10;
const MIN_SOUND_SECONDS = 1;

/** The text fields of the Audio tab: the Music style and lyrics, the Speech text and the Sound prompt. */
const AUDIO_TEXT_DEFAULTS = { audioStyle: '', lyrics: '', speechText: '', soundPrompt: '' } as const satisfies Partial<DraftState>;

/** The other fields the Audio tab keeps in the draft: the length and the voice. */
const AUDIO_SETTING_DEFAULTS = {
  audioSeconds: DEFAULT_SOUND_SECONDS,
  voiceMode: 'any',
  voicePath: null,
  voiceUpload: null,
  voiceText: '',
} as const satisfies Partial<DraftState>;

/** The fields the Audio tab keeps in the draft. The kind and the model of each kind are kept apart. */
export const AUDIO_FIELD_DEFAULTS = { ...AUDIO_TEXT_DEFAULTS, ...AUDIO_SETTING_DEFAULTS } as const satisfies Partial<DraftState>;

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

/** The audio model with this id, or null. */
export function audioModelById(ctx: WorkspaceContext | null, id: string): AudioModelOption | null {
  return ctx?.audio_models?.find((model) => model.id === id) ?? null;
}

/** The range a sound's length can take on a model: from its shortest to its longest. */
export function secondsRange(model: AudioModelOption | null): { min: number; max: number } {
  return { min: model?.limits.min_seconds ?? MIN_SOUND_SECONDS, max: model?.limits.max_seconds ?? 600 };
}

/** Move a length into a model's range. */
export function clampSeconds(seconds: number, model: AudioModelOption | null): number {
  const { min, max } = secondsRange(model);
  return Math.min(max, Math.max(min, Number.isFinite(seconds) ? seconds : DEFAULT_SOUND_SECONDS));
}

/**
 * Return the state settled on a kind and one of its models: the kind falls to the first one the machine runs when it runs
 * none of the state's, the model is the preferred one, else the state's, else the one last chosen for the kind, else the
 * kind's first, and a sound's length moves into the model's range.
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
    audioSeconds: chosen?.capabilities.duration ? clampSeconds(state.audioSeconds, chosen) : state.audioSeconds,
  };
}

/**
 * Whether Workspace prefill params replace the Speech voice: a Use as → Voice naming a recording, a Reuse of a take (it names
 * the prompt, and a take without a recording is in *Any* voice) or any params that name a recording.
 */
export function replacesVoice(params: Record<string, string | undefined>, kind: AudioWorkflow): boolean {
  if (kind !== 'speech') return false;
  if (params.use_as === 'voice') return Boolean(params.voice_path);
  return params.prompt !== undefined || params.voice_path !== undefined;
}

/** What a prefill's audio params change in the draft, and what they say about the voice's consent. */
interface AudioPrefill {
  patch: WorkspacePrefill;
  /** The voice was replaced: the tick is cleared, and the voice needs none when it is a take the model chose the voice of. */
  voice: { exempt: boolean } | null;
}

/**
 * Read the audio fields of Workspace prefill params (a Reuse, a Use as, a link, a job's settings) for the kind they open.
 *
 * A Use as → Voice (`use_as=voice`) changes the voice alone: the recording and its transcript, and it needs no consent when
 * the take was made in the model's own voice (no `voice_clone`). Any other prefill sets the fields it names, and a Reuse
 * (it names the prompt) sets the whole kind: a song without lyrics is an instrumental, speech without a recording is in
 * *Any* voice. A param whose control the kind does not show is skipped, unless *ungated*. The consent is never a param.
 */
export function audioPrefill(params: Record<string, string | undefined>, ctx: WorkspaceContext | null, kind: AudioWorkflow, ungated = false): AudioPrefill {
  const visible = ctx ? new Set(ctx.workflow_contract.definitions[kind]?.visible_controls ?? []) : null;
  const shows = (control: string): boolean => ungated || visible === null || visible.has(control);
  const patch: WorkspacePrefill = {};
  let voice: AudioPrefill['voice'] = null;

  const setVoice = (): void => {
    const path = params.voice_path;
    if (path && shows('voice_path')) Object.assign(patch, { voiceMode: 'recording', voicePath: path, voiceUpload: null });
    else if (shows('voice')) Object.assign(patch, { voiceMode: 'any', voicePath: null, voiceUpload: null });
    if (params.voice_text !== undefined && shows('voice_text')) patch.voiceText = params.voice_text;
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
  if (reuses && shows(TEXT_CONTROLS[kind])) patch[textKey] = params.prompt;
  if (kind === 'music' && shows('lyrics') && (params.lyrics !== undefined || reuses)) patch.lyrics = params.lyrics ?? '';
  if (kind === 'sound' && params.seconds !== undefined && shows('length')) {
    const seconds = Number(params.seconds);
    if (Number.isFinite(seconds) && seconds > 0) patch.audioSeconds = seconds;
  }
  if (replacesVoice(params, kind)) setVoice();
  if (params.seed && shows('seed')) patch.seed = Number(params.seed);
  if (params.runs && shows('runs')) patch.runs = Math.max(1, Number(params.runs) || 1);
  return { patch, voice };
}
