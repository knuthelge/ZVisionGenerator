import audioEnhanceMatrix from '../../../tests/fixtures/audio_enhance_matrix.json';
import audioSettingsContract from '../../../tests/fixtures/audio_settings_contract.json';
import { AUDIO_KINDS, type AudioEnhanceMatrix, type AudioModelOption, type AudioSettingsVocabulary, type AudioVoice, type AudioWorkflow, type GalleryAsset, type ModelLicence, type WorkflowContractEntry, type WorkspaceContext } from '$lib/types';
import { withEnhancer } from './workspace';

/** A take's stored peaks: 100 values, a swell in the middle. */
export const PEAKS: number[] = Array.from({ length: 100 }, (_, index) => 40 + Math.round(180 * Math.sin((index / 99) * Math.PI)));

export const LYRICS = '[verse]\nStreetlights hum a quiet tune\n\n[chorus]\nHold the night, don\'t let it go';

/** A gallery audio take as the server lists it: a song with its lyrics by default. */
export function makeAudioAsset(overrides: Partial<GalleryAsset> = {}): GalleryAsset {
  return {
    id: 'out/song.flac',
    url: '/media/out/song.flac',
    thumbnail_url: '/media/out/song.flac',
    filename: 'song.flac',
    created_at: '2026-11-01T12:00:00Z',
    project: null,
    workflow: 'music',
    prompt: 'Dreamy synth-pop, warm analog pads',
    model: 'yue2',
    seed: 4127,
    media_type: 'audio',
    kind: 'music',
    duration: 120,
    peaks: PEAKS,
    channels: 2,
    sample_rate: 24000,
    file_bytes: 1_000_000,
    has_reference_voice: false,
    lyrics: LYRICS,
    cut: false,
    file_path: '/outputs/out/song.flac',
    has_reusable_config: true,
    reuse_workspace_url: '#/workspace?workflow=music&model=yue2&prompt=Dreamy',
    upscale: null,
    extend: null,
    ...overrides,
  };
}

/** A speech take, made with no reference voice unless *overrides* say so. */
export function makeSpeechAsset(overrides: Partial<GalleryAsset> = {}): GalleryAsset {
  return makeAudioAsset({
    id: 'out/speech.flac',
    url: '/media/out/speech.flac',
    thumbnail_url: '/media/out/speech.flac',
    filename: 'speech.flac',
    workflow: 'speech',
    kind: 'speech',
    prompt: 'Welcome back to the workshop.',
    model: 'qwen3_tts',
    duration: 21,
    lyrics: null,
    file_path: '/outputs/out/speech.flac',
    reuse_workspace_url: '#/workspace?workflow=speech&model=qwen3_tts',
    ...overrides,
  });
}

/** An audio file the app did not make: no kind, no settings, so no Reuse. */
export function makeImportedAudio(overrides: Partial<GalleryAsset> = {}): GalleryAsset {
  return makeAudioAsset({
    id: 'out/imported_song.wav',
    url: '/media/out/imported_song.wav',
    thumbnail_url: '/media/out/imported_song.wav',
    filename: 'imported_song.wav',
    workflow: 'music',
    kind: null,
    prompt: 'imported_song.wav',
    model: 'Unavailable',
    seed: null,
    lyrics: null,
    has_reusable_config: false,
    file_path: '/outputs/out/imported_song.wav',
    reuse_workspace_url: '#/workspace?workflow=music',
    ...overrides,
  });
}

/** The built-in voices of Qwen3-TTS, as the server lists them. */
export const QWEN_VOICES: AudioVoice[] = [
  { id: 'ryan', label: 'Ryan', language: 'English' },
  { id: 'aiden', label: 'Aiden', language: 'English' },
  { id: 'vivian', label: 'Vivian', language: 'Chinese' },
  { id: 'serena', label: 'Serena', language: 'Chinese' },
  { id: 'uncle_fu', label: 'Uncle Fu', language: 'Chinese' },
  { id: 'dylan', label: 'Dylan', language: 'Beijing Chinese' },
  { id: 'eric', label: 'Eric', language: 'Sichuan Chinese' },
  { id: 'ono_anna', label: 'Ono Anna', language: 'Japanese' },
  { id: 'sohee', label: 'Sohee', language: 'Korean' },
];

/** The audio models of a machine that runs all three kinds. */
export const AUDIO_MODELS: AudioModelOption[] = [
  { id: 'yue2', label: 'YuE2 3B', kind: 'music', type: 'audio', hidden_controls: ['length'], limits: { max_seconds: 360 }, instrumental_sections: '[Intro] [Instrumental] [Solo] [Instrumental] [Solo] [Bridge] [Instrumental] [Outro]', licences: [] },
  { id: 'qwen3_tts', label: 'Qwen3-TTS 1.7B', kind: 'speech', type: 'audio', hidden_controls: [], limits: { max_text_chars: 10000, min_voice_seconds: 3, max_voice_seconds: 30 }, voices: QWEN_VOICES, default_voice: 'ryan', voice_download: { downloaded: false }, licences: [] },
  { id: 'sa3', label: 'Stable Audio 3 Medium', kind: 'sound', type: 'audio', hidden_controls: [], limits: { min_seconds: 1, max_seconds: 380, default_seconds: 12 }, licences: [] },
];

/** Stable Audio as a model of two kinds: this is its music row, which sets its length (1 to 380 seconds, 30 by default) and sings no lyrics. Its sound row is in `AUDIO_MODELS`. */
export const SA3_MUSIC: AudioModelOption = { id: 'sa3', label: 'Stable Audio 3 Medium', kind: 'music', type: 'audio', hidden_controls: ['lyrics'], limits: { min_seconds: 1, max_seconds: 380, default_seconds: 30 }, licences: [] };

/** The vocabulary of the audio settings the server serves as `audio_settings`; the file is the served contract, checked by `tests/audio/test_audio_settings.py`. */
export const AUDIO_SETTINGS = audioSettingsContract as AudioSettingsVocabulary;

/** The audio option axes the server serves as `prompt_enhancer.audio_matrix`; the file is the served contract, checked by `tests/prompts/test_audio_enhance_matrix.py`. */
export const AUDIO_MATRIX = audioEnhanceMatrix as AudioEnhanceMatrix;

/** The licence records of the audio models the packaged config ships, as the server lists them. */
export const YUE2_LICENCE: ModelLicence = { name: 'CC BY-NC 4.0 with additional creator permission', supplier: 'M-A-P', summary: 'Non-commercial; individual creators may monetise their outputs, companies need a licence', url: 'https://github.com/multimodal-art-projection/YuE/blob/3acdec8cf1b97603bb696e4c6c4560348ffbb0b7/MODEL_LICENSE', url_label: 'Licence', attribution: null };
export const SA3_LICENCE: ModelLicence = { name: 'Stability AI Community License', supplier: 'Stability AI', summary: 'Free under $1M revenue; commercial use needs a free registration', url: 'https://huggingface.co/stabilityai/stable-audio-3-medium', url_label: 'Model page', attribution: 'Powered by Stability AI' };
export const GEMMA_AUDIO_LICENCE: ModelLicence = { name: 'Gemma Terms of Use', supplier: 'Google', summary: "Free to use, also commercially; Gemma's prohibited use policy applies", url: 'https://huggingface.co/google/gemma-3-12b-it', url_label: 'Model page', attribution: null };

const AUDIO_CONTROLS = ['workflow', 'model', 'runs', 'seed'];
const AUDIO_ENTRY: Omit<WorkflowContractEntry, 'visible_controls'> = { mode: 'audio', model_kind: 'audio', requires_reference_image: false, requires_source_video: false, clear_fields: [] };

/** The workflow contract entries of the three kinds, with the controls the server lists for each. */
export const AUDIO_DEFINITIONS: Record<AudioWorkflow, WorkflowContractEntry> = {
  music: { ...AUDIO_ENTRY, visible_controls: [...AUDIO_CONTROLS, 'music_style', 'lyrics', 'tempo', 'key', 'length'] },
  speech: { ...AUDIO_ENTRY, visible_controls: [...AUDIO_CONTROLS, 'speech_text', 'voice', 'voice_path', 'voice_text', 'voice_consent', 'delivery'] },
  sound: { ...AUDIO_ENTRY, visible_controls: [...AUDIO_CONTROLS, 'sound_prompt', 'sound_type', 'length'] },
};

/**
 * Return *context* for a machine that runs audio models: the models, and the workflow contract listing their kinds after the
 * image and video workflows. Without *models* it runs the three of `AUDIO_MODELS`.
 */
export function withAudio(context: WorkspaceContext, models: AudioModelOption[] = AUDIO_MODELS): WorkspaceContext {
  const kinds = AUDIO_KINDS.filter((kind) => models.some((model) => model.kind === kind));
  return {
    ...context,
    audio_models: models,
    audio_settings: AUDIO_SETTINGS,
    workflow_contract: {
      ...context.workflow_contract,
      values: [...context.workflow_contract.values, ...kinds],
      definitions: { ...context.workflow_contract.definitions, ...Object.fromEntries(kinds.map((kind) => [kind, AUDIO_DEFINITIONS[kind]])) },
    },
  };
}

/** Add the prompt enhancer, with the audio option axes, to *context*. */
export function withAudioEnhancer(context: WorkspaceContext): WorkspaceContext {
  const enhanced = withEnhancer(context);
  enhanced.prompt_enhancer!.audio_matrix = AUDIO_MATRIX;
  return enhanced;
}
