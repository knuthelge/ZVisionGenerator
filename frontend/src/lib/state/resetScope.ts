import type { DraftState, WorkspaceTab } from '$lib/types';

/**
 * Who owns a field of the draft. Every transition that touches part of the draft (a Reset, a Reuse) is a list of these scopes,
 * so a field never sits in a hand-kept list of its own.
 *
 * - `slotsImage`, `slotsVideo`, `slotsExtend`: the images, upload markers and audio track of one tab or kind (Image's also holds Strength).
 * - `settings`: the open mode's (image or video) values for the Settings pane; the Audio tab keeps none of these.
 * - `extendSettings`: the Extend kind's length.
 * - `shared`: Batch and Seed, one pair for every tab.
 * - `model`: the model, its LoRAs and the quantization.
 * - `text`: the open mode's typed text, its Enhanced text and its prompt source.
 * - `promptFile`: the prompts file and the prompts chosen from it.
 * - `enhanceImage`, `enhanceVideo`: the enhancement options of each mode.
 * - `extendSource`: the video an extension continues.
 * - `audioText`, `audioSettings`, `voice`: the four texts of the Audio tab, its length and the Speech voice.
 * - `memory`: what the draft remembers for coming back to a tab or a mode.
 * - `ui`: layout flags and bookkeeping.
 * - `preset`: the settings preset the form started from.
 * - `derived`: values recomputed from other fields, never set by themselves.
 */
export type Scope =
  | 'slotsImage' | 'slotsVideo' | 'slotsExtend' | 'settings' | 'extendSettings' | 'shared' | 'model' | 'text' | 'promptFile'
  | 'enhanceImage' | 'enhanceVideo' | 'extendSource' | 'audioText' | 'audioSettings' | 'voice'
  | 'memory' | 'ui' | 'preset' | 'derived';

/** The scope of every draft field. A field added to `DraftState` without a scope here fails the type check. */
export const FIELD_SCOPE: Record<keyof DraftState, Scope> = {
  workflow: 'derived',
  fps: 'derived',
  missingReferencePaths: 'derived',

  referenceImagePath: 'slotsImage',
  referenceImageUpload: 'slotsImage',
  referenceImagePaths: 'slotsImage',

  firstFramePath: 'slotsVideo',
  firstFrameUpload: 'slotsVideo',
  endImagePath: 'slotsVideo',
  endImageUpload: 'slotsVideo',
  audioTrackPath: 'slotsVideo',
  audioTrackUpload: 'slotsVideo',
  audioTrackStart: 'slotsVideo',

  extendAudioTrackPath: 'slotsExtend',
  extendAudioTrackUpload: 'slotsExtend',
  extendAudioTrackStart: 'slotsExtend',

  ratio: 'settings',
  dimensionMode: 'settings',
  size: 'settings',
  width: 'settings',
  height: 'settings',
  steps: 'settings',
  guidance: 'settings',
  referenceImageStrength: 'slotsImage',
  scheduler: 'settings',
  firstSigma: 'settings',
  frameCount: 'settings',
  audio: 'settings',
  lowMemory: 'settings',
  upscaleEnabled: 'settings',
  upscaleFactor: 'settings',
  upscaleDenoise: 'settings',
  upscaleSteps: 'settings',
  upscaleGuidance: 'settings',
  upscaleSharpen: 'settings',
  upscaleReview: 'settings',
  postprocessSharpenEnabled: 'settings',
  postprocessSharpenAmount: 'settings',
  postprocessContrastEnabled: 'settings',
  postprocessContrastAmount: 'settings',
  postprocessSaturationEnabled: 'settings',
  postprocessSaturationAmount: 'settings',
  postprocessTransparent: 'settings',
  videoUpscaleEnabled: 'settings',
  videoUpscaleFactor: 'settings',
  videoUpscaleReview: 'settings',

  extendFrames: 'extendSettings',

  runs: 'shared',
  seed: 'shared',

  model: 'model',
  loraString: 'model',
  quantize: 'model',

  promptSource: 'text',
  prompt: 'text',
  scene: 'text',
  jsonPromptEnabled: 'text',
  jsonPrompt: 'text',
  negativePrompt: 'text',
  promptFilePath: 'promptFile',
  promptFileOptionIds: 'promptFile',
  promptFileLoaded: 'promptFile',
  enhancedPrompt: 'text',
  enhancedScene: 'text',
  enhancedFrom: 'text',

  enhanceSettings: 'enhanceImage',
  enhanceSettingsVideo: 'enhanceVideo',

  extendSourcePath: 'extendSource',
  missingSourceVideo: 'extendSource',
  extendPrefill: 'extendSource',

  audioStyle: 'audioText',
  lyrics: 'audioText',
  speechText: 'audioText',
  soundPrompt: 'audioText',
  audioSeconds: 'audioSettings',
  voiceMode: 'voice',
  voicePath: 'voice',
  voiceUpload: 'voice',
  voiceText: 'voice',

  enhancedHistory: 'memory',
  lorasByFamily: 'memory',
  lastImageModel: 'memory',
  lastVideoModel: 'memory',
  lastImageQuantize: 'memory',
  savedText: 'memory',
  enhanceAuto: 'memory',
  enhanceModel: 'memory',
  enhanceFormat: 'memory',
  videoKind: 'memory',
  audioKind: 'memory',
  audioModels: 'memory',

  historyCollapsed: 'ui',
  sidebarCollapsed: 'ui',
  lastGeneratedAt: 'ui',
  version: 'ui',

  appliedPreset: 'preset',
};

/** What a Reset puts back on each tab. Adding a tab to `WorkspaceTab` fails the type check until it has a scope here. */
export const RESET_SCOPE: Record<WorkspaceTab, readonly Scope[]> = {
  image: ['slotsImage', 'settings', 'shared', 'enhanceImage', 'preset'],
  video: ['slotsVideo', 'settings', 'shared', 'enhanceVideo', 'preset'],
  extend: ['slotsExtend', 'extendSettings', 'settings', 'shared', 'enhanceVideo', 'preset'],
  audio: ['audioSettings', 'voice', 'shared', 'preset'],
};

/**
 * What applying a take's settings exactly (a Reuse, a Load settings) puts back before it sets the take's own: the tab's Reset scope,
 * its text, the model with its LoRAs and quantization and, on Extend, its source. The prompt file, the rewrites remembered for
 * the text and the choices of enhancer stay. The Audio tab's kinds each keep their own text, length and voice, so a take puts
 * back only Batch and Seed there and sets the fields of its kind.
 */
export const REUSE_SCOPE: Record<WorkspaceTab, readonly Scope[]> = {
  image: [...RESET_SCOPE.image, 'text', 'model'],
  video: [...RESET_SCOPE.video, 'text', 'model'],
  extend: [...RESET_SCOPE.extend, 'text', 'model', 'extendSource'],
  audio: ['shared'],
};

/** The draft keys that belong to any of the scopes. */
export function keysOf(scopes: readonly Scope[]): (keyof DraftState)[] {
  return (Object.keys(FIELD_SCOPE) as (keyof DraftState)[]).filter((key) => scopes.includes(FIELD_SCOPE[key]));
}

/** The fields of `state` that belong to any of the scopes. */
export function pickScopes(state: DraftState, scopes: readonly Scope[]): Partial<DraftState> {
  return Object.fromEntries(keysOf(scopes).map((key) => [key, state[key]]));
}
