import type { WorkspaceCapabilities } from './capabilities';
import type { ChipKey } from './settingsChips';
import { sameDelivery } from '$lib/state/speechDelivery';
import type { DraftState } from '$lib/types';

/** A setting the Settings pane shows as one row: the draft keys it edits, when it shows, and the phone chip it belongs to. */
interface SettingRow {
  id: string;
  /** The draft keys the row compares with their defaults and puts back. */
  keys: readonly (keyof DraftState)[];
  /** The phone chip whose popover edits this row; none for a row only the Settings pane has. */
  chip?: ChipKey;
  visible: (caps: WorkspaceCapabilities, state: DraftState) => boolean;
}

/**
 * Every row of the Settings pane. The Settings pane's changed dots and Reset badge, the phone's chips and its "All" count, and the
 * popovers' Reset all read this one table, so they agree on what has changed and on what a reset puts back.
 */
export const SETTING_ROWS = [
  { id: 'size', keys: ['ratio', 'size', 'dimensionMode'], chip: 'size', visible: (caps) => caps.showDimensions },
  { id: 'runs', keys: ['runs'], chip: 'runs', visible: (caps) => caps.showRuns },
  { id: 'duration', keys: ['frameCount'], chip: 'seconds', visible: (caps) => caps.showFrameCount },
  { id: 'soundType', keys: ['soundType'], chip: 'soundType', visible: (caps) => caps.showSoundType },
  { id: 'length', keys: ['audioSeconds'], chip: 'length', visible: (caps) => caps.showLength && caps.audioKind !== 'music' },
  { id: 'tempo', keys: ['audioTempo'], chip: 'tempo', visible: (caps) => caps.showTempo },
  { id: 'key', keys: ['audioKey'], chip: 'key', visible: (caps) => caps.showKey },
  { id: 'musicLength', keys: ['musicSeconds'], chip: 'length', visible: (caps) => caps.showLength && caps.audioKind === 'music' },
  { id: 'delivery', keys: ['speechDelivery'], chip: 'delivery', visible: (caps) => caps.showDelivery },
  { id: 'voice', keys: ['voiceMode', 'speechSpeaker', 'voicePath', 'voiceUpload', 'voiceText'], chip: 'voice', visible: (caps) => caps.showVoice },
  { id: 'extendLength', keys: ['extendFrames'], chip: 'extendSeconds', visible: (caps) => caps.showExtendLength },
  { id: 'steps', keys: ['steps'], chip: 'steps', visible: (caps) => caps.showSteps },
  { id: 'guidance', keys: ['guidance'], chip: 'guidance', visible: (caps) => caps.showGuidance },
  { id: 'strength', keys: ['referenceImageStrength'], chip: 'images', visible: (caps) => caps.showI2IStrength },
  { id: 'seed', keys: ['seed'], chip: 'seed', visible: (caps) => caps.showSeed },
  { id: 'scheduler', keys: ['scheduler'], visible: (caps) => caps.showScheduler },
  { id: 'firstSigma', keys: ['firstSigma'], visible: (caps) => caps.supportsFirstSigma },
  { id: 'sharpen', keys: ['postprocessSharpenEnabled', 'postprocessSharpenAmount'], visible: (caps) => caps.showPostprocessSharpen },
  { id: 'contrast', keys: ['postprocessContrastEnabled', 'postprocessContrastAmount'], visible: (caps) => caps.showPostprocessContrast },
  { id: 'saturation', keys: ['postprocessSaturationEnabled', 'postprocessSaturationAmount'], visible: (caps) => caps.showPostprocessSaturation },
  { id: 'transparent', keys: ['postprocessTransparent'], chip: 'transparent', visible: (caps) => caps.showPostprocessTransparent },
  { id: 'upscale', keys: ['upscaleEnabled', 'upscaleFactor'], chip: 'upscale', visible: (caps) => caps.showImageUpscale },
  { id: 'upscaleReview', keys: ['upscaleReview'], chip: 'upscale', visible: (caps, s) => caps.showUpscaleReview && s.upscaleEnabled },
  { id: 'upscaleDenoise', keys: ['upscaleDenoise'], visible: (caps, s) => caps.showUpscaleDenoise && s.upscaleEnabled },
  { id: 'upscaleSteps', keys: ['upscaleSteps'], visible: (caps, s) => caps.showUpscaleSteps && s.upscaleEnabled },
  { id: 'upscaleGuidance', keys: ['upscaleGuidance'], visible: (caps, s) => caps.showUpscaleGuidance && s.upscaleEnabled },
  { id: 'upscaleSharpen', keys: ['upscaleSharpen'], visible: (caps, s) => caps.showUpscaleSharpen && s.upscaleEnabled },
  { id: 'audio', keys: ['audio'], chip: 'audio', visible: (caps) => caps.showAudio },
  { id: 'lowMemory', keys: ['lowMemory'], chip: 'audio', visible: (caps) => caps.showLowMemory },
  { id: 'videoUpscale', keys: ['videoUpscaleEnabled', 'videoUpscaleFactor'], chip: 'audio', visible: (caps) => caps.showVideoUpscale },
  { id: 'videoUpscaleReview', keys: ['videoUpscaleReview'], chip: 'audio', visible: (caps, s) => caps.showVideoUpscaleReview && s.videoUpscaleEnabled },
] as const satisfies readonly SettingRow[];

export type RowId = (typeof SETTING_ROWS)[number]['id'];

const _ROWS: readonly SettingRow[] = SETTING_ROWS;

function _row(id: RowId): SettingRow {
  return _ROWS.find((row) => row.id === id) as SettingRow;
}

/** Whether a draft setting equals its default: a value by identity, the Delivery's five values by what they say. */
export function settingChanged(key: keyof DraftState, state: DraftState, defaults: DraftState): boolean {
  return key === 'speechDelivery' ? !sameDelivery(state.speechDelivery, defaults.speechDelivery) : !Object.is(state[key], defaults[key]);
}

function _differs(row: SettingRow, state: DraftState, defaults: DraftState): boolean {
  return row.keys.some((key) => settingChanged(key, state, defaults));
}

/** Whether a row's settings differ from their defaults. */
export function rowChanged(id: RowId, state: DraftState, defaults: DraftState): boolean {
  return _differs(_row(id), state, defaults);
}

/** The row's settings at their defaults, to put back. */
export function rowDefaults(id: RowId, defaults: DraftState): Partial<DraftState> {
  return Object.fromEntries(_row(id).keys.map((key) => [key, defaults[key]]));
}

/** The rows the open tab shows. */
function _visible(caps: WorkspaceCapabilities, state: DraftState): SettingRow[] {
  return _ROWS.filter((row) => row.visible(caps, state));
}

/** How many of the shown rows differ from their defaults: the Reset badge, and the phone's count of changes. */
export function changedRowCount(caps: WorkspaceCapabilities, state: DraftState, defaults: DraftState): number {
  return _visible(caps, state).filter((row) => _differs(row, state, defaults)).length;
}

/** The draft keys of a phone chip's shown rows: what its popover's Reset puts back. */
export function chipKeys(chip: ChipKey, caps: WorkspaceCapabilities, state: DraftState): (keyof DraftState)[] {
  return _visible(caps, state).filter((row) => row.chip === chip).flatMap((row) => row.keys);
}

/** Whether a phone chip's settings differ from their defaults. */
export function chipChanged(chip: ChipKey, caps: WorkspaceCapabilities, state: DraftState, defaults: DraftState): boolean {
  return _visible(caps, state).some((row) => row.chip === chip && _differs(row, state, defaults));
}
