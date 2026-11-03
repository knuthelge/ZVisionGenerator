import { formatTimestamp } from '$lib/state/audioTrack';
import type { AudioTrackView } from './audioTrackView';
import type { WorkspaceCapabilities } from './capabilities';
import type { ImagesSummary } from './imagesSummary';
import type { IconName } from '$lib/components/atoms/Icon.svelte';
import type { AudioWorkflow, DraftState, VideoKind, Workflow, WorkspaceWorkflow } from '$lib/types';
import { AUDIO_KIND_INFO, VIDEO_KIND_INFO } from './kindInfo';
import type { VoiceView } from './audioView';

/** The settings a phone shows as chips, each opening one control. */
export type ChipKey =
  | 'preset' | 'seconds' | 'extendSeconds' | 'steps' | 'guidance' | 'seed' | 'runs'
  | 'source' | 'images' | 'track' | 'size' | 'audio' | 'loras' | 'upscale' | 'transparent' | 'model'
  | 'kind' | 'voice' | 'length';

export interface SettingChip {
  key: ChipKey;
  /** Short name before the value; empty when the value speaks for itself (the size, the model). */
  label: string;
  value: string;
  /** Differs from the model's default. */
  changed: boolean;
  /** The value is "on" (a locked seed, LoRAs present, upscale on, transparent on): shown in the accent colour. */
  on?: boolean;
  /** Something to look at: a clip that may not fit, a model not downloaded. */
  warn?: boolean;
  /** A glyph before the label and value (the Audio tab's Kind chip). */
  icon?: IconName;
  /** The preset chip's applied preset slug, empty when none is applied. */
  applied?: string;
  /** The LoRA chip's files: one is downloading (how far, or null while its size is unknown) or one failed. */
  download?: { kind: 'downloading'; percent: number | null } | { kind: 'failed' };
}

export interface ChipFacts {
  caps: WorkspaceCapabilities;
  state: DraftState;
  /** Whether a chip's settings differ from their defaults (`SETTING_ROWS` decides which settings are the chip's). */
  changed: (chip: ChipKey) => boolean;
  /** The dimensions the run submits (a preset's, or the custom ones). */
  width: number;
  height: number;
  /** The clip length in seconds, when the workflow has one. */
  seconds: number | null;
  /** Extend: the seconds added; null until a source is known. */
  extendSeconds: number | null;
  /** The LoRA names in use. */
  loraNames: string[];
  /** What the LoRA files are doing, when a download is on its way or failed. */
  loraDownload?: SettingChip['download'];
  modelLabel: string;
  modelDownloaded: boolean;
  /** Extend: the source clip's name, or null. */
  sourceName: string | null;
  /** The Video tab's open kind when the machine offers more than one (New, Extend); null where there is nothing to choose. */
  videoKind: VideoKind | null;
  /** The Images section's summary; null where the section is not offered (a model that can use no image, the Extend kind). */
  images: ImagesSummary | null;
  /** The audio track view; null where the section is not offered (Image, a model that follows no track). */
  track: Pick<AudioTrackView, 'set' | 'start' | 'gone'> | null;
  /** The clip or extension may not fit memory. */
  memoryWarn: boolean;
  /** The applied settings preset's name, or null when none is applied. */
  presetName: string | null;
  /** The applied settings preset's slug, or null. */
  presetSlug: string | null;
  /** The form differs from the applied preset. */
  presetChanged: boolean;
  /** The chip whose popover is open; Transparent stays while its popover is, so switching it off there does not pull the chip away. */
  openKey?: ChipKey | null;
}

const fmt = (value: number): string => (Number.isInteger(value) ? String(value) : String(Number(value.toFixed(2))));

/**
 * Chips in the order their settings change in the loop: every run (length, steps, guidance, seed, batch),
 * per series (images or source, size, audio, LoRAs), at the end (upscale, then Transparent, which only shows while on), at the start (model). The preset, which
 * sets many of them at once, comes before all.
 */
export function settingsChips(f: ChipFacts): SettingChip[] {
  const { caps, state: s } = f;
  const chips: (SettingChip | null)[] = [
    f.videoKind ? { key: 'kind', label: '', value: VIDEO_KIND_INFO[f.videoKind].label, changed: false, icon: VIDEO_KIND_INFO[f.videoKind].icon } : null,
    { key: 'preset', label: 'Preset', value: f.presetName ?? 'none', changed: f.presetChanged, on: f.presetName !== null, applied: f.presetSlug ?? '' },
    caps.showFrameCount && f.seconds !== null ? { key: 'seconds', label: 'Length', value: `${fmt(f.seconds)} s`, changed: f.changed('seconds'), warn: f.memoryWarn } : null,
    // The Kind chip already says Extend, so the length chip reads Length in both kinds.
    caps.showExtendLength ? { key: 'extendSeconds', label: 'Length', value: f.extendSeconds === null ? '—' : `+${fmt(f.extendSeconds)} s`, changed: f.changed('extendSeconds'), warn: f.memoryWarn } : null,
    caps.showSteps ? { key: 'steps', label: 'Steps', value: fmt(s.steps), changed: f.changed('steps') } : null,
    caps.showGuidance ? { key: 'guidance', label: 'Guidance', value: fmt(s.guidance), changed: f.changed('guidance') } : null,
    caps.showSeed ? { key: 'seed', label: 'Seed', value: s.seed === null ? 'random' : String(s.seed), changed: f.changed('seed'), on: s.seed !== null } : null,
    caps.showRuns ? { key: 'runs', label: 'Batch', value: `×${s.runs}`, changed: f.changed('runs') } : null,
    // One chip for every image the tab takes; its value is the section's summary, and its dot shows a Strength off its default.
    f.images ? { key: 'images', label: 'Images', value: f.images.text, changed: f.changed('images'), on: f.images.tone === 'active', warn: f.images.tone === 'error' } : null,
    caps.showSourceVideo ? { key: 'source', label: 'Source', value: f.sourceName ?? 'none', changed: false, on: f.sourceName !== null, warn: s.missingSourceVideo } : null,
    // One chip for the audio track; its value is where in the track the clip starts when that isn't the beginning.
    f.track ? { key: 'track', label: 'Track', value: !f.track.set ? 'none' : f.track.start > 0 ? formatTimestamp(f.track.start) : 'on', changed: false, on: f.track.set, warn: f.track.gone } : null,
    caps.showDimensions ? { key: 'size', label: '', value: `${f.width}×${f.height}`, changed: f.changed('size') } : null,
    caps.showAudio || caps.showLowMemory || caps.showVideoUpscale
      ? { key: 'audio', label: 'Audio', value: caps.showAudio ? (s.audio ? 'on' : 'off') : s.lowMemory ? 'low mem' : 'off', changed: f.changed('audio'), on: caps.showAudio ? s.audio : s.lowMemory }
      : null,
    { key: 'loras', label: 'LoRA', value: f.loraNames.length ? f.loraNames.join(', ') : 'none', changed: false, on: f.loraNames.length > 0, download: f.loraDownload },
    caps.showImageUpscale ? { key: 'upscale', label: 'Upscale', value: s.upscaleEnabled ? `${s.upscaleFactor}×` : 'off', changed: f.changed('upscale'), on: s.upscaleEnabled } : null,
    caps.showPostprocessTransparent && (s.postprocessTransparent || f.openKey === 'transparent')
      ? { key: 'transparent', label: 'Transparent', value: s.postprocessTransparent ? 'on' : 'off', changed: f.changed('transparent'), on: s.postprocessTransparent }
      : null,
    { key: 'model', label: '', value: f.modelLabel, changed: false, warn: !f.modelDownloaded },
  ];
  return chips.filter((chip): chip is SettingChip => chip !== null);
}

/** What the Audio tab's chips are made from. */
export interface AudioChipFacts {
  kind: AudioWorkflow;
  state: DraftState;
  modelLabel: string;
  modelDownloaded: boolean;
  /** The Speech voice, when the model can follow a recording; null otherwise. */
  voice: Pick<VoiceView, 'summary' | 'recording' | 'chip'> | null;
  /** Whether a chip's settings differ from their defaults (`SETTING_ROWS` decides which settings are the chip's). */
  changed: (chip: ChipKey) => boolean;
  /** The model takes a length (Sound). */
  showLength: boolean;
  showSeed: boolean;
  showRuns: boolean;
}

/**
 * The chips of the Audio tab, in the order its settings change in the loop: the Kind and the Model first, then the Voice (Speech)
 * or the Length (Sound), the Seed and the Batch. Presets, LoRAs and size have none here.
 */
export function audioChips(f: AudioChipFacts): SettingChip[] {
  const { state: s } = f;
  const chips: (SettingChip | null)[] = [
    { key: 'kind', label: '', value: AUDIO_KIND_INFO[f.kind].label, changed: false, icon: AUDIO_KIND_INFO[f.kind].icon },
    { key: 'model', label: '', value: f.modelLabel, changed: false, warn: !f.modelDownloaded },
    f.voice ? { key: 'voice', label: 'Voice', value: f.voice.chip, changed: false, on: f.voice.recording && f.voice.summary.tone === 'active', warn: f.voice.summary.tone === 'error' } : null,
    f.showLength ? { key: 'length', label: 'Length', value: `${fmt(s.audioSeconds)} s`, changed: f.changed('length') } : null,
    f.showSeed ? { key: 'seed', label: 'Seed', value: s.seed === null ? 'random' : String(s.seed), changed: f.changed('seed'), on: s.seed !== null } : null,
    f.showRuns ? { key: 'runs', label: 'Batch', value: `×${s.runs}`, changed: f.changed('runs') } : null,
  ];
  return chips.filter((chip): chip is SettingChip => chip !== null);
}

/** The chips the strip shows before the context is known, so the frame is complete from the first frame. */
const PENDING_IMAGE_CHIPS: [ChipKey, string][] = [['steps', 'Steps'], ['guidance', 'Guidance'], ['seed', 'Seed'], ['runs', 'Batch'], ['images', 'Images'], ['size', ''], ['loras', 'LoRA'], ['upscale', 'Upscale'], ['model', '']];
const PENDING_VIDEO_CHIPS: [ChipKey, string][] = [['seconds', 'Length'], ['steps', 'Steps'], ['guidance', 'Guidance'], ['seed', 'Seed'], ['images', 'Images'], ['track', 'Track'], ['size', ''], ['audio', 'Audio'], ['loras', 'LoRA'], ['model', '']];
const PENDING_AUDIO_CHIPS: Record<AudioWorkflow, [ChipKey, string][]> = {
  music: [['kind', ''], ['model', ''], ['seed', 'Seed'], ['runs', 'Batch']],
  speech: [['kind', ''], ['model', ''], ['voice', 'Voice'], ['seed', 'Seed'], ['runs', 'Batch']],
  sound: [['kind', ''], ['model', ''], ['length', 'Length'], ['seed', 'Seed'], ['runs', 'Batch']],
};
const PENDING_CHIPS: Record<Workflow, [ChipKey, string][]> = {
  txt2img: PENDING_IMAGE_CHIPS,
  img2img: PENDING_IMAGE_CHIPS,
  txt2vid: PENDING_VIDEO_CHIPS,
  img2vid: PENDING_VIDEO_CHIPS,
  extend: [['extendSeconds', 'Length'], ['steps', 'Steps'], ['guidance', 'Guidance'], ['seed', 'Seed'], ['source', 'Source'], ['track', 'Track'], ['audio', 'Audio'], ['loras', 'LoRA'], ['model', '']],
};

/**
 * Placeholder chips for a workflow while the context loads: the usual chips in their usual order with no value, so
 * the strip has its height and width before the request answers and nothing under the stage moves when it does.
 */
export function pendingChips(workflow: WorkspaceWorkflow): SettingChip[] {
  if (workflow === 'music' || workflow === 'speech' || workflow === 'sound') {
    return PENDING_AUDIO_CHIPS[workflow].map(([key, label]) => ({ key, label, value: '—', changed: false }));
  }
  const kind: [ChipKey, string][] = workflow === 'txt2vid' || workflow === 'img2vid' || workflow === 'extend' ? [['kind', '']] : [];
  const keys: [ChipKey, string][] = [...kind, ['preset', 'Preset'], ...(PENDING_CHIPS[workflow] ?? PENDING_CHIPS.txt2img)];
  return keys.map(([key, label]) => ({ key, label, value: '—', changed: false }));
}
