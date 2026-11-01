import type { WorkspaceCapabilities } from './capabilities';
import type { ImagesSummary } from './imagesSummary';
import type { DraftState, Workflow } from '$lib/types';

/** The settings a phone shows as chips, each opening one control. */
export type ChipKey =
  | 'preset' | 'seconds' | 'extendSeconds' | 'steps' | 'guidance' | 'seed' | 'runs'
  | 'source' | 'images' | 'size' | 'audio' | 'loras' | 'upscale' | 'transparent' | 'model';

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
  /** The preset chip's applied preset slug, empty when none is applied. */
  applied?: string;
  /** The LoRA chip's files: one is downloading (how far, or null while its size is unknown) or one failed. */
  download?: { kind: 'downloading'; percent: number | null } | { kind: 'failed' };
}

export interface ChipFacts {
  caps: WorkspaceCapabilities;
  state: DraftState;
  /** Whether each setting differs from its default, by the keys SettingsPane uses. */
  changed: (...keys: (keyof DraftState)[]) => boolean;
  /** The dimensions the run submits (a preset's, or the custom ones). */
  width: number;
  height: number;
  /** The clip length in seconds, when the workflow has one. */
  seconds: number | null;
  /** Extend: the seconds added; null until a source is known. */
  extendSeconds: number | null;
  extendChanged: boolean;
  /** The LoRA names in use. */
  loraNames: string[];
  /** What the LoRA files are doing, when a download is on its way or failed. */
  loraDownload?: SettingChip['download'];
  modelLabel: string;
  modelDownloaded: boolean;
  /** Extend: the source clip's name, or null. */
  sourceName: string | null;
  /** The Images section's summary; null where the section is not offered (a model that can use no image, Extend Video). */
  images: ImagesSummary | null;
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
    { key: 'preset', label: 'Preset', value: f.presetName ?? 'none', changed: f.presetChanged, on: f.presetName !== null, applied: f.presetSlug ?? '' },
    caps.showFrameCount && f.seconds !== null ? { key: 'seconds', label: 'Length', value: `${fmt(f.seconds)} s`, changed: f.changed('frameCount'), warn: f.memoryWarn } : null,
    caps.showExtendLength ? { key: 'extendSeconds', label: 'Extend', value: f.extendSeconds === null ? '—' : `+${fmt(f.extendSeconds)} s`, changed: f.extendChanged, warn: f.memoryWarn } : null,
    caps.showSteps ? { key: 'steps', label: 'Steps', value: fmt(s.steps), changed: f.changed('steps') } : null,
    caps.showGuidance ? { key: 'guidance', label: 'Guidance', value: fmt(s.guidance), changed: f.changed('guidance') } : null,
    caps.showSeed ? { key: 'seed', label: 'Seed', value: s.seed === null ? 'random' : String(s.seed), changed: f.changed('seed'), on: s.seed !== null } : null,
    caps.showRuns ? { key: 'runs', label: 'Batch', value: `×${s.runs}`, changed: f.changed('runs') } : null,
    // One chip for every image the tab takes; its value is the section's summary, and its dot shows a Strength off its default.
    f.images ? { key: 'images', label: 'Images', value: f.images.text, changed: s.workflow === 'img2img' && caps.showI2IStrength && f.changed('referenceImageStrength'), on: f.images.tone === 'active', warn: f.images.tone === 'error' } : null,
    caps.showSourceVideo ? { key: 'source', label: 'Source', value: f.sourceName ?? 'none', changed: false, on: f.sourceName !== null, warn: s.missingSourceVideo } : null,
    caps.showDimensions ? { key: 'size', label: '', value: `${f.width}×${f.height}`, changed: f.changed('ratio', 'size', 'dimensionMode') } : null,
    caps.showAudio || caps.showLowMemory || caps.showVideoUpscale
      ? { key: 'audio', label: 'Audio', value: caps.showAudio ? (s.audio ? 'on' : 'off') : s.lowMemory ? 'low mem' : 'off', changed: f.changed('audio', 'lowMemory', 'videoUpscaleEnabled'), on: caps.showAudio ? s.audio : s.lowMemory }
      : null,
    { key: 'loras', label: 'LoRA', value: f.loraNames.length ? f.loraNames.join(', ') : 'none', changed: false, on: f.loraNames.length > 0, download: f.loraDownload },
    caps.showImageUpscale ? { key: 'upscale', label: 'Upscale', value: s.upscaleEnabled ? `${s.upscaleFactor}×` : 'off', changed: f.changed('upscaleEnabled', 'upscaleFactor'), on: s.upscaleEnabled } : null,
    caps.showPostprocessTransparent && (s.postprocessTransparent || f.openKey === 'transparent')
      ? { key: 'transparent', label: 'Transparent', value: s.postprocessTransparent ? 'on' : 'off', changed: f.changed('postprocessTransparent'), on: s.postprocessTransparent }
      : null,
    { key: 'model', label: '', value: f.modelLabel, changed: false, warn: !f.modelDownloaded },
  ];
  return chips.filter((chip): chip is SettingChip => chip !== null);
}

/** The chips the strip shows before the context is known, so the frame is complete from the first frame. */
const PENDING_IMAGE_CHIPS: [ChipKey, string][] = [['steps', 'Steps'], ['guidance', 'Guidance'], ['seed', 'Seed'], ['runs', 'Batch'], ['images', 'Images'], ['size', ''], ['loras', 'LoRA'], ['upscale', 'Upscale'], ['model', '']];
const PENDING_VIDEO_CHIPS: [ChipKey, string][] = [['seconds', 'Length'], ['steps', 'Steps'], ['guidance', 'Guidance'], ['seed', 'Seed'], ['images', 'Images'], ['size', ''], ['audio', 'Audio'], ['loras', 'LoRA'], ['model', '']];
const PENDING_CHIPS: Record<Workflow, [ChipKey, string][]> = {
  txt2img: PENDING_IMAGE_CHIPS,
  img2img: PENDING_IMAGE_CHIPS,
  txt2vid: PENDING_VIDEO_CHIPS,
  img2vid: PENDING_VIDEO_CHIPS,
  extend: [['extendSeconds', 'Extend'], ['steps', 'Steps'], ['guidance', 'Guidance'], ['seed', 'Seed'], ['source', 'Source'], ['audio', 'Audio'], ['loras', 'LoRA'], ['model', '']],
};

/**
 * Placeholder chips for a workflow while the context loads: the usual chips in their usual order with no value, so
 * the strip has its height and width before the request answers and nothing under the stage moves when it does.
 */
export function pendingChips(workflow: Workflow): SettingChip[] {
  const keys: [ChipKey, string][] = [['preset', 'Preset'], ...(PENDING_CHIPS[workflow] ?? PENDING_CHIPS.txt2img)];
  return keys.map(([key, label]) => ({ key, label, value: '—', changed: false }));
}
