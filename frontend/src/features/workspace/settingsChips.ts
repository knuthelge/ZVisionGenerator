import type { WorkspaceCapabilities } from './capabilities';
import type { DraftState, Workflow } from '$lib/types';

/** The settings a phone shows as chips, each opening one control. */
export type ChipKey =
  | 'preset' | 'seconds' | 'extendSeconds' | 'steps' | 'guidance' | 'strength' | 'seed' | 'runs'
  | 'reference' | 'source' | 'size' | 'audio' | 'loras' | 'upscale' | 'model';

export interface SettingChip {
  key: ChipKey;
  /** Short name before the value; empty when the value speaks for itself (the size, the model). */
  label: string;
  value: string;
  /** Differs from the model's default. */
  changed: boolean;
  /** The value is "on" (a locked seed, LoRAs present, upscale on): shown in the accent colour. */
  on?: boolean;
  /** Something to look at: a clip that may not fit, a model not downloaded. */
  warn?: boolean;
  /** The preset chip's applied preset slug, empty when none is applied. */
  applied?: string;
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
  modelLabel: string;
  modelDownloaded: boolean;
  /** The reference image's name, or null. */
  referenceName: string | null;
  /** Extend: the source clip's name, or null. */
  sourceName: string | null;
  /** The clip or extension may not fit memory. */
  memoryWarn: boolean;
  /** The applied settings preset's name, or null when none is applied. */
  presetName: string | null;
  /** The applied settings preset's slug, or null. */
  presetSlug: string | null;
  /** The form differs from the applied preset. */
  presetChanged: boolean;
}

const fmt = (value: number): string => (Number.isInteger(value) ? String(value) : String(Number(value.toFixed(2))));

/**
 * Chips in the order their settings change in the loop: every run (length, steps, guidance, strength, seed, batch),
 * per series (reference or source, size, audio, LoRAs), at the end (upscale), at the start (model). The preset, which
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
    caps.showI2IStrength ? { key: 'strength', label: 'Strength', value: fmt(s.referenceImageStrength), changed: f.changed('referenceImageStrength') } : null,
    caps.showSeed ? { key: 'seed', label: 'Seed', value: s.seed === null ? 'random' : String(s.seed), changed: f.changed('seed'), on: s.seed !== null } : null,
    caps.showRuns ? { key: 'runs', label: 'Batch', value: `×${s.runs}`, changed: f.changed('runs') } : null,
    caps.showRefImage ? { key: 'reference', label: 'Reference', value: f.referenceName ?? 'none', changed: false, on: f.referenceName !== null } : null,
    caps.showSourceVideo ? { key: 'source', label: 'Source', value: f.sourceName ?? 'none', changed: false, on: f.sourceName !== null, warn: s.missingSourceVideo } : null,
    caps.showDimensions ? { key: 'size', label: '', value: `${f.width}×${f.height}`, changed: f.changed('ratio', 'size', 'dimensionMode') } : null,
    caps.showAudio || caps.showLowMemory || caps.showVideoUpscale
      ? { key: 'audio', label: 'Audio', value: caps.showAudio ? (s.audio ? 'on' : 'off') : s.lowMemory ? 'low mem' : 'off', changed: f.changed('audio', 'lowMemory', 'videoUpscaleEnabled'), on: caps.showAudio ? s.audio : s.lowMemory }
      : null,
    { key: 'loras', label: 'LoRA', value: f.loraNames.length ? f.loraNames.join(', ') : 'none', changed: false, on: f.loraNames.length > 0 },
    caps.showImageUpscale ? { key: 'upscale', label: 'Upscale', value: s.upscaleEnabled ? `${s.upscaleFactor}×` : 'off', changed: f.changed('upscaleEnabled', 'upscaleFactor'), on: s.upscaleEnabled } : null,
    { key: 'model', label: '', value: f.modelLabel, changed: false, warn: !f.modelDownloaded },
  ];
  return chips.filter((chip): chip is SettingChip => chip !== null);
}

/** The chips the strip shows before the context is known, so the frame is complete from the first frame. */
const PENDING_CHIPS: Record<Workflow, [ChipKey, string][]> = {
  txt2img: [['steps', 'Steps'], ['guidance', 'Guidance'], ['seed', 'Seed'], ['runs', 'Batch'], ['size', ''], ['loras', 'LoRA'], ['upscale', 'Upscale'], ['model', '']],
  img2img: [['steps', 'Steps'], ['guidance', 'Guidance'], ['strength', 'Strength'], ['seed', 'Seed'], ['runs', 'Batch'], ['reference', 'Reference'], ['size', ''], ['loras', 'LoRA'], ['upscale', 'Upscale'], ['model', '']],
  txt2vid: [['seconds', 'Length'], ['steps', 'Steps'], ['guidance', 'Guidance'], ['seed', 'Seed'], ['size', ''], ['audio', 'Audio'], ['loras', 'LoRA'], ['model', '']],
  img2vid: [['seconds', 'Length'], ['steps', 'Steps'], ['guidance', 'Guidance'], ['seed', 'Seed'], ['reference', 'Reference'], ['size', ''], ['audio', 'Audio'], ['loras', 'LoRA'], ['model', '']],
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
