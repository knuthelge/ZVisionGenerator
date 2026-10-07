import { workflowLabel } from '$lib/state/assetActions';
import type { GalleryAsset } from '$lib/types';

export interface DetailFact {
  label: string;
  value: string;
  wide?: boolean;
  /** Hover text when it differs from the value, e.g. a full path. */
  title?: string;
}

export interface AssetDetailSections {
  negativePrompt: string | null;
  generation: DetailFact[];
  postProcessing: DetailFact[];
  file: DetailFact[];
}

/** Group everything recorded about an asset into the viewer's detail sections; unrecorded fields are left out. */
export function assetDetailSections(asset: GalleryAsset): AssetDetailSections {
  const details = asset.details ?? {};
  const generation = details.generation ?? {};
  const upscale = generation.upscale;
  const isVideo = asset.media_type === 'video';

  return {
    negativePrompt: details.negative_prompt || null,
    generation: compact([
      { label: 'Model', value: asset.model || '—' },
      details.model_family ? { label: 'Family', value: details.model_family } : null,
      { label: 'Workflow', value: workflowLabel(details.recorded_workflow || asset.workflow) },
      { label: 'Dimensions', value: asset.width && asset.height ? `${asset.width}×${asset.height}` : '—' },
      asset.ratio ? { label: 'Ratio', value: asset.size ? `${asset.ratio} · ${asset.size}` : asset.ratio } : null,
      { label: 'Seed', value: asset.seed != null ? String(asset.seed) : '—' },
      { label: 'Steps', value: asset.steps != null ? String(asset.steps) : '—' },
      isVideo ? null : { label: 'Guidance', value: asset.guidance != null ? String(asset.guidance) : '—' },
      details.scheduler ? { label: 'Scheduler', value: details.scheduler } : null,
      generation.quantize ? { label: 'Quantize', value: `${generation.quantize}-bit` } : null,
      asset.frame_count ? { label: 'Frames', value: String(asset.frame_count) } : null,
      ...(asset.lora ? loraFacts(asset.lora) : []),
      asset.image_path ? { label: 'Reference', value: fileName(asset.image_path), wide: details.image_strength == null } : null,
      asset.image_path && details.image_strength != null ? { label: 'Strength', value: String(details.image_strength) } : null,
    ]),
    postProcessing: compact([
      upscale?.factor ? { label: 'Upscale', value: describeUpscale(upscale), wide: true } : null,
      generation.sharpen != null ? { label: 'Sharpen', value: String(generation.sharpen) } : null,
      generation.contrast != null ? { label: 'Contrast', value: String(generation.contrast) } : null,
      generation.saturation != null ? { label: 'Saturation', value: String(generation.saturation) } : null,
      generation.audio != null ? { label: 'Audio', value: generation.audio ? 'On' : 'Off' } : null,
      generation.output_format ? { label: 'Format', value: generation.output_format.toUpperCase() } : null,
    ]),
    file: compact([
      generation.time ? { label: 'Generation time', value: formatSeconds(generation.time) } : null,
      { label: 'Created', value: new Date(asset.created_at).toLocaleString(), wide: !generation.time },
    ]),
  };
}

/** List recorded LoRAs (`path:weight` entries joined by commas) one per row as `name · weight`, full entry on hover. */
export function loraFacts(lora: string): DetailFact[] {
  const entries = lora.split(',').map((entry) => entry.trim()).filter(Boolean);
  return entries.map((entry, index) => {
    const separator = entry.lastIndexOf(':');
    const weight = separator > 0 ? entry.slice(separator + 1) : '';
    const hasWeight = weight !== '' && Number.isFinite(Number(weight));
    const name = fileName(hasWeight ? entry.slice(0, separator) : entry).replace(/\.safetensors$/i, '');
    return {
      label: entries.length > 1 ? `LoRA ${index + 1}` : 'LoRA',
      value: hasWeight ? `${name} · ${weight}` : name,
      wide: true,
      title: entry,
    };
  });
}

/** Describe a recorded upscale, e.g. `2× · denoise 0.4 · 3 steps`. */
export function describeUpscale(upscale: NonNullable<NonNullable<GalleryAsset['details']>['generation']>['upscale']): string {
  if (!upscale?.factor) return '';
  const parts = [`${upscale.factor}×`];
  if (upscale.denoise != null) parts.push(`denoise ${upscale.denoise}`);
  if (upscale.steps != null) parts.push(`${upscale.steps} steps`);
  if (upscale.guidance != null) parts.push(`guidance ${upscale.guidance}`);
  if (upscale.pre_sharpen != null) parts.push(`pre-sharpen ${upscale.pre_sharpen}`);
  return parts.join(' · ');
}

/** Format seconds as `12.3 s` or `2m 05s`. */
export function formatSeconds(seconds: number): string {
  if (seconds < 59.95) return `${seconds.toFixed(1)} s`;
  const total = Math.round(seconds);
  return `${Math.floor(total / 60)}m ${String(total % 60).padStart(2, '0')}s`;
}

/** Return the last path segment of a POSIX or Windows path. */
export function fileName(path: string): string {
  return path.split(/[\\/]/).pop() || path;
}

function compact(facts: (DetailFact | null)[]): DetailFact[] {
  return facts.filter((fact): fact is DetailFact => fact !== null);
}
