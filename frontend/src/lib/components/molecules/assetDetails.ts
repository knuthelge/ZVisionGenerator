import { reuseParams, workflowLabel } from '$lib/state/assetActions';
import { audioModelLabel, audioRowOf, formatFact, kindFact, lengthFact } from '$lib/state/audioAsset';
import { formatTimestamp } from '$lib/state/audioTrack';
import { sizeLabel } from '$lib/state/downloadProgress';
import { keyNameOf } from '$lib/state/musicSettings';
import { DELIVERY_FIELDS } from '$lib/state/audioDraft';
import { deliverySummary } from '$lib/state/speechDelivery';
import type { AudioModelOption, AudioSettingsVocabulary, GalleryAsset, SpeechDeliveryValues, UpscaleMethod } from '$lib/types';

export interface DetailFact {
  label: string;
  value: string;
  wide?: boolean;
  /** Let a long value run over several lines instead of cutting it short. */
  wrap?: boolean;
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
export function assetDetailSections(asset: GalleryAsset, audioModels?: readonly AudioModelOption[], vocabulary?: AudioSettingsVocabulary): AssetDetailSections {
  if (asset.media_type === 'audio') return audioDetailSections(asset, audioModels, vocabulary);
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
      asset.image_path ? { label: isVideo ? 'First frame' : 'Start image', value: fileName(asset.image_path), wide: details.image_strength == null && !asset.end_image_path } : null,
      asset.end_image_path ? { label: 'Last frame', value: fileName(asset.end_image_path) } : null,
      asset.audio_track_path ? { label: 'Audio track', value: describeTrack(asset), wide: true, title: asset.audio_track_path } : null,
      asset.reference_images?.length ? { label: 'References', value: asset.reference_images.map(fileName).join(', '), wide: true } : null,
      asset.image_path && details.image_strength != null ? { label: 'Strength', value: String(details.image_strength) } : null,
    ]),
    postProcessing: compact([
      upscale?.factor ? { label: 'Upscale', value: describeUpscale(upscale), wide: true } : null,
      generation.sharpen != null ? { label: 'Sharpen', value: String(generation.sharpen) } : null,
      generation.contrast != null ? { label: 'Contrast', value: String(generation.contrast) } : null,
      generation.saturation != null ? { label: 'Saturation', value: String(generation.saturation) } : null,
      generation.transparent ? { label: 'Transparent background', value: 'On' } : null,
      generation.audio != null ? { label: 'Audio', value: generation.audio ? 'On' : 'Off' } : null,
      generation.output_format ? { label: 'Format', value: generation.output_format.toUpperCase() } : null,
    ]),
    file: compact([
      { label: 'Project', value: asset.project ?? 'No project', wide: true, wrap: true },
      generation.time ? { label: 'Generation time', value: formatSeconds(generation.time) } : null,
      { label: 'Created', value: new Date(asset.created_at).toLocaleString(), wide: !generation.time },
    ]),
  };
}

/** Group what is recorded about an audio take: how it was made, and its file. A file without this app's settings has no Generation. */
function audioDetailSections(asset: GalleryAsset, audioModels: readonly AudioModelOption[] | undefined, vocabulary: AudioSettingsVocabulary | undefined): AssetDetailSections {
  const recorded = asset.kind != null;
  return {
    negativePrompt: null,
    generation: recorded
      ? compact([
          { label: 'Kind', value: kindFact(asset) },
          { label: 'Model', value: audioModelLabel(asset, audioModels) || '—' },
          asset.kind === 'speech' ? { label: 'Voice', value: voiceFact(asset, audioModels) } : null,
          { label: 'Length', value: lengthFact(asset) },
          musicFact(asset, vocabulary),
          deliveryFact(asset, vocabulary),
          soundTypeFact(asset, vocabulary),
          { label: 'Seed', value: asset.seed != null ? String(asset.seed) : '—' },
          asset.notes?.length ? { label: 'Note', value: asset.notes.join(' '), wide: true, wrap: true } : null,
        ])
      : [],
    postProcessing: [],
    file: compact([
      { label: 'Project', value: asset.project ?? 'No project', wide: true, wrap: true },
      { label: 'Format', value: formatFact(asset), wide: true },
      asset.file_bytes != null ? { label: 'Size', value: sizeLabel(asset.file_bytes) } : null,
      { label: 'Created', value: new Date(asset.created_at).toLocaleString(), wide: asset.file_bytes == null },
    ]),
  };
}

/** The Tempo · Key fact of a song, with only the parts the take set (`92 BPM · A minor`); null when it set neither. */
function musicFact(asset: GalleryAsset, vocabulary: AudioSettingsVocabulary | undefined): DetailFact | null {
  const tempo = asset.tempo != null ? `${asset.tempo} BPM` : null;
  const key = asset.key ? keyNameOf(vocabulary, asset.key) : null;
  const parts = [tempo, key].filter((part) => part !== null);
  if (parts.length === 0) return null;
  return { label: tempo !== null && key !== null ? 'Tempo · Key' : tempo !== null ? 'Tempo' : 'Key', value: parts.join(' · ') };
}

/** The Delivery fact of a speech take read in a built-in voice: the Vibe it matches, else Custom; null when it recorded none of the five values or the vocabulary is not at hand. */
function deliveryFact(asset: GalleryAsset, vocabulary: AudioSettingsVocabulary | undefined): DetailFact | null {
  if (asset.kind !== 'speech' || vocabulary === undefined || DELIVERY_FIELDS.every((field) => !asset[field])) return null;
  const values: SpeechDeliveryValues = { ...vocabulary.delivery_defaults };
  for (const field of DELIVERY_FIELDS) values[field] = asset[field] || values[field];
  return { label: 'Delivery', value: deliverySummary(values, vocabulary) };
}

/** The Sound type fact of a sound: its label in the vocabulary, else its id; null when the take recorded none. */
function soundTypeFact(asset: GalleryAsset, vocabulary: AudioSettingsVocabulary | undefined): DetailFact | null {
  if (!asset.sound_type) return null;
  return { label: 'Sound type', value: vocabulary?.sound_types.find((type) => type.id === asset.sound_type)?.label ?? asset.sound_type };
}

/** The Voice fact of a speech take: the name of the saved voice it followed (while that voice exists), else the recording's file name, else the name of the built-in voice it was read in (its id when this machine does not list it), else a dash. */
function voiceFact(asset: GalleryAsset, audioModels: readonly AudioModelOption[] | undefined): string {
  if (asset.voice_name) return asset.voice_name;
  const params = reuseParams(asset);
  if (asset.has_reference_voice === true) return typeof params.voice_path === 'string' && params.voice_path !== '' ? fileName(params.voice_path) : 'A recording';
  const speaker = typeof params.speaker === 'string' && params.speaker !== '' ? params.speaker : null;
  if (speaker === null) return '—';
  const model = audioRowOf(asset, audioModels);
  return (model?.kind === 'speech' ? model.voices.find((voice) => voice.id === speaker)?.label : undefined) ?? speaker;
}

/** Describe the track a clip followed: `song.mp3 · 1:30–1:34.2`, or `song.mp3 · from 1:30` when the end was not recorded. */
function describeTrack(asset: GalleryAsset): string {
  const start = asset.audio_track_start ?? 0;
  const end = asset.audio_track_end;
  const range = end != null && end > start ? `${formatTimestamp(start)}–${formatTimestamp(end)}` : `from ${formatTimestamp(start)}`;
  return `${fileName(asset.audio_track_path ?? '')} · ${range}`;
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

const UPSCALE_METHOD_LABELS: Record<UpscaleMethod, string> = { seedvr2: 'SeedVR2', creative: 'creative', ltx: 'LTX' };

/** Describe a recorded upscale, e.g. `2× · denoise 0.4 · 3 steps`. */
export function describeUpscale(upscale: NonNullable<NonNullable<GalleryAsset['details']>['generation']>['upscale']): string {
  if (!upscale?.factor) return '';
  const parts = [`${upscale.factor}×`];
  if (upscale.method) parts.push(UPSCALE_METHOD_LABELS[upscale.method]);
  if (upscale.model) parts.push(upscale.quantize ? `${upscale.model} q${upscale.quantize}` : upscale.model);
  if (upscale.softness != null) parts.push(`softness ${upscale.softness}`);
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
  // A folder path may end in a separator; its name is the last non-empty part.
  return path.split(/[\\/]/).filter(Boolean).pop() || path;
}

function compact(facts: (DetailFact | null)[]): DetailFact[] {
  return facts.filter((fact): fact is DetailFact => fact !== null);
}
