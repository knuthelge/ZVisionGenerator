import { fileName } from '$lib/components/molecules/assetDetails';
import type { GalleryAsset, LoraInfo, WorkspaceContext } from '$lib/types';

export interface LoraChip {
  /** The installed LoRA's name, or the recorded path when no installed LoRA matches it. */
  name: string;
  weight: number;
}

/**
 * Parse a `ref:weight` LoRA list into chips, naming each by its installed LoRA.
 *
 * Reads the list as the server does: spaces around a ref or weight are ignored and an empty weight means 1.
 * Reused settings record LoRAs by file path; a path that is an installed LoRA's file becomes that LoRA's name.
 * Anything else (a Hugging Face repo, a file elsewhere) is kept as written. A LoRA listed twice keeps its first entry.
 */
export function parseLoraString(value: string, installed: LoraInfo[]): LoraChip[] {
  const chips: LoraChip[] = [];
  for (const raw of value.split(',')) {
    const entry = raw.trim();
    const separator = entry.lastIndexOf(':');
    const weightText = separator > 0 ? entry.slice(separator + 1).trim() : '';
    const hasWeight = separator > 0 && (weightText === '' || Number.isFinite(Number(weightText)));
    const ref = (hasWeight ? entry.slice(0, separator) : entry).trim();
    if (!ref) continue;
    const name = installed.find((lora) => lora.path === ref || lora.name === ref)?.name ?? ref;
    if (chips.some((chip) => chip.name === name)) continue;
    chips.push({ name, weight: hasWeight && weightText !== '' ? Number(weightText) : 1.0 });
  }
  return chips;
}

/** Write chips back as the `name:weight` list the server reads, so a reused path goes out as the LoRA it was matched to. */
export function formatLoraString(chips: LoraChip[]): string {
  return chips.map((chip) => `${chip.name}:${chip.weight}`).join(',');
}

/** Keep only the LoRAs in a `ref:weight` list that are still installed, written back by name. */
export function keepInstalledLoras(value: string, installed: LoraInfo[]): string {
  return formatLoraString(parseLoraString(value, installed).filter((chip) => installed.some((lora) => lora.name === chip.name)));
}

/** Show a LoRA by name: a path is shortened to its file name without the extension. */
export function loraLabel(ref: string): string {
  return /[\\/]/.test(ref) ? fileName(ref).replace(/\.safetensors$/i, '') : ref;
}

/** The family and size a model is, as far as the server detected them. */
export interface ModelLoraFamily {
  family: string | null;
  sizeVariant: string | null;
}

/** LoRAs split for the picker: `main` is shown, `other` folds away. */
export interface LoraGroups {
  main: LoraInfo[];
  other: LoraInfo[];
}

const FAMILY_LABELS: Record<string, string> = {
  zimage: 'Z-Image',
  'flux2_klein:4b': 'Klein 4B',
  'flux2_klein:9b': 'Klein 9B',
  flux2_klein: 'Klein',
  flux1: 'Flux 1',
  krea2: 'Krea 2',
  ideogram4: 'Ideogram 4',
  moss: 'MOSS-Image',
  ltx: 'LTX',
  h3: 'MiniMax H3',
};

/** Return the family and size of the selected model, or null when the context does not list the model. */
export function modelLoraFamily(ctx: WorkspaceContext, model: string): ModelLoraFamily | null {
  const image = ctx.image_model_defaults?.[model];
  if (image) return { family: image.family ?? null, sizeVariant: image.size_variant ?? null };
  const video = ctx.video_model_defaults?.[model];
  if (video) return { family: video.family ?? null, sizeVariant: null };
  return null;
}

/** Tell whether a LoRA was trained for a model: the families are equal and the sizes do not differ. */
export function loraMatches(lora: LoraInfo, model: ModelLoraFamily): boolean {
  if (!model.family || lora.family !== model.family) return false;
  return !lora.size_variant || !model.sizeVariant || lora.size_variant === model.sizeVariant;
}

/**
 * Return the installed LoRAs used with a model family in loaded history, newest first.
 *
 * `assets` are newest first, as the history store keeps them.
 */
export function recentLoraNames(assets: GalleryAsset[], family: string | null, installed: LoraInfo[]): string[] {
  if (!family) return [];
  const names: string[] = [];
  for (const asset of assets) {
    if (!asset.lora || asset.details?.model_family !== family) continue;
    for (const chip of parseLoraString(asset.lora, installed)) {
      if (installed.some((lora) => lora.name === chip.name) && !names.includes(chip.name)) names.push(chip.name);
    }
  }
  return names;
}

function _byName(a: LoraInfo, b: LoraInfo): number {
  return a.name.localeCompare(b.name);
}

/**
 * Order LoRAs for a model's picker.
 *
 * `main` holds the matches (recently used first, then by name). `other` holds every other LoRA by name, those of
 * unknown family included. A model of unknown family gets every LoRA in `main`, by name.
 *
 * @param recent Names of the LoRAs used lately with the model's family, newest first.
 */
export function groupLorasForModel(loras: LoraInfo[], model: ModelLoraFamily | null, recent: string[]): LoraGroups {
  if (!model?.family) return { main: [...loras].sort(_byName), other: [] };
  const matches = loras.filter((lora) => loraMatches(lora, model));
  const other = loras.filter((lora) => !loraMatches(lora, model)).sort(_byName);
  const recentMatches = recent.flatMap((name) => matches.filter((lora) => lora.name === name));
  const rest = matches.filter((lora) => !recent.includes(lora.name)).sort(_byName);
  return { main: [...recentMatches, ...rest], other };
}

/** Name a model family for a person: `flux2_klein` with size `4b` is "Klein 4B". Any unlisted family is shown as it is. */
export function familyLabel(family: string, sizeVariant?: string | null): string {
  return (sizeVariant ? FAMILY_LABELS[`${family}:${sizeVariant}`] : undefined) ?? FAMILY_LABELS[family] ?? family;
}
