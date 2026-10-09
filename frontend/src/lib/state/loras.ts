import { fileName } from '$lib/components/molecules/assetDetails';
import type { LoraInfo } from '$lib/types';

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

/** Show a LoRA by name: a path is shortened to its file name without the extension. */
export function loraLabel(ref: string): string {
  return /[\\/]/.test(ref) ? fileName(ref).replace(/\.safetensors$/i, '') : ref;
}
