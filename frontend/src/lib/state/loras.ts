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
 * Reused settings record LoRAs by file path; a path that matches an installed LoRA becomes that LoRA's name.
 */
export function parseLoraString(value: string, installed: LoraInfo[]): LoraChip[] {
  return value.split(',').flatMap((raw) => {
    const entry = raw.trim();
    const separator = entry.lastIndexOf(':');
    const weightText = separator > 0 ? entry.slice(separator + 1) : '';
    const hasWeight = weightText !== '' && Number.isFinite(Number(weightText));
    const ref = hasWeight ? entry.slice(0, separator) : entry;
    if (!ref) return [];
    const match = installed.find((lora) => lora.path === ref || lora.name === ref)
      ?? installed.find((lora) => lora.name === loraLabel(ref));
    return [{ name: match?.name ?? ref, weight: hasWeight ? Number(weightText) : 1.0 }];
  });
}

/** Show a LoRA by name: a path is shortened to its file name without the extension. */
export function loraLabel(ref: string): string {
  return /[\\/]/.test(ref) ? fileName(ref).replace(/\.safetensors$/i, '') : ref;
}
