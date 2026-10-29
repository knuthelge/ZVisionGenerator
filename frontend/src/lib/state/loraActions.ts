import { draft } from './draft.svelte';
import { formatLoraString, type LoraChip } from './loras';
import { pendingTriggers } from './loraTriggers.svelte';

/** Use a LoRA at *weight*; a LoRA already in use keeps its place and takes the new weight. */
export function addLoraChip(chips: readonly LoraChip[], name: string, weight: number): void {
  const exists = chips.some((chip) => chip.name === name);
  const next = exists ? chips.map((chip) => (chip.name === name ? { name, weight } : chip)) : [...chips, { name, weight }];
  draft.update('loraString', formatLoraString(next));
}

/** Stop using a LoRA: its chip goes, its file stays, and a download still on its way keeps going. */
export function removeLoraChip(chips: readonly LoraChip[], name: string): void {
  pendingTriggers.forget(name);
  draft.update('loraString', formatLoraString(chips.filter((chip) => chip.name !== name)));
}
