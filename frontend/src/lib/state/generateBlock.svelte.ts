import type { LoraInfo } from '$lib/types';
import { downloadStore } from './downloads.svelte';
import { loraBlock, type LoraBlock } from './loraDownloads';
import { parseLoraString } from './loras';

/**
 * What the LoRA chips in a `ref:weight` list hold Generate back for: a download on its way, or one that failed.
 * Reads the download lane, so a `$derived` that calls it follows the downloads.
 */
export function generateBlock(loraString: string, installed: LoraInfo[]): LoraBlock {
  return loraBlock(parseLoraString(loraString, installed), installed, downloadStore.items, downloadStore.unlisted);
}
