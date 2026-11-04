import type { DraftState, LoraInfo } from '$lib/types';
import { downloadStore } from './downloads.svelte';
import { loraBlock, type LoraBlock } from './loraDownloads';
import { parseLoraString } from './loras';
import { isAudioWorkflow } from './workspaceTab';

/**
 * What the LoRA chips in a `ref:weight` list hold Generate back for: a download on its way, or one that failed. The Audio tab
 * has no LoRAs, so the chips it carries from the image or video model hold nothing back.
 * Reads the download lane, so a `$derived` that calls it follows the downloads.
 */
export function generateBlock(state: Pick<DraftState, 'workflow' | 'loraString'>, installed: LoraInfo[]): LoraBlock {
  const loraString = isAudioWorkflow(state.workflow) ? '' : state.loraString;
  return loraBlock(parseLoraString(loraString, installed), installed, downloadStore.items, downloadStore.unlisted);
}
