import { copyText } from '$lib/clipboard';
import { takeWords } from '$lib/state/audioAsset';
import { addToast } from '$lib/state/toasts.svelte';
import type { GalleryAsset } from '$lib/types';

/** The text Copy prompt puts on the clipboard: the prompt of an image or video, the style, text or prompt of a take with settings. */
export function copyableText(asset: GalleryAsset): string {
  return asset.media_type === 'audio' ? takeWords(asset) : (asset.prompt ?? '');
}

/** Put *text* on the clipboard and say so; a refused write is reported as an error. */
export async function copyPrompt(text: string): Promise<void> {
  if (await copyText(text)) addToast('Prompt copied', 'success');
  else addToast('Could not copy the prompt', 'error');
}
