import { flushSync } from 'svelte';
import type { BrowseInfo, BrowseListing, BrowsePlaces, BrowsePreviewKind, BrowsePurpose, BrowsePurposeName } from '$lib/types';

const PURPOSES: Record<BrowsePurposeName, Omit<BrowsePurpose, 'name'>> = {
  output_directory: { kind: 'directory', label: 'Output folder', accepted_extensions: [], preview: null },
  prompt_file: { kind: 'existing_file', label: 'Prompt file', accepted_extensions: ['.yaml', '.yml'], preview: null },
  prompt_file_folder: { kind: 'directory', label: 'Prompt file folder', accepted_extensions: [], preview: null },
  checkpoint_file: { kind: 'existing_file', label: 'Checkpoint file', accepted_extensions: ['.safetensors'], preview: null },
  lora_file: { kind: 'existing_file', label: 'LoRA file', accepted_extensions: ['.safetensors'], preview: null },
  source_video: { kind: 'existing_file', label: 'Source video', accepted_extensions: ['.mp4'], preview: 'video' },
  reference_image: { kind: 'existing_file', label: 'Reference image', accepted_extensions: ['.png', '.jpg', '.jpeg', '.webp'], preview: 'image' },
  audio_track: { kind: 'existing_file', label: 'Audio track', accepted_extensions: ['.wav', '.mp3', '.m4a', '.flac', '.ogg'], preview: 'audio' },
};

/** Return what `/api/browse/places` answers for *purpose*, with no places. */
export function placesFor(purpose: BrowsePurposeName): BrowsePlaces {
  return { purpose: { name: purpose, ...PURPOSES[purpose] }, places: [] };
}

/** The stand-in for `browsePreviewUrl`: a recognisable URL carrying the purpose, path and frame. */
export function stubBrowsePreviewUrl(purpose: BrowsePurposeName, path: string, options: { frame?: 'last'; at?: number; size?: 'large' } = {}): string {
  const query = options.frame ? '?frame=last' : options.at !== undefined ? `?at=${options.at}` : '';
  return `/preview/${purpose}${path}${query}`;
}

/** The stand-in for `browseMediaUrl`. */
export function stubBrowseMediaUrl(purpose: BrowsePurposeName, path: string): string {
  return `/media/${purpose}${path}`;
}

/** The stand-in for `browseSpriteUrl`. */
export function stubBrowseSpriteUrl(purpose: BrowsePurposeName, path: string): string {
  return `/sprite/${purpose}${path}`;
}

/** Return the facts `browseInfo` answers for a file of *kind*, with *overrides* on top. */
export function infoFor(kind: BrowsePreviewKind, overrides: BrowseInfo = {}): BrowseInfo {
  const base: Record<BrowsePreviewKind, BrowseInfo> = {
    image: { width: 640, height: 480 },
    audio: { seconds: 12.5, sample_rate: 44100, channels: 2, peaks: Array.from({ length: 100 }, (_, index) => 20 + (index % 10) * 20) },
    video: { width: 1280, height: 720, fps: 24, frames: 240, seconds: 10 },
  };
  return { ...base[kind], ...overrides };
}

/** The stand-in for `browseInfo`: the facts of the kind the purpose previews. */
export async function stubBrowseInfo(purpose: BrowsePurposeName): Promise<BrowseInfo> {
  const kind = PURPOSES[purpose].preview;
  if (!kind) throw new Error(`The ${purpose} purpose has no previews.`);
  return infoFor(kind);
}

/** Return a listing of the folder that holds *path*: one selectable entry for it (a folder when it ends in `/`). */
export function listingWith(path: string): BrowseListing {
  const isFolder = path.endsWith('/');
  const trimmed = isFolder ? path.slice(0, -1) : path;
  const cut = trimmed.lastIndexOf('/');
  const folder = cut > 0 ? trimmed.slice(0, cut) : '/';
  const name = trimmed.slice(cut + 1);
  return {
    path: folder,
    parent: folder === '/' ? null : folder.slice(0, folder.lastIndexOf('/')) || '/',
    separator: '/',
    focus: null,
    entries: [{ name, path: trimmed, kind: isFolder ? 'directory' : 'file', size: isFolder ? null : 1, modified: null, selectable: true }],
    hidden_count: 0,
  };
}

/** Return the open path browser dialog, or null. */
export function openBrowser(root: ParentNode = document): HTMLElement | null {
  return root.querySelector<HTMLElement>('[data-testid="path-browser"]');
}

/** Choose what the open browser offers: the named row for a file, or the current folder in folder mode. */
export function chooseInBrowser(name?: string, root: ParentNode = document): void {
  const browser = openBrowser(root);
  if (!browser) throw new Error('The path browser is not open.');
  if (name) {
    const row = Array.from(browser.querySelectorAll<HTMLElement>('[role="option"]')).find((item) => item.textContent?.includes(name));
    if (!row) throw new Error(`No row named ${name} in the path browser.`);
    flushSync(() => row.click());
  }
  const main = browser.querySelector<HTMLButtonElement>('.footer .main, .toolbar .main');
  flushSync(() => main?.click());
}
