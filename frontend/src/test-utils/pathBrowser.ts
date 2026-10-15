import { flushSync } from 'svelte';
import type { BrowseListing, BrowsePlaces, BrowsePurpose, BrowsePurposeName } from '$lib/types';

const PURPOSES: Record<BrowsePurposeName, Omit<BrowsePurpose, 'name'>> = {
  output_directory: { kind: 'directory', label: 'Output folder', accepted_extensions: [] },
  prompt_file: { kind: 'existing_file', label: 'Prompt file', accepted_extensions: ['.yaml', '.yml'] },
  prompt_file_folder: { kind: 'directory', label: 'Prompt file folder', accepted_extensions: [] },
  checkpoint_file: { kind: 'existing_file', label: 'Checkpoint file', accepted_extensions: ['.safetensors'] },
  lora_file: { kind: 'existing_file', label: 'LoRA file', accepted_extensions: ['.safetensors'] },
  source_video: { kind: 'existing_file', label: 'Source video', accepted_extensions: ['.mp4'] },
  reference_image: { kind: 'existing_file', label: 'Reference image', accepted_extensions: ['.png', '.jpg', '.jpeg', '.webp'] },
};

/** Return what `/api/browse/places` answers for *purpose*, with no places. */
export function placesFor(purpose: BrowsePurposeName): BrowsePlaces {
  return { purpose: { name: purpose, ...PURPOSES[purpose] }, places: [] };
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
  const main = browser.querySelector<HTMLButtonElement>('.footer .main');
  flushSync(() => main?.click());
}
