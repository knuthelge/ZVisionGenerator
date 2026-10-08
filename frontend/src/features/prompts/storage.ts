import type { PromptDocument } from '$lib/types';

/** Browser-storage keys of the prompt builder. */
export const UNSAVED_KEY = 'ziv.promptBuilder.unsaved';
export const RECENT_KEY = 'ziv.promptBuilder.recent';
const RECENT_LIMIT = 8;

export interface UnsavedDocument {
  revision: string;
  document: PromptDocument;
}

function read<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : fallback;
  } catch {
    return fallback;
  }
}

function write(key: string, value: unknown): void {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // storage full or unavailable: unsaved work just isn't kept across visits
  }
}

/** Unsaved edits of `path`, made on the given revision. */
export function loadUnsaved(path: string): UnsavedDocument | null {
  return read<Record<string, UnsavedDocument>>(UNSAVED_KEY, {})[path] ?? null;
}

export function saveUnsaved(path: string, unsaved: UnsavedDocument): void {
  write(UNSAVED_KEY, { ...read<Record<string, UnsavedDocument>>(UNSAVED_KEY, {}), [path]: unsaved });
}

export function clearUnsaved(path: string): void {
  const all = read<Record<string, UnsavedDocument>>(UNSAVED_KEY, {});
  if (!(path in all)) return;
  delete all[path];
  write(UNSAVED_KEY, all);
}

/** Recently opened prompt files, newest first. */
export function recentFiles(): string[] {
  const list = read<unknown>(RECENT_KEY, []);
  return Array.isArray(list) ? list.filter((item): item is string => typeof item === 'string') : [];
}

export function rememberRecent(path: string): void {
  write(RECENT_KEY, [path, ...recentFiles().filter((item) => item !== path)].slice(0, RECENT_LIMIT));
}

export function forgetRecent(path: string): void {
  write(RECENT_KEY, recentFiles().filter((item) => item !== path));
}
