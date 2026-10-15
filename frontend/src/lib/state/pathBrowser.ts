import type { BrowseEntry, BrowsePurposeName } from '$lib/types';

export interface Crumb {
  label: string;
  path: string;
}

export interface CompletionTarget {
  /** The typed folder, up to and including its last separator. */
  dir: string;
  /** The typed start of a name inside that folder. */
  prefix: string;
}

const RECENT_KEY = 'ziv-recent-paths-v1';
const HIDDEN_KEY = 'ziv-browse-hidden-v1';
const RECENT_LIMIT = 5;
const COMPLETION_LIMIT = 8;
// Absolute host paths: POSIX, home-relative, Windows drive or UNC.
const ABSOLUTE_RE = /^(\/|~|[A-Za-z]:[\\/]|\\\\)/;

/** Split typed text into the folder to list and the name prefix to match, or null when it isn't an absolute path yet. */
export function completionTarget(text: string): CompletionTarget | null {
  if (text === '~') return { dir: '~/', prefix: '' };
  const cut = Math.max(text.lastIndexOf('/'), text.lastIndexOf('\\'));
  if (cut < 0) return null;
  const dir = text.slice(0, cut + 1);
  if (!ABSOLUTE_RE.test(dir)) return null;
  return { dir, prefix: text.slice(cut + 1) };
}

/** Return the entries whose names start with *prefix* (ignoring case), at most *limit*. */
export function matchCompletions(entries: BrowseEntry[], prefix: string, limit = COMPLETION_LIMIT): BrowseEntry[] {
  const lower = prefix.toLowerCase();
  return entries.filter((entry) => entry.name.toLowerCase().startsWith(lower)).slice(0, limit);
}

/** Return the text a completion produces: a folder ends in the separator the user typed so completion can go on. */
export function completedPath(dir: string, entry: BrowseEntry): string {
  const separator = dir.endsWith('\\') ? '\\' : '/';
  return entry.kind === 'directory' ? `${dir}${entry.name}${separator}` : `${dir}${entry.name}`;
}

/** Split an absolute host path into clickable breadcrumbs, from the root down. */
export function breadcrumbs(path: string, separator: string): Crumb[] {
  const parts = path.split(separator).filter(Boolean);
  if (separator === '/') {
    const crumbs: Crumb[] = [{ label: '/', path: '/' }];
    parts.forEach((part, index) => crumbs.push({ label: part, path: `/${parts.slice(0, index + 1).join('/')}` }));
    return crumbs;
  }
  // Windows: the first part is the drive (or the UNC server), shown without its trailing separator.
  const uncPrefix = path.startsWith('\\\\') ? '\\\\' : '';
  return parts.map((part, index) => {
    const joined = uncPrefix + parts.slice(0, index + 1).join(separator);
    return { label: part, path: index === 0 ? joined + separator : joined };
  });
}

/** Return the index of the next entry after *from* whose name starts with *letter*, wrapping; -1 when none does. */
export function jumpToLetter(entries: BrowseEntry[], letter: string, from: number): number {
  const lower = letter.toLowerCase();
  for (let step = 1; step <= entries.length; step += 1) {
    const index = (from + step) % entries.length;
    if (entries[index].name.toLowerCase().startsWith(lower)) return index;
  }
  return -1;
}

/** Format a file size for the list: whole bytes and KB, one decimal below 10 MB or GB. */
export function formatBytes(bytes: number): string {
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  const digits = unit >= 2 && value < 10 ? 1 : 0;
  return `${value.toFixed(digits)} ${units[unit]}`;
}

/** Format a modification time (seconds since the epoch) as a local date, YYYY-MM-DD. */
export function formatModified(seconds: number): string {
  const date = new Date(seconds * 1000);
  const pad = (value: number) => String(value).padStart(2, '0');
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}`;
}

function readJson(key: string): unknown {
  try {
    const raw = localStorage.getItem(key);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

function writeJson(key: string, value: unknown): void {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // Storage can be unavailable in private or restricted browser contexts.
  }
}

/** Return the paths last chosen for *purpose*, newest first. */
export function readRecentPaths(purpose: BrowsePurposeName): string[] {
  const stored = readJson(RECENT_KEY);
  const list = stored && typeof stored === 'object' ? (stored as Record<string, unknown>)[purpose] : null;
  return Array.isArray(list) ? list.filter((item): item is string => typeof item === 'string').slice(0, RECENT_LIMIT) : [];
}

/** Put *path* first in the recent paths for *purpose*, keeping the last five. */
export function rememberRecentPath(purpose: BrowsePurposeName, path: string): void {
  const stored = readJson(RECENT_KEY);
  const all = stored && typeof stored === 'object' ? { ...(stored as Record<string, unknown>) } : {};
  all[purpose] = [path, ...readRecentPaths(purpose).filter((item) => item !== path)].slice(0, RECENT_LIMIT);
  writeJson(RECENT_KEY, all);
}

export function readShowHidden(): boolean {
  return readJson(HIDDEN_KEY) === true;
}

export function writeShowHidden(value: boolean): void {
  writeJson(HIDDEN_KEY, value);
}
