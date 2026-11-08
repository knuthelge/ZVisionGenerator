import type { BrowseEntry, BrowseInfo, BrowsePreviewKind, BrowsePurposeName } from '$lib/types';
import { formatClock } from './clock';

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
const VIEW_KEY = 'ziv-browse-view-v1';
/** The first order, until the user picks another: the file just made or downloaded is on top. */
export const DEFAULT_BROWSE_VIEW: BrowseView = { sort: 'modified', dir: 'desc', view: 'list', preview: true };
const RECENT_LIMIT = 5;
const COMPLETION_LIMIT = 8;
// Absolute host paths: POSIX, home-relative, Windows drive or UNC.
const ABSOLUTE_RE = /^(\/|~|[A-Za-z]:[\\/]|\\\\)/;

/** Return the folder a picker opens in when its field is empty: the picked project's folder for media a project holds, else null (the purpose's own place). */
export function pickerStart(purpose: BrowsePurposeName, projectPath: string | null): string | null {
  return purpose === 'reference_image' || purpose === 'audio_track' || purpose === 'source_video' ? projectPath : null;
}

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

const pad = (value: number) => String(value).padStart(2, '0');

function dayStamp(date: Date): string {
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}`;
}

function clockStamp(date: Date): string {
  return `${pad(date.getHours())}:${pad(date.getMinutes())}`;
}

/** Format a modification time (seconds since the epoch) in local time: "Today 09:12", "Yesterday 21:40", else "2026-11-03 14:05". */
export function formatModified(seconds: number, now: Date = new Date()): string {
  const date = new Date(seconds * 1000);
  const day = dayStamp(date);
  if (day === dayStamp(now)) return `Today ${clockStamp(date)}`;
  if (day === dayStamp(new Date(now.getFullYear(), now.getMonth(), now.getDate() - 1))) return `Yesterday ${clockStamp(date)}`;
  return `${day} ${clockStamp(date)}`;
}

/** Format a modification time for a grid tile: Today and Yesterday as `formatModified`, this year's older dates as "3 Nov 14:05", other years in full. */
export function formatModifiedShort(seconds: number, now: Date = new Date()): string {
  const label = formatModified(seconds, now);
  const date = new Date(seconds * 1000);
  if (!label.startsWith(`${now.getFullYear()}-`)) return label;
  return `${date.getDate()} ${date.toLocaleString('en-GB', { month: 'short' })} ${clockStamp(date)}`;
}

/** Format a modification time (seconds since the epoch) as the full local date and time with seconds, "2026-11-03 14:05:09". */
export function formatModifiedFull(seconds: number): string {
  const date = new Date(seconds * 1000);
  return `${dayStamp(date)} ${clockStamp(date)}:${pad(date.getSeconds())}`;
}

/** Shorten *name* to at most *max* characters by cutting the middle, so both the start and the end (with the extension) stay readable. */
export function middleTruncate(name: string, max: number): string {
  if (name.length <= max || max < 3) return name;
  const kept = max - 1;
  const head = Math.ceil(kept / 2);
  return `${name.slice(0, head)}…${name.slice(name.length - (kept - head))}`;
}

export type BrowseSortKey = 'name' | 'size' | 'modified';
export type BrowseSortDir = 'asc' | 'desc';

/** What the listing is ordered by: a column and a direction. */
export interface BrowseOrder {
  sort: BrowseSortKey;
  dir: BrowseSortDir;
}

/** The remembered look of the browser, shared by every picker. */
export interface BrowseView extends BrowseOrder {
  view: 'list' | 'grid';
  preview: boolean;
}

const nameCollator = new Intl.Collator(undefined, { numeric: true, sensitivity: 'base' });

/** Compare two names the way a person reads them: digits as numbers (`take-2` before `take-10`), ignoring case. */
function compareNames(a: string, b: string): number {
  return nameCollator.compare(a, b) || (a < b ? -1 : a > b ? 1 : 0);
}

/** Return *entries* in *order* as a new list: folders first in every order, and names compared naturally as the tie-break. */
export function sortEntries(entries: BrowseEntry[], order: BrowseOrder): BrowseEntry[] {
  const sign = order.dir === 'asc' ? 1 : -1;
  return [...entries].sort((a, b) => {
    const aFolder = a.kind === 'directory';
    if (aFolder !== (b.kind === 'directory')) return aFolder ? -1 : 1;
    if (order.sort === 'name') return sign * compareNames(a.name, b.name);
    // Folders carry no size, so by size they fall back to name order.
    if (order.sort === 'size' && aFolder) return compareNames(a.name, b.name);
    const key = order.sort === 'size' ? 'size' : 'modified';
    return sign * ((a[key] ?? 0) - (b[key] ?? 0)) || compareNames(a.name, b.name);
  });
}

/** Return the order a click on the *key* heading gives: the same column reverses; another column starts at its natural direction (names A-Z, size and date largest or newest first). */
export function orderAfterClick(current: BrowseOrder, key: BrowseSortKey): BrowseOrder {
  if (current.sort === key) return { sort: key, dir: current.dir === 'asc' ? 'desc' : 'asc' };
  return { sort: key, dir: key === 'name' ? 'asc' : 'desc' };
}

/** One of the six orders the sort menu offers. */
export interface SortChoice {
  id: string;
  label: string;
  order: BrowseOrder;
}

/** The six orders of the sort menu above the grid (and the phone's sort sheet), in menu order. */
export const SORT_CHOICES: readonly SortChoice[] = [
  { id: 'name-asc', label: 'Name A–Z', order: { sort: 'name', dir: 'asc' } },
  { id: 'name-desc', label: 'Name Z–A', order: { sort: 'name', dir: 'desc' } },
  { id: 'modified-desc', label: 'Newest first', order: { sort: 'modified', dir: 'desc' } },
  { id: 'modified-asc', label: 'Oldest first', order: { sort: 'modified', dir: 'asc' } },
  { id: 'size-desc', label: 'Largest first', order: { sort: 'size', dir: 'desc' } },
  { id: 'size-asc', label: 'Smallest first', order: { sort: 'size', dir: 'asc' } },
];

/** Return the menu label of *order* ("Newest first"). */
export function sortChoiceLabel(order: BrowseOrder): string {
  return (SORT_CHOICES.find((choice) => choice.order.sort === order.sort && choice.order.dir === order.dir) ?? SORT_CHOICES[2]).label;
}

/** Return how many items a listing holds, with the noun of the picker: "10 images", "1 audio file", "3 folders", "2 files". */
export function itemCountLabel(count: number, noun: BrowsePreviewKind | 'folder' | null): string {
  const names: Record<BrowsePreviewKind | 'folder' | 'file', [string, string]> = {
    image: ['image', 'images'],
    audio: ['audio file', 'audio files'],
    video: ['video', 'videos'],
    folder: ['folder', 'folders'],
    file: ['file', 'files'],
  };
  const [one, many] = names[noun ?? 'file'];
  return `${count} ${count === 1 ? one : many}`;
}

/** Return the index *key* moves to in a grid of *columns* columns holding *count* tiles: left and right by one, up and down by a row, staying put at an edge (down from a short last row lands on the last tile). */
export function gridMove(index: number, key: string, columns: number, count: number): number {
  if (count <= 0 || columns < 1) return index;
  const last = count - 1;
  switch (key) {
    case 'ArrowLeft':
      return Math.max(0, index - 1);
    case 'ArrowRight':
      return Math.min(last, index + 1);
    case 'ArrowUp':
      return index - columns >= 0 ? index - columns : index;
    case 'ArrowDown':
      if (index + columns <= last) return index + columns;
      return Math.floor(index / columns) === Math.floor(last / columns) ? index : last;
    default:
      return index;
  }
}

/** One labelled fact in the preview. */
export interface PreviewFact {
  label: string;
  value: string;
}

const MISSING = '—';

function fileType(name: string): string {
  const dot = name.lastIndexOf('.');
  return dot > 0 ? name.slice(dot + 1).toUpperCase() : '';
}

function pixels(info: BrowseInfo | null): string {
  return info?.width && info.height ? `${info.width} × ${info.height}` : MISSING;
}

function channelLabel(channels: number): string {
  if (channels === 1) return 'mono';
  if (channels === 2) return 'stereo';
  return `${channels} ch`;
}

/**
 * Return the facts the preview lists for a file of *kind*: the file's own (size, type, date) from *entry* and the rest from *info*.
 *
 * A fact *info* has not delivered (yet, or at all) reads "—", so the list keeps its height while the facts load.
 */
export function previewFacts(kind: BrowsePreviewKind, entry: Pick<BrowseEntry, 'name' | 'size' | 'modified'>, info: BrowseInfo | null): PreviewFact[] {
  const type = fileType(entry.name);
  const bytes = entry.size === null ? MISSING : formatBytes(entry.size);
  const length = info?.seconds != null ? formatClock(info.seconds) : MISSING;
  if (kind === 'audio') {
    const parts = [type, info?.sample_rate ? `${Number((info.sample_rate / 1000).toFixed(1))} kHz` : '', info?.channels ? channelLabel(info.channels) : ''].filter(Boolean);
    return [
      { label: 'Length', value: length },
      { label: 'Format', value: parts.length > 0 ? parts.join(' · ') : MISSING },
      { label: 'File', value: bytes },
    ];
  }
  const file = { label: 'File', value: [bytes, type].filter(Boolean).join(' · ') };
  if (kind === 'video') {
    return [
      { label: 'Length', value: length },
      { label: 'Frames', value: info?.frames != null ? String(info.frames) : MISSING },
      { label: 'Frame rate', value: info?.fps ? `${Number(info.fps.toFixed(2))} fps` : MISSING },
      { label: 'Pixels', value: pixels(info) },
      file,
    ];
  }
  return [
    { label: 'Pixels', value: pixels(info) },
    file,
    { label: 'Modified', value: entry.modified === null ? MISSING : formatModifiedFull(entry.modified) },
  ];
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

/** Return the remembered order, list or grid, and preview switch; any field that is missing or invalid takes its default. */
export function readBrowseView(): BrowseView {
  const stored = readJson(VIEW_KEY);
  const fields = stored && typeof stored === 'object' ? (stored as Record<string, unknown>) : {};
  const defaults = DEFAULT_BROWSE_VIEW;
  return {
    sort: fields.sort === 'name' || fields.sort === 'size' || fields.sort === 'modified' ? fields.sort : defaults.sort,
    dir: fields.dir === 'asc' || fields.dir === 'desc' ? fields.dir : defaults.dir,
    view: fields.view === 'list' || fields.view === 'grid' ? fields.view : defaults.view,
    preview: typeof fields.preview === 'boolean' ? fields.preview : defaults.preview,
  };
}

export function writeBrowseView(view: BrowseView): void {
  writeJson(VIEW_KEY, view);
}
