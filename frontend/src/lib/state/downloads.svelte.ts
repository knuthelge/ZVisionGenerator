// Module-level download lane using Svelte 5 runes.
// One poller keeps the lane current for every page that shows downloads (Models, Workspace); the request flow
// (plan, free-space confirmation, start) and the done/failed toasts live here so they outlast the page that began them.

import { cancelDownload, getDownloadPlan, getDownloads, startDownload } from '$lib/api/models';
import { errorText, isRetryable } from '$lib/api/client';
import { requestConfirm } from '$lib/components/molecules/confirm.svelte';
import type { DownloadItem } from '$lib/types';
import { downloadConfirmText } from './downloadConfirm';
import { addToast } from './toasts.svelte';

export type DownloadKind = DownloadItem['kind'];

/** How a `request` ended: `present` when nothing was left to fetch, `declined` when the user said no (or the row was busy). */
export type DownloadRequestResult = 'queued' | 'present' | 'declined' | 'failed';

const POLL_MS = 1000;

/** The lane's items, newest state from the server; finished ones stay until the server drops them. */
let _items = $state<DownloadItem[]>([]);
/** Rows (by key) whose plan is being read or whose download is being requested or cancelled. */
let _busy = $state<ReadonlySet<string>>(new Set());
/** Whether the lane has been read once: what was already final at that point is not announced. */
let _seeded = $state(false);

let _syncCount = 0;
/** Bumped by `reset` so reads and cleanups from before it do nothing. */
let _epoch = 0;
let _timer: ReturnType<typeof setTimeout> | null = null;
let _reading = false;
let _readAgain = false;
const _knownStates = new Map<string, DownloadItem['state']>();
const _downloadedCallbacks = new Set<() => void>();

function isActive(item: DownloadItem): boolean {
  return item.state === 'queued' || item.state === 'running';
}

function rowKey(kind: DownloadKind, name: string): string {
  return `${kind}:${name}`;
}

function setBusy(key: string, busy: boolean): void {
  const next = new Set(_busy);
  if (busy) next.add(key);
  else next.delete(key);
  _busy = next;
}

function notifyDownloaded(): void {
  for (const callback of [..._downloadedCallbacks]) callback();
}

/** Raise an error toast whose Retry runs the Download flow again. */
function addRetryToast(message: string, kind: DownloadKind, name: string, label: string): void {
  addToast(message, 'error', { action: { label: 'Retry', run: () => void downloadStore.request(kind, name, label) } });
}

/** Report an item that was seen queued or running and has since finished or failed. The first read only records states. */
function reportTransition(item: DownloadItem): void {
  const before = _knownStates.get(item.id);
  _knownStates.set(item.id, item.state);
  if (!_seeded || before === item.state || !(before === 'queued' || before === 'running')) return;
  if (item.state === 'done') {
    addToast(`${item.label} downloaded`, 'success');
    notifyDownloaded();
  } else if (item.state === 'failed') {
    addRetryToast(item.message || `Downloading ${item.label} failed.`, item.kind, item.name, item.label);
  }
}

/** Add or replace one item the server answered with, keeping the rest of the lane. */
function applyItem(item: DownloadItem): void {
  if (!_knownStates.has(item.id)) _knownStates.set(item.id, 'queued');
  _items = [..._items.filter((other) => other.id !== item.id), item];
  reportTransition(item);
}

function applyItems(items: DownloadItem[]): void {
  _items = items;
  for (const item of items) reportTransition(item);
  _seeded = true;
}

/** Read the lane; while a page is syncing and anything is queued or running, read it again in a second. */
async function refresh(): Promise<void> {
  if (_reading) {
    // A read already on its way may predate what the caller just changed: read again right after it.
    _readAgain = true;
    return;
  }
  _reading = true;
  if (_timer) clearTimeout(_timer);
  _timer = null;
  const epoch = _epoch;
  try {
    const { items } = await getDownloads();
    if (epoch === _epoch) applyItems(items);
  } catch {
    // A missed read says nothing about the downloads: try again while one is known to be active.
  }
  if (epoch !== _epoch) return;
  _reading = false;
  if (_readAgain) {
    _readAgain = false;
    await refresh();
    return;
  }
  if (_syncCount > 0 && _items.some(isActive)) _timer = setTimeout(() => void refresh(), POLL_MS);
}

function onVisibilityChange(): void {
  if (!document.hidden) void refresh();
}

function onFocus(): void {
  void refresh();
}

export const downloadStore = {
  /** The lane's items: queued, running and recently finished. */
  get items(): DownloadItem[] { return _items; },
  /** Whether the lane has been read at least once. */
  get seeded(): boolean { return _seeded; },

  /** The queued or running item for a row, or null. */
  active(kind: DownloadKind, name: string): DownloadItem | null {
    return _items.find((item) => isActive(item) && item.kind === kind && item.name === name) ?? null;
  },

  /** Whether a request or cancel for the row is on its way. */
  busy(kind: DownloadKind, name: string): boolean {
    return _busy.has(rowKey(kind, name));
  },

  refresh,

  /**
   * Keep the lane current while a page shows downloads: read now, on focus, and every second while anything is
   * active. Pages share one poller; the last cleanup stops it.
   */
  startSync(): () => void {
    _syncCount += 1;
    if (_syncCount === 1) {
      window.addEventListener('focus', onFocus);
      document.addEventListener('visibilitychange', onVisibilityChange);
    }
    if (!_reading) void refresh();
    const epoch = _epoch;
    let stopped = false;
    return () => {
      if (stopped || epoch !== _epoch) return;
      stopped = true;
      _syncCount -= 1;
      if (_syncCount > 0) return;
      if (_timer) clearTimeout(_timer);
      _timer = null;
      window.removeEventListener('focus', onFocus);
      document.removeEventListener('visibilitychange', onVisibilityChange);
    };
  },

  /**
   * Read the plan, ask the user to confirm its size and free space, then queue the download.
   * *label* names the row in the error toast until the plan supplies its own.
   */
  async request(kind: DownloadKind, name: string, label: string): Promise<DownloadRequestResult> {
    const key = rowKey(kind, name);
    if (_busy.has(key)) return 'declined';
    setBusy(key, true);
    let shown = label;
    try {
      const plan = await getDownloadPlan(kind, name);
      shown = plan.label;
      if (plan.parts.length === 0) {
        addToast(`${plan.label} is already downloaded.`, 'info');
        notifyDownloaded();
        return 'present';
      }
      const { info, confirmLabel } = downloadConfirmText(plan);
      const approved = await requestConfirm({ question: `Download ${plan.label}?`, info, confirmLabel, danger: false });
      if (!approved) return 'declined';
      applyItem(await startDownload(kind, name));
      void refresh();
      return 'queued';
    } catch (err) {
      const message = errorText(err, `Could not download ${shown}.`);
      if (isRetryable(err)) addRetryToast(message, kind, name, shown);
      else addToast(message, 'error');
      return 'failed';
    } finally {
      setBusy(key, false);
    }
  },

  /** Cancel a queued or running download; a running one that a job started keeps downloading for the job. */
  async cancel(item: DownloadItem): Promise<void> {
    const key = rowKey(item.kind, item.name);
    if (_busy.has(key)) return;
    setBusy(key, true);
    try {
      applyItem(await cancelDownload(item.id));
    } catch (err) {
      addToast(errorText(err, 'Could not cancel the download.'), 'error');
    } finally {
      setBusy(key, false);
    }
  },

  /** Call *callback* whenever a download this store saw queued or running finishes, or a request finds nothing left to fetch. */
  onDownloaded(callback: () => void): () => void {
    _downloadedCallbacks.add(callback);
    return () => {
      _downloadedCallbacks.delete(callback);
    };
  },

  /** Forget the lane and every subscriber, for tests. */
  reset(): void {
    if (_timer) clearTimeout(_timer);
    _timer = null;
    if (_syncCount > 0) {
      window.removeEventListener('focus', onFocus);
      document.removeEventListener('visibilitychange', onVisibilityChange);
    }
    _syncCount = 0;
    _epoch += 1;
    _reading = false;
    _readAgain = false;
    _items = [];
    _busy = new Set();
    _seeded = false;
    _knownStates.clear();
    _downloadedCallbacks.clear();
  },
};
