// Module-level download lane using Svelte 5 runes.
// One poller keeps the lane current for every page that shows downloads (Models, Workspace); the request flow
// (plan, free-space confirmation, start) and the done/failed toasts live here so they outlast the page that began them.

import { cancelDownload, getDownloadPlan, getDownloads, importLoraHF, startDownload } from '$lib/api/models';
import { startDiscoverDownload, type DiscoverSource } from '$lib/api/discover';
import { errorText, isRetryable } from '$lib/api/client';
import { requestConfirm } from '$lib/components/molecules/confirm.svelte';
import type { DownloadItem } from '$lib/types';
import { downloadConfirmText } from './downloadConfirm';
import { addToast } from './toasts.svelte';

export type DownloadKind = DownloadItem['kind'];
/** The kinds the Models page plans and starts itself; a LoRA is started by Discover or Import. */
export type PlannedKind = Exclude<DownloadKind, 'lora'>;

/** How a `request` ended: `present` when nothing was left to fetch, `declined` when the user said no (or the row was busy). */
export type DownloadRequestResult = 'queued' | 'present' | 'declined' | 'failed';

const POLL_MS = 1000;

/** The lane's items, newest state from the server; finished ones stay until the server drops them. */
let _items = $state<DownloadItem[]>([]);
/** Rows (by key) whose plan is being read or whose download is being requested or cancelled. */
let _busy = $state<ReadonlySet<string>>(new Set());
/** Ids of finished LoRA downloads whose file the page gave up waiting to see listed. */
let _unlisted = $state<ReadonlySet<string>>(new Set());
/** Names of the LoRAs whose failed download the Workspace shows in its own error slot, so no toast says it again. */
let _inlineFailures: ReadonlySet<string> = new Set();
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

/**
 * What starts a LoRA's download again: the request it started with. An Import (which downloads with the user's login) goes
 * through the Import route; a Discover download repeats its curated entry or repository file, anonymously. Null when the
 * item does not say where it came from.
 */
function loraRetry(item: DownloadItem): (() => Promise<DownloadItem | null>) | null {
  if (item.anonymous === false) {
    if (!item.source_repo || !item.source_file) return null;
    const request = { repo_id: item.source_repo, filename: item.source_file, name: item.name };
    return async () => (await importLoraHF(request)).download;
  }
  const source: DiscoverSource | null = item.curated_id ? { curated_id: item.curated_id } : item.source_repo && item.source_file ? { repo: item.source_repo, file: item.source_file, name: item.name } : null;
  return source ? async () => (await startDiscoverDownload(source)).item ?? null : null;
}

/** Raise an error toast whose Retry runs the Download flow again. */
function addRetryToast(message: string, item: Pick<DownloadItem, 'kind' | 'name' | 'label'> & Partial<DownloadItem>): void {
  const run = item.kind === 'lora' ? () => void downloadStore.retryLora(item as DownloadItem) : () => void downloadStore.request(item.kind as PlannedKind, item.name, item.label);
  addToast(message, 'error', { action: { label: 'Retry', run } });
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
    if (item.kind === 'lora' && _inlineFailures.has(item.name)) return;
    addRetryToast(item.message || `Downloading ${item.label} failed.`, item);
  }
}

/** Add or replace one item the server answered with, keeping the rest of the lane. */
function applyItem(item: DownloadItem): void {
  if (!_knownStates.has(item.id)) _knownStates.set(item.id, 'queued');
  _items = [..._items.filter((other) => other.id !== item.id), item];
  reportTransition(item);
}

/**
 * The failed or cancelled LoRA downloads the page holds that a lane read no longer lists, for LoRAs the read says nothing
 * else about. The server drops finished items after a while (and all of them on a restart), but a failed LoRA chip must
 * keep holding Generate back until the user retries the download, deletes it or removes the chip.
 */
function keptLoraFailures(items: readonly DownloadItem[]): DownloadItem[] {
  return _items.filter((kept) => kept.kind === 'lora' && (kept.state === 'failed' || kept.state === 'cancelled') && !items.some((item) => item.id === kept.id || (item.kind === 'lora' && item.name === kept.name)));
}

function applyItems(items: DownloadItem[]): void {
  _items = [...items, ...keptLoraFailures(items)];
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

  /** Ids of finished LoRA downloads whose file never appeared in the LoRA list; their chips count as ready. */
  get unlisted(): ReadonlySet<string> { return _unlisted; },

  /** Record that the page stopped waiting for the file of the finished download *id* to be listed. */
  markUnlisted(id: string): void {
    _unlisted = new Set([..._unlisted, id]);
  },

  /** The names of the LoRAs the lane knows: downloading, failed, or done and not listed yet. */
  loraNames(): string[] {
    return _items.filter((item) => item.kind === 'lora').map((item) => item.name);
  },

  /**
   * Say which LoRAs a page shows a failed download for in its own place (the Workspace's error slot): their failure
   * raises no toast. Pass an empty list when that place goes away, so a failure on any other page is announced again.
   */
  showFailuresInline(names: Iterable<string>): void {
    _inlineFailures = new Set(names);
  },

  /** Take an item the server just answered with into the lane at once, and read the lane again. */
  adopt(item: DownloadItem): void {
    applyItem(item);
    void refresh();
  },

  /** Forget the finished downloads of the LoRA *name* once its file is deleted, so it can be downloaded again and nothing waits for the file. */
  forgetLora(name: string): void {
    const gone = new Set(_items.filter((item) => item.kind === 'lora' && item.name === name && !isActive(item)).map((item) => item.id));
    if (gone.size === 0) return;
    _items = _items.filter((item) => !gone.has(item.id));
    _unlisted = new Set([..._unlisted].filter((id) => !gone.has(id)));
    // A read that began before the delete may still bring the finished item back: read again once it is over.
    void refresh();
  },

  /**
   * Start a failed LoRA download again with the request it started with (an Import through the Import route, a Discover
   * download through Discover); the new item replaces the failed one.
   */
  async retryLora(item: DownloadItem): Promise<boolean> {
    const restart = loraRetry(item);
    if (!restart) {
      addToast(`Could not retry ${item.label}. Download it again from Discover.`, 'error');
      return false;
    }
    const key = rowKey('lora', item.name);
    if (_busy.has(key)) return false;
    setBusy(key, true);
    try {
      const started = await restart();
      if (started) this.adopt(started);
      return true;
    } catch (err) {
      addToast(errorText(err, `Could not download ${item.label}.`), 'error');
      return false;
    } finally {
      setBusy(key, false);
    }
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
  async request(kind: PlannedKind, name: string, label: string): Promise<DownloadRequestResult> {
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
      if (isRetryable(err)) addRetryToast(message, { kind, name, label: shown });
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
    _unlisted = new Set();
    _inlineFailures = new Set();
    _seeded = false;
    _knownStates.clear();
    _downloadedCallbacks.clear();
  },
};
