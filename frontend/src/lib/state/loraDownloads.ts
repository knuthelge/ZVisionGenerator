import type { DownloadItem, LoraInfo } from '$lib/types';
import type { LoraChip } from './loras';

/** What a LoRA chip shows about its file: there, on its way, or a download that did not finish. */
export type ChipDownload =
  | { state: 'ready' }
  | { state: 'downloading'; item: DownloadItem | null; percent: number | null }
  | { state: 'failed'; item: DownloadItem; reason: string };

function isActive(item: DownloadItem): boolean {
  return item.state === 'queued' || item.state === 'running';
}

/** Order lane items by when they were submitted: the lane numbers its ids `dl-<n>`, counting up, so a retry has the larger number. */
function submitted(item: DownloadItem): number {
  const order = Number(item.id.split('-').pop());
  return Number.isFinite(order) ? order : -1;
}

function newest(items: readonly DownloadItem[]): DownloadItem | null {
  return items.reduce<DownloadItem | null>((found, item) => (found === null || submitted(item) >= submitted(found) ? item : found), null);
}

/**
 * The lane item that speaks for a LoRA: the newest queued or running one, else the newest finished one. The newest is
 * found by the item's id, not its place in the list: the server lists finished items newest first, while an item the
 * page was just handed is added last.
 */
export function loraItem(name: string, items: readonly DownloadItem[]): DownloadItem | null {
  const named = items.filter((item) => item.kind === 'lora' && item.name === name);
  return newest(named.filter(isActive)) ?? newest(named);
}

/** How far a download is, 0 to 100, or null while its size is not known. */
export function downloadPercent(item: DownloadItem): number | null {
  if (item.state === 'queued') return 0;
  if (item.state === 'done') return 100;
  const total = item.bytes_total ?? item.size_bytes ?? null;
  if (item.bytes_done === undefined || !total || total <= 0) return null;
  return Math.max(0, Math.min(100, Math.floor((item.bytes_done / total) * 100)));
}

/** The ids of finished downloads whose file the page stopped waiting to see listed. */
export type UnlistedItems = ReadonlySet<string>;

const NONE_UNLISTED: UnlistedItems = new Set();

/**
 * Tell what a chip's file is doing. A file that is installed is ready whatever the lane says. A download that is
 * done but whose file the page has not listed yet still counts as on its way, so Generate never races the refresh,
 * until the page gives up waiting (*unlisted* holds its id): then the chip is ready and the server checks the file
 * when Generate is pressed.
 */
export function chipDownload(name: string, installed: readonly LoraInfo[], items: readonly DownloadItem[], unlisted: UnlistedItems = NONE_UNLISTED): ChipDownload {
  if (installed.some((lora) => lora.name === name)) return { state: 'ready' };
  const item = loraItem(name, items);
  if (!item) return { state: 'ready' };
  if (item.state === 'done' && unlisted.has(item.id)) return { state: 'ready' };
  if (item.state === 'failed') return { state: 'failed', item, reason: item.message || `${name} didn't download.` };
  if (item.state === 'cancelled') return { state: 'failed', item, reason: `The download of ${name} was cancelled.` };
  return { state: 'downloading', item, percent: downloadPercent(item) };
}

/** The first download Generate waits for, and every failure that blocks it. */
export interface LoraBlock {
  /** The first chip still downloading; null while any chip has failed, since a failure is what the user must act on. */
  waiting: { name: string; percent: number | null } | null;
  failures: { name: string; reason: string }[];
  /** Generate cannot run: a download is on its way or has failed. */
  blocked: boolean;
}

export function loraBlock(chips: readonly LoraChip[], installed: readonly LoraInfo[], items: readonly DownloadItem[], unlisted: UnlistedItems = NONE_UNLISTED): LoraBlock {
  let waiting: LoraBlock['waiting'] = null;
  const failures: LoraBlock['failures'] = [];
  for (const chip of chips) {
    const status = chipDownload(chip.name, installed, items, unlisted);
    if (status.state === 'downloading' && !waiting) waiting = { name: chip.name, percent: status.percent };
    if (status.state === 'failed') failures.push({ name: chip.name, reason: status.reason });
  }
  // A failed chip holds Generate back for good until the user retries or removes it, so it wins over a download still on its way.
  return { waiting: failures.length > 0 ? null : waiting, failures, blocked: waiting !== null || failures.length > 0 };
}

/** What Generate reads while it waits: "Waiting for name · 40 %" (name cut by the layout), or on a phone "Waiting · 40 %". */
export function waitText(waiting: NonNullable<LoraBlock['waiting']>, compact: boolean): { lead: string; percent: string; full: string } {
  const lead = compact ? 'Waiting' : `Waiting for ${waiting.name}`;
  const percent = waiting.percent === null ? '' : `${waiting.percent} %`;
  const full = `Waiting for ${waiting.name}${percent ? ` · ${percent}` : ''}`;
  return { lead, percent, full };
}

/** The sentence shown above Generate for a failed download. */
export function failureText(failure: { name: string }): string {
  return `${failure.name} didn't download. Retry it or remove it from LoRAs.`;
}
