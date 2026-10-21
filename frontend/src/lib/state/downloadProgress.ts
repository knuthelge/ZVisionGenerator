import type { DownloadProgressFields } from '$lib/types';

type DownloadLineFields = Pick<DownloadProgressFields, 'bytes_done' | 'bytes_total' | 'bytes_per_sec' | 'download_eta_secs'>;

const GB = 1e9;
const MB = 1e6;

function unitFor(bytes: number): { divisor: number; unit: 'GB' | 'MB' | 'KB'; digits: number } {
  if (bytes >= GB) return { divisor: GB, unit: 'GB', digits: 1 };
  if (bytes >= MB) return { divisor: MB, unit: 'MB', digits: 0 };
  return { divisor: 1e3, unit: 'KB', digits: 0 };
}

function speedLabel(bytesPerSec: number): string {
  const mb = bytesPerSec / MB;
  return `${mb.toFixed(mb < 10 ? 1 : 0)} MB/s`;
}

function etaLabel(secs: number): string {
  if (secs < 60) return 'less than a minute';
  const minutes = Math.round(secs / 60);
  if (secs < 90 * 60) return `about ${minutes} min`;
  const hours = Math.floor(minutes / 60);
  return `about ${hours} h ${minutes % 60} min`;
}

/** A byte count in decimal units like the server's disk size labels: "34.7 GB", "92 MB" or "4 KB". */
export function sizeLabel(bytes: number): string {
  const { divisor, unit, digits } = unitFor(bytes);
  const amount = unit === 'KB' ? Math.max(1, Math.round(bytes / divisor)) : (bytes / divisor).toFixed(digits);
  return `${amount} ${unit}`;
}

/** One-line byte progress, e.g. "12.4 / 34.7 GB · 48 MB/s · about 8 min"; unknown speed, ETA or total segments are left out. */
export function downloadLine(p: DownloadLineFields): string {
  const { divisor, unit, digits } = unitFor(p.bytes_total ?? p.bytes_done);
  const done = (p.bytes_done / divisor).toFixed(digits);
  const amount = p.bytes_total === null ? `${done} ${unit}` : `${done} / ${(p.bytes_total / divisor).toFixed(digits)} ${unit}`;
  const segments = [amount];
  if (p.bytes_per_sec !== null) segments.push(speedLabel(p.bytes_per_sec));
  if (p.bytes_total !== null && p.download_eta_secs !== null) segments.push(etaLabel(p.download_eta_secs));
  return segments.join(' · ');
}

/** "part 1 of 2" for a download with several parts, otherwise an empty string. */
export function partLabel(p: Pick<DownloadProgressFields, 'part' | 'parts'>): string {
  return p.parts > 1 ? `part ${p.part} of ${p.parts}` : '';
}
