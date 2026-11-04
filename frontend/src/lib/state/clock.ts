/** m:ss, or h:mm:ss from an hour. */
export function formatClock(secs: number): string {
  const rounded = Math.max(0, Math.round(Number.isFinite(secs) ? secs : 0));
  const h = Math.floor(rounded / 3600);
  const m = Math.floor((rounded % 3600) / 60);
  const s = rounded % 60;
  return h > 0 ? `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}` : `${m}:${String(s).padStart(2, '0')}`;
}
