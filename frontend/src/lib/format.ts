const MINUTE_S = 60;
const HOUR_S = 3600;
const DAY_S = 86400;
const RELATIVE_LIMIT_DAYS = 7;

/** Return the first non-blank line of a prompt, trimmed. */
export function promptFirstLine(prompt: string): string {
  return prompt.split(/\r?\n/).map((line) => line.trim()).find(Boolean) ?? '';
}

/** Describe how long ago *epochSecs* was at *nowMs*: "just now", "5 min ago", "3 h ago", "2 d ago", or the date after a week. */
export function timeAgo(epochSecs: number, nowMs: number): string {
  const elapsed = nowMs / 1000 - epochSecs;
  if (elapsed < MINUTE_S) return 'just now';
  if (elapsed < HOUR_S) return `${Math.floor(elapsed / MINUTE_S)} min ago`;
  if (elapsed < DAY_S) return `${Math.floor(elapsed / HOUR_S)} h ago`;
  const days = Math.floor(elapsed / DAY_S);
  if (days <= RELATIVE_LIMIT_DAYS) return `${days} d ago`;
  return new Date(epochSecs * 1000).toLocaleDateString();
}
