import type { JobMemory } from '$lib/types';

/** The estimated generation time of a clip or an extension, in whole minutes and labelled as an estimate; null where the server has none (only MiniMax H3 does). */
export function clipTime(memory: Pick<JobMemory, 'expected_seconds'> | null): string | null {
  const seconds = memory?.expected_seconds;
  if (seconds === undefined || seconds === null || !(seconds > 0)) return null;
  return `About ${Math.max(1, Math.round(seconds / 60))} min (estimated).`;
}
