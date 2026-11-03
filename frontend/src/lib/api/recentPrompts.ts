import { api } from './client';
import type { RecentMode, RecentPrompt } from '$lib/types';

/** Fetch the newest recent prompts of *mode* (an audio kind for the Audio tab) whose prompt or scene contains *query* (all of them when blank), newest first. */
export function getRecentPrompts(query: string, mode: RecentMode): Promise<{ entries: RecentPrompt[] }> {
  return api.get<{ entries: RecentPrompt[] }>(`/api/recent-prompts?${new URLSearchParams({ q: query, mode })}`);
}

/** Remove every recent prompt of *mode* (an audio kind for the Audio tab); resolves to how many there were. */
export function clearRecentPrompts(mode: RecentMode): Promise<{ deleted_count: number }> {
  return api.delete(`/api/recent-prompts?${new URLSearchParams({ mode })}`);
}
