import { api } from './client';
import type { RecentPrompt, WorkflowMode } from '$lib/types';

/** Fetch the newest recent prompts of *mode* whose prompt or scene contains *query* (all of them when blank), newest first. */
export function getRecentPrompts(query: string, mode: WorkflowMode): Promise<{ entries: RecentPrompt[] }> {
  const params = new URLSearchParams({ q: query, mode });
  return api.get<{ entries: RecentPrompt[] }>(`/api/recent-prompts?${params}`);
}

/** Remove every recent prompt of *mode*; resolves to how many there were. */
export function clearRecentPrompts(mode: WorkflowMode): Promise<{ deleted_count: number }> {
  return api.delete(`/api/recent-prompts?${new URLSearchParams({ mode })}`);
}
