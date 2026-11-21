import { api } from './client';
import type { JobMemory } from '$lib/types';
import type { MemoryRequest } from '$lib/state/memoryRequest';

/** Ask whether the job a form would submit fits this machine's memory, level by level. Nothing is queued; null when the machine has no memory budget or the model's weights are unknown. */
export function checkMemory(body: MemoryRequest): Promise<JobMemory | null> {
  return api.post<JobMemory | null>('/api/memory', body);
}
